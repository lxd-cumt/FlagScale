# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configure FlagOS backend selection before backend-dependent initialization."""

import os
import sys


import warnings

# The TransformerEngine plugin reads every selection knob under the
# NVTE_PLUGIN_ prefix; MG_FL_PREFER is Megatron-side and keeps its own name.
# The third element is the argument's default, used only to tell an explicit
# user setting apart from argparse filling in a default.
_BACKEND_ENVIRONMENT_ARGUMENTS = (
    ("mg_fl_prefer", "MG_FL_PREFER", ""),
    ("nvte_plugin_prefer", "NVTE_PLUGIN_PREFER", "vendor"),
    ("nvte_plugin_per_op", "NVTE_PLUGIN_PER_OP", None),
    ("nvte_plugin_allow_vendors", "NVTE_PLUGIN_ALLOW_VENDORS", None),
    ("nvte_plugin_deny_vendors", "NVTE_PLUGIN_DENY_VENDORS", None),
)

# Master switch, injected by the runner from experiment.task.nvte_plugin. It has
# to predate the Transformer Engine import, so it cannot be set here.
_TE_PLUGIN_ENV = "NVTE_PLUGIN"

# Policy module of the TransformerEngine plugin package, whose env-backed policy
# is cached on first read.
_TE_POLICY_MODULE = "transformer_engine_plugin_fl.policy"


def plugin_enabled() -> bool:
    """Whether the TransformerEngine plugin is active in this process."""
    return bool(os.environ.get(_TE_PLUGIN_ENV, "").strip())


def configure_backend_environment(args) -> None:
    """Apply backend-selection arguments before the first FL-dispatched call.

    The TransformerEngine plugin reads its environment-backed selection policy
    lazily and caches it. Avoid importing TransformerEngine here, but invalidate
    the policy if it was imported earlier in the process so that the next
    dispatch sees the new environment.

    Without the plugin the NVTE_PLUGIN_* knobs have nothing to configure, so
    they are left unset rather than exported into an environment that no one
    reads. An explicitly configured knob is reported, because a setting that is
    silently ignored is worse than one that is refused.
    """
    plugin_on = plugin_enabled()
    ignored = []
    te_policy_configured = False

    for argument_name, environment_name, default in _BACKEND_ENVIRONMENT_ARGUMENTS:
        value = getattr(args, argument_name, None)
        if value is None or value == "":
            continue

        if not plugin_on and environment_name.startswith("NVTE_PLUGIN_"):
            if value != default:
                ignored.append(f"{argument_name}={value}")
            continue

        os.environ[environment_name] = str(value)
        if environment_name.startswith("NVTE_PLUGIN_"):
            te_policy_configured = True

    if ignored:
        warnings.warn(
            "TransformerEngine plugin is disabled, ignoring: "
            + ", ".join(ignored)
            + ". Set experiment.task.nvte_plugin in the training config to enable it.",
            RuntimeWarning,
            stacklevel=2,
        )

    _check_transformer_config_coupling(args)

    if not te_policy_configured:
        return

    policy_module = sys.modules.get(_TE_POLICY_MODULE)
    reset_global_policy = getattr(policy_module, "reset_global_policy", None)
    if callable(reset_global_policy):
        reset_global_policy()


# Arguments that TransformerConfig picks up by matching its own field name
# against an attribute of args.
_CONFIG_COUPLED_ARGUMENTS = ("nvte_plugin_prefer",)


def _check_transformer_config_coupling(args) -> None:
    """Fail loudly if a config field no longer matches its argument name.

    core_transformer_config_from_args() copies values with
    ``if hasattr(args, f.name)``, so a field whose name drifts from the
    argument's is skipped without an error and silently keeps its dataclass
    default. That turned a rename into a wrong-backend bug once already:
    nvte_plugin_prefer fell back to "vendor", and the fused CUDA wgrad path in
    tensor_parallel/layers.py ran even though the user asked for flagos.

    Checked here rather than in Megatron's generic copy loop so the assertion
    stays next to the arguments it is about.
    """
    try:
        import dataclasses

        from megatron.core.transformer.transformer_config import TransformerConfig
    except ImportError:
        # Megatron is optional for some entry points; nothing to verify.
        return

    field_names = {f.name for f in dataclasses.fields(TransformerConfig)}
    for argument_name in _CONFIG_COUPLED_ARGUMENTS:
        if not hasattr(args, argument_name):
            continue
        assert argument_name in field_names, (
            f"'{argument_name}' is a training argument but not a TransformerConfig"
            " field, so core_transformer_config_from_args() would silently drop it"
            " and the config would keep its default. Rename the field in"
            " megatron/core/transformer/transformer_config.py to match."
        )
