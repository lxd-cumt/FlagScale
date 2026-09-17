#!/usr/bin/env python3
"""Verify checkpoint directory naming matches Megatron-LM-FL conventions.

Tests save_megatron_release_checkpoint() output against the expected naming from
Megatron's get_checkpoint_name():
    PP>1:  mp_rank_{tp:02d}_{pp:03d}[_{ep:03d}]
    PP=1:  mp_rank_{tp:02d}[_{ep:03d}]

EP segment is included only when EP>1.
TP is always included (even when TP=1).
ETP never appears in directory names.

Usage:
    python test_checkpoint_naming.py
"""

import os
import shutil
import sys
import tempfile

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen35.config import Config
from qwen35.io import save_megatron_release_checkpoint


def _make_cfg(tmpdir, tp, pp, ep, expert_tp=None):
    """Create a minimal Config for naming tests."""
    if expert_tp is None:
        expert_tp = tp
    cfg_dict = {
        "tensor_model_parallel_size": tp,
        "pipeline_model_parallel_size": pp,
        "expert_model_parallel_size": ep,
        "expert_tensor_parallel_size": expert_tp,
        "num_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_query_groups": 2,
        "kv_channels": 16,
        "attention_output_gate": False,
        "untie_embeddings_and_output_weights": True,
        "linear_attention_freq": 999,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "num_experts": 4 if ep > 1 or expert_tp != tp else 0,
        "ffn_hidden_size": 128,
        "no_enable_vision": True,
        "mtp_num_layers": 0,
    }
    if cfg_dict["num_experts"] > 0:
        cfg_dict["moe_ffn_hidden_size"] = 32
        cfg_dict["moe_shared_expert_intermediate_size"] = 64

    yaml_path = os.path.join(tmpdir, "cfg.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(cfg_dict, f)
    return Config(yaml_path)


def _build_shards_dict(tp, pp, ep):
    """Build a minimal shards_dict with dummy tensors."""
    shards = {}
    dummy = torch.zeros(1)
    for pp_rank in range(pp):
        for tp_rank in range(tp):
            if ep > 1:
                for ep_rank in range(ep):
                    shards[(pp_rank, tp_rank, ep_rank)] = {"dummy": dummy}
            else:
                shards[(pp_rank, tp_rank)] = {"dummy": dummy}
    return shards


def _get_shard_dirs(release_dir):
    """List all mp_rank_* directory names under release/."""
    if not os.path.isdir(release_dir):
        return set()
    return {d for d in os.listdir(release_dir) if d.startswith("mp_rank_")}


def _expected_dirs(tp, pp, ep):
    """Generate the expected set of directory names per Megatron convention."""
    dirs = set()
    for pp_rank in range(pp):
        for tp_rank in range(tp):
            if ep > 1:
                for ep_rank in range(ep):
                    if pp > 1:
                        name = f"mp_rank_{tp_rank:02d}_{pp_rank:03d}_{ep_rank:03d}"
                    else:
                        name = f"mp_rank_{tp_rank:02d}_{ep_rank:03d}"
                    dirs.add(name)
            else:
                if pp > 1:
                    name = f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
                else:
                    name = f"mp_rank_{tp_rank:02d}"
                dirs.add(name)
    return dirs


def run_test(tp, pp, ep, expert_tp=None):
    """Run a single naming test."""
    if expert_tp is None:
        expert_tp = tp
    label = f"TP={tp}, PP={pp}, EP={ep}, ETP={expert_tp}"

    tmpdir = tempfile.mkdtemp(prefix="test_naming_")
    try:
        cfg = _make_cfg(tmpdir, tp, pp, ep, expert_tp)
        shards = _build_shards_dict(tp, pp, ep)
        save_dir = os.path.join(tmpdir, "ckpt")

        release_dir = save_megatron_release_checkpoint(shards, save_dir, cfg)
        actual = _get_shard_dirs(release_dir)
        expected = _expected_dirs(tp, pp, ep)

        if actual == expected:
            print(f"  ✓ {label:50s} PASS  ({len(actual)} dirs)")
            return True
        else:
            print(f"  ✗ {label:50s} FAIL")
            missing = expected - actual
            extra = actual - expected
            if missing:
                print(f"    Missing: {sorted(missing)[:5]}")
            if extra:
                print(f"    Extra:   {sorted(extra)[:5]}")
            return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    print("=" * 70)
    print("Checkpoint Naming Convention Tests")
    print("=" * 70)
    print()

    test_cases = [
        # (tp, pp, ep, expert_tp) — expert_tp=None defaults to tp
        # --- PP=1, EP=1: mp_rank_{tp:02d} ---
        (1, 1, 1, None),    # simplest: mp_rank_00
        (4, 1, 1, None),    # tp only: mp_rank_00..03
        (4, 1, 1, 2),       # etp!=tp but naming unchanged: mp_rank_00..03
        (4, 1, 1, 1),       # etp=1 but naming unchanged: mp_rank_00..03

        # --- PP>1, EP=1: mp_rank_{tp:02d}_{pp:03d} ---
        (1, 2, 1, None),    # pp only: mp_rank_00_000, mp_rank_00_001
        (4, 2, 1, None),    # tp+pp
        (4, 2, 1, 2),       # tp+pp, etp!=tp

        # --- PP=1, EP>1: mp_rank_{tp:02d}_{ep:03d} ---
        (1, 1, 8, None),    # ep only: mp_rank_00_000..007
        (4, 1, 8, None),    # tp+ep
        (4, 1, 8, 2),       # tp+ep, etp!=tp
        (4, 1, 8, 1),       # tp+ep, etp=1

        # --- PP>1, EP>1: mp_rank_{tp:02d}_{pp:03d}_{ep:03d} ---
        (1, 2, 8, None),    # pp+ep
        (4, 2, 8, None),    # tp+pp+ep (full)
        (4, 2, 8, 2),       # tp+pp+ep, etp!=tp
        (4, 2, 8, 1),       # tp+pp+ep, etp=1
    ]

    results = []
    for tp, pp, ep, etp in test_cases:
        ok = run_test(tp, pp, ep, etp)
        results.append(ok)

    print()
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"ALL {total} TESTS PASSED")
    else:
        print(f"{passed}/{total} PASSED, {total - passed} FAILED")
    print("=" * 70)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
