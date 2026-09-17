#!/usr/bin/env python3
"""Round-trip test for expert_tensor_parallel_size support.

Creates a small mock MoE model in HF format, converts:
    HF -> Megatron (with expert_tp != tp) -> HF
and verifies the weights are numerically identical.

Usage:
    python test_expert_tp.py
"""

import os
import shutil
import sys
import tempfile

import torch
import yaml

# Add parent directory so qwen35 package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen35.config import Config
from qwen35.converter import MoEConverter

# ---------------------------------------------------------------------------
# Test model dimensions (tiny, just enough to exercise the logic)
# ---------------------------------------------------------------------------
NUM_LAYERS = 2
HIDDEN_SIZE = 64
NUM_ATTENTION_HEADS = 4
NUM_QUERY_GROUPS = 2
KV_CHANNELS = 16  # hidden_size / num_attention_heads
FFN_HIDDEN_SIZE = 128
MOE_FFN_HIDDEN_SIZE = 32
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE = 64
NUM_EXPERTS = 4
# PLACEHOLDER_FOR_MORE_CONSTANTS


def _make_yaml(path, tp, pp, ep, expert_tp):
    """Write a minimal training YAML for the mock model."""
    cfg = {
        "tensor_model_parallel_size": tp,
        "pipeline_model_parallel_size": pp,
        "expert_model_parallel_size": ep,
        "expert_tensor_parallel_size": expert_tp,
        "num_layers": NUM_LAYERS,
        "hidden_size": HIDDEN_SIZE,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "num_query_groups": NUM_QUERY_GROUPS,
        "kv_channels": KV_CHANNELS,
        "attention_output_gate": False,
        "untie_embeddings_and_output_weights": True,
        "linear_attention_freq": 999,  # no GDN layers
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "num_experts": NUM_EXPERTS,
        "moe_ffn_hidden_size": MOE_FFN_HIDDEN_SIZE,
        "moe_shared_expert_intermediate_size": MOE_SHARED_EXPERT_INTERMEDIATE_SIZE,
        "ffn_hidden_size": FFN_HIDDEN_SIZE,
        "no_enable_vision": True,
        "mtp_num_layers": 0,
    }
    with open(path, "w") as f:
        yaml.dump(cfg, f)


def _build_hf_state_dict():
    """Build a synthetic HF-format state dict for a tiny MoE model."""
    sd = {}
    H = HIDDEN_SIZE

    # Embeddings
    sd["model.language_model.embed_tokens.weight"] = torch.randn(1000, H)
    sd["lm_head.weight"] = torch.randn(1000, H)

    for layer in range(NUM_LAYERS):
        pfx = f"model.language_model.layers.{layer}"

        # Attention
        sd[f"{pfx}.input_layernorm.weight"] = torch.randn(H)
        q_size = NUM_ATTENTION_HEADS * KV_CHANNELS
        kv_size = NUM_QUERY_GROUPS * KV_CHANNELS
        sd[f"{pfx}.self_attn.q_proj.weight"] = torch.randn(q_size, H)
        sd[f"{pfx}.self_attn.k_proj.weight"] = torch.randn(kv_size, H)
        sd[f"{pfx}.self_attn.v_proj.weight"] = torch.randn(kv_size, H)
        sd[f"{pfx}.self_attn.o_proj.weight"] = torch.randn(H, q_size)
        sd[f"{pfx}.self_attn.q_norm.weight"] = torch.randn(KV_CHANNELS)
        sd[f"{pfx}.self_attn.k_norm.weight"] = torch.randn(KV_CHANNELS)

        # MoE MLP
        sd[f"{pfx}.post_attention_layernorm.weight"] = torch.randn(H)
        sd[f"{pfx}.mlp.gate.weight"] = torch.randn(NUM_EXPERTS, H)

        # Per-expert weights
        for e in range(NUM_EXPERTS):
            sd[f"{pfx}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(MOE_FFN_HIDDEN_SIZE, H)
            sd[f"{pfx}.mlp.experts.{e}.up_proj.weight"] = torch.randn(MOE_FFN_HIDDEN_SIZE, H)
            sd[f"{pfx}.mlp.experts.{e}.down_proj.weight"] = torch.randn(H, MOE_FFN_HIDDEN_SIZE)

        # Shared expert
        sd[f"{pfx}.mlp.shared_expert.gate_proj.weight"] = torch.randn(
            MOE_SHARED_EXPERT_INTERMEDIATE_SIZE, H
        )
        sd[f"{pfx}.mlp.shared_expert.up_proj.weight"] = torch.randn(
            MOE_SHARED_EXPERT_INTERMEDIATE_SIZE, H
        )
        sd[f"{pfx}.mlp.shared_expert.down_proj.weight"] = torch.randn(
            H, MOE_SHARED_EXPERT_INTERMEDIATE_SIZE
        )
        sd[f"{pfx}.mlp.shared_expert_gate.weight"] = torch.randn(1)

    # Final layernorm
    sd["model.language_model.norm.weight"] = torch.randn(H)

    return sd


def _normalize_hf_experts(sd):
    """Normalize expert format: convert stacked format to per-expert format for comparison.

    HF has two representations:
      - Per-expert:  experts.{e}.gate_proj.weight, experts.{e}.up_proj.weight, experts.{e}.down_proj.weight
      - Stacked:     experts.gate_up_proj (num_experts, 2*ffn, hidden), experts.down_proj (num_experts, hidden, ffn)

    The round-trip may change format. Normalize both to per-expert for fair comparison.
    """
    import re

    normalized = {}
    stacked_gate_up = re.compile(r"^(.+\.mlp\.experts)\.gate_up_proj$")
    stacked_down = re.compile(r"^(.+\.mlp\.experts)\.down_proj$")

    for k, v in sd.items():
        m = stacked_gate_up.match(k)
        if m:
            prefix = m.group(1)
            # v shape: (num_experts, 2*ffn, hidden)
            num_experts = v.shape[0]
            ffn = v.shape[1] // 2
            for e in range(num_experts):
                gate = v[e, :ffn, :]
                up = v[e, ffn:, :]
                normalized[f"{prefix}.{e}.gate_proj.weight"] = gate
                normalized[f"{prefix}.{e}.up_proj.weight"] = up
            continue

        m = stacked_down.match(k)
        if m:
            prefix = m.group(1)
            num_experts = v.shape[0]
            for e in range(num_experts):
                normalized[f"{prefix}.{e}.down_proj.weight"] = v[e]
            continue

        normalized[k] = v

    return normalized


def _compare_state_dicts(sd_a, sd_b, label):
    """Compare two state dicts, return True if identical.

    Only checks expert-related keys to focus on expert_tp logic.
    """
    # Filter to only expert-related keys
    expert_keys_a = {k for k in sd_a.keys() if ".mlp." in k}
    expert_keys_b = {k for k in sd_b.keys() if ".mlp." in k}

    missing = expert_keys_a - expert_keys_b
    extra = expert_keys_b - expert_keys_a

    ok = True
    if missing:
        print(f"  [{label}] FAIL: expert keys missing after round-trip ({len(missing)}):")
        for k in sorted(missing)[:5]:
            print(f"    {k}")
        ok = False
    if extra:
        print(f"  [{label}] FAIL: extra expert keys after round-trip ({len(extra)}):")
        for k in sorted(extra)[:5]:
            print(f"    {k}")
        ok = False

    mismatches = 0
    for k in sorted(expert_keys_a & expert_keys_b):
        a, b = sd_a[k], sd_b[k]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            continue
        if a.shape != b.shape:
            print(f"  [{label}] Shape mismatch: {k}  {a.shape} vs {b.shape}")
            mismatches += 1
            ok = False
        elif not torch.allclose(a, b, rtol=1e-5, atol=1e-6):
            diff = (a - b).abs()
            print(
                f"  [{label}] Value mismatch: {k}  "
                f"max_diff={diff.max().item():.2e}, mean_diff={diff.mean().item():.2e}"
            )
            mismatches += 1
            ok = False

    if ok:
        print(f"  [{label}] PASS: {len(expert_keys_a & expert_keys_b)} expert keys matched exactly")
    else:
        print(f"  [{label}] FAIL: {mismatches} mismatches")
    return ok


def run_round_trip_test(tp, pp, ep, expert_tp):
    """Run a single HF -> Megatron -> HF round-trip test."""
    label = f"TP={tp}, PP={pp}, EP={ep}, Expert-TP={expert_tp}"
    print(f"\n{'='*70}")
    print(f"Test: {label}")
    print(f"{'='*70}")

    tmpdir = tempfile.mkdtemp(prefix="test_expert_tp_")
    try:
        yaml_path = os.path.join(tmpdir, "train.yaml")
        meg_dir = os.path.join(tmpdir, "meg")
        hf_out_dir = os.path.join(tmpdir, "hf_out")

        _make_yaml(yaml_path, tp, pp, ep, expert_tp)
        cfg = Config(yaml_path)

        converter = MoEConverter(cfg)

        # Build original HF weights
        original_hf_sd = _build_hf_state_dict()

        # Step 1: HF -> Megatron (in memory)
        print("  Converting HF -> Megatron...")
        meg_sd = converter.convert_to_megatron(original_hf_sd)

        # Step 2: Split by PP -> EP -> TP (including expert_tp)
        from qwen35.sharding import split_ep_experts, split_pp_layers

        pp_stages = split_pp_layers(meg_sd, cfg)

        shards_dict = {}
        for pp_rank in range(pp):
            ep_shards = split_ep_experts(pp_stages[pp_rank], cfg)
            for ep_rank in range(ep):
                tp_shards = converter._split_tp(ep_shards[ep_rank])
                # tp_shards is always a list indexed by tp_rank (1D, no expert_tp dimension)
                for tp_rank in range(tp):
                    shard = converter._add_extra_states(tp_shards[tp_rank])
                    shards_dict[(pp_rank, tp_rank, ep_rank)] = shard

        total_shards = len(shards_dict)
        print(f"  Generated {total_shards} shards")

        # Verify shard count: always pp * tp * ep (no extra expert_tp dimension)
        expected = pp * tp * ep
        assert total_shards == expected, f"Expected {expected} shards, got {total_shards}"

        # Step 3: Merge back: TP -> EP -> PP
        from qwen35.sharding import merge_ep_experts, merge_pp_layers

        print("  Merging back Megatron shards...")
        pp_merged = {}
        for pp_rank in range(pp):
            tp_merged = {}
            for tp_rank in range(tp):
                ep_merged_list = []
                for ep_rank in range(ep):
                    ep_merged_list.append(shards_dict[(pp_rank, tp_rank, ep_rank)])

                tp_merged[tp_rank] = merge_ep_experts(ep_merged_list, cfg)

            pp_merged[pp_rank] = converter._merge_tp([tp_merged[r] for r in range(tp)])

        full_sd = merge_pp_layers(pp_merged, cfg)

        # Step 4: Megatron -> HF
        print("  Converting Megatron -> HF...")
        recovered_hf_sd = converter.convert_to_hf(full_sd)

        # Step 5: Compare (normalize expert format for fair comparison)
        norm_original = _normalize_hf_experts(original_hf_sd)
        norm_recovered = _normalize_hf_experts(recovered_hf_sd)
        return _compare_state_dicts(norm_original, norm_recovered, label)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    print("=" * 70)
    print("Expert Tensor Parallel - Round-Trip Test Suite")
    print("=" * 70)

    test_cases = [
        # (tp, pp, ep, expert_tp)
        # Baseline: expert_tp == tp (should behave identically to before)
        (1, 1, 1, 1),
        (2, 1, 1, 2),
        (2, 1, 2, 2),

        # Core: expert_tp != tp
        (2, 1, 1, 1),   # tp=2, expert_tp=1: experts not split, rest split by 2
        (4, 1, 1, 2),   # tp=4, expert_tp=2: different split sizes
        (4, 1, 2, 1),   # tp=4, ep=2, expert_tp=1: with EP
        (2, 2, 1, 1),   # tp=2, pp=2, expert_tp=1: with PP
        (2, 2, 2, 1),   # tp=2, pp=2, ep=2, expert_tp=1: all dimensions
        (4, 1, 1, 1),   # tp=4, expert_tp=1: max difference
    ]

    results = []
    for tp, pp, ep, expert_tp in test_cases:
        ok = run_round_trip_test(tp, pp, ep, expert_tp)
        results.append((tp, pp, ep, expert_tp, ok))

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    all_pass = True
    for tp, pp, ep, expert_tp, ok in results:
        status = "PASS" if ok else "FAIL"
        marker = "✓" if ok else "✗"
        label = f"TP={tp}, PP={pp}, EP={ep}, Expert-TP={expert_tp}"
        suffix = "" if expert_tp == tp else "  <-- expert_tp != tp"
        print(f"  {marker} {label:45s} {status}{suffix}")
        if not ok:
            all_pass = False

    print(f"\n{'='*70}")
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print(f"{'='*70}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
