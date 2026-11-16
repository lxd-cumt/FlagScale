#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

# Preserve the vendor Python environment; do not create a CUDA/CPU torch venv.
export PATH="/usr/local/PPU_SDK/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export FLAGSCALE_TEST_DIST_BACKEND=nccl
export FLAGSCALE_TEST_TORCH_DEVICE_TYPE=cuda
# Initial correctness baseline uses the plugin's PyTorch reference implementation.
# This is not a claim that a native PPU TE or FlagGems backend is validated.
export NVTE_PLUGIN_SKIP_CUDA=1
# Master switch: must predate the Transformer Engine import, and without it
# NVTE_PLUGIN_PREFER has no effect.
export NVTE_PLUGIN=transformer_engine_plugin_fl
export NVTE_PLUGIN_PREFER=reference
export NVTE_FRAMEWORK=pytorch
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=1
export NVTE_WITH_NCCL_EP=0
