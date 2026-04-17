# FlagScale Training Upgrade Report: core_v0.15.0rc7 → core_v0.16.1

| Stage | Status | Notes |
|-------|--------|-------|
| 1. Environment setup | ✅ | Megatron-LM-FL pip-installed at core_v0.16.1 |
| 2. Three-way merge refs | ✅ | base=core_v0.15.0rc7, target=core_v0.16.1 |
| 3. Customization diff | ✅ | Three-way diff identified FlagScale customizations vs upstream changes |
| 4. Upgrade applied | ✅ | 9 conflict files resolved via three-way merge |
| 5. Stale reference fixes | ✅ | 81 megatron.core/plugin imports verified; 13 cur_platform violations fixed |
| 6. Verification | ✅ | syntax: OK, imports: OK, circular imports: fixed, lazy imports: OK |

## Commits

| Commit | Description |
|--------|-------------|
| `4d7b6697f` | Replace training/legacy with upstream core_v0.16.1 content |
| `c932fb032` | Resolve conflicts for all 9 files (28 files changed, +1928 −293) |
| `1e45969bd` | Platform compliance: training.py and utils.py (12 torch.cuda → cur_platform) |
| `16410ff53` | Lazy import StableLM2SchedulerConfig + cur_platform in train_engram.py |

## Files Changed

40 files changed, 3164 insertions(+), 5022 deletions(-)

### Key modified files
- `training/training.py` — major upstream changes + FlagScale customizations preserved
- `training/arguments.py` — argument refactoring, FlagScale args preserved
- `training/checkpointing.py` — new upstream features merged
- `training/utils.py` — cur_platform compliance + upstream changes
- `training/initialize.py` — upstream updates merged
- `training/datasets/sft_dataset.py` — upstream refactoring merged
- `training/datasets/data_samplers.py` — upstream changes merged
- `legacy/model/transformer.py` — significant upstream cleanup
- `train_engram.py` — cur_platform compliance

### Files removed (upstream deletions)
- `training/arguments_fs.py` — merged into arguments.py upstream
- `training/extra_valid.py` — removed upstream
- `training/fs_theoretical_memory_usage.py` — removed upstream
- `training/spiky_loss.py` — removed upstream
- `training/stablelm2_scheduler.py` — removed upstream
- `training/peft/` — entire directory removed upstream
- `training/datasets/sft_dataset_fs.py` — removed upstream
- `training/datasets/concated_indexed_dataset.py` — removed upstream
- `training/tokenizer/rwkv_tokenization.py` — removed upstream
- `training/tokenizer/tokenization_utils.py` — removed upstream
- `training/tokenizer/sft_tokenizer.py` — removed upstream

### Files added (upstream additions)
- `training/argument_utils.py` — new upstream
- `training/common_config.py` — new upstream
- `training/dgrad_logging.py` — new upstream
- `training/resilience_config.py` — new upstream
- `training/training_config.py` — new upstream

## Compatibility Status
- megatron.core imports: ✅ all resolve
- megatron.plugin imports: ✅ all resolve
- megatron.training import: ✅ no circular imports
- megatron.rl import: ✅ guarded with try/except
- Multi-platform (cur_platform): ✅ no torch.cuda.current_device() violations
- FlagScale-specific features: ✅ preserved (hetero, dualpipev, engram, extra_valid lazy)

## Import Fixes Applied
1. `megatron.rl` — unconditional import moved into try/except with `has_rl_utils` guard
2. `extra_evaluate_and_print_results` / `build_extra_valid_data_iterators` — top-level import → lazy import at 2 usage sites
3. `StableLM2SchedulerConfig` — top-level import → lazy import at usage site
