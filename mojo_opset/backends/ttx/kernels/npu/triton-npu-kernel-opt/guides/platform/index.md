# Triton-Ascend Documentation Index

Use this file to load only the needed local documentation. Prefer English docs
only. Do not route normal skill usage through the Chinese docs.

Platform guide root:

- `guides/platform/`

## Start here

- `quick_start.md`
  Basic environment and first-run flow.
- `programming_guide.md`
  Core NPU programming guidance. Read for migration and hardware-specific rules.
- `architecture_design_and_core_features.md`
  Read when the kernel design depends on hardware execution model or backend
  architecture.
- `FAQ.md`
  Useful for recurring limitations and troubleshooting clues.

## Migration and optimization

- `migration_guide/architecture_difference.md`
  Use when porting a Triton GPU kernel to NPU.
- `migration_guide/migrate_from_gpu.md`
  Use for API and model migration notes.
- `migration_guide/performance_guidelines.md`
  Use for tiling, layout, and scheduling direction.

## Debugging and profiling

- `debug_guide/debugging.md`
  Use for compiler/runtime failures, bad outputs, or launch/debug workflow.
- `debug_guide/profiling.md`
  Use for performance bottleneck analysis and profiling workflow.
- `environment_variable_reference.md`
  Use when a compile/runtime issue depends on env settings.

## Examples

- `examples/01_vector_add_example.md`
- `examples/02_fused_softmax_example.md`
- `examples/03_layer_norm_example.md`
- `examples/04_fused_attention_example.md`
- `examples/05_matrix_multiplication_example.md`
- `examples/06_autotune_example.md`
- `examples/07_accuracy_comparison_example.md`

Read examples selectively. Do not load all of them by default.

## API references

English guides in this tree cover the higher-level material. For lower-level API
shape details, use `api_refs/python-api/` before considering any other
source.

Useful local anchors:

- `api_refs/python-api/outline.md`
- `api_refs/python-api/triton.language.rst`
- `api_refs/python-api/triton.rst`
- `api_refs/python-api/triton.testing.rst`

## Related repo guides

These are the curated local documents that should be preferred over duplicate or
raw note dumps:

- [../tuning/ascend-910b-gemm.md](../tuning/ascend-910b-gemm.md)
- [../tuning/best_practice.md](../tuning/best_practice.md)
- [../tuning/compile_option.md](../tuning/compile_option.md)
- [../tuning/optimization-log.md](../tuning/optimization-log.md)
- [../repo/kernel_inventory.md](../repo/kernel_inventory.md)
- [../repo/integration_and_acceptance.md](../repo/integration_and_acceptance.md)

## Suggested loading order by task

### New GEMM-like kernel

1. `programming_guide.md`
2. `migration_guide/performance_guidelines.md`
3. [../tuning/ascend-910b-gemm.md](../tuning/ascend-910b-gemm.md)
4. `examples/05_matrix_multiplication_example.md`
5. profiling/debugging docs only if needed

### New normalization or reduction kernel

1. `programming_guide.md`
2. `migration_guide/performance_guidelines.md`
3. `examples/03_layer_norm_example.md`
4. [../tuning/best_practice.md](../tuning/best_practice.md)

### New attention kernel

1. `programming_guide.md`
2. `architecture_design_and_core_features.md`
3. `examples/04_fused_attention_example.md`
4. `migration_guide/performance_guidelines.md`

### Debugging compile/runtime issues

1. `debug_guide/debugging.md`
2. `environment_variable_reference.md`
3. `FAQ.md`

### Performance investigation

1. `debug_guide/profiling.md`
2. `migration_guide/performance_guidelines.md`
3. [../tuning/optimization-log.md](../tuning/optimization-log.md)
