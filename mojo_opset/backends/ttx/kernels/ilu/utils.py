import triton
import triton.language as tl

try:
    from triton.runtime.libentry import libentry as _libentry_impl
except ImportError:

    def _libentry_impl():
        def _decorator(fn):
            return fn

        return _decorator


libentry = _libentry_impl

VEC_ALIGN_BYTES = 256

# LayerNorm / RMSNorm Triton tile heuristics (shared across ILU norm kernels).
COL_BLOCKING_THRESHOLD = 2048

TOKEN_BLOCK_SIZE_TABLE = {
    2048: 4,
    1024: 8,
    # NOTE: tl.arange range must be power-of-2 on some backends.
    512: 16,
    256: 16,
    128: 32,
}


def _block_size_n_pow2(n_cols: int) -> int:
    # ILU backend requires tl.arange range to be power-of-2.
    if n_cols <= 128:
        return 128
    if n_cols <= 256:
        return 256
    if n_cols <= 512:
        return 512
    if n_cols <= 1024:
        return 1024
    return 2048


def norm_fwd_heuristics(args):
    """BLOCK_SIZE_M heuristic for row tiling; shared by LayerNorm and RMSNorm kernels."""
    hidden_dim = args.get("n_cols", args.get("N_COLS"))
    if hidden_dim is None:
        raise KeyError("n_cols")
    if hidden_dim <= COL_BLOCKING_THRESHOLD:
        if hidden_dim in TOKEN_BLOCK_SIZE_TABLE:
            return TOKEN_BLOCK_SIZE_TABLE[hidden_dim]

        for dim_thresh, block_size in sorted(TOKEN_BLOCK_SIZE_TABLE.items()):
            if hidden_dim <= dim_thresh:
                return block_size
        return 1
    else:
        return 4


layer_norm_fwd_heuristics = norm_fwd_heuristics
rms_norm_fwd_heuristics = norm_fwd_heuristics


def ilu_grid_dim_from_row_tasks(n_row_tasks: int) -> int:
    """
    ILU Triton may fail to compile some kernels when grid.x is very small (e.g. 1).
    Match historical behavior by using at least num_vectorcore programs while still
    allowing larger grids when ceil(n_rows / BLOCK_M) exceeds that count.
    """
    n_tasks = int(n_row_tasks)
    if n_tasks <= 0:
        return 0
    nvc = 256
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(0)
        raw = props.get("num_vectorcore", props.get("num_aicore"))
        if raw is not None:
            nvc = int(raw)
    except Exception:
        pass
    return max(n_tasks, nvc)