import torch
import triton
import triton.language as tl



# linear
configs = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}),
    # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}),
    # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}),
    # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}),
    # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}),
]
@triton.autotune(configs=configs, key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel_2d(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    acc_dtype: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_total = num_pid_m * num_pid_n
    pid_m = (pid % num_pid_total) // num_pid_n
    pid_n = pid % num_pid_n

    # offset cal
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A point: [M, K]
    a_ptrs = a_ptr +\
             offs_am[:, None] * stride_am + \
             offs_k[None, :] * stride_ak

    # B point: [K, N]
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + \
             offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=acc_dtype)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator
    c_ptrs = c_ptr + \
             offs_am[:, None] * stride_cm + \
             offs_bn[None, :] * stride_cn
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b, bias = None):
    # Check constraints. 
    assert b.dim() ==2, '2D Matrix'
    assert a.shape[-1] == b.shape[-2], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    input_shape = a.shape
    a = a.reshape(-1, b.shape[-2])
    M, K = a.shape
    N = b.shape[-1]
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    acc_dtype = tl.float32
    matmul_kernel_2d[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        acc_dtype,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    if bias is not None:
        c += bias.unsqueeze(0)
    new_shape = input_shape[:-1] + (b.shape[-1],)
    return c.reshape(new_shape)
# def naive_softmax(x):
#     """Compute row-wise softmax of X using native pytorch

#     We subtract the maximum element in order to avoid overflows. Softmax is invariant to
#     this shift.
#     """
#     # read  MN elements ; write M  elements
#     x_max = x.max(dim=1)[0]
#     # read MN + M elements ; write MN elements
#     z = x - x_max[:, None]
#     # read  MN elements ; write MN elements
#     numerator = torch.exp(z)
#     # read  MN elements ; write M  elements
#     denominator = numerator.sum(dim=1)
#     # read MN + M elements ; write MN elements
#     ret = numerator / denominator[:, None]
#     # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
#     return ret

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)

        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


target = triton.runtime.driver.active.get_current_target()
#缓存预编译的核函数：key=BLOCK_SIZE，value=(编译后的核函数, 推荐程序数)
kernels = {}

def softmax(x):
    shape = x.shape
    dim = shape[-1]
    x_2d = x.reshape(-1, dim)
    n_rows, n_cols = x_2d.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    # BLOCK_SIZE = 大于 n_cols 的最小2的幂（Triton 最优实践）
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Allocate output
    y = torch.empty_like(x_2d)

    # pre-compile kernel to get register usage and compute thread occupancy.
    # 从缓存中获取已编译的核函数和推荐程序数
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        num_programs = 32
        kernel = softmax_kernel
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    # grid = lambda meta: (triton.cdiv(n_rows, meta["XBLOCK"]), 1, 1)
    # 网格大小grid=(num_programs, 1, 1) → 1D 网格，num_programs 个线程块，num_programs × 1 × 1 个线程块
    softmax_kernel[(num_programs, 1, 1)](
        y,
        x_2d,
        x_2d.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE
    )

    return y.reshape(*shape)

def MoeGating_forward(hidden_states,gate_weight,top_k):
    logits = matmul(hidden_states, gate_weight) #[T,E]/BSE
    probs = softmax(logits) #[T,E]/BSE
    values, indices = torch.topk(probs, top_k, dim=-1)
    gate_weights = values / torch.sum(values, dim=-1, keepdim=True)
    return indices, gate_weights