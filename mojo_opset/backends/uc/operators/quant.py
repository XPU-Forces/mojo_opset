import torch

from mojo_opset.core import MojoDynamicQuant
from mojo_opset.core import MojoStaticQuant
from mojo_opset.core import MojoMoEDynamicQuant

from ._utils import _matrix_shape
from ._utils import run_kernel


class UCStaticQuant(MojoStaticQuant):
    supported_platforms_list = ["npu"]

    # P1-G4: small-shape gate (best-practices §C.1 / §I.4).
    # Below the UC launch-overhead floor (~80-95 us per kernel call),
    # the torch native fallback is faster and avoids the
    # OOB-write-on-small-M risk of the X=16 kernel.
    _UC_MIN_NUMEL = 64 * 1024

    # The v2 kernel writes a full X=16-row block per program; M must be
    # a multiple of this on entry.  Common transformer shapes
    # (M = batch * seq) are; ragged shapes (e.g. (33, 4096)) fall back
    # to the torch parent path.  Same value as UCDynamicQuant.
    _UC_ROW_TILE = 16

    def forward(self, input: torch.Tensor):
        if self.quant_dtype != torch.int8:
            raise NotImplementedError(f"UCStaticQuant only supports torch.int8, got {self.quant_dtype}.")
        if input.dim() < len(self.input_size):
            raise ValueError(
                f"input must have at least {len(self.input_size)} dims for scale shape "
                f"{self.input_size}, got {tuple(input.shape)}."
            )
        if tuple(input.shape[-len(self.input_size):]) != self.input_size:
            raise ValueError(
                f"input trailing dims {tuple(input.shape[-len(self.input_size):])} must "
                f"match scale shape {self.input_size}."
            )
        if input.numel() == 0:
            return torch.empty_like(input, dtype=self.quant_dtype), self.scale

        # P1-G4 small-shape fallback.
        if input.numel() < self._UC_MIN_NUMEL:
            return super().forward(input)

        kernel_input = input.contiguous()
        scale = self.scale.to(device=kernel_input.device, dtype=torch.float32).contiguous()
        cols = scale.numel()
        rows = kernel_input.numel() // cols

        # P1-G4 ragged-M fallback.  We deliberately do not pad with
        # torch.zeros + copy_ here: experimentally on torch_npu 2.x the
        # freshly-allocated padding DMA does not retire before the
        # kernel reads its DRAM ptr on the very next dispatch, producing
        # one corrupted result per process.  See sibling
        # docs/project-ops/perf-debug/op-MojoDynamicQuant-2026-06-11.md
        # § 5 row "ragged-M padding".
        if rows % self._UC_ROW_TILE != 0:
            return super().forward(input)

        kernel_input_2d = kernel_input.reshape(rows, cols)
        scale_1d = scale.reshape(cols)
        kernel_output = torch.empty_like(kernel_input_2d, dtype=self.quant_dtype)

        run_kernel(
            "mojo_static_quant",
            kernel_input.dtype,
            kernel_input_2d,
            scale_1d,
            kernel_output,
            rows,
            cols,
        )
        return kernel_output.reshape(input.shape), self.scale


class UCDynamicQuant(MojoDynamicQuant):
    supported_platforms_list = ["npu"]

    def forward(self, input: torch.Tensor):
        if input.dim() < 1:
            raise ValueError("input must have at least one dimension.")
        if input.numel() == 0:
            output = torch.empty_like(input, dtype=torch.int8)
            scale = torch.empty((*input.shape[:-1], 1), dtype=torch.float32, device=input.device)
            return output, scale

        kernel_input = input.contiguous()
        rows, cols = _matrix_shape(kernel_input)
        kernel_input_2d = kernel_input.reshape(rows, cols)
        kernel_output = torch.empty_like(kernel_input_2d, dtype=torch.int8)
        kernel_scale = torch.empty((rows,), dtype=torch.float32, device=kernel_input.device)
        if self.inv_smooth_scale is None:
            inv_smooth_scale = torch.ones((cols,), dtype=torch.float32, device=kernel_input.device)
        else:
            inv_smooth_scale = self.inv_smooth_scale.to(device=kernel_input.device, dtype=torch.float32).contiguous()

        run_kernel(
            "mojo_dynamic_quant",
            kernel_input.dtype,
            kernel_input_2d,
            inv_smooth_scale,
            kernel_output,
            kernel_scale,
            rows,
            cols,
        )
        return kernel_output.reshape(input.shape), kernel_scale.reshape(*input.shape[:-1], 1)


class UCMoEDynamicQuant(MojoMoEDynamicQuant):
    supported_platforms_list = ["npu"]

    def forward(self, input: torch.Tensor, token_count: torch.Tensor):
        if input.dim() < 2:
            raise ValueError(f"input must have at least 2 dimensions for MoE dynamic quant, got {input.dim()}.")
        if token_count.dim() != 1:
            raise ValueError(f"token_count must be 1D, got shape {tuple(token_count.shape)}.")
        if token_count.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"token_count must be int32 or int64, got {token_count.dtype}.")
        if token_count.numel() != self.expert_num:
            raise ValueError(f"token_count length must equal expert_num {self.expert_num}, got {token_count.numel()}.")
        if bool(torch.any(token_count < 0).item()):
            raise ValueError("token_count must be non-negative.")

        kernel_input = input.contiguous()
        rows, cols = _matrix_shape(kernel_input)
        if cols != self.input_size:
            raise ValueError(f"input last dimension must equal input_size {self.input_size}, got {cols}.")

        token_total = int(token_count.sum().item())
        if token_total != rows:
            raise ValueError(f"token_count sum must equal flattened row count {rows}, got {token_total}.")

        if input.numel() == 0:
            output = torch.empty_like(input, dtype=torch.int8)
            scale = torch.empty((*input.shape[:-1], 1), dtype=torch.float32, device=input.device)
            return output, scale

        inv_smooth_scale = self.inv_smooth_scale.to(device=kernel_input.device, dtype=torch.float32).contiguous()
        token_count_device = token_count.to(device=kernel_input.device, dtype=torch.int64)
        expanded_inv_smooth_scale = inv_smooth_scale.repeat_interleave(token_count_device, dim=0).contiguous()
        scaled_input = (kernel_input.float().reshape(rows, cols) * expanded_inv_smooth_scale).to(kernel_input.dtype)

        kernel_input_2d = scaled_input.contiguous()
        kernel_output = torch.empty_like(kernel_input_2d, dtype=torch.int8)
        kernel_scale = torch.empty((rows,), dtype=torch.float32, device=kernel_input.device)
        dynamic_inv_smooth_scale = torch.ones((cols,), dtype=torch.float32, device=kernel_input.device)

        run_kernel(
            "mojo_dynamic_quant",
            kernel_input_2d.dtype,
            kernel_input_2d,
            dynamic_inv_smooth_scale,
            kernel_output,
            kernel_scale,
            rows,
            cols,
        )
        return kernel_output.reshape(input.shape), kernel_scale.reshape(*input.shape[:-1], 1)
