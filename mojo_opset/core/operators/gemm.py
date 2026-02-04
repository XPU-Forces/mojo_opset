import torch

from ..operator import MojoOperator


class MojoGroupGemm(MojoOperator):
    def __init__(
        self,
        weight: torch.Tensor,
        trans_weight=False,
    ):
        super().__init__()

        if not isinstance(trans_weight, bool):
            raise TypeError("trans_weight must be bool.")
        self.trans_weight = trans_weight
        self.weight = weight

    def forward(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        """
        Grouped linear forward over variable-length segments.

        Splits the 2D input into contiguous groups defined by `group_list`,
        applies a per-group weight, and concatenates outputs.

        Args:
            input (torch.Tensor): 2D tensor of shape (N, Din); rows are grouped
                contiguously. Sum(group_list) must equal N.
            group_list (torch.Tensor): 1D tensor of length G with row counts per group.

        Returns:
            torch.Tensor: 2D tensor of shape (N, Dout), concatenated per-group outputs.

        Notes:
            - Expects `self.weight` of shape (G, Din, Dout). If `trans_weight` is True,
            weights are transposed from (G, Dout, Din) to (G, Din, Dout).
            - Each group's output is computed as `input_g @ weight_g`.
        """
        assert input.dim() == 2, "input must be 2D"
        assert self.weight.dim() == 3, "weight must be 3D"
        num_groups = group_list.numel()
        assert self.weight.size(0) == num_groups, "self.weight must have same group count as group_list"

        if self.trans_weight:
            self.weight = self.weight.transpose(1, 2).contiguous()

        group_start = group_list.cumsum(0) - group_list
        group_end = group_list.cumsum(0)

        out_list = []
        for g, (start, end) in enumerate(zip(group_start.tolist(), group_end.tolist())):
            a_g = input[start:end, :]
            b_g = self.weight[g, :, :]
            out_g = a_g @ b_g
            out_list.append(out_g)

        return torch.cat(out_list, dim=0)


class MojoQuantGroupLinear(MojoOperator):
    def __init__(
        self,
        weight: torch.Tensor,
        trans_weight: bool = False,
    ):
        super().__init__()

        if not isinstance(trans_weight, bool):
            raise TypeError("trans_weight must be bool.")
        self.trans_weight = trans_weight
        self.weight = weight

    def forward(self, input: torch.Tensor, x1_scale: torch.Tensor, x2_scale: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 3, "input must be 3D"
        assert self.weight.dim() == 3, "weight must be 3D"

        if self.trans_weight:
            weight = self.weight.transpose(1, 2).contiguous()
        else:
            weight = self.weight

        b, m, k = input.shape
        b_w, k_w, n = weight.shape
        assert b == b_w, "input and weight must have same batch size"
        assert k == k_w, "K of input should be equal to K of weight"

        out = torch.bmm(input.float(), weight.float()).to(torch.float32)
        out = x2_scale[None, None, :] * out
        out = x1_scale[:, :, None] * out

        out_1 = torch.zeros(m, n, dtype=torch.bfloat16, device=out.device)
        for i in range(b):
            out_1 += out[i, ...].to(torch.bfloat16)

        return out_1


class MojoDequantGroupLinear(MojoOperator):
    def __init__(
        self,
        weight: torch.Tensor,
        trans_weight: bool = False,
    ):
        super().__init__()

        if not isinstance(trans_weight, bool):
            raise TypeError("trans_weight must be bool.")
        self.trans_weight = trans_weight
        self.weight = weight

    def forward(
        self,
        input: torch.Tensor,
        group_list: torch.Tensor,
        antiquant_scale: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        antiquant_offset: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        assert input.dim() == 2, "input must be 2D"
        assert self.weight.dim() == 3, "weight must be 3D"
        num_groups = group_list.numel()
        assert self.weight.size(0) == num_groups, "self.weight must have same group count as group_list"

        if self.trans_weight:
            weight = self.weight.transpose(1, 2).contiguous()
        else:
            weight = self.weight

        group_start = group_list.cumsum(0) - group_list
        group_end = group_list.cumsum(0)

        out_list = []
        for g, (start, end) in enumerate(zip(group_start.tolist(), group_end.tolist())):
            a_g = input[start:end, :]
            b_g = weight[g, :, :]

            scale_g = None
            if antiquant_scale is not None:
                if isinstance(antiquant_scale, list):
                    if len(antiquant_scale) != num_groups:
                        raise ValueError("antiquant_scale must match group count")
                    scale_g = antiquant_scale[g]
                elif antiquant_scale.dim() > 0 and antiquant_scale.size(0) == num_groups:
                    scale_g = antiquant_scale[g]
                else:
                    scale_g = antiquant_scale

            offset_g = None
            if antiquant_offset is not None:
                if isinstance(antiquant_offset, list):
                    if len(antiquant_offset) != num_groups:
                        raise ValueError("antiquant_offset must match group count")
                    offset_g = antiquant_offset[g]
                elif antiquant_offset.dim() > 0 and antiquant_offset.size(0) == num_groups:
                    offset_g = antiquant_offset[g]
                else:
                    offset_g = antiquant_offset

            if scale_g is not None:
                if isinstance(scale_g, torch.Tensor):
                    scale_g = scale_g.to(b_g.device)
                if offset_g is None:
                    offset_g = b_g.new_tensor(0)
                elif isinstance(offset_g, torch.Tensor):
                    offset_g = offset_g.to(b_g.device)
                b_g = (b_g.float() - offset_g) * scale_g

            out_g = a_g @ b_g
            out_list.append(out_g)

        return torch.cat(out_list, dim=0)


class MojoGemmAllReduce(MojoOperator):
    pass


class MojoAllGatherGemm(MojoOperator):
    pass


class MojoAllGatherGemm(MojoOperator):
    pass


class MojoGemmAll2All(MojoOperator):
    pass


class MojoGemmReduceScatter(MojoOperator):
    pass
