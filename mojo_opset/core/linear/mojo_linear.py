import os
import torch
from typing import Optional

from ..mojo_operator import MojoOperator


class MojoLinear(MojoOperator):
    def __init__(
        self,
        input_layout: str = "NZ",
        weight: torch.Tensor = None,
        bias: Optional[torch.Tensor] = None,
        is_varlen: bool = True,
        op_name: str = "",
    ):
        """
        Linear 的通用参数定义。

        Init 参数：
        - input_layout (str)：输入布局枚举，取值 {"KN","NZ"}，默认 "NZ"。
        - weight (torch.Tensor)：权重张量，形状 [in_dim, out_dim]。
        - bias (Optional[torch.Tensor])：偏置张量，形状与输出维度对齐；可选。
        - is_varlen (bool)：为 True 时优先按 TND（逐 token）计算；为 False 时按 BSND；默认 True。
        - op_name (str)：算子名称占位。
        """
        super().__init__(op_name)

        if input_layout not in {"KN", "NZ"}:
            raise ValueError('input_layout 需为 {"KN","NZ"}')
        self.input_layout = input_layout

        if weight is None or not isinstance(weight, torch.Tensor):
            raise TypeError("weight 必须为 torch.Tensor 且不可为 None")
        if weight.ndim not in (2, ):
            raise ValueError(f"weight 需为 2-D，实际为 {tuple(weight.shape)}")
        self.weight = weight

        if bias is not None:
            if not isinstance(bias, torch.Tensor):
                raise TypeError("bias 必须为 torch.Tensor 或 None")
            # 对齐输出维度（仅做轻量检查，不做广播/重排）
            if weight.ndim == 2:
                out_dim = weight.shape[1]
                if bias.ndim != 1 or bias.shape[0] != out_dim:
                    raise ValueError("bias 形状需为 [out_dim]，并与 weight 的输出维一致")
        self.bias = bias
        if not isinstance(is_varlen, bool):
            raise TypeError("is_varlen 必须为 bool 类型")
        self.is_varlen = is_varlen

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        输入：
        - input：输入张量。

        输出：
        - output: 输出张量，形状需遵循矩阵乘法规则。

        """
        
        raise NotImplementedError("MojoGroupLinear forward 仅做通用参数校验，不包含具体计算")


    def forward_ref(self, input: torch.Tensor) -> torch.Tensor:
        """
        参考实现（golden）：标准线性变换，严格区分 TND/BNSD 输入。
        输入布局契约：
        - 当 is_varlen=True（TND）：仅接受 [T, in_dim] 或 [T, G, in_dim]
        - 当 is_varlen=False（BNSD）：仅接受 [B, S, in_dim] 或 [B, S, G, in_dim]
        - 否则报错（Expected TND/BNSD ...）。
        公式：Y = X · W + b，其中 W=[in_dim,out_dim]，b=[out_dim]
        返回：形状遵循矩阵乘法规则，最后一维为 out_dim，dtype 与输入一致。
        """
        in_dim = self.weight.shape[0]
        if input.shape[-1] != in_dim:
            raise ValueError("input 的最后一维需与 weight 的 in_dim 对齐")
        if self.is_varlen:
            # 仅接受 TND
            if not (input.ndim in (2, 3)):
                raise ValueError(f"Expected TND when is_varlen=True; got shape {tuple(input.shape)}")
            return torch.nn.functional.linear(input, self.weight.t(), self.bias)
        else:
            # 仅接受 BNSD
            if not (input.ndim in (3, 4)):
                raise ValueError(f"Expected BNSD when is_varlen=False; got shape {tuple(input.shape)}")
            return torch.nn.functional.linear(input, self.weight.t(), self.bias)


class MojoBatchLinear(MojoOperator):
    pass


class MojoGroupLinear(MojoOperator):
    def __init__(
        self,
        input_layout: str = "NZ",
        weight: torch.Tensor = None,
        bias: Optional[torch.Tensor] = None,
        is_varlen: bool = True,
        op_name: str = "",
    ):
        """
        Group Linear 的通用参数定义。

        Init 参数：
        - input_layout (str)：输入布局枚举，取值 {"KN","NZ"}，默认 "NZ"。
        - weight (torch.Tensor)：权重张量，形状 [G, in_dim_g, out_dim_g] 或 [in_dim, out_dim]。
        - bias (Optional[torch.Tensor])：偏置张量，形状与输出维度对齐；可选。
        - is_varlen (bool)：为 True 时优先按 TND（逐 token/逐组）计算；为 False 时按 BSND；默认 True。
        - op_name (str)：算子名称占位。

        语义与校验：
        - forward 的 group_list 需与 weight 对齐。
        """
        super().__init__(op_name)

        if input_layout not in {"KN", "NZ"}:
            raise ValueError('input_layout 需为 {"KN","NZ"}')
        self.input_layout = input_layout

        if weight is None or not isinstance(weight, torch.Tensor):
            raise TypeError("weight 必须为 torch.Tensor 且不可为 None")
        if weight.ndim not in (2, 3):
            raise ValueError(f"weight 需为 2-D 或 3-D，实际为 {tuple(weight.shape)}")
        self.weight = weight

        if bias is not None:
            if not isinstance(bias, torch.Tensor):
                raise TypeError("bias 必须为 torch.Tensor 或 None")
            # 对齐输出维度（仅做轻量检查，不做广播/重排）
            if weight.ndim == 2:
                out_dim = weight.shape[1]
                if bias.ndim != 1 or bias.shape[0] != out_dim:
                    raise ValueError("bias 形状需为 [out_dim]，并与 weight 的输出维一致")
            else:
                out_dim_g = weight.shape[-1]
                if bias.ndim not in (1, 2):
                    raise ValueError("分组 bias 需为 1-D 或 2-D")
                if bias.ndim == 1 and bias.shape[0] != out_dim_g:
                    raise ValueError("分组 bias 形状需与每组 out_dim 对齐")
                if bias.ndim == 2 and bias.shape[1] != out_dim_g:
                    raise ValueError("分组 bias 的第二维需与 out_dim 对齐")
        self.bias = bias
        if not isinstance(is_varlen, bool):
            raise TypeError("is_varlen 必须为 bool 类型")
        self.is_varlen = is_varlen

    def forward(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        """
        输入：
        - input：输入张量。
        - group_list：分组列表，形状 [G] 或 [B, S, G]，dtype=int32；用于指示分组映射。

        输出：
        - output: 输出张量，形状需遵循矩阵乘法规则。

        """
        
        raise NotImplementedError("MojoGroupLinear forward 仅做通用参数校验，不包含具体计算")

    def forward_ref(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        """
        参考实现（golden）：分组线性变换，严格区分 TND/BNSD 输入。
        输入布局契约：
        - 当 is_varlen=True（TND）：
          · weight 2-D：[in_dim,out_dim] → 仅接受 [T,in_dim] 或 [T,G,in_dim]
          · weight 3-D：[G,in_dim_g,out_dim_g] → 仅接受 [T,G,in_dim_g] 或 [T,in_dim_g]（需提供 group_list[T]）
        - 当 is_varlen=False（BNSD）：
          · weight 2-D → 仅接受 [B,S,in_dim] 或 [B,S,G,in_dim]
          · weight 3-D → 仅接受 [B,S,G,in_dim_g] 或 [B,S,in_dim_g]（需提供 group_list[B,S]）
        公式：Y = concat_g (X_g · W_g + b_g)
        否则报错（Expected TND/BNSD ...）。
        """
        if self.weight.ndim == 2:
            if input.shape[-1] != self.weight.shape[0]:
                raise ValueError("input 的最后一维需与 weight 的 in_dim 对齐")
            if self.is_varlen:
                if input.ndim not in (2, 3):
                    raise ValueError(f"Expected TND when is_varlen=True; got shape {tuple(input.shape)}")
                return torch.nn.functional.linear(input, self.weight.t(), self.bias)
            else:
                if input.ndim not in (3, 4):
                    raise ValueError(f"Expected BNSD when is_varlen=False; got shape {tuple(input.shape)}")
                return torch.nn.functional.linear(input, self.weight.t(), self.bias)
        else:
            G, in_g, out_g = self.weight.shape
            if self.is_varlen:
                # TND 仅接受 [T,G,in_g] 或 [T,in_g] + group_list[T]
                if input.ndim == 3 and input.shape[-2] == G and input.shape[-1] == in_g:
                    T = input.shape[0]
                    y = torch.empty((T, G, out_g), dtype=input.dtype, device=input.device)
                    for g in range(G):
                        xg = input[:, g, :]  # [T,in_g]
                        yg = torch.matmul(xg, self.weight[g])  # [T,out_g]
                        if self.bias is not None:
                            if self.bias.ndim == 2:
                                yg = yg + self.bias[g]
                            else:
                                yg = yg + self.bias
                        y[:, g, :] = yg
                    return y
                elif input.ndim == 2 and input.shape[-1] == in_g:
                    if group_list is None or group_list.ndim != 1 or group_list.shape[0] != input.shape[0]:
                        raise ValueError("TND 模式下需提供 group_list[T] 与输入对齐")
                    T = input.shape[0]
                    y = torch.empty((T, out_g), dtype=input.dtype, device=input.device)
                    for t in range(T):
                        g = int(group_list[t].item())
                        ys = torch.matmul(input[t], self.weight[g])
                        if self.bias is not None:
                            if self.bias.ndim == 2:
                                ys = ys + self.bias[g]
                            else:
                                ys = ys + self.bias
                        y[t] = ys
                    return y
                else:
                    raise ValueError(f"Expected TND when is_varlen=True; got shape {tuple(input.shape)}")
            else:
                # BNSD 仅接受 [B,S,G,in_g] 或 [B,S,in_g] + group_list[B,S]
                if input.ndim == 4 and input.shape[-2] == G and input.shape[-1] == in_g:
                    B, S = input.shape[0], input.shape[1]
                    y = torch.empty((B, S, G, out_g), dtype=input.dtype, device=input.device)
                    for g in range(G):
                        xg = input[:, :, g, :]  # [B,S,in_g]
                        yg = torch.matmul(xg, self.weight[g])  # [B,S,out_g]
                        if self.bias is not None:
                            if self.bias.ndim == 2:
                                yg = yg + self.bias[g]
                            else:
                                yg = yg + self.bias
                        y[:, :, g, :] = yg
                    return y
                elif input.ndim == 3 and input.shape[-1] == in_g:
                    if group_list is None or group_list.ndim != 2 or group_list.shape[:2] != input.shape[:2]:
                        raise ValueError("BNSD 模式下需提供 group_list[B,S] 与输入对齐")
                    B, S = input.shape[0], input.shape[1]
                    y = torch.empty((B, S, out_g), dtype=input.dtype, device=input.device)
                    for b in range(B):
                        for s in range(S):
                            g = int(group_list[b, s].item())
                            ys = torch.matmul(input[b, s], self.weight[g])
                            if self.bias is not None:
                                if self.bias.ndim == 2:
                                    ys = ys + self.bias[g]
                                else:
                                    ys = ys + self.bias
                            y[b, s] = ys
                    return y
                else:
                    raise ValueError(f"Expected BNSD when is_varlen=False; got shape {tuple(input.shape)}")
