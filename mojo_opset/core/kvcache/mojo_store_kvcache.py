import os
import torch
from typing import Optional

from ..mojo_operator import MojoOperator


class MojoStoreKVCache(MojoOperator):
    pass


class MojoStorePagedKVCache(MojoOperator):
    def __init__(
        self,
        kv_layout: str = "ND",
        kv_dim: int = None,
        is_varlen: bool = True,
        op_name: str = "",
    ):
        """
        StoreKVPaged 的通用参数定义。

        Init 参数：
        - kv_layout (str)：KV 计算布局，取值 {"ND","NZ","CB"}，默认 "ND"。
        - kv_dim (int)：KV 隐层维度 D_kv，用于 MLA 等场景区分 KV 压缩维与 K 的维度；正整数。
        - is_varlen (bool)：为 True 时按 TND 连续 token 视角优先；为 False 时按 BSND；默认 True。
        - op_name (str)：算子名称占位。

        范围与说明：
        - 仅覆盖通用参数；不涉及分页策略细节与量化参数（QuantMode/QuantParam）。
        """
        super().__init__(op_name)
        if kv_layout not in {"ND", "NZ", "CB"}:
            raise ValueError('kv_layout 需为 {"ND","NZ","CB"}')
        if kv_dim is None or not isinstance(kv_dim, int) or kv_dim <= 0:
            raise ValueError("kv_dim 必须为正整数")
        if not isinstance(is_varlen, bool):
            raise TypeError("is_varlen 必须为 bool 类型")
        self.kv_layout = kv_layout
        self.kv_dim = kv_dim
        self.is_varlen = is_varlen

    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        kv_lens: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Forward 参数（通用层面）：
        - key：形状 [B, S_new, H_kv, D_kv]，dtype 为 float16/bfloat16。
        - value：形状 [B, S_new, H_kv, D_kv]，dtype 为 float16/bfloat16。
        - key_cache：形状 [B, S_cap, H_kv, D_kv]，dtype 与 key 一致；用于写入。
        - value_cache：形状 [B, S_cap, H_kv, D_kv]，dtype 与 value 一致；用于写入。
        - kv_lens：可选，形状 [B]，dtype=int32，表示历史已存储长度。
        - block_table：可选，占位 [B, T]（T>=1），dtype=int32；仅作分页占位，不实现页管理。

        校验与约束：
        - dtype：key/value 为 float16 或 bfloat16；cache 与之匹配。
        - 形状一致：key/value 的 batch 与 H_kv、D_kv 一致；cache 的 H_kv、D_kv 与输入一致；S_new 与 kv_lens 增量逻辑不在本方法处理范围。
        - kv_lens：如提供，需为 [B]、int32，元素非负且不超过 S_cap。
        - block_table：如提供，需为 [B,T]、int32、T>=1、元素非负。
        """
        
        # 写入逻辑占位
        raise NotImplementedError("MojoStorePagedKVCache forward 仅进行通用参数校验，不包含具体写入逻辑")

    def forward_ref(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        kv_lens: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
    ) -> None:
        """
        参考实现（golden）：按通用语义将新增的 key/value 写入 cache，严格区分 TND/BNSD 输入。
        输入布局契约：
        - 当 is_varlen=True（TND）：仅接受 key/value/key_cache/value_cache 为 [T, H_kv, D_kv]
          · 否则报错：ValueError("Expected TND when is_varlen=True; got shape ...")
        - 当 is_varlen=False（BNSD）：仅接受 key/value 为 [B, S_new, H_kv, D_kv]，cache 为 [B, S_cap, H_kv, D_kv]
          · 否则报错：ValueError("Expected BNSD when is_varlen=False; got shape ...")
        公式语义（不涉及分页表 block_table 的细节，仅校验占位）：
        - TND：offset = int(kv_lens)（若未提供则 0）；
          * key_cache[offset:offset+T_new, :, :] = key[:T_new, :, :]
          * value_cache[offset:offset+T_new, :, :] = value[:T_new, :, :]
        - BNSD：对每个 batch b，offset = kv_lens[b]（若未提供则 0）；按 [offset, offset+S_new) 写入。
        返回：None。
        """
        # 公共 dtype/维度检查
        if key.dtype != value.dtype or key_cache.dtype != value_cache.dtype or key.dtype != key_cache.dtype:
            raise TypeError("key/value 与 cache 的 dtype 需一致")
        if key.shape[-1] != self.kv_dim or value.shape[-1] != self.kv_dim:
            raise ValueError("key/value 的最后一维需等于 kv_dim")
        if key_cache.shape[-1] != self.kv_dim or value_cache.shape[-1] != self.kv_dim:
            raise ValueError("cache 的最后一维需等于 kv_dim")
        if self.is_varlen:
            # 仅接受 TND
            if not (key.ndim == value.ndim == key_cache.ndim == value_cache.ndim == 3):
                raise ValueError(f"Expected TND when is_varlen=True; got shapes key={tuple(key.shape)}, value={tuple(value.shape)}, key_cache={tuple(key_cache.shape)}, value_cache={tuple(value_cache.shape)}")
            T_new, Hkv, Dkv = key.shape
            T_cap, Hc, Dc = key_cache.shape
            if Hc != Hkv or Dc != Dkv:
                raise ValueError("cache 的 H/D 需与输入一致")
            if value.shape != key.shape or value_cache.shape != key_cache.shape:
                raise ValueError("value 与 key 形状需一致，value_cache 与 key_cache 形状需一致")
            # kv_lens 允许为 None 或 0-D/1-D 单元素整型
            if kv_lens is None:
                offset = 0
            else:
                if kv_lens.ndim not in (0, 1):
                    raise ValueError("TND 下 kv_lens 需为标量或单元素张量")
                offset = int(kv_lens.item()) if kv_lens.numel() == 1 else int(kv_lens[0].item())
                if offset < 0 or offset > T_cap:
                    raise ValueError("kv_lens 需在 [0, T_cap]")
            end = offset + T_new
            if end > T_cap:
                raise ValueError("写入越界：offset+T_new 超过 T_cap")
            key_cache[offset:end, :, :] = key[:, :, :]
            value_cache[offset:end, :, :] = value[:, :, :]
            return None
        else:
            # 仅接受 BNSD
            if not (key.ndim == value.ndim == key_cache.ndim == value_cache.ndim == 4):
                raise ValueError(f"Expected BNSD when is_varlen=False; got shapes key={tuple(key.shape)}, value={tuple(value.shape)}, key_cache={tuple(key_cache.shape)}, value_cache={tuple(value_cache.shape)}")
            B, S_new, Hkv, Dkv = key.shape
            Bc, S_cap, Hc, Dc = key_cache.shape
            if Bc != B or Hc != Hkv or Dc != Dkv:
                raise ValueError("cache 的 B/H/D 需与输入一致")
            if value.shape != key.shape or value_cache.shape != key_cache.shape:
                raise ValueError("value 与 key 形状需一致，value_cache 与 key_cache 形状需一致")
            if kv_lens is None:
                kv_lens = torch.zeros((B,), dtype=torch.int32, device=key.device)
            else:
                if kv_lens.ndim != 1 or kv_lens.shape[0] != B:
                    raise ValueError("kv_lens 需为 [B]")
                if kv_lens.dtype not in (torch.int32, torch.int64):
                    raise TypeError("kv_lens 需为整型")
                if torch.any(kv_lens < 0) or torch.any(kv_lens > S_cap):
                    raise ValueError("kv_lens 元素需在 [0, S_cap]")
            for b in range(B):
                offset = int(kv_lens[b].item())
                end = offset + S_new
                if end > S_cap:
                    raise ValueError(f"写入越界：batch {b} 的 offset+S_new 超过 S_cap")
                key_cache[b, offset:end, :, :] = key[b, :, :, :]
                value_cache[b, offset:end, :, :] = value[b, :, :, :]
            return None


class MojoStoreMLAKVCache(MojoOperator):
    pass


class MojoStorePagedMLAKVCache(MojoOperator):
    pass
