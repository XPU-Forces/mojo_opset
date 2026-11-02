"""A5（950 系列）平台的 pos_embed 插值实现 —— A2 实现的薄复用。

复用依据（实测，见 `docs/03-校验/P13-三平台验证完成-20261102.md`）：
  * 本 kernel 为 Vector-only（无 tl.dot），不依赖 A2/A5 的 Cube 差异；
  * 三平台官方精度套件各 27/27 通过（A2 `Ascend910B3` / A3 `Ascend910_9382`
    / A5 `Ascend950PR_9579`）；
  * 唯一的平台差异是广播 tile 的 ``block_rows``（A2/A3=8，A5=52，因 UB 192KB vs
    256KB），该值在 A2 实现内部按 ``get_device_name`` 自动选择，无需 A5 独立副本。

因此 A5 不维护 kernel 副本，直接转发 A2 实现，避免双份代码漂移。
``config/npu_a5.yaml`` 的 ``pos_embed_interp: triton`` 由分派层解析到本包。

⚠️ 必须显式转发 ``{op_id}_fwd``：``_dispatch._load_kernel_binding`` 按
``getattr(module, f"{op_id}_fwd")`` 取入口，单纯的 ``import *`` 或按旧名
``pos_embed_interp`` 转发都会在 A5 上解析失败（实测 26/27 全红）。
"""

from mojo_opset.kernels.npu_a2_triton.pos_embed_interp import (  # noqa: F401
    pos_embed_interp,
    pos_embed_interp_broadcast_impl,
    pos_embed_interp_fwd,
    pos_embed_interp_impl,
    pos_embed_interp_multi_fwd,
    pos_embed_interp_multi_impl,
    vision_pos_embed_interp_fwd,
)

__all__ = [
    # provider contract（分派层按 {op_id}_fwd 取）
    "vision_pos_embed_interp_fwd",
    "pos_embed_interp_fwd",
    "pos_embed_interp_multi_fwd",
    # 直接调用入口
    "pos_embed_interp",
    "pos_embed_interp_impl",
    "pos_embed_interp_broadcast_impl",
    "pos_embed_interp_multi_impl",
]
