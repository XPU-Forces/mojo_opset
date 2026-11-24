"""Stateful modules built directly on :mod:`mojo_opset.functions`."""

from mojo_opset.utils._exports import extend_exports as _extend_exports

from .activation import GELU
from .activation import SiLU
from .activation import SwiGLU
from .attention import SWAInfer
from .attention import VarlenFAInfer
from .convolution import CausalConv1dUpdateStateInfer
from .fused_quantization import DequantSwiGLUQuant
from .fused_quantization import LayerNormQuant
from .fused_quantization import MoEDynamicQuant
from .fused_quantization import RMSNormQuant
from .fused_quantization import ResidualAddLayerNormQuant
from .fused_quantization import ResidualAddRMSNormQuant
from .gemm import GroupGemm
from .gemm import QuantBatchGemmReduceSum
from .gemm import QuantGemm
from .indexer import Indexer
from .indexer import LightningIndexer
from .kv_cache import StorePagedKVCache
from .multimodal_rope_infer import MultimodalRoPEInfer
from .multimodal_rope_infer import MultimodalRoPEInplaceInfer
from .normalization import GroupRMSNormInfer
from .normalization import LayerNormInfer
from .normalization import RMSNorm
from .normalization import RMSNormInfer
from .normalization import ResidualAddLayerNormInfer
from .normalization import ResidualAddRMSNormInfer
from .over_encoding import NF4DequantEmbedding
from .over_encoding import OverEncoding
from .over_encoding import OverEncodingNGram
from .paged_attention import PagedDecodeGQAInfer
from .paged_attention import PagedDecodeSWAInfer
from .paged_attention import PagedPrefillGQAInfer
from .paged_attention import PagedPrefillSWAInfer
from .paged_attention import PrefillGQAInfer
from .quantization import Dequant
from .quantization import DynamicQuant
from .quantization import StaticQuant
from .rope import RoPE
from .rope_cos_sin import RoPECosSin
from .rope_cos_sin import VisionRoPECosSin2D
from .rotary import RoPEInfer
from .rotary import VisionRoPE2DInfer
from .rotate_activation import RotateActivation
from .sampling import ApplyPenaltiesTemperature
from .sampling import JoinProbRejectSampling
from .sampling import RejectSampling
from .sampling import TopKSampling
from .sampling import TopPFilter
from .sampling import TopPSampling
from .sdpa import Sdpa
from .store_lowrank import StoreLowrank

__all__ = [
    "ApplyPenaltiesTemperature",
    "CausalConv1dUpdateStateInfer",
    "Dequant",
    "DequantSwiGLUQuant",
    "DynamicQuant",
    "GELU",
    "GroupGemm",
    "GroupRMSNormInfer",
    "Indexer",
    "JoinProbRejectSampling",
    "LayerNormInfer",
    "LayerNormQuant",
    "LightningIndexer",
    "MoEDynamicQuant",
    "MultimodalRoPEInfer",
    "MultimodalRoPEInplaceInfer",
    "NF4DequantEmbedding",
    "OverEncoding",
    "OverEncodingNGram",
    "PagedDecodeGQAInfer",
    "PagedDecodeSWAInfer",
    "PagedPrefillGQAInfer",
    "PagedPrefillSWAInfer",
    "PrefillGQAInfer",
    "QuantBatchGemmReduceSum",
    "QuantGemm",
    "RMSNorm",
    "RMSNormInfer",
    "RMSNormQuant",
    "RejectSampling",
    "ResidualAddLayerNormInfer",
    "ResidualAddLayerNormQuant",
    "ResidualAddRMSNormInfer",
    "ResidualAddRMSNormQuant",
    "RoPE",
    "RoPECosSin",
    "RoPEInfer",
    "RotateActivation",
    "SWAInfer",
    "Sdpa",
    "SiLU",
    "StaticQuant",
    "StoreLowrank",
    "StorePagedKVCache",
    "SwiGLU",
    "TopKSampling",
    "TopPFilter",
    "TopPSampling",
    "VarlenFAInfer",
    "VisionRoPE2DInfer",
    "VisionRoPECosSin2D",
]


_extend_exports(globals())
del _extend_exports
