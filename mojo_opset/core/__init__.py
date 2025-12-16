"""
All Mojo Operators contained in Mojo Opsets listed here.
"""

"""mojo bigop"""
from .bigop.m8_bigop import MojoM8BigOp

""" mojo activation """
from .activation.mojo_gelu import MojoGelu, MojoGeluQuant
from .activation.mojo_silu import MojoSilu, MojoSiluQuant, MojoSiluMul, MojoSwiglu

""" mojo attn """
from .attn.mojo_prefill_gqa import MojoPrefillGQA, MojoPagedPrefillGQA
from .attn.mojo_prefill_mla import MojoPrefillMLA, MojoPagedPrefillMLA
from .attn.mojo_prefill_nsa import MojoPrefillNSA, MojoPagedPrefillNSA
from .attn.mojo_decode_gqa import MojoDecodeGQA, MojoPagedDecodeGQA
from .attn.mojo_decode_mla import MojoDecodeMLA, MojoPagedDecodeMLA
from .attn.mojo_decode_nsa import MojoDecodeNSA, MojoPagedDecodeNSA

""" mojo kvcache """
from .kvcache.mojo_store_kvcache import (
    MojoStoreKVCache,
    MojoStorePagedKVCache,
    MojoStoreMLAKVCache,
    MojoStorePagedMLAKVCache,
)
from .kvcache.mojo_kv_cast import MojoKVCacheCast


""" mojo linear """
from .linear.mojo_linear import MojoLinear, MojoBatchLinear, MojoGroupLinear
from .linear.mojo_linear_cc import (
    MojoLinearAllReduce,
    MojoLinearAll2All,
    MojoAllGatherLinear,
    MojoLinearReduceScatter,
    MojoLinearReduceScatter,
)

""" mojo misc """
from .misc.mojo_quant import MojoQuant, MojoDequant
from .misc.mojo_wte import MojoEmbedding, MojoParallelEmbedding

""" mojo moe """
from .moe.mojo_moe_gate import MojoMoEGate
from .moe.mojo_moe_dispatch import MojoMoEDispatch, MojoBigEPDispatch
from .moe.mojo_moe_combine import MojoMoECombine, MojoBigEPCombine


""" mojo norm """
from .norm.mojo_norm import MojoNorm, MojoNormQuant
from .norm.mojo_add_norm import (
    MojoResidualAddNorm,
    MojoResidualAddNormQuant,
    MojoResidualAddNormCast,
)

""" mojo pos_emb """
from .pos_emb.mojo_rope import (
    MojoRoPE,
    MojoRoPEStoreKV,
    MojoNormRoPE,
    MojoNormRoPEStoreKV,
)

""" mojo matmul """
from .matmul.mojo_group_quant_matmul_reduce_sum import (
    MojoGroupQuantMatmulReduceSum,
)

from .matmul.mojo_grouped_matmul import (
    MojoGroupedMatmul,
)

""" mojo sampling """
from .sampling.mojo_sampling import (
    MojoTopPSampling,
    MojoTopKSampling,
    MojoRejectSampling,
)

from .functions.mojo_silu_func import MojoSiluFunction
from .functions.mojo_rms_norm_func import MojoRMSNormFunction
from .functions.mojo_rope_func import MojoRoPEFunction
from .functions.mojo_linear_crossentropy_func import MojoFusedLinearCrossEntropyFunction


# fmt: off
__all__ = [
    "MojoM8BigOp",

    "MojoGelu",
    "MojoGeluQuant",
    "MojoSilu",
    "MojoSiluQuant",
    "MojoSiluMul",
    "MojoSwiglu",

    "MojoPrefillGQA",
    "MojoPagedPrefillGQA",
    "MojoPrefillMLA",
    "MojoPagedPrefillMLA",
    "MojoPrefillNSA",
    "MojoPagedPrefillNSA",
    "MojoDecodeGQA",
    "MojoPagedDecodeGQA",
    "MojoDecodeMLA",
    "MojoPagedDecodeMLA",
    "MojoDecodeNSA",
    "MojoPagedDecodeNSA",

    "MojoStoreKVCache",
    "MojoStorePagedKVCache",
    "MojoStoreMLAKVCache",
    "MojoStorePagedMLAKVCache",
    "MojoKVCacheCast",

    "MojoLinear",
    "MojoBatchLinear",
    "MojoGroupLinear",
    "MojoLinearAllReduce",
    "MojoLinearAll2All",
    "MojoAllGatherLinear",
    "MojoLinearReduceScatter",

    "MojoQuant",
    "MojoDequant",

    "MojoEmbedding",
    "MojoParallelEmbedding",

    "MojoMoEGate",
    "MojoMoEDispatch",
    "MojoBigEPDispatch",
    "MojoMoECombine",
    "MojoBigEPCombine",

    "MojoNorm",
    "MojoNormQuant",
    "MojoResidualAddNorm",
    "MojoResidualAddNormQuant",
    "MojoResidualAddNormCast",

    "MojoRoPE",
    "MojoRoPEStoreKV",
    "MojoNormRoPE",
    "MojoNormRoPEStoreKV",

    "MojoGroupQuantMatmulReduceSum",
    "MojoGroupedMatmul",

    "MojoTopPSampling",
    "MojoTopKSampling",
    "MojoRejectSampling",

    "MojoSiluFunction",
    "MojoRMSNormFunction",
    "MojoRoPEFunction",
    "MojoFusedLinearCrossEntropyFunction",
]
# fmt: on
