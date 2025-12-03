"""
All Mojo Operators contained in Mojo Opsets listed here.
"""

# Set of all valid KV layouts for parameter validation (sorted for consistent ordering)
VALID_KV_LAYOUTS = sorted({"NPU_ND", "NPU_NZ", "AMD_CB"})

"""mojo bigop"""
from .bigop.m8_bigop import MojoM8BigOp

""" mojo activation """
from .activation.mojo_gelu import MojoGelu
from .activation.mojo_gelu import MojoGeluQuant
from .activation.mojo_silu import MojoSilu
from .activation.mojo_silu import MojoSiluQuant
from .activation.mojo_swiglu import MojoSwiGLU

""" mojo attn """
from .attn.mojo_decode_gqa import MojoDecodeGQA
from .attn.mojo_decode_gqa import MojoPagedDecodeGQA
from .attn.mojo_decode_mla import MojoDecodeMLA
from .attn.mojo_decode_mla import MojoPagedDecodeMLA
from .attn.mojo_decode_nsa import MojoDecodeNSA
from .attn.mojo_decode_nsa import MojoPagedDecodeNSA
from .attn.mojo_prefill_gqa import MojoPagedPrefillGQA
from .attn.mojo_prefill_gqa import MojoPrefillGQA
from .attn.mojo_prefill_mla import MojoPagedPrefillMLA
from .attn.mojo_prefill_mla import MojoPrefillMLA
from .attn.mojo_prefill_nsa import MojoPagedPrefillNSA
from .attn.mojo_prefill_nsa import MojoPrefillNSA

""" mojo kvcache """
from .kvcache.mojo_kv_cast import MojoKVCacheCast
from .kvcache.mojo_store_kvcache import MojoStoreKVCache
from .kvcache.mojo_store_kvcache import MojoStoreMLAKVCache
from .kvcache.mojo_store_kvcache import MojoStorePagedKVCache
from .kvcache.mojo_store_kvcache import MojoStorePagedMLAKVCache

""" mojo linear """
from .linear.mojo_linear import MojoBatchLinear
from .linear.mojo_linear import MojoGroupLinear
from .linear.mojo_linear import MojoLinear
from .linear.mojo_linear_cc import MojoAllGatherLinear
from .linear.mojo_linear_cc import MojoLinearAll2All
from .linear.mojo_linear_cc import MojoLinearAllReduce
from .linear.mojo_linear_cc import MojoLinearReduceScatter

""" mojo misc """
from .misc.mojo_quant import MojoDequant
from .misc.mojo_quant import MojoQuant
from .misc.mojo_wte import MojoEmbedding
from .misc.mojo_wte import MojoParallelEmbedding

""" mojo moe """
from .moe.mojo_moe_combine import MojoBigEPCombine
from .moe.mojo_moe_combine import MojoMoECombine
from .moe.mojo_moe_dispatch import MojoBigEPDispatch
from .moe.mojo_moe_dispatch import MojoMoEDispatch
from .moe.mojo_moe_gate import MojoMoEGate

""" mojo norm """
from .norm.mojo_add_norm import MojoResidualAddNorm
from .norm.mojo_add_norm import MojoResidualAddNormCast
from .norm.mojo_add_norm import MojoResidualAddNormQuant
from .norm.mojo_norm import MojoNorm
from .norm.mojo_norm import MojoNormQuant

""" mojo pos_emb """
from .pos_emb.mojo_rope import MojoNormRoPE
from .pos_emb.mojo_rope import MojoNormRoPEStoreKV
from .pos_emb.mojo_rope import MojoRoPE
from .pos_emb.mojo_rope import MojoRoPEStoreKV

""" mojo sampling """
from .sampling.mojo_sampling import MojoApplyPenalties
from .sampling.mojo_sampling import MojoRejectSampling
from .sampling.mojo_sampling import MojoTopKSampling
from .sampling.mojo_sampling import MojoTopPSampling

""" mojo func """
from .functions.mojo_gated_delta_rule_func import MojoGatedDeltaRuleFunction
from .functions.mojo_linear_crossentropy_func import MojoFusedLinearCrossEntropyFunction
from .functions.mojo_rmsnorm_func import MojoRMSNormFunction
from .functions.mojo_rope_func import MojoRoPEFunction

# fmt: off
__all__ = [
    "MojoM8BigOp",

    "MojoGelu",
    "MojoGeluQuant",
    "MojoSilu",
    "MojoSiluQuant",
    "MojoSwiGLU",

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

    "MojoTopPSampling",
    "MojoTopKSampling",
    "MojoRejectSampling",
    "MojoApplyPenalties",

    "MojoSiluFunction",
    "MojoRMSNormFunction",
    "MojoRoPEFunction",
    "MojoFusedLinearCrossEntropyFunction",
]
# fmt: on
