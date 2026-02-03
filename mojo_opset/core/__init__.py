"""
All Mojo Operators contained in Mojo Opsets listed here.
"""

# Set of all valid KV layouts for parameter validation (sorted for consistent ordering)
VALID_KV_LAYOUTS = sorted({"NPU_ND", "NPU_NZ", "AMD_CB"})

""" base class """
from .function import MojoFunction
from .operator import MojoOperator

""" activation """
from .operators.activation import MojoGelu
from .operators.activation import MojoSilu
from .operators.activation import MojoSwiGLU

""" attention """
from .operators.attention import MojoDecodeGQA
from .operators.attention import MojoDecodeMLA
from .operators.attention import MojoDecodeNSA
from .operators.attention import MojoPagedDecodeGQA
from .operators.attention import MojoPagedDecodeMLA
from .operators.attention import MojoPagedDecodeNSA
from .operators.attention import MojoPagedPrefillGQA
from .operators.attention import MojoPagedPrefillMLA
from .operators.attention import MojoPagedPrefillNSA
from .operators.attention import MojoPrefillGQA
from .operators.attention import MojoPrefillMLA
from .operators.attention import MojoPrefillNSA
from .operators.attention import MojoSdpa
from .operators.attention import MojoT5FullAttention

""" kvcache """
from .operators.kv_cache import MojoStoreMLAKVCache
from .operators.kv_cache import MojoStorePagedKVCache
from .operators.kv_cache import MojoStorePagedMLAKVCache

""" linear """
from .operators.linear import MojoAllGatherLinear
from .operators.linear import MojoBatchLinear
from .operators.linear import MojoGroupLinear
from .operators.linear import MojoLinear
from .operators.linear import MojoLinearAll2All
from .operators.linear import MojoLinearAllReduce
from .operators.linear import MojoLinearReduceScatter

""" embedding """
from .operators.embedding import MojoEmbedding
from .operators.embedding import MojoParallelEmbedding
from .operators.embedding import MojoRelativeEmbedding

""" quantize """
from .operators.quantize import MojoDequant
from .operators.quantize import MojoQuant

""" moe """
from .operators.moe import MojoMoECombine
from .operators.moe import MojoMoEDispatch
from .operators.moe import MojoMoEGating

""" normalization """
from .operators.normalization import MojoLayerNorm
from .operators.normalization import MojoNormQuant
from .operators.normalization import MojoResidualAddLayerNorm
from .operators.normalization import MojoResidualAddNormCast
from .operators.normalization import MojoResidualAddNormQuant
from .operators.normalization import MojoResidualAddRMSNorm
from .operators.normalization import MojoRMSNorm
from .operators.normalization import MojoChannelRMSNorm

""" position_embedding """
from .operators.position_embedding import MojoNormRoPE
from .operators.position_embedding import MojoNormRoPEStoreKV
from .operators.position_embedding import MojoRoPE
from .operators.position_embedding import MojoRoPEStoreKV
from .operators.position_embedding import MojoGridRoPE

""" sampling """
from .operators.sampling import MojoApplyPenaltiesTempurate
from .operators.sampling import MojoJoinProbRejectSampling
from .operators.sampling import MojoRejectSampling
from .operators.sampling import MojoTopKSampling
from .operators.sampling import MojoTopPFilter
from .operators.sampling import MojoTopPSampling

""" convolution"""
from .operators.convolution import MojoCausalConv1dUpdateState

""" functions """
from .functions.activation import MojoSiluFunction
from .functions.convolution import MojoCausalConv1dFunction
from .functions.loss_function import MojoFusedLinearCrossEntropyFunction
from .functions.loss_function import MojoFusedLinearCrossEntropyLoss
from .functions.normalization import MojoRMSNormFunction
from .functions.position_embedding import MojoRoPEFunction

# fmt: off
__all__ = [
    "MojoFunction",
    "MojoOperator",

    "MojoGelu",
    "MojoSilu",
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
    "MojoSdpa",
    "MojoT5FullAttention",

    "MojoStorePagedKVCache",
    "MojoStoreMLAKVCache",
    "MojoStorePagedMLAKVCache",

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
    "MojoRelativeEmbedding",

    "MojoMoEGating",
    "MojoMoEDispatch",
    "MojoMoECombine",

    "MojoLayerNorm",
    "MojoRMSNorm",
    "MojoChannelRMSNorm",
    "MojoResidualAddRMSNorm",
    "MojoResidualAddLayerNorm",
    "MojoNormQuant",
    "MojoResidualAddNormQuant",
    "MojoResidualAddNormCast",

    "MojoRoPE",
    "MojoRoPEStoreKV",
    "MojoNormRoPE",
    "MojoNormRoPEStoreKV",
    "MojoGridRoPE",

    "MojoTopPSampling",
    "MojoTopKSampling",
    "MojoRejectSampling",
    "MojoJoinProbRejectSampling",
    "MojoApplyPenaltiesTempurate",
    "MojoTopPFilter",

    "MojoCausalConv1dUpdateState",

    "MojoSiluFunction",
    "MojoRMSNormFunction",
    "MojoRoPEFunction",
    "MojoFusedLinearCrossEntropyFunction",
    "MojoCausalConv1dFunction",

    "MojoFusedLinearCrossEntropyLoss",
]
# fmt: on
