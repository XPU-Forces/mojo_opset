"""
All Mojo Operators contained in Mojo Opsets listed here.
"""

# Set of all valid KV layouts for parameter validation (sorted for consistent ordering)
VALID_KV_LAYOUTS = sorted({"NPU_ND", "NPU_NZ", "AMD_CB"})

""" activation """
from .operators.activation import MojoGelu
from .operators.activation import MojoGeluQuant
from .operators.activation import MojoSilu
from .operators.activation import MojoSiluQuant
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
from .operators.attention import MojoBlockSparseAttention
from .operators.attention import MojoPagedPrefillBlockSparseAttention
from .operators.attention import MojoPrefillGQA
from .operators.attention import MojoPrefillMLA
from .operators.attention import MojoPrefillNSA
from .operators.attention import MojoSdpa

""" kvcache """
from .operators.kvcache import MojoKVCacheCast
from .operators.kvcache import MojoStoreKVCache
from .operators.kvcache import MojoStoreMLAKVCache
from .operators.kvcache import MojoStorePagedKVCache
from .operators.kvcache import MojoStorePagedMLAKVCache

""" linear """
from .operators.linear import MojoAllGatherLinear
from .operators.linear import MojoBatchLinear
from .operators.linear import MojoGroupLinear
from .operators.linear import MojoLinear
from .operators.linear import MojoLinearAll2All
from .operators.linear import MojoLinearAllReduce
from .operators.linear import MojoLinearReduceScatter

""" mojo misc """
from .operators.misc import MojoDequant
from .operators.misc import MojoEmbedding
from .operators.misc import MojoParallelEmbedding
from .operators.misc import MojoQuant
from .operators.misc import MojoQuest
from .operators.misc import MojoPagedPrefillBlockQuest

""" moe """
from .operators.moe import MojoBigEPCombine
from .operators.moe import MojoBigEPDispatch
from .operators.moe import MojoMoECombine
from .operators.moe import MojoMoEDispatch
from .operators.moe import MojoMoEGate

""" normalization """
from .operators.normalization import MojoNorm
from .operators.normalization import MojoNormQuant
from .operators.normalization import MojoResidualAddNorm
from .operators.normalization import MojoResidualAddNormCast
from .operators.normalization import MojoResidualAddNormQuant

""" position_embedding """
from .operators.position_embedding import MojoNormRoPE
from .operators.position_embedding import MojoNormRoPEStoreKV
from .operators.position_embedding import MojoRoPE
from .operators.position_embedding import MojoRoPEStoreKV

""" sampling """
from .operators.sampling import MojoApplyPenaltiesTempurate
from .operators.sampling import MojoJoinProbRejectSampling
from .operators.sampling import MojoRejectSampling
from .operators.sampling import MojoTopKSampling
from .operators.sampling import MojoTopPFilter
from .operators.sampling import MojoTopPSampling

""" functions """
from .functions.activation import MojoSiluFunction
from .functions.activation import mojo_silu
from .functions.attention import MojoDiffusionAttentionFunction
from .functions.attention import mojo_diffusion_attention
from .functions.convolution import MojoCausalConv1dFunction
from .functions.convolution import mojo_causal_conv1d
from .functions.loss_function import MojoFusedLinearCrossEntropyFunction
from .functions.loss_function import MojoFusedLinearCrossEntropyLoss
from .functions.normalization import MojoRMSNorm
from .functions.normalization import MojoRMSNormFunction
from .functions.position_embedding import MojoRoPEFunction
from .functions.position_embedding import mojo_rope

# fmt: off
__all__ = [
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
    "MojoSdpa",

    "MojoQuest",
    "MojoPagedPrefillBlockQuest",
    "MojoBlockSparseAttention",
    "MojoPagedPrefillBlockSparseAttention",

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
    "MojoJoinProbRejectSampling",
    "MojoApplyPenaltiesTempurate",
    "MojoTopPFilter",

    "MojoSiluFunction",
    "MojoRMSNormFunction",
    "MojoRoPEFunction",
    "MojoFusedLinearCrossEntropyFunction",
    "MojoCausalConv1dFunction",
    "MojoDiffusionAttentionFunction",

    "mojo_causal_conv1d",
    "mojo_silu",
    "mojo_diffusion_attention",
    "MojoFusedLinearCrossEntropyLoss",
    "mojo_rope",
    "MojoRMSNorm",
]
# fmt: on
