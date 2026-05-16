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
from .operators.attention import MojoPagedDecodeGQA
from .operators.attention import MojoPagedDecodeSWA
from .operators.attention import MojoPagedPrefillGQA
from .operators.attention import MojoPagedPrefillSWA
from .operators.attention import MojoPrefillGQA
from .operators.attention import MojoSdpa
from .operators.attention import MojoSWA

""" kvcache """
from .operators.kv_cache import MojoStorePagedKVCache

""" gemm """
from .operators.gemm import MojoGemm
from .operators.gemm import MojoQuantGemm
from .operators.gemm import MojoGroupGemm

""" compute + comm """
from .operators.compute_with_comm import MojoGemmAll2All
from .operators.compute_with_comm import MojoAllGatherGemm
from .operators.compute_with_comm import MojoGemmAllReduce
from .operators.compute_with_comm import MojoGemmReduceScatter

""" embedding """
from .operators.embedding import MojoEmbedding
from .operators.embedding import MojoParallelEmbedding

""" over_encoding """
from .operators.over_encoding import MojoOverEncoding
from .operators.over_encoding import MojoOverEncodingNGram
from .operators.over_encoding import MojoNF4DequantEmbedding

""" quantize """
from .operators.quantize import MojoDequant
from .operators.quantize import MojoDequantSwiGLUQuant
from .operators.quantize import MojoDynamicQuant
from .operators.quantize import MojoMoEDynamicQuant
from .operators.quantize import MojoStaticQuant

""" moe """
from .operators.moe import MojoExperts
from .operators.moe import MojoMoE
from .operators.moe import MojoMoECombine
from .operators.moe import MojoMoEDispatch
from .operators.moe import MojoMoEGating
from .operators.moe import MojoQuantExperts
from .operators.moe import MojoQuantMoE

""" normalization """
from .operators.normalization import MojoGroupRMSNorm
from .operators.normalization import MojoLayerNorm
from .operators.normalization import MojoLayerNormQuant
from .operators.normalization import MojoResidualAddLayerNorm
from .operators.normalization import MojoResidualAddLayerNormQuant
from .operators.normalization import MojoResidualAddRMSNorm
from .operators.normalization import MojoResidualAddRMSNormQuant
from .operators.normalization import MojoRMSNorm
from .operators.normalization import MojoRMSNormQuant

""" position_embedding """
from .operators.position_embedding import MojoApplyRoPE
from .operators.position_embedding import MojoApplyVisionRoPE2D
from .operators.position_embedding import MojoMRoPE
from .operators.position_embedding import MojoRotaryEmbedding
from .operators.position_embedding import MojoVisionRotaryEmbedding2D

""" sampling """
from .operators.sampling import MojoApplyPenaltiesTempurate
from .operators.sampling import MojoJoinProbRejectSampling
from .operators.sampling import MojoRejectSampling
from .operators.sampling import MojoTopKSampling
from .operators.sampling import MojoTopPFilter
from .operators.sampling import MojoTopPSampling

""" convolution"""
from .operators.convolution import MojoCausalConv1dUpdateState

""" mlp"""
from .operators.mlp import MojoSwiGLUMLP

""" functions """
from .functions.activation import MojoSiluFunction
from .functions.attention import MojoSWAFunction
from .functions.convolution import MojoCausalConv1dFunction
from .functions.loss_function import MojoFusedLinearCrossEntropyFunction
from .functions.loss_function import MojoFusedLinearCrossEntropyLoss
from .functions.normalization import MojoRMSNormFunction
from .functions.position_embedding import MojoApplyRoPEFunction

# fmt: off
__all__ = [
    "MojoFunction",
    "MojoOperator",

    "MojoGelu",
    "MojoSilu",
    "MojoSwiGLU",

    "MojoPrefillGQA",
    "MojoPagedPrefillGQA",
    "MojoDecodeGQA",
    "MojoPagedDecodeGQA",
    "MojoSdpa",
    "MojoPagedPrefillSWA",
    "MojoPagedDecodeSWA",
    "MojoSWA",

    "MojoStorePagedKVCache",

    "MojoGemm",
    "MojoQuantGemm",
    "MojoGroupGemm",
    "MojoGemmAll2All",
    "MojoAllGatherGemm",
    "MojoGemmAllReduce",
    "MojoGemmReduceScatter",

    "MojoStaticQuant",
    "MojoDequant",
    "MojoDynamicQuant",
    "MojoMoEDynamicQuant",
    "MojoDequantSwiGLUQuant",

    "MojoEmbedding",
    "MojoParallelEmbedding",
    "MojoNF4DequantEmbedding",
    "MojoOverEncoding",
    "MojoOverEncodingNGram",

    "MojoMoE",
    "MojoMoEGating",
    "MojoMoEDispatch",
    "MojoExperts",
    "MojoMoECombine",
    "MojoQuantExperts",
    "MojoQuantMoE",

    "MojoLayerNorm",
    "MojoRMSNorm",
    "MojoGroupRMSNorm",
    "MojoRMSNormQuant",
    "MojoLayerNormQuant",
    "MojoResidualAddRMSNorm",
    "MojoResidualAddLayerNorm",
    "MojoResidualAddRMSNormQuant",
    "MojoResidualAddLayerNormQuant",

    "MojoRotaryEmbedding",
    "MojoApplyRoPE",
    "MojoApplyVisionRoPE2D",
    "MojoVisionRotaryEmbedding2D",
    "MojoMRoPE",

    "MojoTopPSampling",
    "MojoTopKSampling",
    "MojoRejectSampling",
    "MojoJoinProbRejectSampling",
    "MojoApplyPenaltiesTempurate",
    "MojoTopPFilter",

    "MojoCausalConv1dUpdateState",

    "MojoSwiGLUMLP",

    "MojoSiluFunction",
    "MojoRMSNormFunction",
    "MojoApplyRoPEFunction",
    "MojoFusedLinearCrossEntropyFunction",
    "MojoCausalConv1dFunction",

    "MojoFusedLinearCrossEntropyLoss",

    "MojoSWAFunction",
]
# fmt: on
