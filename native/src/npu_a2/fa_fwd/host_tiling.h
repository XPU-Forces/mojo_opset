#pragma once
#include <tiling/tiling_api.h>
namespace optiling {
BEGIN_TILING_DATA_DEF(FACopyTransposeTiling)
TILING_DATA_FIELD_DEF(uint32_t, dstShapeB);
TILING_DATA_FIELD_DEF(uint32_t, dstShapeN);
TILING_DATA_FIELD_DEF(uint32_t, dstShapeS);
TILING_DATA_FIELD_DEF(uint32_t, dstShapeHN);
TILING_DATA_FIELD_DEF(uint32_t, dstShapeH);
TILING_DATA_FIELD_DEF(uint32_t, srcShapeB);
TILING_DATA_FIELD_DEF(uint32_t, srcShapeN);
TILING_DATA_FIELD_DEF(uint32_t, srcShapeS);
TILING_DATA_FIELD_DEF(uint32_t, srcShapeHN);
TILING_DATA_FIELD_DEF(uint32_t, originalShapeNLen);
TILING_DATA_FIELD_DEF(uint32_t, shapeSHValue);
TILING_DATA_FIELD_DEF(uint32_t, shapeNsValue);
TILING_DATA_FIELD_DEF(uint32_t, shapeNsnValue);
TILING_DATA_FIELD_DEF(uint32_t, invalidParamCopyTransposeTiling);
TILING_DATA_FIELD_DEF(uint32_t, shapeBHValue);
TILING_DATA_FIELD_DEF(uint32_t, paramsAlign);
END_TILING_DATA_DEF;
BEGIN_TILING_DATA_DEF(FAInputParams)
TILING_DATA_FIELD_DEF(int64_t, bSize);
TILING_DATA_FIELD_DEF(int64_t, n2Size);
TILING_DATA_FIELD_DEF(int64_t, gSize);
TILING_DATA_FIELD_DEF(int64_t, s1Size);
TILING_DATA_FIELD_DEF(int64_t, s2Size);
TILING_DATA_FIELD_DEF(int64_t, alignedS2);
TILING_DATA_FIELD_DEF(int64_t, dSize);
TILING_DATA_FIELD_DEF(float, keepProb);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(int64_t, preTokens);
TILING_DATA_FIELD_DEF(int64_t, nextTokens);
// in pse encoding scenes, s1 and s2 might not equal with s1, s2 in Q, K
TILING_DATA_FIELD_DEF(int64_t, pseS1Size);
TILING_DATA_FIELD_DEF(int64_t, pseS2Size);
TILING_DATA_FIELD_DEF(uint32_t, pseBSize);
TILING_DATA_FIELD_DEF(uint32_t, bandIndex);

// 1: BSH/BSND, 2: SBH, 3: BNSD
TILING_DATA_FIELD_DEF(uint8_t, layoutType);
// 0: (B,N2,G,S1,S2), 1: (B,N2,G,1,S2)
TILING_DATA_FIELD_DEF(uint8_t, pseShapeType);
// 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
TILING_DATA_FIELD_DEF(uint8_t, attenMaskShapeType);
// 0: fp16, 1: bool(uint8)
TILING_DATA_FIELD_DEF(uint8_t, attenMaskDataType);
// ALL: 0, NONE: 1, ANY: 2, CAUSAL: 3, BAND: 4 };
TILING_DATA_FIELD_DEF(uint8_t, attenMaskCompressMode);
// 0: high precise, 1: high performance, 2: invalid line high precise
TILING_DATA_FIELD_DEF(uint8_t, implMode);
TILING_DATA_FIELD_DEF(uint8_t, sparseType);
TILING_DATA_FIELD_DEF(uint8_t, needDropMaskOp);
TILING_DATA_FIELD_DEF(uint8_t, pseEncodeType);
TILING_DATA_FIELD_DEF(uint8_t, rsv);
TILING_DATA_FIELD_DEF(uint16_t, remain);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskS2Size);
TILING_DATA_FIELD_DEF(uint32_t, pseType);
TILING_DATA_FIELD_DEF(uint32_t, rsv1);
TILING_DATA_FIELD_DEF(int64_t, qStartIdx);
TILING_DATA_FIELD_DEF(int64_t, kvStartIdx);
END_TILING_DATA_DEF;
BEGIN_TILING_DATA_DEF(FAMultiCoreParams)
TILING_DATA_FIELD_DEF(int32_t, coreNum);
TILING_DATA_FIELD_DEF(int32_t, reserve);
// BN2GS1.o
TILING_DATA_FIELD_DEF(int64_t, totalSize);
// BN2GS1.o / core_num
TILING_DATA_FIELD_DEF(int64_t, splitFactorSize);
TILING_DATA_FIELD_DEF(int64_t, splitFactorTailSize);
TILING_DATA_FIELD_DEF_ARR(int64_t, 48, sparseStartIdx);
END_TILING_DATA_DEF;
BEGIN_TILING_DATA_DEF(FACoreParams)
TILING_DATA_FIELD_DEF(int32_t, s1BaseSize);
TILING_DATA_FIELD_DEF(int32_t, s1BaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, s1OuterSize);
TILING_DATA_FIELD_DEF(int32_t, s1Vec2BaseSize);
TILING_DATA_FIELD_DEF(int32_t, s1Vec2BaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, s1Vec2OuterSize);
TILING_DATA_FIELD_DEF(int32_t, s2BaseSize);
TILING_DATA_FIELD_DEF(int32_t, s2BaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, s2OuterSize);
TILING_DATA_FIELD_DEF(int32_t, dBaseSize);
TILING_DATA_FIELD_DEF(int32_t, dBaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, dOuterSize);
TILING_DATA_FIELD_DEF(int32_t, bBaseSize);
TILING_DATA_FIELD_DEF(int32_t, bBaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, bOuterSize);
TILING_DATA_FIELD_DEF(int32_t, n2BaseSize);
TILING_DATA_FIELD_DEF(int32_t, n2BaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, n2OuterSize);
TILING_DATA_FIELD_DEF(int32_t, gBaseSize);
TILING_DATA_FIELD_DEF(int32_t, gBaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, gOuterSize);
TILING_DATA_FIELD_DEF(int32_t, nRatio);
TILING_DATA_FIELD_DEF(int32_t, rsvd);
TILING_DATA_FIELD_DEF(int64_t, s1SparseValidSize);
TILING_DATA_FIELD_DEF(int64_t, s2SparseValidSize);
TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS1);
TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS2);

END_TILING_DATA_DEF;
BEGIN_TILING_DATA_DEF(FATensorSizeParams)
TILING_DATA_FIELD_DEF(int32_t, bmm1ResUbSize);
TILING_DATA_FIELD_DEF(int32_t, attenMaskUbSize);
TILING_DATA_FIELD_DEF(int32_t, pseUbSize);
TILING_DATA_FIELD_DEF(int32_t, dropMaskUbSize);
TILING_DATA_FIELD_DEF(int32_t, castUbSize);
TILING_DATA_FIELD_DEF(int32_t, softmaxMaxUbSize);
TILING_DATA_FIELD_DEF(int32_t, softmaxSumUbSize);
TILING_DATA_FIELD_DEF(int32_t, softmaxExpUbSize);
TILING_DATA_FIELD_DEF(int32_t, apiTmpBufferBytes);
TILING_DATA_FIELD_DEF(int32_t, bmm2ResUbSize);
TILING_DATA_FIELD_DEF(int32_t, inputQueBytes);
TILING_DATA_FIELD_DEF(int32_t, outputQueBytes);
// API buffer use remain space of ub
TILING_DATA_FIELD_DEF(int32_t, tmpBufBytes);
TILING_DATA_FIELD_DEF(int32_t, softmaxMaxOffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, softmaxSumOffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, maxSumApiOffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, customSoftmaxApiOffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, pseTbufOffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, dropoutApiOffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, maxSumApiSize);
TILING_DATA_FIELD_DEF(int32_t, customSoftmaxApiSize);
TILING_DATA_FIELD_DEF(int32_t, dropoutApiSize);
TILING_DATA_FIELD_DEF(int32_t, attenMaskApiSize);
TILING_DATA_FIELD_DEF(int32_t, attenMaskApiOffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, bmm1ProcessTInStage2Size);
TILING_DATA_FIELD_DEF(int32_t, bmm1ProcessTInStage2OffsetBytes);
// workspace
TILING_DATA_FIELD_DEF(int32_t, wkspSection1OffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, wkspSection2OffsetBytes);
END_TILING_DATA_DEF;
BEGIN_TILING_DATA_DEF(FADropmaskParams)
TILING_DATA_FIELD_DEF(int32_t, multiCoreFactorSize);
TILING_DATA_FIELD_DEF(int32_t, baseUbCalSize);
TILING_DATA_FIELD_DEF(int64_t, multiCoreTotalSize);
TILING_DATA_FIELD_DEF(int64_t, shapeTotalSize);
END_TILING_DATA_DEF;
BEGIN_TILING_DATA_DEF(FlashAttentionTrainingGeneralTilingData)
TILING_DATA_FIELD_DEF_STRUCT(FAInputParams, inputParams);
TILING_DATA_FIELD_DEF_STRUCT(FAMultiCoreParams, multiCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(FACoreParams, coreParams);
TILING_DATA_FIELD_DEF_STRUCT(FATensorSizeParams, tensorSizeParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxFlashTilingData);
TILING_DATA_FIELD_DEF_STRUCT(FACopyTransposeTiling, transposeTilingData);
TILING_DATA_FIELD_DEF_STRUCT(FACopyTransposeTiling, transposeTilingDataTailCore);
TILING_DATA_FIELD_DEF_STRUCT(FADropmaskParams, dropmaskParams);
END_TILING_DATA_DEF;
} // namespace optiling