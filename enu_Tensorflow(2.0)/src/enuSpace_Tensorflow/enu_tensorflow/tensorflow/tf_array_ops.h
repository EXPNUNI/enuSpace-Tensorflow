#pragma once

#ifndef _TF_ARRAY_OPS_HEADER_
#define _TF_ARRAY_OPS_HEADER_

#include <string>
#include "jsoncpp/json.h"


void* Create_BatchToSpace(std::string id, Json::Value pInputItem);
void* Create_BatchToSpaceND(std::string id, Json::Value pInputItem);
void* Create_Bitcast(std::string id, Json::Value pInputItem);
void* Create_BroadcastDynamicShape(std::string id, Json::Value pInputItem);
void* Create_CheckNumerics(std::string id, Json::Value pInputItem);
void* Create_Concat(std::string id, Json::Value pInputItem);
void* Create_DebugGradientIdentity(std::string id, Json::Value pInputItem);
void* Create_DepthToSpace(std::string id, Json::Value pInputItem);
void* Create_Dequantize(std::string id, Json::Value pInputItem);
void* Create_Diag(std::string id, Json::Value pInputItem);
void* Create_DiagPart(std::string id, Json::Value pInputItem);
void* Create_EditDistance(std::string id, Json::Value pInputItem);
void* Create_ExpandDims(std::string id, Json::Value pInputItem);
void* Create_ExtractImagePatches(std::string id, Json::Value pInputItem);
void* Create_FakeQuantWithMinMaxArgs(std::string id, Json::Value pInputItem);
void* Create_FakeQuantWithMinMaxArgsGradient(std::string id, Json::Value pInputItem);
void* Create_FakeQuantWithMinMaxVars(std::string id, Json::Value pInputItem);
void* Create_FakeQuantWithMinMaxVarsGradient(std::string id, Json::Value pInputItem);
void* Create_FakeQuantWithMinMaxVarsPerChannel(std::string id, Json::Value pInputItem);
void* Create_FakeQuantWithMinMaxVarsPerChannelGradient(std::string id, Json::Value pInputItem);
void* Create_Fill(std::string id, Json::Value pInputItem);
void* Create_Gather(std::string id, Json::Value pInputItem);
void* Create_GatherNd(std::string id, Json::Value pInputItem);
void* Create_GatherV2(std::string id, Json::Value pInputItem);
void* Create_Identity(std::string id, Json::Value pInputItem);
void* Create_ImmutableConst(std::string id, Json::Value pInputItem);
void* Create_InvertPermutation(std::string id, Json::Value pInputItem);
void* Create_MatrixBandPart(std::string id, Json::Value pInputItem);
void* Create_MatrixDiag(std::string id, Json::Value pInputItem);
void* Create_MatrixDiagPart(std::string id, Json::Value pInputItem);
void* Create_MatrixSetDiag(std::string id, Json::Value pInputItem);
void* Create_MirrorPad(std::string id, Json::Value pInputItem);
void* Create_OneHot(std::string id, Json::Value pInputItem);
void* Create_OnesLike(std::string id, Json::Value pInputItem);
void* Create_Pad(std::string id, Json::Value pInputItem);
void* Create_PadV2(std::string id, Json::Value pInputItem);
void* Create_ParallelConcat(std::string id, Json::Value pInputItem);
void* Create_Placeholder(std::string id, Json::Value pInputItem);
void* Create_PlaceholderWithDefault(std::string id, Json::Value pInputItem);
void* Create_PreventGradient(std::string id, Json::Value pInputItem);
void* Create_QuantizeAndDequantizeV2(std::string id, Json::Value pInputItem);
void* Create_QuantizeV2(std::string id, Json::Value pInputItem);
void* Create_QuantizedConcat(std::string id, Json::Value pInputItem);
void* Create_QuantizedInstanceNorm(std::string id, Json::Value pInputItem);
void* Create_QuantizedReshape(std::string id, Json::Value pInputItem);
void* Create_Rank(std::string id, Json::Value pInputItem);
void* Create_Reshape(std::string id, Json::Value pInputItem);
void* Create_ResourceStridedSliceAssign(std::string id, Json::Value pInputItem);
void* Create_Reverse(std::string id, Json::Value pInputItem);
void* Create_ReverseSequence(std::string id, Json::Value pInputItem);
void* Create_ScatterNd(std::string id, Json::Value pInputItem);
void* Create_ScatterNdNonAliasingAdd(std::string id, Json::Value pInputItem);
void* Create_SetDiff1D(std::string id, Json::Value pInputItem);
void* Create_Shape(std::string id, Json::Value pInputItem);
void* Create_ShapeN(std::string id, Json::Value pInputItem);
void* Create_Size(std::string id, Json::Value pInputItem);
void* Create_Slice(std::string id, Json::Value pInputItem);
void* Create_SpaceToBatch(std::string id, Json::Value pInputItem);
void* Create_SpaceToBatchND(std::string id, Json::Value pInputItem);
void* Create_SpaceToDepth(std::string id, Json::Value pInputItem);
void* Create_Split(std::string id, Json::Value pInputItem);
void* Create_SplitV(std::string id, Json::Value pInputItem);
void* Create_Squeeze(std::string id, Json::Value pInputItem);
void* Create_Stack(std::string id, Json::Value pInputItem);
void* Create_StopGradient(std::string id, Json::Value pInputItem);
void* Create_StridedSlice(std::string id, Json::Value pInputItem);
void* Create_StridedSliceAssign(std::string id, Json::Value pInputItem);
void* Create_StridedSliceGrad(std::string id, Json::Value pInputItem);
void* Create_Tile(std::string id, Json::Value pInputItem);
void* Create_Transpose(std::string id, Json::Value pInputItem);
void* Create_Unique(std::string id, Json::Value pInputItem);
void* Create_UniqueWithCounts(std::string id, Json::Value pInputItem);
void* Create_Unstack(std::string id, Json::Value pInputItem);
void* Create_Where(std::string id, Json::Value pInputItem);
void* Create_ZerosLike(std::string id, Json::Value pInputItem);

#endif