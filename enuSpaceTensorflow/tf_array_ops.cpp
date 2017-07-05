#include "stdafx.h"
#include "tf_array_ops.h"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "include/json/json.h"

#include "GlobalHeader.h"
#include "tensorflow.h"
#include "utility_functions.h"
#include "enuSpaceToTensorflow.h"

void* Create_BatchToSpace(std::string id, Json::Value pInputItem) {
	BatchToSpace* pBatchToSpace = nullptr;
	Scope* pScope = nullptr;
	return pBatchToSpace;
}

void* Create_BatchToSpaceND(std::string id, Json::Value pInputItem) {
	BatchToSpaceND* pBatchToSpaceND = nullptr;
	Scope* pScope = nullptr;
	return pBatchToSpaceND;
}

void* Create_Bitcast(std::string id, Json::Value pInputItem) {
	Bitcast* pBitcast = nullptr;
	Scope* pScope = nullptr;
	return pBitcast;
}

void* Create_BroadcastDynamicShape(std::string id, Json::Value pInputItem) {
	BroadcastDynamicShape* pBroadcastDynamicShape = nullptr;
	Scope* pScope = nullptr;
	return pBroadcastDynamicShape;
}

void* Create_CheckNumerics(std::string id, Json::Value pInputItem) {
	CheckNumerics* pCheckNumerics = nullptr;
	Scope* pScope = nullptr;
	return pCheckNumerics;
}

void* Create_Concat(std::string id, Json::Value pInputItem) {
	Concat* pConcat = nullptr;
	Scope* pScope = nullptr;
	return pConcat;
}

void* Create_DepthToSpace(std::string id, Json::Value pInputItem) {
	DepthToSpace* pDepthToSpace = nullptr;
	Scope* pScope = nullptr;
	return pDepthToSpace;
}

void* Create_Dequantize(std::string id, Json::Value pInputItem) {
	Dequantize* pDequantize = nullptr;
	Scope* pScope = nullptr;
	return pDequantize;
}

void* Create_Diag(std::string id, Json::Value pInputItem) {
	Diag* pDiag = nullptr;
	Scope* pScope = nullptr;
	return pDiag;
}

void* Create_DiagPart(std::string id, Json::Value pInputItem) {
	DiagPart* pDiagPart = nullptr;
	Scope* pScope = nullptr;
	return pDiagPart;
}

void* Create_EditDistance(std::string id, Json::Value pInputItem) {
	EditDistance* pEditDistance = nullptr;
	Scope* pScope = nullptr;
	return pEditDistance;
}

void* Create_ExpandDims(std::string id, Json::Value pInputItem) {
	ExpandDims* pExpandDims = nullptr;
	Scope* pScope = nullptr;
	return pExpandDims;
}

void* Create_ExtractImagePatches(std::string id, Json::Value pInputItem) {
	ExtractImagePatches* pExtractImagePatches = nullptr;
	Scope* pScope = nullptr;
	return pExtractImagePatches;
}

void* Create_FakeQuantWithMinMaxArgs(std::string id, Json::Value pInputItem) {
	FakeQuantWithMinMaxArgs* pFakeQuantWithMinMaxArgs = nullptr;
	Scope* pScope = nullptr;
	return pFakeQuantWithMinMaxArgs;
}

void* Create_FakeQuantWithMinMaxArgsGradient(std::string id, Json::Value pInputItem) {
	FakeQuantWithMinMaxArgsGradient* pFakeQuantWithMinMaxArgsGradient = nullptr;
	Scope* pScope = nullptr;
	return pFakeQuantWithMinMaxArgsGradient;
}

void* Create_FakeQuantWithMinMaxVars(std::string id, Json::Value pInputItem) {
	FakeQuantWithMinMaxVars* pFakeQuantWithMinMaxVars = nullptr;
	Scope* pScope = nullptr;
	return pFakeQuantWithMinMaxVars;
}

void* Create_FakeQuantWithMinMaxVarsGradient(std::string id, Json::Value pInputItem) {
	FakeQuantWithMinMaxVarsGradient* pFakeQuantWithMinMaxVarsGradient = nullptr;
	Scope* pScope = nullptr;
	return pFakeQuantWithMinMaxVarsGradient;
}

void* Create_FakeQuantWithMinMaxVarsPerChannel(std::string id, Json::Value pInputItem) {
	FakeQuantWithMinMaxVarsPerChannel* pFakeQuantWithMinMaxVarsPerChannel = nullptr;
	Scope* pScope = nullptr;
	return pFakeQuantWithMinMaxVarsPerChannel;
}

void* Create_FakeQuantWithMinMaxVarsPerChannelGradient(std::string id, Json::Value pInputItem) {
	FakeQuantWithMinMaxVarsPerChannelGradient* pFakeQuantWithMinMaxVarsPerChannelGradient = nullptr;
	Scope* pScope = nullptr;
	return pFakeQuantWithMinMaxVarsPerChannelGradient;
}

void* Create_Fill(std::string id, Json::Value pInputItem) {
	Fill* pFill = nullptr;
	Scope* pScope = nullptr;
	return pFill;
}

void* Create_Gather(std::string id, Json::Value pInputItem) {
	Gather* pGather = nullptr;
	Scope* pScope = nullptr;
	return pGather;
}

void* Create_GatherNd(std::string id, Json::Value pInputItem) {
	GatherNd* pGatherNd = nullptr;
	Scope* pScope = nullptr;
	return pGatherNd;
}

void* Create_Identity(std::string id, Json::Value pInputItem) {
	Identity* pIdentity = nullptr;
	Scope* pScope = nullptr;
	return pIdentity;
}

void* Create_ImmutableConst(std::string id, Json::Value pInputItem) {
	ImmutableConst* pImmutableConst = nullptr;
	Scope* pScope = nullptr;
	return pImmutableConst;
}

void* Create_InvertPermutation(std::string id, Json::Value pInputItem) {
	InvertPermutation* pInvertPermutation = nullptr;
	Scope* pScope = nullptr;
	return pInvertPermutation;
}

void* Create_MatrixBandPart(std::string id, Json::Value pInputItem) {
	MatrixBandPart* pMatrixBandPart = nullptr;
	Scope* pScope = nullptr;
	return pMatrixBandPart;
}

void* Create_MatrixDiag(std::string id, Json::Value pInputItem) {
	MatrixDiag* pMatrixDiag = nullptr;
	Scope* pScope = nullptr;
	return pMatrixDiag;
}

void* Create_MatrixDiagPart(std::string id, Json::Value pInputItem) {
	MatrixDiagPart* pMatrixDiagPart = nullptr;
	Scope* pScope = nullptr;
	return pMatrixDiagPart;
}

void* Create_MatrixSetDiag(std::string id, Json::Value pInputItem) {
	MatrixSetDiag* pMatrixSetDiag = nullptr;
	Scope* pScope = nullptr;
	return pMatrixSetDiag;
}

void* Create_MirrorPad(std::string id, Json::Value pInputItem) {
	MirrorPad* pMirrorPad = nullptr;
	Scope* pScope = nullptr;
	return pMirrorPad;
}

void* Create_OneHot(std::string id, Json::Value pInputItem) {
	OneHot* pOneHot = nullptr;
	Scope* pScope = nullptr;
	return pOneHot;
}

void* Create_OnesLike(std::string id, Json::Value pInputItem) {
	//OnesLike* pOnesLike = nullptr;
	//Scope* pScope = nullptr;
	return NULL;
}

void* Create_Pad(std::string id, Json::Value pInputItem) {
	Pad* pPad = nullptr;
	Scope* pScope = nullptr;
	return pPad;
}

void* Create_ParallelConcat(std::string id, Json::Value pInputItem) {
	ParallelConcat* pParallelConcat = nullptr;
	Scope* pScope = nullptr;
	return pParallelConcat;
}

void* Create_Placeholder(std::string id, Json::Value pInputItem) {
	Placeholder* pPlaceholder = nullptr;
	Scope* pScope = nullptr;
	tensorflow::DataType dtype = DT_DOUBLE;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Placeholder - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				if (strPinInitial == "double")
					dtype = DT_DOUBLE;
				else if (strPinInitial == "float")
					dtype = DT_FLOAT;
				else if (strPinInitial == "int")
					dtype = DT_INT32;
				else if (strPinInitial == "bool")
					dtype = DT_BOOL;
				else if (strPinInitial == "string")
					dtype = DT_STRING;
				else
				{
					std::string msg = string_format("warning : Placeholder - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : Placeholder - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}

		else
		{
			std::string msg = string_format("warning : Placeholder pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pPlaceholder = new Placeholder(*pScope, dtype);
		ObjectInfo* pObj = AddObjectMap(pPlaceholder, id, SYMBOL_PLACEHOLDER, "Placeholder", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pPlaceholder->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : Placeholder(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}

	return pPlaceholder;
}

void* Create_PlaceholderV2(std::string id, Json::Value pInputItem) {
	PlaceholderV2* pPlaceholderV2 = nullptr;
	Scope* pScope = nullptr;
	return pPlaceholderV2;
}

void* Create_PlaceholderWithDefault(std::string id, Json::Value pInputItem) {
	PlaceholderWithDefault* pPlaceholderWithDefault = nullptr;
	Scope* pScope = nullptr;
	return pPlaceholderWithDefault;
}

void* Create_PreventGradient(std::string id, Json::Value pInputItem) {
	PreventGradient* pPreventGradient = nullptr;
	Scope* pScope = nullptr;
	return pPreventGradient;
}

void* Create_QuantizeAndDequantizeV2(std::string id, Json::Value pInputItem) {
	QuantizeAndDequantizeV2* pQuantizeAndDequantizeV2 = nullptr;
	Scope* pScope = nullptr;
	return pQuantizeAndDequantizeV2;
}

void* Create_QuantizeV2(std::string id, Json::Value pInputItem) {
	QuantizeV2* pQuantizeV2 = nullptr;
	Scope* pScope = nullptr;
	return pQuantizeV2;
}

void* Create_QuantizedConcat(std::string id, Json::Value pInputItem) {
	QuantizedConcat* pQuantizedConcat = nullptr;
	Scope* pScope = nullptr;
	return pQuantizedConcat;
}

void* Create_QuantizedInstanceNorm(std::string id, Json::Value pInputItem) {
	QuantizedInstanceNorm* pQuantizedInstanceNorm = nullptr;
	Scope* pScope = nullptr;
	return pQuantizedInstanceNorm;
}

void* Create_QuantizedReshape(std::string id, Json::Value pInputItem) {
	QuantizedReshape* pQuantizedReshape = nullptr;
	Scope* pScope = nullptr;
	return pQuantizedReshape;
}

void* Create_Rank(std::string id, Json::Value pInputItem) {
	Rank* pRank = nullptr;
	Scope* pScope = nullptr;
	return pRank;
}

void* Create_Reshape(std::string id, Json::Value pInputItem) {
	Reshape* pReshape = nullptr;
	Scope* pScope = nullptr;
	return pReshape;
}

void* Create_ResourceStridedSliceAssign(std::string id, Json::Value pInputItem) {
	//ResourceStridedSliceAssign* pResourceStridedSliceAssign = nullptr;
	Scope* pScope = nullptr;
	return NULL;
}

void* Create_Reverse(std::string id, Json::Value pInputItem) {
	Reverse* pReverse = nullptr;
	Scope* pScope = nullptr;
	return pReverse;
}

void* Create_ReverseSequence(std::string id, Json::Value pInputItem) {
	ReverseSequence* pReverseSequence = nullptr;
	Scope* pScope = nullptr;
	return pReverseSequence;
}

void* Create_ScatterNd(std::string id, Json::Value pInputItem) {
	ScatterNd* pScatterNd = nullptr;
	Scope* pScope = nullptr;
	return pScatterNd;
}

void* Create_SetDiff1D(std::string id, Json::Value pInputItem) {
	SetDiff1D* pSetDiff1D = nullptr;
	Scope* pScope = nullptr;
	return pSetDiff1D;
}

void* Create_Shape(std::string id, Json::Value pInputItem) {
	Shape* pShape = nullptr;
	Scope* pScope = nullptr;
	return pShape;
}

void* Create_ShapeN(std::string id, Json::Value pInputItem) {
	ShapeN* pShapeN = nullptr;
	Scope* pScope = nullptr;
	return pShapeN;
}

void* Create_Size(std::string id, Json::Value pInputItem) {
	Size* pSize = nullptr;
	Scope* pScope = nullptr;
	return pSize;
}

void* Create_Slice(std::string id, Json::Value pInputItem) {
	Slice* pSlice = nullptr;
	Scope* pScope = nullptr;
	return pSlice;
}

void* Create_SpaceToBatch(std::string id, Json::Value pInputItem) {
	SpaceToBatch* pSpaceToBatch = nullptr;
	Scope* pScope = nullptr;
	return pSpaceToBatch;
}

void* Create_SpaceToBatchND(std::string id, Json::Value pInputItem) {
	SpaceToBatchND* pSpaceToBatchND = nullptr;
	Scope* pScope = nullptr;
	return pSpaceToBatchND;
}

void* Create_SpaceToDepth(std::string id, Json::Value pInputItem) {
	SpaceToDepth* pSpaceToDepth = nullptr;
	Scope* pScope = nullptr;
	return pSpaceToDepth;
}

void* Create_Split(std::string id, Json::Value pInputItem) {
	Split* pSplit = nullptr;
	Scope* pScope = nullptr;
	return pSplit;
}

void* Create_SplitV(std::string id, Json::Value pInputItem) {
	SplitV* pSplitV = nullptr;
	Scope* pScope = nullptr;
	return pSplitV;
}

void* Create_Squeeze(std::string id, Json::Value pInputItem) {
	Squeeze* pSqueeze = nullptr;
	Scope* pScope = nullptr;
	return pSqueeze;
}

void* Create_Stack(std::string id, Json::Value pInputItem) {
	Stack* pStack = nullptr;
	Scope* pScope = nullptr;
	return pStack;
}

void* Create_StopGradient(std::string id, Json::Value pInputItem) {
	StopGradient* pStopGradient = nullptr;
	Scope* pScope = nullptr;
	return pStopGradient;
}

void* Create_StridedSlice(std::string id, Json::Value pInputItem) {
	StridedSlice* pStridedSlice = nullptr;
	Scope* pScope = nullptr;
	return pStridedSlice;
}

void* Create_StridedSliceAssign(std::string id, Json::Value pInputItem) {
	StridedSliceAssign* pStridedSliceAssign = nullptr;
	Scope* pScope = nullptr;
	return pStridedSliceAssign;
}

void* Create_StridedSliceGrad(std::string id, Json::Value pInputItem) {
	StridedSliceGrad* pStridedSliceGrad = nullptr;
	Scope* pScope = nullptr;
	return pStridedSliceGrad;
}

void* Create_Tile(std::string id, Json::Value pInputItem) {
	Tile* pTile = nullptr;
	Scope* pScope = nullptr;
	return pTile;
}

void* Create_Transpose(std::string id, Json::Value pInputItem) {
	Transpose* pTranspose = nullptr;
	Scope* pScope = nullptr;
	return pTranspose;
}

void* Create_Unique(std::string id, Json::Value pInputItem) {
	Unique* pUnique = nullptr;
	Scope* pScope = nullptr;
	return pUnique;
}

void* Create_UniqueWithCounts(std::string id, Json::Value pInputItem) {
	UniqueWithCounts* pUniqueWithCounts = nullptr;
	Scope* pScope = nullptr;
	return pUniqueWithCounts;
}

void* Create_Unstack(std::string id, Json::Value pInputItem) {
	Unstack* pUnstack = nullptr;
	Scope* pScope = nullptr;
	return pUnstack;
}

void* Create_Where(std::string id, Json::Value pInputItem) {
	Where* pWhere = nullptr;
	Scope* pScope = nullptr;
	return pWhere;
}

void* Create_ZerosLike(std::string id, Json::Value pInputItem) {
	ZerosLike* pZerosLike = nullptr;
	Scope* pScope = nullptr;
	return pZerosLike;
}
