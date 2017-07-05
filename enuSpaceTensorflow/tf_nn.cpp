#include "stdafx.h"
#include "tf_nn.h"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "include/json/json.h"

#include "GlobalHeader.h"
#include "tensorflow.h"
#include "utility_functions.h"
#include "enuSpaceToTensorflow.h"

void* Create_AvgPool(std::string id, Json::Value pInputItem) {
	AvgPool* pAvgPool = nullptr;
	Scope* pScope = nullptr;
	return pAvgPool;
}

void* Create_AvgPool3D(std::string id, Json::Value pInputItem) {
	AvgPool3D* pAvgPool3D = nullptr;
	Scope* pScope = nullptr;
	return pAvgPool3D;
}

void* Create_AvgPool3DGrad(std::string id, Json::Value pInputItem) {
	AvgPool3DGrad* pAvgPool3DGrad = nullptr;
	Scope* pScope = nullptr;
	return pAvgPool3DGrad;
}

void* Create_BiasAdd(std::string id, Json::Value pInputItem) {
	BiasAdd* pBiasAdd = nullptr;
	Scope* pScope = nullptr;
	return pBiasAdd;
}

void* Create_BiasAddGrad(std::string id, Json::Value pInputItem) {
	BiasAddGrad* pBiasAddGrad = nullptr;
	Scope* pScope = nullptr;
	return pBiasAddGrad;
}

void* Create_Conv2D(std::string id, Json::Value pInputItem) {
	Conv2D* pConv2D = nullptr;
	Scope* pScope = nullptr;
	return pConv2D;
}

void* Create_Conv2DBackpropFilter(std::string id, Json::Value pInputItem) {
	Conv2DBackpropFilter* pConv2DBackpropFilter = nullptr;
	Scope* pScope = nullptr;
	return pConv2DBackpropFilter;
}

void* Create_Conv2DBackpropInput(std::string id, Json::Value pInputItem) {
	Conv2DBackpropInput* pConv2DBackpropInput = nullptr;
	Scope* pScope = nullptr;
	return pConv2DBackpropInput;
}

void* Create_Conv3D(std::string id, Json::Value pInputItem) {
	Conv3D* pConv3D = nullptr;
	Scope* pScope = nullptr;
	return pConv3D;
}

void* Create_Conv3DBackpropFilterV2(std::string id, Json::Value pInputItem) {
	Conv3DBackpropFilterV2* pConv3DBackpropFilterV2 = nullptr;
	Scope* pScope = nullptr;
	return pConv3DBackpropFilterV2;
}

void* Create_Conv3DBackpropInputV2(std::string id, Json::Value pInputItem) {
	Conv3DBackpropInputV2* pConv3DBackpropInputV2 = nullptr;
	Scope* pScope = nullptr;
	return pConv3DBackpropInputV2;
}

void* Create_DepthwiseConv2dNative(std::string id, Json::Value pInputItem) {
	DepthwiseConv2dNative* pDepthwiseConv2dNative = nullptr;
	Scope* pScope = nullptr;
	return pDepthwiseConv2dNative;
}

void* Create_DepthwiseConv2dNativeBackpropFilter(std::string id, Json::Value pInputItem) {
	DepthwiseConv2dNativeBackpropFilter* pDepthwiseConv2dNativeBackpropFilter = nullptr;
	Scope* pScope = nullptr;
	return pDepthwiseConv2dNativeBackpropFilter;
}

void* Create_DepthwiseConv2dNativeBackpropInput(std::string id, Json::Value pInputItem) {
	DepthwiseConv2dNativeBackpropInput* pDepthwiseConv2dNativeBackpropInput = nullptr;
	Scope* pScope = nullptr;
	return pDepthwiseConv2dNativeBackpropInput;
}

void* Create_Dilation2D(std::string id, Json::Value pInputItem) {
	Dilation2D* pDilation2D = nullptr;
	Scope* pScope = nullptr;
	return pDilation2D;
}

void* Create_Dilation2DBackpropFilter(std::string id, Json::Value pInputItem) {
	Dilation2DBackpropFilter* pDilation2DBackpropFilter = nullptr;
	Scope* pScope = nullptr;
	return pDilation2DBackpropFilter;
}

void* Create_Dilation2DBackpropInput(std::string id, Json::Value pInputItem) {
	Dilation2DBackpropInput* pDilation2DBackpropInput = nullptr;
	Scope* pScope = nullptr;
	return pDilation2DBackpropInput;
}

void* Create_Elu(std::string id, Json::Value pInputItem) {
	Elu* pElu = nullptr;
	Scope* pScope = nullptr;
	return pElu;
}

void* Create_FractionalAvgPool(std::string id, Json::Value pInputItem) {
	FractionalAvgPool* pFractionalAvgPool = nullptr;
	Scope* pScope = nullptr;
	return pFractionalAvgPool;
}

void* Create_FractionalMaxPool(std::string id, Json::Value pInputItem) {
	FractionalMaxPool* pFractionalMaxPool = nullptr;
	Scope* pScope = nullptr;
	return pFractionalMaxPool;
}

void* Create_FusedBatchNorm(std::string id, Json::Value pInputItem) {
	FusedBatchNorm* pFusedBatchNorm = nullptr;
	Scope* pScope = nullptr;
	return pFusedBatchNorm;
}

void* Create_FusedBatchNormGrad(std::string id, Json::Value pInputItem) {
	FusedBatchNormGrad* pFusedBatchNormGrad = nullptr;
	Scope* pScope = nullptr;
	return pFusedBatchNormGrad;
}

void* Create_FusedPadConv2D(std::string id, Json::Value pInputItem) {
	FusedPadConv2D* pFusedPadConv2D = nullptr;
	Scope* pScope = nullptr;
	return pFusedPadConv2D;
}

void* Create_FusedResizeAndPadConv2D(std::string id, Json::Value pInputItem) {
	FusedResizeAndPadConv2D* pFusedResizeAndPadConv2D = nullptr;
	Scope* pScope = nullptr;
	return pFusedResizeAndPadConv2D;
}

void* Create_InTopK(std::string id, Json::Value pInputItem) {
	InTopK* pInTopK = nullptr;
	Scope* pScope = nullptr;
	return pInTopK;
}

void* Create_L2Loss(std::string id, Json::Value pInputItem) {
	L2Loss* pL2Loss = nullptr;
	Scope* pScope = nullptr;
	return pL2Loss;
}

void* Create_LRN(std::string id, Json::Value pInputItem) {
	LRN* pLRN = nullptr;
	Scope* pScope = nullptr;
	return pLRN;
}

void* Create_LogSoftmax(std::string id, Json::Value pInputItem) {
	LogSoftmax* pLogSoftmax = nullptr;
	Scope* pScope = nullptr;
	return pLogSoftmax;
}

void* Create_MaxPool(std::string id, Json::Value pInputItem) {
	MaxPool* pMaxPool = nullptr;
	Scope* pScope = nullptr;
	return pMaxPool;
}

void* Create_MaxPool3D(std::string id, Json::Value pInputItem) {
	MaxPool3D* pMaxPool3D = nullptr;
	Scope* pScope = nullptr;
	return pMaxPool3D;
}

void* Create_MaxPool3DGrad(std::string id, Json::Value pInputItem) {
	MaxPool3DGrad* pMaxPool3DGrad = nullptr;
	Scope* pScope = nullptr;
	return pMaxPool3DGrad;
}

void* Create_MaxPool3DGradGrad(std::string id, Json::Value pInputItem) {
	//MaxPool3DGradGrad* pMaxPool3DGradGrad = nullptr;
	Scope* pScope = nullptr;
	return NULL;
}

void* Create_MaxPoolGradGrad(std::string id, Json::Value pInputItem) {
	//MaxPoolGradGrad* pMaxPoolGradGrad = nullptr;
	Scope* pScope = nullptr;
	return NULL;
}

void* Create_MaxPoolGradGradWithArgmax(std::string id, Json::Value pInputItem) {
	//MaxPoolGradGradWithArgmax* pMaxPoolGradGradWithArgmax = nullptr;
	Scope* pScope = nullptr;
	return NULL;
}

void* Create_MaxPoolWithArgmax(std::string id, Json::Value pInputItem) {
	MaxPoolWithArgmax* pMaxPoolWithArgmax = nullptr;
	Scope* pScope = nullptr;
	return pMaxPoolWithArgmax;
}

void* Create_QuantizedAvgPool(std::string id, Json::Value pInputItem) {
	QuantizedAvgPool* pQuantizedAvgPool = nullptr;
	Scope* pScope = nullptr;
	return pQuantizedAvgPool;
}

void* Create_QuantizedBatchNormWithGlobalNormalization(std::string id, Json::Value pInputItem) {
	QuantizedBatchNormWithGlobalNormalization* pQuantizedBatchNormWithGlobalNormalization = nullptr;
	Scope* pScope = nullptr;
	return pQuantizedBatchNormWithGlobalNormalization;
}

void* Create_QuantizedBiasAdd(std::string id, Json::Value pInputItem) {
	QuantizedBiasAdd* pQuantizedBiasAdd = nullptr;
	Scope* pScope = nullptr;
	return pQuantizedBiasAdd;
}

void* Create_QuantizedConv2D(std::string id, Json::Value pInputItem) {
	QuantizedConv2D* pQuantizedConv2D = nullptr;
	Scope* pScope = nullptr;
	return pQuantizedConv2D;
}

void* Create_QuantizedMaxPool(std::string id, Json::Value pInputItem) {
	QuantizedMaxPool* pQuantizedMaxPool = nullptr;
	Scope* pScope = nullptr;
	return pQuantizedMaxPool;
}

void* Create_QuantizedRelu(std::string id, Json::Value pInputItem) {
	QuantizedRelu* pQuantizedRelu = nullptr;
	Scope* pScope = nullptr;
	return pQuantizedRelu;
}

void* Create_QuantizedRelu6(std::string id, Json::Value pInputItem) {
	QuantizedRelu6* pQuantizedRelu6 = nullptr;
	Scope* pScope = nullptr;
	return pQuantizedRelu6;
}

void* Create_QuantizedReluX(std::string id, Json::Value pInputItem) {
	QuantizedReluX* pQuantizedReluX = nullptr;
	Scope* pScope = nullptr;
	return pQuantizedReluX;
}

void* Create_Relu(std::string id, Json::Value pInputItem) {
	Relu* pRelu = nullptr;
	Scope* pScope = nullptr;
	return pRelu;
}

void* Create_Relu6(std::string id, Json::Value pInputItem) {
	Relu6* pRelu6 = nullptr;
	Scope* pScope = nullptr;
	return pRelu6;
}

void* Create_Softmax(std::string id, Json::Value pInputItem) {
	Softmax* pSoftmax = nullptr;
	Scope* pScope = nullptr;
	return pSoftmax;
}

void* Create_SoftmaxCrossEntropyWithLogits(std::string id, Json::Value pInputItem) {
	SoftmaxCrossEntropyWithLogits* pSoftmaxCrossEntropyWithLogits = nullptr;
	Scope* pScope = nullptr;
	return pSoftmaxCrossEntropyWithLogits;
}

void* Create_Softplus(std::string id, Json::Value pInputItem) {
	Softplus* pSoftplus = nullptr;
	Scope* pScope = nullptr;
	return pSoftplus;
}

void* Create_Softsign(std::string id, Json::Value pInputItem) {
	Softsign* pSoftsign = nullptr;
	Scope* pScope = nullptr;
	return pSoftsign;
}

void* Create_SparseSoftmaxCrossEntropyWithLogits(std::string id, Json::Value pInputItem) {
	SparseSoftmaxCrossEntropyWithLogits* pSparseSoftmaxCrossEntropyWithLogits = nullptr;
	Scope* pScope = nullptr;
	return pSparseSoftmaxCrossEntropyWithLogits;
}

void* Create_TopK(std::string id, Json::Value pInputItem) {
	TopK* pTopK = nullptr;
	Scope* pScope = nullptr;
	return pTopK;
}

