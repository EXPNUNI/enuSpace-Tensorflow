#pragma once



#ifndef _TF_NN_HEADER_
#define _TF_NN_HEADER_

#include <string>
#include "include/json/json.h"

void* Create_AvgPool(std::string id, Json::Value pInputItem);
void* Create_AvgPool3D(std::string id, Json::Value pInputItem);
void* Create_AvgPool3DGrad(std::string id, Json::Value pInputItem);
void* Create_BiasAdd(std::string id, Json::Value pInputItem);
void* Create_BiasAddGrad(std::string id, Json::Value pInputItem);
void* Create_Conv2D(std::string id, Json::Value pInputItem);
void* Create_Conv2DBackpropFilter(std::string id, Json::Value pInputItem);
void* Create_Conv2DBackpropInput(std::string id, Json::Value pInputItem);
void* Create_Conv3D(std::string id, Json::Value pInputItem);
void* Create_Conv3DBackpropFilterV2(std::string id, Json::Value pInputItem);
void* Create_Conv3DBackpropInputV2(std::string id, Json::Value pInputItem);
void* Create_DepthwiseConv2dNative(std::string id, Json::Value pInputItem);
void* Create_DepthwiseConv2dNativeBackpropFilter(std::string id, Json::Value pInputItem);
void* Create_DepthwiseConv2dNativeBackpropInput(std::string id, Json::Value pInputItem);
void* Create_Dilation2D(std::string id, Json::Value pInputItem);
void* Create_Dilation2DBackpropFilter(std::string id, Json::Value pInputItem);
void* Create_Dilation2DBackpropInput(std::string id, Json::Value pInputItem);
void* Create_Elu(std::string id, Json::Value pInputItem);
void* Create_FractionalAvgPool(std::string id, Json::Value pInputItem);
void* Create_FractionalMaxPool(std::string id, Json::Value pInputItem);
void* Create_FusedBatchNorm(std::string id, Json::Value pInputItem);
void* Create_FusedBatchNormGrad(std::string id, Json::Value pInputItem);
void* Create_FusedPadConv2D(std::string id, Json::Value pInputItem);
void* Create_FusedResizeAndPadConv2D(std::string id, Json::Value pInputItem);
void* Create_InTopK(std::string id, Json::Value pInputItem);
void* Create_L2Loss(std::string id, Json::Value pInputItem);
void* Create_LRN(std::string id, Json::Value pInputItem);
void* Create_LogSoftmax(std::string id, Json::Value pInputItem);
void* Create_MaxPool(std::string id, Json::Value pInputItem);
void* Create_MaxPool3D(std::string id, Json::Value pInputItem);
void* Create_MaxPool3DGrad(std::string id, Json::Value pInputItem);
void* Create_MaxPool3DGradGrad(std::string id, Json::Value pInputItem);
void* Create_MaxPoolGradGrad(std::string id, Json::Value pInputItem);
void* Create_MaxPoolGradGradWithArgmax(std::string id, Json::Value pInputItem);
void* Create_MaxPoolWithArgmax(std::string id, Json::Value pInputItem);
void* Create_QuantizedAvgPool(std::string id, Json::Value pInputItem);
void* Create_QuantizedBatchNormWithGlobalNormalization(std::string id, Json::Value pInputItem);
void* Create_QuantizedBiasAdd(std::string id, Json::Value pInputItem);
void* Create_QuantizedConv2D(std::string id, Json::Value pInputItem);
void* Create_QuantizedMaxPool(std::string id, Json::Value pInputItem);
void* Create_QuantizedRelu(std::string id, Json::Value pInputItem);
void* Create_QuantizedRelu6(std::string id, Json::Value pInputItem);
void* Create_QuantizedReluX(std::string id, Json::Value pInputItem);
void* Create_Relu(std::string id, Json::Value pInputItem);
void* Create_Relu6(std::string id, Json::Value pInputItem);
void* Create_Softmax(std::string id, Json::Value pInputItem);
void* Create_SoftmaxCrossEntropyWithLogits(std::string id, Json::Value pInputItem);
void* Create_Softplus(std::string id, Json::Value pInputItem);
void* Create_Softsign(std::string id, Json::Value pInputItem);
void* Create_SparseSoftmaxCrossEntropyWithLogits(std::string id, Json::Value pInputItem);
void* Create_TopK(std::string id, Json::Value pInputItem);

#endif