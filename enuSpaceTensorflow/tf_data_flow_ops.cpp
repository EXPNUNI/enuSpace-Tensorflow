#include "stdafx.h"
#include "tf_data_flow_ops.h"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "include/json/json.h"

#include "GlobalHeader.h"
#include "tensorflow.h"
#include "utility_functions.h"
#include "enuSpaceToTensorflow.h"

void* Create_AccumulatorApplyGradient(std::string id, Json::Value pInputItem) {
	AccumulatorApplyGradient* pAccumulatorApplyGradient = nullptr;
	Scope* pScope = nullptr;
	return pAccumulatorApplyGradient;
}

void* Create_AccumulatorNumAccumulated(std::string id, Json::Value pInputItem) {
	AccumulatorNumAccumulated* pAccumulatorNumAccumulated = nullptr;
	Scope* pScope = nullptr;
	return pAccumulatorNumAccumulated;
}

void* Create_AccumulatorSetGlobalStep(std::string id, Json::Value pInputItem) {
	AccumulatorSetGlobalStep* pAccumulatorSetGlobalStep = nullptr;
	Scope* pScope = nullptr;
	return pAccumulatorSetGlobalStep;
}

void* Create_AccumulatorTakeGradient(std::string id, Json::Value pInputItem) {
	AccumulatorTakeGradient* pAccumulatorTakeGradient = nullptr;
	Scope* pScope = nullptr;
	return pAccumulatorTakeGradient;
}

void* Create_Barrier(std::string id, Json::Value pInputItem) {
	Barrier* pBarrier = nullptr;
	Scope* pScope = nullptr;
	return pBarrier;
}

void* Create_BarrierClose(std::string id, Json::Value pInputItem) {
	BarrierClose* pBarrierClose = nullptr;
	Scope* pScope = nullptr;
	return pBarrierClose;
}

void* Create_BarrierIncompleteSize(std::string id, Json::Value pInputItem) {
	BarrierIncompleteSize* pBarrierIncompleteSize = nullptr;
	Scope* pScope = nullptr;
	return pBarrierIncompleteSize;
}

void* Create_BarrierInsertMany(std::string id, Json::Value pInputItem) {
	BarrierInsertMany* pBarrierInsertMany = nullptr;
	Scope* pScope = nullptr;
	return pBarrierInsertMany;
}

void* Create_BarrierReadySize(std::string id, Json::Value pInputItem) {
	BarrierReadySize* pBarrierReadySize = nullptr;
	Scope* pScope = nullptr;
	return pBarrierReadySize;
}

void* Create_BarrierTakeMany(std::string id, Json::Value pInputItem) {
	BarrierTakeMany* pBarrierTakeMany = nullptr;
	Scope* pScope = nullptr;
	return pBarrierTakeMany;
}

void* Create_ConditionalAccumulator(std::string id, Json::Value pInputItem) {
	ConditionalAccumulator* pConditionalAccumulator = nullptr;
	Scope* pScope = nullptr;
	return pConditionalAccumulator;
}

void* Create_DeleteSessionTensor(std::string id, Json::Value pInputItem) {
	DeleteSessionTensor* pDeleteSessionTensor = nullptr;
	Scope* pScope = nullptr;
	return pDeleteSessionTensor;
}

void* Create_DynamicPartition(std::string id, Json::Value pInputItem) {
	DynamicPartition* pDynamicPartition = nullptr;
	Scope* pScope = nullptr;
	return pDynamicPartition;
}

void* Create_DynamicStitch(std::string id, Json::Value pInputItem) {
	DynamicStitch* pDynamicStitch = nullptr;
	Scope* pScope = nullptr;
	return pDynamicStitch;
}

void* Create_FIFOQueue(std::string id, Json::Value pInputItem) {
	FIFOQueue* pFIFOQueue = nullptr;
	Scope* pScope = nullptr;
	return pFIFOQueue;
}

void* Create_GetSessionHandle(std::string id, Json::Value pInputItem) {
	GetSessionHandle* pGetSessionHandle = nullptr;
	Scope* pScope = nullptr;
	return pGetSessionHandle;
}

void* Create_GetSessionHandleV2(std::string id, Json::Value pInputItem) {
	GetSessionHandleV2* pGetSessionHandleV2 = nullptr;
	Scope* pScope = nullptr;
	return pGetSessionHandleV2;
}

void* Create_GetSessionTensor(std::string id, Json::Value pInputItem) {
	GetSessionTensor* pGetSessionTensor = nullptr;
	Scope* pScope = nullptr;
	return pGetSessionTensor;
}

void* Create_PaddingFIFOQueue(std::string id, Json::Value pInputItem) {
	PaddingFIFOQueue* pPaddingFIFOQueue = nullptr;
	Scope* pScope = nullptr;
	return pPaddingFIFOQueue;
}

void* Create_PriorityQueue(std::string id, Json::Value pInputItem) {
	PriorityQueue* pPriorityQueue = nullptr;
	Scope* pScope = nullptr;
	return pPriorityQueue;
}

void* Create_QueueClose(std::string id, Json::Value pInputItem) {
	QueueClose* pQueueClose = nullptr;
	Scope* pScope = nullptr;
	return pQueueClose;
}

void* Create_QueueDequeue(std::string id, Json::Value pInputItem) {
	QueueDequeue* pQueueDequeue = nullptr;
	Scope* pScope = nullptr;
	return pQueueDequeue;
}

void* Create_QueueDequeueMany(std::string id, Json::Value pInputItem) {
	QueueDequeueMany* pQueueDequeueMany = nullptr;
	Scope* pScope = nullptr;
	return pQueueDequeueMany;
}

void* Create_QueueDequeueUpTo(std::string id, Json::Value pInputItem) {
	QueueDequeueUpTo* pQueueDequeueUpTo = nullptr;
	Scope* pScope = nullptr;
	return pQueueDequeueUpTo;
}

void* Create_QueueEnqueue(std::string id, Json::Value pInputItem) {
	QueueEnqueue* pQueueEnqueue = nullptr;
	Scope* pScope = nullptr;
	return pQueueEnqueue;
}

void* Create_QueueEnqueueMany(std::string id, Json::Value pInputItem) {
	QueueEnqueueMany* pQueueEnqueueMany = nullptr;
	Scope* pScope = nullptr;
	return pQueueEnqueueMany;
}

void* Create_QueueSize(std::string id, Json::Value pInputItem) {
	QueueSize* pQueueSize = nullptr;
	Scope* pScope = nullptr;
	return pQueueSize;
}

void* Create_RandomShuffleQueue(std::string id, Json::Value pInputItem) {
	RandomShuffleQueue* pRandomShuffleQueue = nullptr;
	Scope* pScope = nullptr;
	return pRandomShuffleQueue;
}

void* Create_RecordInput(std::string id, Json::Value pInputItem) {
	RecordInput* pRecordInput = nullptr;
	Scope* pScope = nullptr;
	return pRecordInput;
}

void* Create_SparseAccumulatorApplyGradient(std::string id, Json::Value pInputItem) {
	SparseAccumulatorApplyGradient* pSparseAccumulatorApplyGradient = nullptr;
	Scope* pScope = nullptr;
	return pSparseAccumulatorApplyGradient;
}

void* Create_SparseAccumulatorTakeGradient(std::string id, Json::Value pInputItem) {
	SparseAccumulatorTakeGradient* pSparseAccumulatorTakeGradient = nullptr;
	Scope* pScope = nullptr;
	return pSparseAccumulatorTakeGradient;
}

void* Create_SparseConditionalAccumulator(std::string id, Json::Value pInputItem) {
	SparseConditionalAccumulator* pSparseConditionalAccumulator = nullptr;
	Scope* pScope = nullptr;
	return pSparseConditionalAccumulator;
}

void* Create_Stage(std::string id, Json::Value pInputItem) {
	Stage* pStage = nullptr;
	Scope* pScope = nullptr;
	return pStage;
}

void* Create_TensorArray(std::string id, Json::Value pInputItem) {
	TensorArray* pTensorArray = nullptr;
	Scope* pScope = nullptr;
	return pTensorArray;
}

void* Create_TensorArrayClose(std::string id, Json::Value pInputItem) {
	TensorArrayClose* pTensorArrayClose = nullptr;
	Scope* pScope = nullptr;
	return pTensorArrayClose;
}

void* Create_TensorArrayConcat(std::string id, Json::Value pInputItem) {
	TensorArrayConcat* pTensorArrayConcat = nullptr;
	Scope* pScope = nullptr;
	return pTensorArrayConcat;
}

void* Create_TensorArrayGather(std::string id, Json::Value pInputItem) {
	TensorArrayGather* pTensorArrayGather = nullptr;
	Scope* pScope = nullptr;
	return pTensorArrayGather;
}

void* Create_TensorArrayGrad(std::string id, Json::Value pInputItem) {
	TensorArrayGrad* pTensorArrayGrad = nullptr;
	Scope* pScope = nullptr;
	return pTensorArrayGrad;
}

void* Create_TensorArrayRead(std::string id, Json::Value pInputItem) {
	TensorArrayRead* pTensorArrayRead = nullptr;
	Scope* pScope = nullptr;
	return pTensorArrayRead;
}

void* Create_TensorArrayScatter(std::string id, Json::Value pInputItem) {
	TensorArrayScatter* pTensorArrayScatter = nullptr;
	Scope* pScope = nullptr;
	return pTensorArrayScatter;
}

void* Create_TensorArraySize(std::string id, Json::Value pInputItem) {
	TensorArraySize* pTensorArraySize = nullptr;
	Scope* pScope = nullptr;
	return pTensorArraySize;
}

void* Create_TensorArraySplit(std::string id, Json::Value pInputItem) {
	TensorArraySplit* pTensorArraySplit = nullptr;
	Scope* pScope = nullptr;
	return pTensorArraySplit;
}

void* Create_TensorArrayWrite(std::string id, Json::Value pInputItem) {
	TensorArrayWrite* pTensorArrayWrite = nullptr;
	Scope* pScope = nullptr;
	return pTensorArrayWrite;
}

void* Create_Unstage(std::string id, Json::Value pInputItem) {
	Unstage* pUnstage = nullptr;
	Scope* pScope = nullptr;
	return pUnstage;
}
