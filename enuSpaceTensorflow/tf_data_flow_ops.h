#pragma once

#ifndef _TF_DATA_FLOW_OPS_HEADER_
#define _TF_DATA_FLOW_OPS_HEADER_

#include <string>
#include "include/json/json.h"


void* Create_AccumulatorApplyGradient(std::string id, Json::Value pInputItem);
void* Create_AccumulatorNumAccumulated(std::string id, Json::Value pInputItem);
void* Create_AccumulatorSetGlobalStep(std::string id, Json::Value pInputItem);
void* Create_AccumulatorTakeGradient(std::string id, Json::Value pInputItem);
void* Create_Barrier(std::string id, Json::Value pInputItem);
void* Create_BarrierClose(std::string id, Json::Value pInputItem);
void* Create_BarrierIncompleteSize(std::string id, Json::Value pInputItem);
void* Create_BarrierInsertMany(std::string id, Json::Value pInputItem);
void* Create_BarrierReadySize(std::string id, Json::Value pInputItem);
void* Create_BarrierTakeMany(std::string id, Json::Value pInputItem);
void* Create_ConditionalAccumulator(std::string id, Json::Value pInputItem);
void* Create_DeleteSessionTensor(std::string id, Json::Value pInputItem);
void* Create_DynamicPartition(std::string id, Json::Value pInputItem);
void* Create_DynamicStitch(std::string id, Json::Value pInputItem);
void* Create_FIFOQueue(std::string id, Json::Value pInputItem);
void* Create_GetSessionHandle(std::string id, Json::Value pInputItem);
void* Create_GetSessionHandleV2(std::string id, Json::Value pInputItem);
void* Create_GetSessionTensor(std::string id, Json::Value pInputItem);

void* Create_MapClear(std::string id, Json::Value pInputItem);
void* Create_MapIncompleteSize(std::string id, Json::Value pInputItem);
void* Create_MapPeek(std::string id, Json::Value pInputItem);
void* Create_MapSize(std::string id, Json::Value pInputItem);
void* Create_MapStage(std::string id, Json::Value pInputItem);
void* Create_MapUnstage(std::string id, Json::Value pInputItem);
void* Create_MapUnstageNoKey(std::string id, Json::Value pInputItem);
void* Create_OrderedMapClear(std::string id, Json::Value pInputItem);
void* Create_OrderedMapIncompleteSize(std::string id, Json::Value pInputItem);
void* Create_OrderedMapPeek(std::string id, Json::Value pInputItem);
void* Create_OrderedMapSize(std::string id, Json::Value pInputItem);
void* Create_OrderedMapStage(std::string id, Json::Value pInputItem);
void* Create_OrderedMapUnstage(std::string id, Json::Value pInputItem);
void* Create_OrderedMapUnstageNoKey(std::string id, Json::Value pInputItem);

void* Create_PaddingFIFOQueue(std::string id, Json::Value pInputItem);
void* Create_PriorityQueue(std::string id, Json::Value pInputItem);
void* Create_QueueClose(std::string id, Json::Value pInputItem);
void* Create_QueueDequeue(std::string id, Json::Value pInputItem);
void* Create_QueueDequeueMany(std::string id, Json::Value pInputItem);
void* Create_QueueDequeueUpTo(std::string id, Json::Value pInputItem);
void* Create_QueueEnqueue(std::string id, Json::Value pInputItem);
void* Create_QueueEnqueueMany(std::string id, Json::Value pInputItem);
void* Create_QueueSize(std::string id, Json::Value pInputItem);
void* Create_RandomShuffleQueue(std::string id, Json::Value pInputItem);
void* Create_RecordInput(std::string id, Json::Value pInputItem);
void* Create_SparseAccumulatorApplyGradient(std::string id, Json::Value pInputItem);
void* Create_SparseAccumulatorTakeGradient(std::string id, Json::Value pInputItem);
void* Create_SparseConditionalAccumulator(std::string id, Json::Value pInputItem);
void* Create_Stage(std::string id, Json::Value pInputItem);

void* Create_StageClear(std::string id, Json::Value pInputItem);
void* Create_StagePeek(std::string id, Json::Value pInputItem);
void* Create_StageSize(std::string id, Json::Value pInputItem);

void* Create_TensorArray(std::string id, Json::Value pInputItem);
void* Create_TensorArrayClose(std::string id, Json::Value pInputItem);
void* Create_TensorArrayConcat(std::string id, Json::Value pInputItem);
void* Create_TensorArrayGather(std::string id, Json::Value pInputItem);
void* Create_TensorArrayGrad(std::string id, Json::Value pInputItem);
void* Create_TensorArrayRead(std::string id, Json::Value pInputItem);
void* Create_TensorArrayScatter(std::string id, Json::Value pInputItem);
void* Create_TensorArraySize(std::string id, Json::Value pInputItem);
void* Create_TensorArraySplit(std::string id, Json::Value pInputItem);
void* Create_TensorArrayWrite(std::string id, Json::Value pInputItem);
void* Create_Unstage(std::string id, Json::Value pInputItem);

#endif