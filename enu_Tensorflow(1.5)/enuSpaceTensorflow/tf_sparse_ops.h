#pragma once

#ifndef _TF_SPARSE_OPS_HEADER_
#define _TF_SPARSE_OPS_HEADER_

#include <string>
#include "include/json/json.h"


void* Create_AddManySparseToTensorsMap(std::string id, Json::Value pInputItem);
void* Create_AddSparseToTensorsMap(std::string id, Json::Value pInputItem);
void* Create_DeserializeManySparse(std::string id, Json::Value pInputItem);
void* Create_SerializeManySparse(std::string id, Json::Value pInputItem);
void* Create_SerializeSparse(std::string id, Json::Value pInputItem);
void* Create_SparseAdd(std::string id, Json::Value pInputItem);
void* Create_SparseAddGrad(std::string id, Json::Value pInputItem);
void* Create_SparseConcat(std::string id, Json::Value pInputItem);
void* Create_SparseCross(std::string id, Json::Value pInputItem);
void* Create_SparseDenseCwiseAdd(std::string id, Json::Value pInputItem);
void* Create_SparseDenseCwiseDiv(std::string id, Json::Value pInputItem);
void* Create_SparseDenseCwiseMul(std::string id, Json::Value pInputItem);
void* Create_SparseReduceSum(std::string id, Json::Value pInputItem);
void* Create_SparseReduceSumSparse(std::string id, Json::Value pInputItem);
void* Create_SparseReorder(std::string id, Json::Value pInputItem);
void* Create_SparseReshape(std::string id, Json::Value pInputItem);
void* Create_SparseSoftmax(std::string id, Json::Value pInputItem);
void* Create_SparseSparseMaximum(std::string id, Json::Value pInputItem);
void* Create_SparseSparseMinimum(std::string id, Json::Value pInputItem);
void* Create_SparseSplit(std::string id, Json::Value pInputItem);
void* Create_SparseTensorDenseAdd(std::string id, Json::Value pInputItem);
void* Create_SparseTensorDenseMatMul(std::string id, Json::Value pInputItem);
void* Create_SparseToDense(std::string id, Json::Value pInputItem);
void* Create_TakeManySparseFromTensorsMap(std::string id, Json::Value pInputItem);

void* Create_SparseFillEmptyRows(std::string id, Json::Value pInputItem);
void* Create_SparseFillEmptyRowsGrad(std::string id, Json::Value pInputItem);
void* Create_SparseReduceMax(std::string id, Json::Value pInputItem);
void* Create_SparseReduceMaxSparse(std::string id, Json::Value pInputItem);
void* Create_SparseSlice(std::string id, Json::Value pInputItem);
#endif