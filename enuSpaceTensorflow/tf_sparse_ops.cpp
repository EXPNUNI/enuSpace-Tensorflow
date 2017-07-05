#include "stdafx.h"
#include "tf_sparse_ops.h"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "include/json/json.h"

#include "GlobalHeader.h"
#include "tensorflow.h"
#include "utility_functions.h"
#include "enuSpaceToTensorflow.h"

void* Create_AddManySparseToTensorsMap(std::string id, Json::Value pInputItem) {
	AddManySparseToTensorsMap* pAddManySparseToTensorsMap = nullptr;
	Scope* pScope = nullptr;
	return pAddManySparseToTensorsMap;
}

void* Create_AddSparseToTensorsMap(std::string id, Json::Value pInputItem) {
	AddSparseToTensorsMap* pAddSparseToTensorsMap = nullptr;
	Scope* pScope = nullptr;
	return pAddSparseToTensorsMap;
}

void* Create_DeserializeManySparse(std::string id, Json::Value pInputItem) {
	DeserializeManySparse* pDeserializeManySparse = nullptr;
	Scope* pScope = nullptr;
	return pDeserializeManySparse;
}

void* Create_SerializeManySparse(std::string id, Json::Value pInputItem) {
	SerializeManySparse* pSerializeManySparse = nullptr;
	Scope* pScope = nullptr;
	return pSerializeManySparse;
}

void* Create_SerializeSparse(std::string id, Json::Value pInputItem) {
	SerializeSparse* pSerializeSparse = nullptr;
	Scope* pScope = nullptr;
	return pSerializeSparse;
}

void* Create_SparseAdd(std::string id, Json::Value pInputItem) {
	SparseAdd* pSparseAdd = nullptr;
	Scope* pScope = nullptr;
	return pSparseAdd;
}

void* Create_SparseAddGrad(std::string id, Json::Value pInputItem) {
	SparseAddGrad* pSparseAddGrad = nullptr;
	Scope* pScope = nullptr;
	return pSparseAddGrad;
}

void* Create_SparseConcat(std::string id, Json::Value pInputItem) {
	SparseConcat* pSparseConcat = nullptr;
	Scope* pScope = nullptr;
	return pSparseConcat;
}

void* Create_SparseCross(std::string id, Json::Value pInputItem) {
	//SparseCross* pSparseCross = nullptr;
	Scope* pScope = nullptr;
	return NULL;
}

void* Create_SparseDenseCwiseAdd(std::string id, Json::Value pInputItem) {
	SparseDenseCwiseAdd* pSparseDenseCwiseAdd = nullptr;
	Scope* pScope = nullptr;
	return pSparseDenseCwiseAdd;
}

void* Create_SparseDenseCwiseDiv(std::string id, Json::Value pInputItem) {
	SparseDenseCwiseDiv* pSparseDenseCwiseDiv = nullptr;
	Scope* pScope = nullptr;
	return pSparseDenseCwiseDiv;
}

void* Create_SparseDenseCwiseMul(std::string id, Json::Value pInputItem) {
	SparseDenseCwiseMul* pSparseDenseCwiseMul = nullptr;
	Scope* pScope = nullptr;
	return pSparseDenseCwiseMul;
}

void* Create_SparseReduceSum(std::string id, Json::Value pInputItem) {
	SparseReduceSum* pSparseReduceSum = nullptr;
	Scope* pScope = nullptr;
	return pSparseReduceSum;
}

void* Create_SparseReduceSumSparse(std::string id, Json::Value pInputItem) {
	SparseReduceSumSparse* pSparseReduceSumSparse = nullptr;
	Scope* pScope = nullptr;
	return pSparseReduceSumSparse;
}

void* Create_SparseReorder(std::string id, Json::Value pInputItem) {
	SparseReorder* pSparseReorder = nullptr;
	Scope* pScope = nullptr;
	return pSparseReorder;
}

void* Create_SparseReshape(std::string id, Json::Value pInputItem) {
	SparseReshape* pSparseReshape = nullptr;
	Scope* pScope = nullptr;
	return pSparseReshape;
}

void* Create_SparseSoftmax(std::string id, Json::Value pInputItem) {
	SparseSoftmax* pSparseSoftmax = nullptr;
	Scope* pScope = nullptr;
	return pSparseSoftmax;
}

void* Create_SparseSparseMaximum(std::string id, Json::Value pInputItem) {
	SparseSparseMaximum* pSparseSparseMaximum = nullptr;
	Scope* pScope = nullptr;
	return pSparseSparseMaximum;
}

void* Create_SparseSparseMinimum(std::string id, Json::Value pInputItem) {
	SparseSparseMinimum* pSparseSparseMinimum = nullptr;
	Scope* pScope = nullptr;
	return pSparseSparseMinimum;
}

void* Create_SparseSplit(std::string id, Json::Value pInputItem) {
	SparseSplit* pSparseSplit = nullptr;
	Scope* pScope = nullptr;
	return pSparseSplit;
}

void* Create_SparseTensorDenseAdd(std::string id, Json::Value pInputItem) {
	SparseTensorDenseAdd* pSparseTensorDenseAdd = nullptr;
	Scope* pScope = nullptr;
	return pSparseTensorDenseAdd;
}

void* Create_SparseTensorDenseMatMul(std::string id, Json::Value pInputItem) {
	SparseTensorDenseMatMul* pSparseTensorDenseMatMul = nullptr;
	Scope* pScope = nullptr;
	return pSparseTensorDenseMatMul;
}

void* Create_SparseToDense(std::string id, Json::Value pInputItem) {
	SparseToDense* pSparseToDense = nullptr;
	Scope* pScope = nullptr;
	return pSparseToDense;
}

void* Create_TakeManySparseFromTensorsMap(std::string id, Json::Value pInputItem) {
	TakeManySparseFromTensorsMap* pTakeManySparseFromTensorsMap = nullptr;
	Scope* pScope = nullptr;
	return pTakeManySparseFromTensorsMap;
}
