#include "stdafx.h"
#include "tf_logging_ops.h"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "include/json/json.h"

#include "GlobalHeader.h"
#include "tensorflow.h"
#include "utility_functions.h"
#include "enuSpaceToTensorflow.h"



void* Create_Assert(std::string id, Json::Value pInputItem) {
	Assert* pAssert = nullptr;
	Scope* pScope = nullptr;
	return pAssert;
}

void* Create_HistogramSummary(std::string id, Json::Value pInputItem) {
	HistogramSummary* pHistogramSummary = nullptr;
	Scope* pScope = nullptr;
	return pHistogramSummary;
}

void* Create_MergeSummary(std::string id, Json::Value pInputItem) {
	MergeSummary* pMergeSummary = nullptr;
	Scope* pScope = nullptr;
	return pMergeSummary;
}

void* Create_Print(std::string id, Json::Value pInputItem) {
	Print* pPrint = nullptr;
	Scope* pScope = nullptr;
	return pPrint;
}

void* Create_ScalarSummary(std::string id, Json::Value pInputItem) {
	ScalarSummary* pScalarSummary = nullptr;
	Scope* pScope = nullptr;
	return pScalarSummary;
}

void* Create_TensorSummary(std::string id, Json::Value pInputItem) {
	TensorSummary* pTensorSummary = nullptr;
	Scope* pScope = nullptr;
	return pTensorSummary;
}