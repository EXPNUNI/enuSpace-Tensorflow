#include "stdafx.h"
#include "tf_candidate_sampling_ops.h"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "include/json/json.h"

#include "GlobalHeader.h"
#include "tensorflow.h"
#include "utility_functions.h"
#include "enuSpaceToTensorflow.h"

void* Create_AllCandidateSampler(std::string id, Json::Value pInputItem) {
	AllCandidateSampler* pAllCandidateSampler = nullptr;
	Scope* pScope = nullptr;
	return pAllCandidateSampler;
}

void* Create_ComputeAccidentalHits(std::string id, Json::Value pInputItem) {
	ComputeAccidentalHits* pComputeAccidentalHits = nullptr;
	Scope* pScope = nullptr;
	return pComputeAccidentalHits;
}

void* Create_FixedUnigramCandidateSampler(std::string id, Json::Value pInputItem) {
	FixedUnigramCandidateSampler* pFixedUnigramCandidateSampler = nullptr;
	Scope* pScope = nullptr;
	return pFixedUnigramCandidateSampler;
}

void* Create_LearnedUnigramCandidateSampler(std::string id, Json::Value pInputItem) {
	LearnedUnigramCandidateSampler* pLearnedUnigramCandidateSampler = nullptr;
	Scope* pScope = nullptr;
	return pLearnedUnigramCandidateSampler;
}

void* Create_LogUniformCandidateSampler(std::string id, Json::Value pInputItem) {
	LogUniformCandidateSampler* pLogUniformCandidateSampler = nullptr;
	Scope* pScope = nullptr;
	return pLogUniformCandidateSampler;
}

void* Create_UniformCandidateSampler(std::string id, Json::Value pInputItem) {
	UniformCandidateSampler* pUniformCandidateSampler = nullptr;
	Scope* pScope = nullptr;
	return pUniformCandidateSampler;
}
