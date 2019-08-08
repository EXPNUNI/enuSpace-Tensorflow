#pragma once

#ifndef _TF_CANDIDATE_SAMPLING_OPS_HEADER_
#define _TF_CANDIDATE_SAMPLING_OPS_HEADER_

#include <string>
#include "jsoncpp/json.h"


void* Create_AllCandidateSampler(std::string id, Json::Value pInputItem);
void* Create_ComputeAccidentalHits(std::string id, Json::Value pInputItem);
void* Create_FixedUnigramCandidateSampler(std::string id, Json::Value pInputItem);
void* Create_LearnedUnigramCandidateSampler(std::string id, Json::Value pInputItem);
void* Create_LogUniformCandidateSampler(std::string id, Json::Value pInputItem);
void* Create_UniformCandidateSampler(std::string id, Json::Value pInputItem);

#endif