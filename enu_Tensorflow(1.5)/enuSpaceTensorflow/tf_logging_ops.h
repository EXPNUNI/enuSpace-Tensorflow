#pragma once

#ifndef _TF_LOGGING_OPS_HEADER_
#define _TF_LOGGING_OPS_HEADER_

#include <string>
#include "include/json/json.h"


void* Create_Assert(std::string id, Json::Value pInputItem);
void* Create_HistogramSummary(std::string id, Json::Value pInputItem);
void* Create_MergeSummary(std::string id, Json::Value pInputItem);
void* Create_Print(std::string id, Json::Value pInputItem);
void* Create_ScalarSummary(std::string id, Json::Value pInputItem);
void* Create_TensorSummary(std::string id, Json::Value pInputItem);


#endif