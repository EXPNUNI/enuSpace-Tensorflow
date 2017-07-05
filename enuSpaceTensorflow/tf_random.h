#pragma once

#ifndef _TF_RANDOM_HEADER_
#define _TF_RANDOM_HEADER_

#include <string>
#include "include/json/json.h"


void* Create_Multinomial(std::string id, Json::Value pInputItem);
void* Create_ParameterizedTruncatedNormal(std::string id, Json::Value pInputItem);
void* Create_RandomGamma(std::string id, Json::Value pInputItem);
void* Create_RandomNormal(std::string id, Json::Value pInputItem);
void* Create_RandomPoisson(std::string id, Json::Value pInputItem);
void* Create_RandomShuffle(std::string id, Json::Value pInputItem);
void* Create_RandomUniform(std::string id, Json::Value pInputItem);
void* Create_RandomUniformInt(std::string id, Json::Value pInputItem);
void* Create_TruncatedNormal(std::string id, Json::Value pInputItem);

void* Create_RandomNormal_ex(std::string id, Json::Value pInputItem);

#endif