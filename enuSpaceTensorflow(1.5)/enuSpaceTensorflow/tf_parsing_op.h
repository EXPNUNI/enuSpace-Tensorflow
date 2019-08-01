#pragma once


#ifndef _TF_PARSING_OPS_HEADER_
#define _TF_PARSING_OPS_HEADER_

#include <string>
#include "include/json/json.h"


void* Create_DecodeCSV(std::string id, Json::Value pInputItem);
void* Create_DecodeJSONExample(std::string id, Json::Value pInputItem);
void* Create_DecodeRaw(std::string id, Json::Value pInputItem);
void* Create_ParseExample(std::string id, Json::Value pInputItem);
void* Create_ParseSingleSequenceExample(std::string id, Json::Value pInputItem);
void* Create_ParseTensor(std::string id, Json::Value pInputItem);
void* Create_StringToNumber(std::string id, Json::Value pInputItem);


#endif