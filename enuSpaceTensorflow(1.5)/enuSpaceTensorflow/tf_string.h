#pragma once

#ifndef _TF_STRING_HEADER_
#define _TF_STRING_HEADER_

#include <string>
#include "include/json/json.h"


void* Create_AsString(std::string id, Json::Value pInputItem);
void* Create_DecodeBase64(std::string id, Json::Value pInputItem);
void* Create_EncodeBase64(std::string id, Json::Value pInputItem);
void* Create_ReduceJoin(std::string id, Json::Value pInputItem);
void* Create_StringJoin(std::string id, Json::Value pInputItem);
void* Create_StringSplit(std::string id, Json::Value pInputItem);
void* Create_StringToHashBucket(std::string id, Json::Value pInputItem);
void* Create_StringToHashBucketFast(std::string id, Json::Value pInputItem);
void* Create_StringToHashBucketStrong(std::string id, Json::Value pInputItem);
void* Create_Substr(std::string id, Json::Value pInputItem);

#endif