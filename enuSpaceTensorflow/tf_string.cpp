#include "stdafx.h"
#include "tf_string.h"


#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "include/json/json.h"

#include "GlobalHeader.h"
#include "tensorflow.h"
#include "utility_functions.h"
#include "enuSpaceToTensorflow.h"

void* Create_AsString(std::string id, Json::Value pInputItem) {
	AsString* pAsString = nullptr;
	Scope* pScope = nullptr;
	return pAsString;
}

void* Create_DecodeBase64(std::string id, Json::Value pInputItem) {
	DecodeBase64* pDecodeBase64 = nullptr;
	Scope* pScope = nullptr;
	return pDecodeBase64;
}

void* Create_EncodeBase64(std::string id, Json::Value pInputItem) {
	EncodeBase64* pEncodeBase64 = nullptr;
	Scope* pScope = nullptr;
	return pEncodeBase64;
}

void* Create_ReduceJoin(std::string id, Json::Value pInputItem) {
	ReduceJoin* pReduceJoin = nullptr;
	Scope* pScope = nullptr;
	return pReduceJoin;
}

void* Create_StringJoin(std::string id, Json::Value pInputItem) {
	StringJoin* pStringJoin = nullptr;
	Scope* pScope = nullptr;
	return pStringJoin;
}

void* Create_StringSplit(std::string id, Json::Value pInputItem) {
	StringSplit* pStringSplit = nullptr;
	Scope* pScope = nullptr;
	return pStringSplit;
}

void* Create_StringToHashBucket(std::string id, Json::Value pInputItem) {
	StringToHashBucket* pStringToHashBucket = nullptr;
	Scope* pScope = nullptr;
	return pStringToHashBucket;
}

void* Create_StringToHashBucketFast(std::string id, Json::Value pInputItem) {
	StringToHashBucketFast* pStringToHashBucketFast = nullptr;
	Scope* pScope = nullptr;
	return pStringToHashBucketFast;
}

void* Create_StringToHashBucketStrong(std::string id, Json::Value pInputItem) {
	StringToHashBucketStrong* pStringToHashBucketStrong = nullptr;
	Scope* pScope = nullptr;
	return pStringToHashBucketStrong;
}

void* Create_Substr(std::string id, Json::Value pInputItem) {
	Substr* pSubstr = nullptr;
	Scope* pScope = nullptr;
	return pSubstr;
}
