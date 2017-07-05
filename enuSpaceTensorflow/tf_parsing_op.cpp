#include "stdafx.h"
#include "tf_parsing_op.h"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "include/json/json.h"

#include "GlobalHeader.h"
#include "tensorflow.h"
#include "utility_functions.h"
#include "enuSpaceToTensorflow.h"

void* Create_DecodeCSV(std::string id, Json::Value pInputItem) {
	DecodeCSV* pDecodeCSV = nullptr;
	Scope* pScope = nullptr;
	return pDecodeCSV;
}

void* Create_DecodeJSONExample(std::string id, Json::Value pInputItem) {
	DecodeJSONExample* pDecodeJSONExample = nullptr;
	Scope* pScope = nullptr;
	return pDecodeJSONExample;
}

void* Create_DecodeRaw(std::string id, Json::Value pInputItem) {
	DecodeRaw* pDecodeRaw = nullptr;
	Scope* pScope = nullptr;
	return pDecodeRaw;
}

void* Create_ParseExample(std::string id, Json::Value pInputItem) {
	ParseExample* pParseExample = nullptr;
	Scope* pScope = nullptr;
	return pParseExample;
}

void* Create_ParseSingleSequenceExample(std::string id, Json::Value pInputItem) {
	ParseSingleSequenceExample* pParseSingleSequenceExample = nullptr;
	Scope* pScope = nullptr;
	return pParseSingleSequenceExample;
}

void* Create_ParseTensor(std::string id, Json::Value pInputItem) {
	ParseTensor* pParseTensor = nullptr;
	Scope* pScope = nullptr;
	return pParseTensor;
}

void* Create_StringToNumber(std::string id, Json::Value pInputItem) {
	StringToNumber* pStringToNumber = nullptr;
	Scope* pScope = nullptr;
	return pStringToNumber;
}