#pragma once

#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include <map>
#include "GlobalHeader.h"
using namespace tensorflow;
using namespace tensorflow::ops;

class CAttributeParser
{
public:

	CAttributeParser(std::string attrName, std::string attrValue);
	~CAttributeParser();

	std::string GetAttribute(std::string attrName);
	bool ConvStrToBool(std::string attrValue);
	long long ConvStrToInt64(std::string attrValue);
	float ConvStrToFloat(std::string attrValue);
	StringPiece ConvStrToStringPiece(std::string attrValue);
	gtl::ArraySlice<PartialTensorShape> ConvStrToArraySliceTensorshape(std::string attrValue);
	DataTypeSlice ConvStrToDataTypeSlice(std::string attrValue);
	gtl::ArraySlice<string> ConvStrToArraySliceString(std::string attrValue);
	gtl::ArraySlice<float> ConvStrToArraySlicefloat(std::string attrValue);
	PartialTensorShape ConvStrToPartialTensorShape(std::string attrValue);
	gtl::ArraySlice<int> ConvStrToArraySliceInt(std::string attrValue);
	TensorShape ConvStrToTensorShape(std::string attrValue);
	DataType ConvStrToDataType(std::string attrValue);

	bool GetValue_bool(std::string name);
	int64 GetValue_int64(std::string name);
	float GetValue_float(std::string name);
	StringPiece GetValue_StringPiece(std::string name);
	gtl::ArraySlice<PartialTensorShape> GetValue_arraySliceTensorshape(std::string name);
	DataTypeSlice GetValue_DataTypeSlice(std::string name);
	gtl::ArraySlice<string> GetValue_arraySliceString(std::string name);
	gtl::ArraySlice<float> GetValue_arraySlicefloat(std::string name);
	PartialTensorShape GetValue_PartialTensorShape(std::string name);
	gtl::ArraySlice<int> GetValue_arraySliceInt(std::string name);
	TensorShape GetValue_TensorShape(std::string name);
	DataType GetValue_DataType(std::string name);
private:
	std::map<std::string, std::string> m_attribute;
};
