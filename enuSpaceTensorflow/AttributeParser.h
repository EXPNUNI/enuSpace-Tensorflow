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
	bool ConvStrToArraySliceTensorshape(std::string attrValue, std::vector<PartialTensorShape>& v_PTS);
	DataTypeSlice ConvStrToDataTypeSlice(std::string attrValue);
	bool ConvStrToArraySliceInt(std::string attrValue, std::vector<int>& v_int);
	bool ConvStrToArraySlicefloat(std::string attrValue, std::vector<float>& v_float);
	bool ConvStrToArraySliceString(std::string attrValue, std::vector<std::string>& v_string);
	
	PartialTensorShape ConvStrToPartialTensorShape(std::string attrValue);
	
	TensorShape ConvStrToTensorShape(std::string attrValue);
	DataType ConvStrToDataType(std::string attrValue);

	bool GetValue_bool(std::string name);
	int64 GetValue_int64(std::string name);
	float GetValue_float(std::string name);
	StringPiece GetValue_StringPiece(std::string name);
	bool GetValue_arraySliceTensorshape(std::string name, std::vector<PartialTensorShape>& v_PTS);
	DataTypeSlice GetValue_DataTypeSlice(std::string name);

	bool GetValue_arraySliceInt(std::string name, std::vector<int>& v_int);
	bool GetValue_arraySliceString(std::string name, std::vector<std::string>& v_string);
	bool GetValue_arraySlicefloat(std::string name, std::vector<float>& v_float);

	PartialTensorShape GetValue_PartialTensorShape(std::string name);
	
	TensorShape GetValue_TensorShape(std::string name);
	DataType GetValue_DataType(std::string name);
private:
	std::map<std::string, std::string> m_attribute;
};
