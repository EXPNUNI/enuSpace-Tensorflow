#include "stdafx.h"
#include "AttributeParser.h"
#include "utility_functions.h"
#include <cstdio>
#include <string>
#include <vector>


CAttributeParser::~CAttributeParser()
{

}

CAttributeParser::CAttributeParser(std::string attrs, std::string attrsValue)
{
	std::string attr;
	std::string value;
	int iType = 0;
	for (std::string::size_type i = 0; i < attrsValue.size(); i++)
	{
		if (attrsValue[i] == '=')
		{
			iType = 1;
		}
		else if (attrsValue[i] == ';')
		{
			iType = 0;
			m_attribute.insert(std::pair<std::string, std::string>(attr, value));

			attr = "";
			value = "";
		}
		else
		{
			if (iType == 0)
				attr = attr + attrsValue[i];
			else
				value = value + attrsValue[i];
		}
	}
	if (attr.length() > 0)
	{
		m_attribute.insert(std::pair<std::string, std::string>(attr, value));
	}
}

std::string CAttributeParser::GetAttribute(std::string attrName)
{
	std::string strValue;
	const std::map<std::string, std::string>::const_iterator aLookup = m_attribute.find(attrName);
	const bool bExists = aLookup != m_attribute.end();
	if (bExists)
	{
		strValue = aLookup->second;
		return strValue;
	}
	else
	{
		return strValue;
	}
}

bool CAttributeParser::ConvStrToBool(std::string strValue)
{
	if (strValue == "true" || strValue == "TRUE" || strValue == "1")
		return true;
	else
		return false;
}

long long CAttributeParser::ConvStrToInt64(std::string strValue)
{
	return std::stoll(strValue);
}

float CAttributeParser::ConvStrToFloat(std::string strValue)
{
	return std::stof(strValue);
}

tensorflow::StringPiece CAttributeParser::ConvStrToStringPiece(std::string attrValue)
{
	StringPiece strpiece(attrValue);
	return strpiece;
}

gtl::ArraySlice<PartialTensorShape> CAttributeParser::ConvStrToArraySliceTensorshape(std::string attrValue)
{
	gtl::ArraySlice<PartialTensorShape> ad;
	//dev_need
	return ad;
}

tensorflow::DataTypeSlice CAttributeParser::ConvStrToDataTypeSlice(std::string attrValue)
{
	std::string val;
	std::vector<DataType> arrayvals;
	for (std::string::size_type i = 0; i < attrValue.size(); i++)
	{
		if (attrValue[i] == ';')
		{
			DataType dtype;
			if (val == "double")
				dtype = DT_DOUBLE;
			else if (val == "float")
				dtype = DT_FLOAT;
			else if (val == "int")
				dtype = DT_INT32;
			arrayvals.push_back(dtype);
			val = "";
		}
		else
		{
			val = val + attrValue[i];
		}
	}

	if (val.length() > 0)
	{
		DataType dtype;
		if (val == "double")
			dtype = DT_DOUBLE;
		else if (val == "float")
			dtype = DT_FLOAT;
		else if (val == "int")
			dtype = DT_INT32;
		arrayvals.push_back(dtype);
	}
	DataTypeSlice dtypeslice(arrayvals);
	return dtypeslice;
}

gtl::ArraySlice<string> CAttributeParser::ConvStrToArraySliceString(std::string attrValue)
{
	std::vector<std::string> v_string;
	GetStringVectorFormInitial(attrValue, v_string);
	gtl::ArraySlice< string > arraySlice(v_string);
	return arraySlice;
}

gtl::ArraySlice<float> CAttributeParser::ConvStrToArraySlicefloat(std::string attrValue)
{
	std::vector<float> v_float;
	GetFloatVectorFormInitial(attrValue, v_float);
	gtl::ArraySlice< float > arraySlice(v_float);
	return arraySlice;
}

tensorflow::PartialTensorShape CAttributeParser::ConvStrToPartialTensorShape(std::string attrValue)
{
	std::vector<int64> arrayslice;
	std::vector<int64> arraydims;
	GetArrayDimsFromShape(attrValue, arraydims, arrayslice);
	gtl::ArraySlice< int64 > arraySlice(arraydims);
	PartialTensorShape TS = PartialTensorShape(arraySlice);
	return TS;
}

gtl::ArraySlice<int> CAttributeParser::ConvStrToArraySliceInt(std::string attrValue)
{
	std::vector<int> v_int;
	GetIntVectorFormInitial(attrValue, v_int);
	gtl::ArraySlice< int > arraySlice(v_int);
	return arraySlice;
}

tensorflow::TensorShape CAttributeParser::ConvStrToTensorShape(std::string attrValue)
{

	std::vector<int64> arrayslice;
	std::vector<int64> arraydims;
	GetArrayDimsFromShape(attrValue, arraydims, arrayslice);
	gtl::ArraySlice< int64 > arraySlice(arraydims);
	TensorShape TS = TensorShape(arraySlice);
	return TS;
}

tensorflow::DataType CAttributeParser::ConvStrToDataType(std::string attrValue)
{
	DataType dtype;
	if (attrValue == "double")
		dtype = DT_DOUBLE;
	else if (attrValue == "float")
		dtype = DT_FLOAT;
	else if (attrValue == "int")
		dtype = DT_INT32;
	else if (attrValue == "bool")
		dtype = DT_BOOL;
	else if (attrValue == "string")
		dtype = DT_STRING;
	return dtype;
}

bool CAttributeParser::GetValue_bool(std::string name)
{
	return ConvStrToBool(GetAttribute(name));
}

tensorflow::int64 CAttributeParser::GetValue_int64(std::string name)
{
	return ConvStrToInt64(GetAttribute(name));
}

float CAttributeParser::GetValue_float(std::string name)
{
	return ConvStrToInt64(GetAttribute(name));
}

StringPiece CAttributeParser::GetValue_StringPiece(std::string name)
{
	return ConvStrToStringPiece(GetAttribute(name));
}

gtl::ArraySlice<PartialTensorShape> CAttributeParser::GetValue_arraySliceTensorshape(std::string name)
{
	return ConvStrToArraySliceTensorshape(GetAttribute(name));
}

DataTypeSlice CAttributeParser::GetValue_DataTypeSlice(std::string name)
{
	return ConvStrToDataTypeSlice(GetAttribute(name));
}

gtl::ArraySlice<string> CAttributeParser::GetValue_arraySliceString(std::string name)
{
	return ConvStrToArraySliceString(GetAttribute(name));
}

gtl::ArraySlice<float> CAttributeParser::GetValue_arraySlicefloat(std::string name)
{
	return ConvStrToArraySlicefloat(GetAttribute(name));
}

PartialTensorShape CAttributeParser::GetValue_PartialTensorShape(std::string name)
{
	return ConvStrToPartialTensorShape(GetAttribute(name));
}

gtl::ArraySlice<int> CAttributeParser::GetValue_arraySliceInt(std::string name)
{
	return ConvStrToArraySliceInt(GetAttribute(name));
}

TensorShape CAttributeParser::GetValue_TensorShape(std::string name)
{
	return ConvStrToTensorShape(GetAttribute(name));
}

tensorflow::DataType CAttributeParser::GetValue_DataType(std::string name)
{
	return ConvStrToDataType(GetAttribute(name));
}
