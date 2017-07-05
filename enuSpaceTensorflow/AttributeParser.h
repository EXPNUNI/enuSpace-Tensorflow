#pragma once

#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include <map>

class CAttributeParser
{
public:

	CAttributeParser(std::string attrName, std::string attrValue);
	~CAttributeParser();

	std::string GetAttribute(std::string attrName);
	bool ConvStrToBool(std::string strValue);
	long long ConvStrToInt64(std::string strValue);

	std::map<std::string, std::string> m_attribute;
};

