#include "stdafx.h"
#include "AttributeParser.h"

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

