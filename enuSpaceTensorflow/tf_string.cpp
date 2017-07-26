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
#include "AttributeParser.h"

void* Create_AsString(std::string id, Json::Value pInputItem) {
	AsString* pAsString = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	AsString::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "tensorflow::Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : AsString - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pinput = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AsString - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "AsString::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("precision_")!="")attrs.Precision(attrParser.GetValue_int64("precision_"));
				if (attrParser.GetAttribute("scientific_") != "")attrs.Scientific(attrParser.GetValue_bool("scientific_"));
				if (attrParser.GetAttribute("shortest_") != "")attrs.Shortest(attrParser.GetValue_bool("shortest_"));
				if (attrParser.GetAttribute("width_") != "")attrs.Width(attrParser.GetValue_int64("width_"));
				if (attrParser.GetAttribute("fill_") != "")attrs.Fill(attrParser.GetValue_StringPiece("fill_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : AsString - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && pinput)
	{
		*pAsString = AsString(*pScope, *pinput,  attrs);
		ObjectInfo* pObj = AddObjectMap(pAsString, id, SYMBOL_ASSTRING, "AsString", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAsString->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : AsString(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAsString;
}

void* Create_DecodeBase64(std::string id, Json::Value pInputItem) {
	DecodeBase64* pDecodeBase64 = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "tensorflow::Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : DecodeBase64 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pInput = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DecodeBase64 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : DecodeBase64 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && pInput)
	{
		*pDecodeBase64 = DecodeBase64(*pScope, *pInput);
		ObjectInfo* pObj = AddObjectMap(pDecodeBase64, id, SYMBOL_DECODEBASE64, "DecodeBase64", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pDecodeBase64->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : DecodeBase64(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pDecodeBase64;
}

void* Create_EncodeBase64(std::string id, Json::Value pInputItem) {
	EncodeBase64* pEncodeBase64 = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	EncodeBase64::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "tensorflow::Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : EncodeBase64 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pInput = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : EncodeBase64 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "EncodeBase64::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("pad_") != "")attrs.Pad(attrParser.GetValue_bool("pad_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : DecodeBase64 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && pInput)
	{
		*pEncodeBase64 = EncodeBase64(*pScope, *pInput,attrs);
		ObjectInfo* pObj = AddObjectMap(pEncodeBase64, id, SYMBOL_ENCODEBASE64, "EncodeBase64", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pEncodeBase64->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : EncodeBase64(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pEncodeBase64;
}

void* Create_ReduceJoin(std::string id, Json::Value pInputItem) {
	ReduceJoin* pReduceJoin = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* reduction_indices = nullptr;
	ReduceJoin::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "tensorflow::Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ReduceJoin - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pInput = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ReduceJoin - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "reduction_indices")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							reduction_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ReduceJoin - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ReduceJoin::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("keep_dims_") != "")attrs.KeepDims(attrParser.GetValue_bool("keep_dims_"));
				if (attrParser.GetAttribute("separator_") != "")attrs.Separator(attrParser.GetValue_StringPiece("separator_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : ReduceJoin - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && pInput && reduction_indices)
	{
		*pReduceJoin = ReduceJoin(*pScope, *pInput, *reduction_indices, attrs);
		ObjectInfo* pObj = AddObjectMap(pReduceJoin, id, SYMBOL_REDUCEJOIN, "ReduceJoin", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pReduceJoin->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : pReduceJoin(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pReduceJoin;
}

void* Create_StringJoin(std::string id, Json::Value pInputItem) {
	StringJoin* pStringJoin = nullptr;
	Scope* pScope = nullptr;
	OutputList* pInputs = nullptr;
	StringJoin::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "tensorflow::Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : StringJoin - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "inputs")
		{
			if (strPinInterface == "InputList")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pInputs = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : StringJoin - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "StringJoin::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("separator_") != "")attrs.Separator(attrParser.GetValue_StringPiece("separator_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : StringJoin - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && pInputs)
	{
		*pStringJoin = StringJoin(*pScope, *pInputs, attrs);
		ObjectInfo* pObj = AddObjectMap(pStringJoin, id, SYMBOL_STRINGJOIN, "StringJoin", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pStringJoin->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : StringJoin(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pStringJoin;
}

void* Create_StringSplit(std::string id, Json::Value pInputItem) {
	StringSplit* pStringSplit = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* delimiter = nullptr;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "tensorflow::Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : StringSplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pInput = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : StringSplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "delimiter")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							delimiter = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : StringSplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : StringSplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && pInput && delimiter)
	{
		*pStringSplit = StringSplit(*pScope, *pInput,*delimiter);
		ObjectInfo* pObj = AddObjectMap(pStringSplit, id, SYMBOL_STRINGSPLIT, "StringSplit", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pStringSplit->indices, OUTPUT_TYPE_OUTPUT, "indices");
			AddOutputInfo(pObj, &pStringSplit->values, OUTPUT_TYPE_OUTPUT, "values");
			AddOutputInfo(pObj, &pStringSplit->shape, OUTPUT_TYPE_OUTPUT, "shape");
		}
	}
	else
	{
		std::string msg = string_format("error : StringSplit(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pStringSplit;
}

void* Create_StringToHashBucket(std::string id, Json::Value pInputItem) {
	StringToHashBucket* pStringToHashBucket = nullptr;
	Scope* pScope = nullptr;
	Output* string_tensor = nullptr;
	int64 num_buckets = 0;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "tensorflow::Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : StringToHashBucket - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "string_tensor")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							string_tensor = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : StringToHashBucket - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_buckets")
		{
			if (strPinInterface == "int64")
			{
				num_buckets = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : StringToHashBucket - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : StringToHashBucket - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && string_tensor )
	{
		*pStringToHashBucket = StringToHashBucket(*pScope, *string_tensor, num_buckets);
		ObjectInfo* pObj = AddObjectMap(pStringToHashBucket, id, SYMBOL_STRINGTOHASHBUCKET, "StringToHashBucket", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pStringToHashBucket->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : StringToHashBucket(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pStringToHashBucket;
}

void* Create_StringToHashBucketFast(std::string id, Json::Value pInputItem) {
	StringToHashBucketFast* pStringToHashBucketFast = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	int64 num_buckets = 0;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "tensorflow::Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : StringToHashBucketFast - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pinput = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : StringToHashBucketFast - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_buckets")
		{
			if (strPinInterface == "int64")
			{
				num_buckets = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : StringToHashBucketFast - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : StringToHashBucketFast - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && pinput)
	{
		*pStringToHashBucketFast = StringToHashBucketFast(*pScope, *pinput, num_buckets);
		ObjectInfo* pObj = AddObjectMap(pStringToHashBucketFast, id, SYMBOL_STRINGTOHASHBUCKET, "StringToHashBucketFast", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pStringToHashBucketFast->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : StringToHashBucketFast(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pStringToHashBucketFast;
}

void* Create_StringToHashBucketStrong(std::string id, Json::Value pInputItem) {
	StringToHashBucketStrong* pStringToHashBucketStrong = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	int64 num_buckets = 0;
	std::vector<int> v_key;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "tensorflow::Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : StringToHashBucketStrong - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pinput = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : StringToHashBucketStrong - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_buckets")
		{
			if (strPinInterface == "int64")
			{
				num_buckets = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : StringToHashBucketStrong - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "key")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_key);
			}
			else
			{
				std::string msg = string_format("warning : StringToHashBucketStrong - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : StringToHashBucketStrong - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && pinput)
	{
		gtl::ArraySlice<int> key(v_key);
		*pStringToHashBucketStrong = StringToHashBucketStrong(*pScope, *pinput, num_buckets,key);
		ObjectInfo* pObj = AddObjectMap(pStringToHashBucketStrong, id, SYMBOL_STRINGTOHASHBUCKETSTRONG, "StringToHashBucketStrong", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pStringToHashBucketStrong->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		key.clear();
	}
	else
	{
		std::string msg = string_format("error : StringToHashBucketStrong(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	
	v_key.clear();
	return pStringToHashBucketStrong;
}

void* Create_Substr(std::string id, Json::Value pInputItem) {
	Substr* pSubstr = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* pos = nullptr;
	Output* len = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "tensorflow::Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Substr - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "pInput")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pInput = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Substr - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "pos")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pos = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Substr - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "len")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							len = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Substr - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Substr - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pInput &&pos&&len )
	{
		*pSubstr = Substr(*pScope, *pInput,*pos,*len);
		ObjectInfo* pObj = AddObjectMap(pSubstr, id, SYMBOL_SUBSTR, "Substr", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSubstr->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("error : Substr(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}

	return pSubstr;
}
