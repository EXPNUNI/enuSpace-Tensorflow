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
#include "AttributeParser.h"

void* Create_DecodeCSV(std::string id, Json::Value pInputItem) {
	DecodeCSV* pDecodeCSV = nullptr;
	Scope* pScope = nullptr;
	Output* precords = nullptr;
	OutputList* precord_defaults = nullptr;
	DecodeCSV::Attrs attrs;
	std::string Temp1 = "";
	std::string msgParam = string_format("#%s", id.c_str());

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
			// 입력심볼 : #Scope, 입력심볼의 핀 : Scope, 연결 핀 : Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : DecodeCSV - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "records")
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
							precords = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DecodeCSV - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "record_defaults")
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
							precord_defaults = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DecodeCSV - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "DecodeCSV::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("field_delim_")!="")
				{
					Temp1 = attrParser.GetAttribute("field_delim_");
					attrs= attrs.FieldDelim(Temp1);
				}
				if (attrParser.GetAttribute("use_quote_delim_") != "")
				{
					attrs =attrs.UseQuoteDelim(attrParser.GetValue_bool("use_quote_delim_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : DecodeCSV - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}
	if (pScope && precords && precord_defaults)
	{
		pDecodeCSV = new DecodeCSV(*pScope, *precords, *precord_defaults,attrs);
		ObjectInfo* pObj = AddObjectMap(pDecodeCSV, id, SYMBOL_DECODECSV, "DecodeCSV", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pDecodeCSV->output, OUTPUT_TYPE_OUTPUTLIST, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : DecodeCSV(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pDecodeCSV;
}

void* Create_DecodeJSONExample(std::string id, Json::Value pInputItem) {
	DecodeJSONExample* pDecodeJSONExample = nullptr;
	Scope* pScope = nullptr;
	Output* pjson_examples = nullptr;
	std::string msgParam = string_format("#%s", id.c_str());

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
			// 입력심볼 : #Scope, 입력심볼의 핀 : Scope, 연결 핀 : Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : DecodeJSONExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "json_examples")
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
							pjson_examples = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DecodeJSONExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else
		{
			std::string msg = string_format("warning : DecodeJSONExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}
	if (pScope && pjson_examples)
	{	
		pDecodeJSONExample = new DecodeJSONExample(*pScope, *pjson_examples);
		ObjectInfo* pObj = AddObjectMap(pDecodeJSONExample, id, SYMBOL_DECODEJSONEXAMPLE, "DecodeJSONExample", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pDecodeJSONExample->binary_examples, OUTPUT_TYPE_OUTPUT, "binary_examples");
		}
	}
	else
	{
		std::string msg = string_format("error : DecodeJSONExample(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pDecodeJSONExample;
}

void* Create_DecodeRaw(std::string id, Json::Value pInputItem) {
	DecodeRaw* pDecodeRaw = nullptr;
	Scope* pScope = nullptr;
	Output* pbytes = nullptr;
	DataType dtype;
	DecodeRaw::Attrs attrs;
	std::string msgParam = string_format("#%s", id.c_str());

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
			// 입력심볼 : #Scope, 입력심볼의 핀 : Scope, 연결 핀 : Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : DecodeRaw - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "bytes")
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
							pbytes = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DecodeRaw - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "out_type")
		{
			if (strPinInterface == "DataType")
			{
			
				dtype = GetDatatypeFromInitial(strPinInitial);
				
			}
			else
			{
				std::string msg = string_format("warning : DecodeRaw - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "DecodeRaw::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("little_endian_")!="")
				{
					attrs =attrs.LittleEndian(attrParser.GetValue_bool("little_endian_"));
				}
				
			}
		}
		else
		{
			std::string msg = string_format("warning : DecodeRaw - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}
	if (pScope && pbytes)
	{
		pDecodeRaw =new DecodeRaw(*pScope, *pbytes,dtype,attrs);
		ObjectInfo* pObj = AddObjectMap(pDecodeRaw, id, SYMBOL_DECODERAW, "DecodeRaw", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pDecodeRaw->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : DecodeRaw(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pDecodeRaw;
}

void* Create_ParseExample(std::string id, Json::Value pInputItem) {
	ParseExample* pParseExample = nullptr;
	Scope* pScope = nullptr;
	Output* pserialized = nullptr;
	Output* pnames = nullptr;
	OutputList* psparse_keys = nullptr;
	OutputList* pdense_keys = nullptr;
	OutputList* pdense_defaults = nullptr;
	vector<DataType> v_sparse_types;
	vector<PartialTensorShape> v_dense_shapes;
	gtl::ArraySlice<PartialTensorShape> dense_shapes;
	std::string msgParam = string_format("#%s", id.c_str());

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
			// 입력심볼 : #Scope, 입력심볼의 핀 : Scope, 연결 핀 : Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ParseExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "serialized")
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
							pserialized = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ParseExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "names")
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
							pnames = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ParseExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "sparse_keys")
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
							psparse_keys = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ParseExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "dense_keys")
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
							pdense_keys= (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ParseExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "dense_defaults")
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
							pdense_defaults = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ParseExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "sparse_types")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				GetDatatypeSliceFromInitial(strPinInitial, v_sparse_types);
			}
			else
			{
				std::string msg = string_format("warning : ParseExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "dense_shapes")
		{
			if (strPinInterface == "gtl::ArraySlice<PartialTensorShape>")
			{
				GetArrayShapeFromInitial(strPinInitial,v_dense_shapes);
			}
			else
			{
				std::string msg = string_format("warning : ParseExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
	}
	if (pScope && pserialized && pnames && psparse_keys && pdense_keys&&pdense_defaults)
	{

		DataTypeSlice sparse_types(v_sparse_types);
		gtl::ArraySlice<PartialTensorShape> dense_shapes(v_dense_shapes);
		pParseExample = new ParseExample(*pScope, *pserialized, *pnames, *psparse_keys, *pdense_keys, *pdense_defaults, sparse_types, dense_shapes);
		ObjectInfo* pObj = AddObjectMap(pParseExample, id, SYMBOL_PARSEEXAMPLE, "ParseExample", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pParseExample->sparse_indices, OUTPUT_TYPE_OUTPUTLIST, "sparse_indices");
			AddOutputInfo(pObj, &pParseExample->sparse_values, OUTPUT_TYPE_OUTPUTLIST, "sparse_values");
			AddOutputInfo(pObj, &pParseExample->sparse_shapes, OUTPUT_TYPE_OUTPUTLIST, "sparse_shapes");
			AddOutputInfo(pObj, &pParseExample->dense_values, OUTPUT_TYPE_OUTPUTLIST, "dense_values");
		}
		sparse_types.clear();
		dense_shapes.clear();

	}
	else
	{
		std::string msg = string_format("error : ParseExample(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pParseExample;
}

void* Create_ParseSingleSequenceExample(std::string id, Json::Value pInputItem) {
	ParseSingleSequenceExample* pParseSingleSequenceExample = nullptr;
	Scope* pScope = nullptr;
	Output* serialized = nullptr;
	Output* feature_list_dense_missing_assumed_empty = nullptr;
	OutputList* context_sparse_keys = nullptr;
	OutputList* context_dense_keys = nullptr;
	OutputList* feature_list_sparse_keys = nullptr;
	OutputList* feature_list_dense_keys = nullptr;
	OutputList* context_dense_defaults = nullptr;
	Output* debug_name = nullptr;
	ParseSingleSequenceExample::Attrs attrs;

	std::vector<tensorflow::DataType> v_context_sparse_types;
	std::vector<tensorflow::DataType> v_feature_list_dense_types;
	std::vector<tensorflow::DataType> v_feature_list_sparse_types;
	std::vector<PartialTensorShape> v_feature_list_dense_shapes;
	std::vector<PartialTensorShape>v_context_dense_shapes;
	std::string msgParam = string_format("#%s", id.c_str());

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
			// 입력심볼 : #Scope, 입력심볼의 핀 : Scope, 연결 핀 : Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ParseSingleSequenceExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "serialized")
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
							serialized = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ParseSingleSequenceExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "feature_list_dense_missing_assumed_empty")
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
							feature_list_dense_missing_assumed_empty = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ParseSingleSequenceExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "context_sparse_keys")
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
							context_sparse_keys = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ParseSingleSequenceExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "context_dense_keys")
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
							context_dense_keys = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ParseSingleSequenceExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "feature_list_sparse_keys")
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
							feature_list_sparse_keys = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ParseSingleSequenceExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "feature_list_dense_keys")
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
							feature_list_dense_keys = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ParseSingleSequenceExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "context_dense_defaults")
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
							context_dense_defaults = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ParseSingleSequenceExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "debug_name")
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
							debug_name = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ParseSingleSequenceExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "DecodeRaw::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("context_sparse_types_")!="")
				{
					attrParser.GetValue_DataTypeSlice("context_sparse_types_", v_context_sparse_types);			
				}
				if (attrParser.GetAttribute("feature_list_dense_types_") != "")
				{
					attrParser.GetValue_DataTypeSlice("feature_list_dense_types_", v_feature_list_dense_types);
				}
				if (attrParser.GetAttribute("context_dense_shapes_") != "")
				{
					attrParser.GetValue_arraySliceTensorshape("context_dense_shapes_",v_context_dense_shapes);
				}
				if (attrParser.GetAttribute("feature_list_sparse_types_") != "")
				{
					attrParser.GetValue_DataTypeSlice("feature_list_sparse_types_",v_feature_list_sparse_types);
				}
				if (attrParser.GetAttribute("feature_list_dense_shapes_") != "")
				{
					attrParser.GetValue_arraySliceTensorshape("feature_list_dense_shapes_", v_feature_list_dense_shapes);
				}
				
			}
		}
		else
		{
			std::string msg = string_format("warning : ParseSingleSequenceExample - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}
	if (pScope && serialized &&feature_list_dense_missing_assumed_empty&&context_sparse_keys&&context_dense_keys &&feature_list_sparse_keys&&feature_list_dense_keys&&context_dense_defaults && debug_name)
	{

		DataTypeSlice context_sparse_types(v_context_sparse_types);
		attrs = attrs.ContextSparseTypes(context_sparse_types);
		DataTypeSlice feature_list_dense_types(v_feature_list_dense_types);
		attrs = attrs.FeatureListDenseTypes(feature_list_dense_types);
		DataTypeSlice feature_list_sparse_types(v_feature_list_sparse_types);
		attrs = attrs.FeatureListSparseTypes(feature_list_sparse_types);
		gtl::ArraySlice<PartialTensorShape> context_dense_shapes(v_context_dense_shapes);
		attrs = attrs.ContextDenseShapes(context_dense_shapes);
		gtl::ArraySlice<PartialTensorShape> feature_list_dense_shapes(v_feature_list_dense_shapes);
		attrs =attrs.FeatureListDenseShapes(feature_list_dense_shapes);

		pParseSingleSequenceExample = new ParseSingleSequenceExample(*pScope, *serialized,
												*feature_list_dense_missing_assumed_empty,
												*context_sparse_keys,
												*context_dense_keys,
												*feature_list_sparse_keys,
												*feature_list_dense_keys,
												*context_dense_defaults,
												*debug_name,attrs);
		ObjectInfo* pObj = AddObjectMap(pParseSingleSequenceExample, id, SYMBOL_PARSESINGLESEQUENCEEXAMPLE, "ParseSingleSequenceExample", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pParseSingleSequenceExample->context_sparse_indices, OUTPUT_TYPE_OUTPUTLIST, "context_sparse_indices");
			AddOutputInfo(pObj, &pParseSingleSequenceExample->context_sparse_values, OUTPUT_TYPE_OUTPUTLIST, "context_sparse_values");
			AddOutputInfo(pObj, &pParseSingleSequenceExample->context_sparse_shapes, OUTPUT_TYPE_OUTPUTLIST, "context_sparse_shapes");
			AddOutputInfo(pObj, &pParseSingleSequenceExample->context_dense_values, OUTPUT_TYPE_OUTPUTLIST, "context_dense_values");
			AddOutputInfo(pObj, &pParseSingleSequenceExample->feature_list_sparse_indices, OUTPUT_TYPE_OUTPUTLIST, "feature_list_sparse_indices");
			AddOutputInfo(pObj, &pParseSingleSequenceExample->feature_list_sparse_values, OUTPUT_TYPE_OUTPUTLIST, "feature_list_sparse_values");
			AddOutputInfo(pObj, &pParseSingleSequenceExample->feature_list_sparse_shapes, OUTPUT_TYPE_OUTPUTLIST, "feature_list_sparse_shapes");
			AddOutputInfo(pObj, &pParseSingleSequenceExample->feature_list_dense_values, OUTPUT_TYPE_OUTPUTLIST, "feature_list_dense_values");
		}
	}
	else
	{
		std::string msg = string_format("error : ParseSingleSequenceExample(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}



	return pParseSingleSequenceExample;
}

void* Create_ParseTensor(std::string id, Json::Value pInputItem) {
	ParseTensor* pParseTensor = nullptr;
	Scope* pScope = nullptr;
	Output* serialized = nullptr;
	DataType dtype;
	std::string msgParam = string_format("#%s", id.c_str());

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
			// 입력심볼 : #Scope, 입력심볼의 핀 : Scope, 연결 핀 : Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ParseTensor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "serialized")
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
							serialized = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ParseTensor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "out_type")
		{
			if (strPinInterface == "DataType")
			{

				dtype = GetDatatypeFromInitial(strPinInitial);
				if(dtype == DT_INVALID)
				{
					std::string msg = string_format("warning : ParseTensor - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg, msgParam);
				}
			}
			else
			{
				std::string msg = string_format("warning : ParseTensor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else
		{
			std::string msg = string_format("warning : ParseTensor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}
	if (pScope && serialized)
	{
		pParseTensor = new ParseTensor(*pScope, *serialized, dtype);
		ObjectInfo* pObj = AddObjectMap(pParseTensor, id, SYMBOL_PARSETENSOR, "ParseTensor", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pParseTensor->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : ParseTensor(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pParseTensor;
}

void* Create_StringToNumber(std::string id, Json::Value pInputItem) {
	StringToNumber* pStringToNumber = nullptr;
	Scope* pScope = nullptr;
	Output* string_tensor = nullptr;
	StringToNumber::Attrs attrs;
	std::string msgParam = string_format("#%s", id.c_str());

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
			// 입력심볼 : #Scope, 입력심볼의 핀 : Scope, 연결 핀 : Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : StringToNumber - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
				std::string msg = string_format("warning : StringToNumber - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "StringToNumber::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("out_type_")!="")
				{
					attrs =attrs.OutType(attrParser.GetValue_DataType("out_type_"));
				}
				
			}
		}
		else
		{
			std::string msg = string_format("warning : StringToNumber - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}
	if (pScope && string_tensor)
	{
		pStringToNumber = new StringToNumber(*pScope, *string_tensor, attrs);
		ObjectInfo* pObj = AddObjectMap(pStringToNumber, id, SYMBOL_DECODERAW, "StringToNumber", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pStringToNumber->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : DecodeRaw(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}

	return pStringToNumber;
}