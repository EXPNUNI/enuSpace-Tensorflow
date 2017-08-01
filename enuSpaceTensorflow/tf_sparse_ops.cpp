#include "stdafx.h"
#include "tf_sparse_ops.h"

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

void* Create_AddManySparseToTensorsMap(std::string id, Json::Value pInputItem) {
	AddManySparseToTensorsMap* pAddManySparseToTensorsMap = nullptr;
	Scope* pScope = nullptr;
	Output* sparse_indices = nullptr;
	Output* sparse_values = nullptr;
	Output* sparse_shape = nullptr;
	AddManySparseToTensorsMap::Attrs attrs;

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
				std::string msg = string_format("warning : AddManySparseToTensorsMap - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sparse_indices")
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
							sparse_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AddManySparseToTensorsMap - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sparse_values")
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
							sparse_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AddManySparseToTensorsMap - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sparse_shape")
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
							sparse_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AddManySparseToTensorsMap - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "AddManySparseToTensorsMap::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("container_")!="")
					attrs.Container(attrParser.GetValue_StringPiece("container_"));
				if (attrParser.GetAttribute("shared_name_") != "")
					attrs.SharedName(attrParser.GetValue_StringPiece("shared_name_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : AddManySparseToTensorsMap - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && sparse_indices && sparse_values && sparse_shape)
	{
		pAddManySparseToTensorsMap = new AddManySparseToTensorsMap(*pScope, *sparse_indices, *sparse_values,*sparse_shape, attrs);
		ObjectInfo* pObj = AddObjectMap(pAddManySparseToTensorsMap, id, SYMBOL_ADDMANYSPARSETOTENSORSMAP, "AddManySparseToTensorsMap", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAddManySparseToTensorsMap->sparse_handles, OUTPUT_TYPE_OUTPUT, "sparse_handles");
		}
	}
	else
	{
		std::string msg = string_format("error : AddManySparseToTensorsMap(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAddManySparseToTensorsMap;
}

void* Create_AddSparseToTensorsMap(std::string id, Json::Value pInputItem) {
	AddSparseToTensorsMap* pAddSparseToTensorsMap = nullptr;
	Scope* pScope = nullptr;
	Output* sparse_indices = nullptr;
	Output* sparse_values = nullptr;
	Output* sparse_shape = nullptr;
	AddSparseToTensorsMap::Attrs attrs;

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
				std::string msg = string_format("warning : AddSparseToTensorsMap - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sparse_indices")
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
							sparse_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AddSparseToTensorsMap - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sparse_values")
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
							sparse_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AddSparseToTensorsMap - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sparse_shape")
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
							sparse_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AddSparseToTensorsMap - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "AddSparseToTensorsMap::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("container_") != "")
					attrs.Container(attrParser.GetValue_StringPiece("container_"));
				if (attrParser.GetAttribute("shared_name_") != "")
					attrs.SharedName(attrParser.GetValue_StringPiece("shared_name_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : AddSparseToTensorsMap - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && sparse_indices && sparse_values && sparse_shape)
	{
		pAddSparseToTensorsMap = new AddSparseToTensorsMap(*pScope, *sparse_indices, *sparse_values, *sparse_shape, attrs);
		ObjectInfo* pObj = AddObjectMap(pAddSparseToTensorsMap, id, SYMBOL_ADDSPARSETOTENSORSMAP, "AddSparseToTensorsMap", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAddSparseToTensorsMap->sparse_handle, OUTPUT_TYPE_OUTPUT, "sparse_handle");
		}
	}
	else
	{
		std::string msg = string_format("error : AddSparseToTensorsMap(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAddSparseToTensorsMap;
}

void* Create_DeserializeManySparse(std::string id, Json::Value pInputItem) {
	DeserializeManySparse* pDeserializeManySparse = nullptr;
	Scope* pScope = nullptr;
	Output* serialized_sparse = nullptr;
	DataType dtype;

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
				std::string msg = string_format("warning : DeserializeManySparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "serialized_sparse")
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
							serialized_sparse = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DeserializeManySparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "out_type")
		{
			if (strPinInterface == "DataType")
			{
				dtype = GetDatatypeFromInitial(strPinInitial);
				if(dtype == DT_INVALID)
				{
					std::string msg = string_format("warning : DeserializeManySparse - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : DeserializeManySparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : DeserializeManySparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && serialized_sparse)
	{
		pDeserializeManySparse = new DeserializeManySparse(*pScope, *serialized_sparse, dtype);
		ObjectInfo* pObj = AddObjectMap(pDeserializeManySparse, id, SYMBOL_DESERIALIZEMANYSPARSE, "DeserializeManySparse", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pDeserializeManySparse->sparse_indices, OUTPUT_TYPE_OUTPUT, "sparse_indices");
			AddOutputInfo(pObj, &pDeserializeManySparse->sparse_values, OUTPUT_TYPE_OUTPUT, "sparse_values");
			AddOutputInfo(pObj, &pDeserializeManySparse->sparse_shape, OUTPUT_TYPE_OUTPUT, "sparse_shape");
		}
	}
	else
	{
		std::string msg = string_format("error : DeserializeManySparse(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pDeserializeManySparse;
}

void* Create_SerializeManySparse(std::string id, Json::Value pInputItem) {
	SerializeManySparse* pSerializeManySparse = nullptr;
	Scope* pScope = nullptr;
	Output* sparse_indices = nullptr;
	Output* sparse_values = nullptr;
	Output* sparse_shape = nullptr;

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
				std::string msg = string_format("warning : SerializeManySparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sparse_indices")
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
							sparse_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SerializeManySparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sparse_values")
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
							sparse_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SerializeManySparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sparse_shape")
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
							sparse_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SerializeManySparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SerializeManySparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && sparse_indices && sparse_values && sparse_shape)
	{
		pSerializeManySparse = new SerializeManySparse(*pScope, *sparse_indices, *sparse_values, *sparse_shape);
		ObjectInfo* pObj = AddObjectMap(pSerializeManySparse, id, SYMBOL_SERIALIZEMANYSPARSE, "SerializeManySparse", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSerializeManySparse->serialized_sparse, OUTPUT_TYPE_OUTPUT, "serialized_sparse");
		}
	}
	else
	{
		std::string msg = string_format("error : SerializeManySparse(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSerializeManySparse;
}

void* Create_SerializeSparse(std::string id, Json::Value pInputItem) {
	SerializeSparse* pSerializeSparse = nullptr;
	Scope* pScope = nullptr;
	Output* sparse_indices = nullptr;
	Output* sparse_values = nullptr;
	Output* sparse_shape = nullptr;

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
				std::string msg = string_format("warning : SerializeSparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sparse_indices")
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
							sparse_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SerializeSparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sparse_values")
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
							sparse_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SerializeSparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sparse_shape")
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
							sparse_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SerializeSparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SerializeSparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && sparse_indices && sparse_values && sparse_shape)
	{
		pSerializeSparse = new SerializeSparse(*pScope, *sparse_indices, *sparse_values, *sparse_shape);
		ObjectInfo* pObj = AddObjectMap(pSerializeSparse, id, SYMBOL_SERIALIZESPARSE, "SerializeSparse", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSerializeSparse->serialized_sparse, OUTPUT_TYPE_OUTPUT, "serialized_sparse");
		}
	}
	else
	{
		std::string msg = string_format("error : SerializeSparse(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSerializeSparse;
}

void* Create_SparseAdd(std::string id, Json::Value pInputItem) {
	SparseAdd* pSparseAdd = nullptr;
	Scope* pScope = nullptr;
	Output* a_indices = nullptr;
	Output* a_values = nullptr;
	Output* a_shape = nullptr;
	Output* b_indices = nullptr;
	Output* b_values = nullptr;
	Output* b_shape = nullptr;
	Output* thresh = nullptr;

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
				std::string msg = string_format("warning : SparseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a_indices")
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
							a_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a_values")
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
							a_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a_shape")
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
							a_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "b_indices")
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
							b_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "b_values")
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
							b_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "b_shape")
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
							b_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "thresh")
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
							thresh = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && a_indices && a_values && a_shape &&  b_indices && b_values && b_shape && thresh)
	{
		pSparseAdd = new SparseAdd(*pScope, *a_indices, *a_values, *a_shape, *b_indices, *b_values, *b_shape,*thresh);
		ObjectInfo* pObj = AddObjectMap(pSparseAdd, id, SYMBOL_SPARSEADD, "SparseAdd", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseAdd->sum_indices, OUTPUT_TYPE_OUTPUT, "sum_indices");
			AddOutputInfo(pObj, &pSparseAdd->sum_values, OUTPUT_TYPE_OUTPUT, "sum_values");
			AddOutputInfo(pObj, &pSparseAdd->sum_shape, OUTPUT_TYPE_OUTPUT, "sum_shape");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseAdd(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseAdd;
}

void* Create_SparseAddGrad(std::string id, Json::Value pInputItem) {
	SparseAddGrad* pSparseAddGrad = nullptr;
	Scope* pScope = nullptr;
	Output* backprop_val_grad = nullptr;
	Output* a_indices = nullptr;
	Output* b_indices = nullptr;
	Output* sum_indices = nullptr;

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
				std::string msg = string_format("warning : SparseAddGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "backprop_val_grad")
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
							backprop_val_grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseAddGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a_indices")
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
							a_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseAddGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "b_indices")
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
							b_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseAddGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sum_indices")
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
							sum_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseAddGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseAddGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && backprop_val_grad && a_indices && b_indices && sum_indices)
	{
		pSparseAddGrad = new SparseAddGrad(*pScope, *backprop_val_grad, *a_indices, *b_indices, *sum_indices);
		ObjectInfo* pObj = AddObjectMap(pSparseAddGrad, id, SYMBOL_SPARSEADDGRAD, "SparseAddGrad", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseAddGrad->a_val_grad, OUTPUT_TYPE_OUTPUT, "a_val_grad");
			AddOutputInfo(pObj, &pSparseAddGrad->b_val_grad, OUTPUT_TYPE_OUTPUT, "b_val_grad");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseAddGrad(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseAddGrad;
}

void* Create_SparseConcat(std::string id, Json::Value pInputItem) {
	SparseConcat* pSparseConcat = nullptr;
	Scope* pScope = nullptr;
	OutputList* indices = nullptr;
	OutputList* values = nullptr;
	OutputList* shapes = nullptr;
	int64 concat_dim = 0;

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
				std::string msg = string_format("warning : SparseConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "indices")
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
							indices = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "values")
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
							values = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "shapes")
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
							shapes = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "concat_dim")
		{
			if (strPinInterface == "int64")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					concat_dim = stoll(strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && indices && values && shapes)
	{
		pSparseConcat = new SparseConcat(*pScope, *indices, *values, *shapes,concat_dim);
		ObjectInfo* pObj = AddObjectMap(pSparseConcat, id, SYMBOL_SPARSECONCAT, "SparseConcat", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseConcat->output_indices, OUTPUT_TYPE_OUTPUT, "output_indices");
			AddOutputInfo(pObj, &pSparseConcat->output_values, OUTPUT_TYPE_OUTPUT, "output_values");
			AddOutputInfo(pObj, &pSparseConcat->output_shape, OUTPUT_TYPE_OUTPUT, "output_shape");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseConcat(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseConcat;
}

void* Create_SparseCross(std::string id, Json::Value pInputItem) {
	SparseCross* pSparseCross = nullptr;
	Scope* pScope = nullptr;
	OutputList* indices = nullptr;
	OutputList* values = nullptr;
	OutputList* shapes = nullptr;
	OutputList* dense_inputs = nullptr;
	bool hashed_output = false;
	int64 num_buckets = 0;
	int64 hash_key = 0;
	DataType out_type = DT_DOUBLE;
	DataType internal_type = DT_DOUBLE;

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
				std::string msg = string_format("warning : SparseCross - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "indices")
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
							indices = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseCross - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "values")
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
							values = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseCross - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "shapes")
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
							shapes = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseCross - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "hashed_output")
		{
			if (strPinInterface == "bool")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					if (strPinInitial =="true")
					{
						hashed_output = true;
					}
					else if (strPinInitial == "false")
					{
						hashed_output = false;
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseCross - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_buckets")
		{
			if (strPinInterface == "int64")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					num_buckets = stoll(strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "hash_key")
		{
			if (strPinInterface == "int64")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					hash_key = stoll(strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseCross - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "out_type")
		{
			if (strPinInterface == "DataType")
			{
				out_type = GetDatatypeFromInitial(strPinInitial);
				if (out_type == DT_INVALID)
				{
					std::string msg = string_format("warning : SparseCross - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseCross - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "internal_type")
		{
			if (strPinInterface == "DataType")
			{
			
				internal_type = GetDatatypeFromInitial(strPinInitial);
				if (internal_type == DT_INVALID)
				{
					std::string msg = string_format("warning : SparseCross - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseCross - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseCross - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && indices && values && shapes && dense_inputs)
	{
		pSparseCross = new SparseCross(*pScope, *indices, *values, *shapes, *dense_inputs, hashed_output, num_buckets, hash_key, out_type, internal_type);
		ObjectInfo* pObj = AddObjectMap(pSparseCross, id, SYMBOL_SPARSECROSS, "SparseCross", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseCross->output_indices, OUTPUT_TYPE_OUTPUT, "output_indices");
			AddOutputInfo(pObj, &pSparseCross->output_values, OUTPUT_TYPE_OUTPUT, "output_values");
			AddOutputInfo(pObj, &pSparseCross->output_shape, OUTPUT_TYPE_OUTPUT, "output_shape");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseCross(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseCross;
}

void* Create_SparseDenseCwiseAdd(std::string id, Json::Value pInputItem) {
	SparseDenseCwiseAdd* pSparseDenseCwiseAdd = nullptr;
	Scope* pScope = nullptr;
	Output* sp_indices = nullptr;
	Output* sp_values = nullptr;
	Output* sp_shape = nullptr;
	Output* dense = nullptr;

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
				std::string msg = string_format("warning : SparseDenseCwiseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sp_indices")
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
							sp_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseDenseCwiseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sp_values")
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
							sp_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseDenseCwiseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sp_shape")
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
							sp_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseDenseCwiseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "dense")
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
							dense = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseDenseCwiseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseDenseCwiseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && sp_indices && sp_values && sp_shape && dense)
	{
		pSparseDenseCwiseAdd = new SparseDenseCwiseAdd(*pScope, *sp_indices, *sp_values, *sp_shape,*dense);
		ObjectInfo* pObj = AddObjectMap(pSparseDenseCwiseAdd, id, SYMBOL_SPARSEDENSECWISEADD, "SparseDenseCwiseAdd", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseDenseCwiseAdd->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseDenseCwiseAdd(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseDenseCwiseAdd;
}

void* Create_SparseDenseCwiseDiv(std::string id, Json::Value pInputItem) {
	SparseDenseCwiseDiv* pSparseDenseCwiseDiv = nullptr;
	Scope* pScope = nullptr;
	Output* sp_indices = nullptr;
	Output* sp_values = nullptr;
	Output* sp_shape = nullptr;
	Output* dense = nullptr;

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
				std::string msg = string_format("warning : SparseDenseCwiseDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sp_indices")
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
							sp_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseDenseCwiseDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sp_values")
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
							sp_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseDenseCwiseDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sp_shape")
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
							sp_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseDenseCwiseDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "dense")
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
							dense = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseDenseCwiseDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseDenseCwiseDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && sp_indices && sp_values && sp_shape && dense)
	{
		pSparseDenseCwiseDiv = new SparseDenseCwiseDiv(*pScope, *sp_indices, *sp_values, *sp_shape, *dense);
		ObjectInfo* pObj = AddObjectMap(pSparseDenseCwiseDiv, id, SYMBOL_SPARSEDENSECWISEDIV, "SparseDenseCwiseDiv", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseDenseCwiseDiv->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseDenseCwiseDiv(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseDenseCwiseDiv;
}

void* Create_SparseDenseCwiseMul(std::string id, Json::Value pInputItem) {
	SparseDenseCwiseMul* pSparseDenseCwiseMul = nullptr;
	Scope* pScope = nullptr;
	Output* sp_indices = nullptr;
	Output* sp_values = nullptr;
	Output* sp_shape = nullptr;
	Output* dense = nullptr;

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
				std::string msg = string_format("warning : SparseDenseCwiseMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sp_indices")
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
							sp_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseDenseCwiseMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sp_values")
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
							sp_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseDenseCwiseMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sp_shape")
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
							sp_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseDenseCwiseMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "dense")
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
							dense = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseDenseCwiseMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseDenseCwiseMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && sp_indices && sp_values && sp_shape && dense)
	{
		pSparseDenseCwiseMul = new SparseDenseCwiseMul(*pScope, *sp_indices, *sp_values, *sp_shape, *dense);
		ObjectInfo* pObj = AddObjectMap(pSparseDenseCwiseMul, id, SYMBOL_SPARSEDENSECWISEMUL, "SparseDenseCwiseMul", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseDenseCwiseMul->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseDenseCwiseMul(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseDenseCwiseMul;
}

void* Create_SparseReduceSum(std::string id, Json::Value pInputItem) {
	SparseReduceSum* pSparseReduceSum = nullptr;
	Scope* pScope = nullptr;
	Output* input_indices = nullptr;
	Output* input_values = nullptr;
	Output* input_shape = nullptr;
	Output* reduction_axes = nullptr;
	SparseReduceSum::Attrs attrs;

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
				std::string msg = string_format("warning : SparseReduceSum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input_indices")
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
							input_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseReduceSum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input_values")
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
							input_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseReduceSum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input_shape")
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
							input_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseReduceSum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "reduction_axes")
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
							reduction_axes = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseReduceSum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "SparseReduceSum::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if(attrParser.GetAttribute("keep_dims_")!="")
					attrs.KeepDims(attrParser.GetValue_bool("keep_dims_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseReduceSum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && input_indices && input_values && input_shape && reduction_axes)
	{
		pSparseReduceSum = new SparseReduceSum(*pScope, *input_indices, *input_values, *input_shape, *reduction_axes,attrs);
		ObjectInfo* pObj = AddObjectMap(pSparseReduceSum, id, SYMBOL_SPARSEREDUCESUM, "SparseReduceSum", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseReduceSum->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseReduceSum(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseReduceSum;
}

void* Create_SparseReduceSumSparse(std::string id, Json::Value pInputItem) {
	SparseReduceSumSparse* pSparseReduceSumSparse = nullptr;
	Scope* pScope = nullptr;
	Output* input_indices = nullptr;
	Output* input_values = nullptr;
	Output* input_shape = nullptr;
	Output* reduction_axes = nullptr;
	SparseReduceSumSparse::Attrs attrs;

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
				std::string msg = string_format("warning : SparseReduceSumSparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input_indices")
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
							input_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseReduceSumSparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input_values")
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
							input_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseReduceSumSparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input_shape")
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
							input_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseReduceSumSparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "reduction_axes")
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
							reduction_axes = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseReduceSumSparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "SparseReduceSumSparse::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("keep_dims_")!="")
					attrs.KeepDims(attrParser.GetValue_bool("keep_dims_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseReduceSumSparse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && input_indices && input_values && input_shape && reduction_axes)
	{
		pSparseReduceSumSparse = new SparseReduceSumSparse(*pScope, *input_indices, *input_values, *input_shape, *reduction_axes, attrs);
		ObjectInfo* pObj = AddObjectMap(pSparseReduceSumSparse, id, SYMBOL_SPARSEREDUCESUMSPARSE, "SparseReduceSumSparse", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseReduceSumSparse->output_indices, OUTPUT_TYPE_OUTPUT, "output_indices");
			AddOutputInfo(pObj, &pSparseReduceSumSparse->output_values, OUTPUT_TYPE_OUTPUT, "output_values");
			AddOutputInfo(pObj, &pSparseReduceSumSparse->output_shape, OUTPUT_TYPE_OUTPUT, "output_shape");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseReduceSum(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseReduceSumSparse;
}

void* Create_SparseReorder(std::string id, Json::Value pInputItem) {
	SparseReorder* pSparseReorder = nullptr;
	Scope* pScope = nullptr;
	Output* input_indices = nullptr;
	Output* input_values = nullptr;
	Output* input_shape = nullptr;

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
				std::string msg = string_format("warning : SparseReorder - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input_indices")
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
							input_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseReorder - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input_values")
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
							input_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseReorder - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input_shape")
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
							input_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseReorder - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseReorder - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && input_indices && input_values && input_shape)
	{
		pSparseReorder = new SparseReorder(*pScope, *input_indices, *input_values, *input_shape);
		ObjectInfo* pObj = AddObjectMap(pSparseReorder, id, SYMBOL_SPARSEREORDER, "SparseReorder", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseReorder->output_indices, OUTPUT_TYPE_OUTPUT, "output_indices");
			AddOutputInfo(pObj, &pSparseReorder->output_values, OUTPUT_TYPE_OUTPUT, "output_values");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseReorder(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseReorder;
}

void* Create_SparseReshape(std::string id, Json::Value pInputItem) {
	SparseReshape* pSparseReshape = nullptr;
	Scope* pScope = nullptr;
	Output* input_indices = nullptr;
	Output* input_shape = nullptr;
	Output* new_shape = nullptr;

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
				std::string msg = string_format("warning : SparseReshape - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input_indices")
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
							input_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseReshape - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input_shape")
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
							input_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseReshape - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "new_shape")
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
							new_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseReshape - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseReshape - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && input_indices && input_shape && new_shape)
	{
		pSparseReshape = new SparseReshape(*pScope, *input_indices, *input_shape, *new_shape);
		ObjectInfo* pObj = AddObjectMap(pSparseReshape, id, SYMBOL_SPARSERESHAPE, "SparseReshape", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseReshape->output_indices, OUTPUT_TYPE_OUTPUT, "output_indices");
			AddOutputInfo(pObj, &pSparseReshape->output_shape, OUTPUT_TYPE_OUTPUT, "output_shape");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseReshape(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseReshape;
}

void* Create_SparseSoftmax(std::string id, Json::Value pInputItem) {
	SparseSoftmax* pSparseSoftmax = nullptr;
	Scope* pScope = nullptr;
	Output* sp_indices = nullptr;
	Output* sp_values = nullptr;
	Output* sp_shape = nullptr;

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
				std::string msg = string_format("warning : SparseSoftmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sp_indices")
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
							sp_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSoftmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sp_values")
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
							sp_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSoftmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sp_shape")
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
							sp_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSoftmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseSoftmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && sp_indices && sp_values && sp_shape)
	{
		pSparseSoftmax = new SparseSoftmax(*pScope, *sp_indices, *sp_values, *sp_shape);
		ObjectInfo* pObj = AddObjectMap(pSparseSoftmax, id, SYMBOL_SPARSESOFTMAX, "SparseSoftmax", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseSoftmax->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseSoftmax(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseSoftmax;
}

void* Create_SparseSparseMaximum(std::string id, Json::Value pInputItem) {
	SparseSparseMaximum* pSparseSparseMaximum = nullptr;
	Scope* pScope = nullptr;
	Output* a_indices = nullptr;
	Output* a_values = nullptr;
	Output* a_shape = nullptr;
	Output* b_indices = nullptr;
	Output* b_values = nullptr;
	Output* b_shape = nullptr;

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
				std::string msg = string_format("warning : SparseSparseMaximum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a_indices")
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
							a_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSparseMaximum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a_values")
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
							a_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSparseMaximum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a_shape")
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
							a_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSparseMaximum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "b_indices")
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
							b_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSparseMaximum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "b_values")
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
							b_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSparseMaximum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "b_shape")
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
							b_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSparseMaximum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseSparseMaximum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && a_indices && a_values && a_shape && b_indices && b_values && b_shape)
	{
		pSparseSparseMaximum = new SparseSparseMaximum(*pScope, *a_indices, *a_values, *a_shape, *b_indices, *b_values, *b_shape);
		ObjectInfo* pObj = AddObjectMap(pSparseSparseMaximum, id, SYMBOL_SPARSESPARSEMAXIMUM, "SparseSparseMaximum", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseSparseMaximum->output_indices, OUTPUT_TYPE_OUTPUT, "output_indices");
			AddOutputInfo(pObj, &pSparseSparseMaximum->output_values, OUTPUT_TYPE_OUTPUT, "output_values");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseSparseMaximum(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseSparseMaximum;
}

void* Create_SparseSparseMinimum(std::string id, Json::Value pInputItem) {
	SparseSparseMinimum* pSparseSparseMinimum = nullptr;
	Scope* pScope = nullptr;
	Output* a_indices = nullptr;
	Output* a_values = nullptr;
	Output* a_shape = nullptr;
	Output* b_indices = nullptr;
	Output* b_values = nullptr;
	Output* b_shape = nullptr;

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
				std::string msg = string_format("warning : SparseSparseMinimum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a_indices")
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
							a_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSparseMinimum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a_values")
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
							a_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSparseMinimum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a_shape")
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
							a_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSparseMinimum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "b_indices")
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
							b_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSparseMinimum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "b_values")
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
							b_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSparseMinimum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "b_shape")
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
							b_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSparseMinimum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseSparseMinimum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && a_indices && a_values && a_shape && b_indices && b_values && b_shape)
	{
		pSparseSparseMinimum = new SparseSparseMinimum(*pScope, *a_indices, *a_values, *a_shape, *b_indices, *b_values, *b_shape);
		ObjectInfo* pObj = AddObjectMap(pSparseSparseMinimum, id, SYMBOL_SPARSESPARSEMINIMUM, "SparseSparseMinimum", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseSparseMinimum->output_indices, OUTPUT_TYPE_OUTPUT, "output_indices");
			AddOutputInfo(pObj, &pSparseSparseMinimum->output_values, OUTPUT_TYPE_OUTPUT, "output_values");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseSparseMinimum(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseSparseMinimum;
}

void* Create_SparseSplit(std::string id, Json::Value pInputItem) {
	SparseSplit* pSparseSplit = nullptr;
	Scope* pScope = nullptr;
	Output* split_dim = nullptr;
	Output* indices = nullptr;
	Output* values = nullptr;
	Output* shape = nullptr;
	int64 num_split = 0;

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
				std::string msg = string_format("warning : SparseSplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "split_dim")
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
							split_dim = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "indices")
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
							indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "values")
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
							values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "shape")
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
							shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_split")
		{
			if (strPinInterface == "int64")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					num_split = stoll(strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseSplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && split_dim &&indices&&values&&shape)
	{
		pSparseSplit = new SparseSplit(*pScope, *split_dim, *indices,*values,*shape,num_split);
		ObjectInfo* pObj = AddObjectMap(pSparseSplit, id, SYMBOL_SPARSESPLIT, "SparseSplit", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseSplit->output_indices, OUTPUT_TYPE_OUTPUT, "output_indices");
			AddOutputInfo(pObj, &pSparseSplit->output_values, OUTPUT_TYPE_OUTPUT, "output_values");
			AddOutputInfo(pObj, &pSparseSplit->output_shape, OUTPUT_TYPE_OUTPUT, "output_shape");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseSplit(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseSplit;
}

void* Create_SparseTensorDenseAdd(std::string id, Json::Value pInputItem) {
	SparseTensorDenseAdd* pSparseTensorDenseAdd = nullptr;
	Scope* pScope = nullptr;
	Output* a_indices = nullptr;
	Output* a_values = nullptr;;
	Output* a_shape = nullptr;
	Output* b = nullptr;

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
				std::string msg = string_format("warning : SparseTensorDenseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a_indices")
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
							a_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseTensorDenseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a_values")
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
							a_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseTensorDenseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a_shape")
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
							a_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseTensorDenseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "b")
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
							b = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseTensorDenseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseTensorDenseAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && a_indices &&a_values && a_shape&&b)
	{
		pSparseTensorDenseAdd = new SparseTensorDenseAdd(*pScope, *a_indices, *a_values, *a_shape,*b);
		ObjectInfo* pObj = AddObjectMap(pSparseTensorDenseAdd, id, SYMBOL_SPARSETENSORDENSEADD, "SparseTensorDenseAdd", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseTensorDenseAdd->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseTensorDenseAdd(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseTensorDenseAdd;
}

void* Create_SparseTensorDenseMatMul(std::string id, Json::Value pInputItem) {
	SparseTensorDenseMatMul* pSparseTensorDenseMatMul = nullptr;
	Scope* pScope = nullptr;
	Output* a_indices = nullptr;
	Output* a_values = nullptr;
	Output* a_shape = nullptr;
	Output* b = nullptr;
	SparseTensorDenseMatMul::Attrs attrs;

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
				std::string msg = string_format("warning : SparseTensorDenseMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a_indices")
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
							a_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseTensorDenseMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a_values")
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
							a_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseTensorDenseMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a_shape")
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
							a_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseTensorDenseMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "b")
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
							b = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseTensorDenseMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "SparseTensorDenseMatMul::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if(attrParser.GetAttribute("adjoint_a_")!="") attrs.AdjointA(attrParser.GetValue_bool("adjoint_a_"));
				if (attrParser.GetAttribute("adjoint_b_") != "") attrs.AdjointB(attrParser.GetValue_bool("adjoint_b_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseTensorDenseMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && a_indices &&a_values && a_shape&&b)
	{
		pSparseTensorDenseMatMul = new SparseTensorDenseMatMul(*pScope, *a_indices, *a_values, *a_shape, *b,attrs);
		ObjectInfo* pObj = AddObjectMap(pSparseTensorDenseMatMul, id, SYMBOL_SPARSETENSORDENSEMATMUL, "SparseTensorDenseMatMul", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseTensorDenseMatMul->product, OUTPUT_TYPE_OUTPUT, "product");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseTensorDenseMatMul(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseTensorDenseMatMul;
}

void* Create_SparseToDense(std::string id, Json::Value pInputItem) {
	SparseToDense* pSparseToDense = nullptr;
	Scope* pScope = nullptr;
	Output* sparse_indices = nullptr;
	Output* output_shape = nullptr;
	Output* sparse_values = nullptr;
	Output* default_value = nullptr;
	SparseToDense::Attrs attrs;

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
				std::string msg = string_format("warning : SparseToDense - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "output_shape")
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
							output_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseToDense - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sparse_values")
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
							sparse_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseToDense - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "default_value")
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
							default_value = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseToDense - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "SparseToDense::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if(attrParser.GetAttribute("validate_indices_")!="") 
					attrs.ValidateIndices(attrParser.GetValue_bool("validate_indices_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseToDense - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && sparse_indices && output_shape && sparse_values && default_value)
	{
		pSparseToDense = new SparseToDense(*pScope, *sparse_indices, *output_shape, *sparse_values,*default_value,attrs);
		ObjectInfo* pObj = AddObjectMap(pSparseToDense, id, SYMBOL_SPARSETODENSE, "SparseToDense", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseToDense->dense, OUTPUT_TYPE_OUTPUT, "dense");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseToDense(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseToDense;
}

void* Create_TakeManySparseFromTensorsMap(std::string id, Json::Value pInputItem) {
	TakeManySparseFromTensorsMap* pTakeManySparseFromTensorsMap = nullptr;
	Scope* pScope = nullptr;
	Output* sparse_handles = nullptr;
	DataType dtype;
	TakeManySparseFromTensorsMap::Attrs attrs;

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
				std::string msg = string_format("warning : TakeManySparseFromTensorsMap - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sparse_handles")
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
							sparse_handles = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TakeManySparseFromTensorsMap - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				dtype = GetDatatypeFromInitial(strPinInitial);
				if (dtype == DT_INVALID)
				{
					std::string msg = string_format("warning : TakeManySparseFromTensorsMap - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : TakeManySparseFromTensorsMap - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "TakeManySparseFromTensorsMap::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if(attrParser.GetAttribute("container_")!="")
					attrs.Container(attrParser.GetValue_StringPiece("container_"));
				if (attrParser.GetAttribute("shared_name_") != "")
					attrs.SharedName(attrParser.GetValue_StringPiece("shared_name_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : TakeManySparseFromTensorsMap - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && sparse_handles)
	{
		pTakeManySparseFromTensorsMap = new TakeManySparseFromTensorsMap(*pScope, *sparse_handles, dtype, attrs);
		ObjectInfo* pObj = AddObjectMap(pTakeManySparseFromTensorsMap, id, SYMBOL_TAKEMANYSPARSEFROMTENSORSMAP, "TakeManySparseFromTensorsMap", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pTakeManySparseFromTensorsMap->sparse_indices, OUTPUT_TYPE_OUTPUT, "sparse_indices");
			AddOutputInfo(pObj, &pTakeManySparseFromTensorsMap->sparse_values, OUTPUT_TYPE_OUTPUT, "sparse_values");
			AddOutputInfo(pObj, &pTakeManySparseFromTensorsMap->sparse_shape, OUTPUT_TYPE_OUTPUT, "sparse_shape");

		}
	}
	else
	{
		std::string msg = string_format("error : TakeManySparseFromTensorsMap(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pTakeManySparseFromTensorsMap;
}
