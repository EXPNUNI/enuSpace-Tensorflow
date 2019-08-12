#include "stdafx.h"
#include "tf_array_ops.h"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "jsoncpp/json.h"

#include "GlobalHeader.h"
#include "tensorflow.h"
#include "utility_functions.h"
#include "enuSpaceToTensorflow.h"
#include "AttributeParser.h"

void* Create_BatchToSpace(std::string id, Json::Value pInputItem) {
	BatchToSpace* pBatchToSpace = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* crops = nullptr;
	int64 block_size = 0;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("BatchToSpace - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					std::string strPinSave = ItemValue.get("pin-save", "").asString();
					if (strPinSave != "binary")
					{
						if (!strPinInitial.empty())
							pinput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
					}
					else
					{
						std::string strPinBinPos = ItemValue.get("pin-binary-pos", "").asString();
						int ipos = stoi(strPinBinPos);
						if (ipos != -1 && m_FileData)
						{
							pinput = (Output*)Create_BinaryToOutput(*m_pScope, strPinType, strPinShape, m_FileData, ipos);
						}
						else
						{
							std::string msg = string_format("%s(%s) binary data missed.", id.c_str(), strPinName.c_str());
							SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("BatchToSpace - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "crops")
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
							crops = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						crops = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("BatchToSpace - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "block_size")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				if (strPinInitial == "")
				{
					block_size = 0;
				}
				else
				{
					block_size = stoll(strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("BatchToSpace - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("BatchToSpace pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pinput && crops)
	{
		pBatchToSpace = new BatchToSpace(*pScope, *pinput, *crops, block_size);
		ObjectInfo* pObj = AddObjectMap(pBatchToSpace, id, SYMBOL_BATCHTOSPACE, "BatchToSpace", pInputItem);
		if (pObj) {
			if (pBatchToSpace->output.node())
			{
				AddOutputInfo(pObj, &pBatchToSpace->output, OUTPUT_TYPE_OUTPUT, "output");
			}
			else
			{
				std::string msg = string_format("BatchToSpace(%s) Object output create failed.", id.c_str());
				SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
			}
		}
	}
	else
	{
		std::string msg = string_format("BatchToSpace(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pBatchToSpace;
}
void* Create_BatchToSpaceND(std::string id, Json::Value pInputItem) {
	BatchToSpaceND* pBatchToSpaceND = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* crops = nullptr;
	Output* block_shape = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("BatchToSpaceND - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pinput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("BatchToSpaceND - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "block_shape")
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
							block_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						block_shape = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("BatchToSpaceND - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "crops")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
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
							crops = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						crops = (Output*)Create_StrToOutput(*m_pScope, "DT_INT32", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("BatchToSpaceND - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("BatchToSpaceND pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pinput && crops && block_shape)
	{
		pBatchToSpaceND = new BatchToSpaceND(*pScope, *pinput, *block_shape, *crops);
		ObjectInfo* pObj = AddObjectMap(pBatchToSpaceND, id, SYMBOL_BATCHTOSPACEND, "BatchToSpaceND", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pBatchToSpaceND->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("BatchToSpaceND(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pBatchToSpaceND;
}
void* Create_Bitcast(std::string id, Json::Value pInputItem) {
	Bitcast* pBitcast = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	DataType dtype;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Bitcast - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Bitcast - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "type")
		{
			if (strPinInterface == "DataType")
			{
				dtype = GetDatatypeFromInitial(strPinInitial);
				if(dtype == DT_INVALID)
				{
					std::string msg = string_format("Bitcast - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
				}
			}
			else
			{
				std::string msg = string_format("Bitcast - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("Bitcast pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput && dtype)
	{
		pBitcast = new Bitcast(*pScope, *pInput, dtype);
		if (!pScope->ok())
		{
			Status st = pScope->status();
			SetLastError(DEF_ERROR, "", 0, st.error_message().c_str(), false, id.c_str());
		}
		ObjectInfo* pObj = AddObjectMap(pBitcast, id, SYMBOL_BITCAST, "Bitcast", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pBitcast->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("Bitcast(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pBitcast;
}
void* Create_BroadcastDynamicShape(std::string id, Json::Value pInputItem) {
	BroadcastDynamicShape* pBroadcastDynamicShape = nullptr;
	Scope* pScope = nullptr;
	Output* s0 = nullptr;
	Output* s1 = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("BroadcastDynamicShape - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "s0")
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
							s0 = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						s0 = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("BroadcastDynamicShape - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "s1")
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
							s1 = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						s1 = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("BroadcastDynamicShape - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("BroadcastDynamicShape pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && s0 && s1)
	{
		pBroadcastDynamicShape = new BroadcastDynamicShape(*pScope, *s0, *s1);
		if (!pScope->ok())
		{
			Status st = pScope->status();
			std::string errors = st.error_message().c_str();
			std::string msg = string_format("%s.", errors);
			SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
		}
		ObjectInfo* pObj = AddObjectMap(pBroadcastDynamicShape, id, SYMBOL_BITCAST, "BroadcastDynamicShape", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pBroadcastDynamicShape->r0, OUTPUT_TYPE_OUTPUT, "r0");
	}
	else
	{
		std::string msg = string_format("BroadcastDynamicShape(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pBroadcastDynamicShape;
}
void* Create_CheckNumerics(std::string id, Json::Value pInputItem) {
	CheckNumerics* pCheckNumerics = nullptr;
	Scope* pScope = nullptr;
	Output* ptensor = nullptr;
	std::string message;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("CheckNumerics - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "tensor")
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
							ptensor = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						ptensor = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("CheckNumerics - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "message")
		{
			if (strPinInterface == "StringPiece")
			{
				message = strPinInitial;
			}
			else
			{
				std::string msg = string_format("CheckNumerics - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("CheckNumerics pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && ptensor)
	{
		pCheckNumerics = new CheckNumerics(*pScope, *ptensor, message);
		ObjectInfo* pObj = AddObjectMap(pCheckNumerics, id, SYMBOL_CHECKNUMERICS, "CheckNumerics", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pCheckNumerics->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("CheckNumerics(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pCheckNumerics;
}
void* Create_Concat(std::string id, Json::Value pInputItem) {
	Concat* pConcat = nullptr;
	Scope* pScope = nullptr;
	OutputList* values = nullptr;
	Output* axis = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Concat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "values")
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
							values = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("Concat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "axis")
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
							axis = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						axis = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Concat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("Concat pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && values && axis)
	{
		pConcat = new Concat(*pScope, *values, *axis);
		ObjectInfo* pObj = AddObjectMap(pConcat, id, SYMBOL_CONCAT, "Concat", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pConcat->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("Concat(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pConcat;
}
void * Create_DebugGradientIdentity(std::string id, Json::Value pInputItem)
{
	DebugGradientIdentity* pDebugGradientIdentity = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("DebugGradientIdentity - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pinput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("DebugGradientIdentity - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("DebugGradientIdentity pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pinput)
	{
		pDebugGradientIdentity = new DebugGradientIdentity(*pScope, *pinput);
		ObjectInfo* pObj = AddObjectMap(pDebugGradientIdentity, id, SYMBOL_DEBUGGRADIENTIDENTITY, "DebugGradientIdentity", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pDebugGradientIdentity->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("DebugGradientIdentity(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pDebugGradientIdentity;
}
void* Create_DepthToSpace(std::string id, Json::Value pInputItem) {
	DepthToSpace* pDepthToSpace = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	int64 block_size = 0;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("DepthToSpace - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("DepthToSpace - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "block_size")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				block_size = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("DepthToSpace - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("DepthToSpace pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput)
	{
		pDepthToSpace = new DepthToSpace(*pScope, *pInput, block_size);
		ObjectInfo* pObj = AddObjectMap(pDepthToSpace, id, SYMBOL_DEPTHTOSPACE, "DepthToSpace", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pDepthToSpace->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("DepthToSpace(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pDepthToSpace;
}
void* Create_Dequantize(std::string id, Json::Value pInputItem) {
	Dequantize* pDequantize = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* min_range = nullptr;
	Output* max_range = nullptr;
	Dequantize::Attrs attrs;
	std::string mode;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Dequantize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Dequantize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "min_range")
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
							min_range = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						min_range = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Dequantize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "max_range")
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
							max_range = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						max_range = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Dequantize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Dequantize::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("mode_") != "")
				{
					mode = attrParser.GetAttribute("mode_");
					attrs.mode_ = mode;
				}
			}
		}
		else
		{
			std::string msg = string_format("Dequantize pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput && min_range && max_range)
	{
		pDequantize = new Dequantize(*pScope, *pInput, *min_range, *max_range, attrs);
		ObjectInfo* pObj = AddObjectMap(pDequantize, id, SYMBOL_DEQUANTIZE, "Dequantize", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pDequantize->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("Dequantize(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pDequantize;
}
void* Create_Diag(std::string id, Json::Value pInputItem) {
	Diag* pDiag = nullptr;
	Scope* pScope = nullptr;
	Output* diagonal = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Diag - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "diagonal")
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
							diagonal = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						diagonal = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Diag - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("Diag pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && diagonal)
	{
		pDiag = new Diag(*pScope, *diagonal);
		ObjectInfo* pObj = AddObjectMap(pDiag, id, SYMBOL_DIAG, "Diag", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pDiag->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("Diag(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pDiag;
}
void* Create_DiagPart(std::string id, Json::Value pInputItem) {
	DiagPart* pDiagPart = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("DiagPart - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("DiagPart - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("DiagPart pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput)
	{
		pDiagPart = new DiagPart(*pScope, *pInput);
		ObjectInfo* pObj = AddObjectMap(pDiagPart, id, SYMBOL_DIAG, "DiagPart", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pDiagPart->diagonal, OUTPUT_TYPE_OUTPUT, "diagonal");
	}
	else
	{
		std::string msg = string_format("DiagPart(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pDiagPart;
}
void* Create_EditDistance(std::string id, Json::Value pInputItem) {
	EditDistance* pEditDistance = nullptr;
	Scope* pScope = nullptr;
	Output* hypothesis_indices = nullptr;
	Output* hypothesis_values = nullptr;
	Output* hypothesis_shape = nullptr;
	Output* truth_indices = nullptr;
	Output* truth_values = nullptr;
	Output* truth_shape = nullptr;
	EditDistance::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("EditDistance - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "hypothesis_indices")
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
							hypothesis_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						hypothesis_indices = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("EditDistance - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "hypothesis_values")
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
							hypothesis_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						hypothesis_values = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("EditDistance - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "hypothesis_shape")
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
							hypothesis_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						hypothesis_shape = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("EditDistance - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "truth_indices")
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
							truth_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						truth_indices = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("EditDistance - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "truth_values")
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
							truth_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						truth_values = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("EditDistance - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "truth_shape")
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
							truth_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						truth_shape = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("EditDistance - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "EditDistance::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("normalize_") != "")
				{
					attrs.normalize_ = attrParser.ConvStrToBool(attrParser.GetAttribute("normalize_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("EditDistance pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && hypothesis_indices && hypothesis_values && hypothesis_shape && truth_indices && truth_values && truth_shape)
	{
		pEditDistance = new EditDistance(*pScope, *hypothesis_indices, *hypothesis_values, *hypothesis_shape, *truth_indices, *truth_values, *truth_shape, attrs);
		ObjectInfo* pObj = AddObjectMap(pEditDistance, id, SYMBOL_EDITDISTANCE, "EditDistance", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pEditDistance->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("EditDistance(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pEditDistance;
}
void* Create_ExpandDims(std::string id, Json::Value pInputItem) {
	ExpandDims* pExpandDims = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* axis = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("ExpandDims - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("ExpandDims - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "axis")
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
							axis = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						axis = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("ExpandDims - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}		
		else
		{
			std::string msg = string_format("ExpandDims pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput && axis)
	{
		pExpandDims = new ExpandDims(*pScope, *pInput, *axis);
		ObjectInfo* pObj = AddObjectMap(pExpandDims, id, SYMBOL_EXPANDDIMS, "ExpandDims", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pExpandDims->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("ExpandDims(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pExpandDims;
}
void* Create_ExtractImagePatches(std::string id, Json::Value pInputItem) {
	ExtractImagePatches* pExtractImagePatches = nullptr;
	Scope* pScope = nullptr;
	Output* images = nullptr;
	std::vector<int> v_ksizes;
	std::vector<int> v_strides;
	std::vector<int> v_rates;
	std::string padding;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("ExtractImagePatches - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "images")
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
							images = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						images = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("ExtractImagePatches - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "ksizes")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_ksizes);
			}
			else
			{
				std::string msg = string_format("ExtractImagePatches - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("ExtractImagePatches - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "rates")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_rates);
			}
			else
			{
				std::string msg = string_format("ExtractImagePatches - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("ExtractImagePatches - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("ExtractImagePatches pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && images)
	{
		gtl::ArraySlice<int> ksizes(v_ksizes);
		gtl::ArraySlice<int> strides(v_strides);
		gtl::ArraySlice<int> rates(v_rates);

		pExtractImagePatches = new ExtractImagePatches(*pScope, *images, ksizes, strides, rates, padding);
		ObjectInfo* pObj = AddObjectMap(pExtractImagePatches, id, SYMBOL_MATMUL, "ExtractImagePatches", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pExtractImagePatches->patches, OUTPUT_TYPE_OUTPUT, "patches");

		v_ksizes.clear();
		v_strides.clear();
		v_rates.clear();
	}
	else
	{
		std::string msg = string_format("ExtractImagePatches(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());;
	}

	return pExtractImagePatches;
}
void* Create_FakeQuantWithMinMaxArgs(std::string id, Json::Value pInputItem) {
	FakeQuantWithMinMaxArgs* pFakeQuantWithMinMaxArgs = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	FakeQuantWithMinMaxArgs::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxArgs - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "inputs")
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxArgs - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "FakeQuantWithMinMaxArgs::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("min_") != "")
				{
					attrs.min_ = attrParser.ConvStrToFloat(attrParser.GetAttribute("min_"));
				}
				if (attrParser.GetAttribute("max_") != "")
				{
					attrs.max_ = attrParser.ConvStrToFloat(attrParser.GetAttribute("max_"));
				}
				if (attrParser.GetAttribute("num_bits_") != "")
				{
					attrs.num_bits_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("num_bits_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("FakeQuantWithMinMaxArgs pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput)
	{
		pFakeQuantWithMinMaxArgs = new FakeQuantWithMinMaxArgs(*pScope, *pInput, attrs);
		ObjectInfo* pObj = AddObjectMap(pFakeQuantWithMinMaxArgs, id, SYMBOL_FAKEQUANTWITHMINMAXARGS, "FakeQuantWithMinMaxArgs", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pFakeQuantWithMinMaxArgs->outputs, OUTPUT_TYPE_OUTPUT, "outputs");
	}
	else
	{
		std::string msg = string_format("FakeQuantWithMinMaxArgs(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pFakeQuantWithMinMaxArgs;
}
void* Create_FakeQuantWithMinMaxArgsGradient(std::string id, Json::Value pInputItem) {
	FakeQuantWithMinMaxArgsGradient* pFakeQuantWithMinMaxArgsGradient = nullptr;
	Scope* pScope = nullptr;
	Output* gradients = nullptr;
	Output* inputs = nullptr;
	FakeQuantWithMinMaxArgsGradient::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxArgsGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "gradients")
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
							gradients = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						gradients = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxArgsGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "inputs")
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
							inputs = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						inputs = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxArgsGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "FakeQuantWithMinMaxArgsGradient::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("min_") != "")
				{
					attrs.min_ = attrParser.ConvStrToFloat(attrParser.GetAttribute("min_"));
				}
				if (attrParser.GetAttribute("max_") != "")
				{
					attrs.max_ = attrParser.ConvStrToFloat(attrParser.GetAttribute("max_"));
				}
				if (attrParser.GetAttribute("num_bits_") != "")
				{
					attrs.num_bits_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("num_bits_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("FakeQuantWithMinMaxArgsGradient pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && gradients && inputs)
	{
		pFakeQuantWithMinMaxArgsGradient = new FakeQuantWithMinMaxArgsGradient(*pScope, *gradients, *inputs, attrs);
		ObjectInfo* pObj = AddObjectMap(pFakeQuantWithMinMaxArgsGradient, id, SYMBOL_FAKEQUANTWITHMINMAXARGSGRADIENT, "FakeQuantWithMinMaxArgsGradient", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pFakeQuantWithMinMaxArgsGradient->backprops, OUTPUT_TYPE_OUTPUT, "backprops");
	}
	else
	{
		std::string msg = string_format("FakeQuantWithMinMaxArgsGradient(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pFakeQuantWithMinMaxArgsGradient;
}
void* Create_FakeQuantWithMinMaxVars(std::string id, Json::Value pInputItem) {
	FakeQuantWithMinMaxVars* pFakeQuantWithMinMaxVars = nullptr;
	Scope* pScope = nullptr;
	Output* inputs = nullptr;
	Output* min = nullptr;
	Output* max = nullptr;
	FakeQuantWithMinMaxVars::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxVars - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "inputs")
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
							inputs = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						inputs = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxVars - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "min")
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
							min = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						min = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxVars - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "max")
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
							max = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						max = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxVars - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "FakeQuantWithMinMaxVars::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("num_bits_") != "")
				{
					attrs.num_bits_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("num_bits_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("FakeQuantWithMinMaxVars pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && inputs && min && max)
	{
		pFakeQuantWithMinMaxVars = new FakeQuantWithMinMaxVars(*pScope, *inputs, *min, *max, attrs);
		ObjectInfo* pObj = AddObjectMap(pFakeQuantWithMinMaxVars, id, SYMBOL_FAKEQUANTWITHMINMAXVARS, "FakeQuantWithMinMaxVars", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pFakeQuantWithMinMaxVars->outputs, OUTPUT_TYPE_OUTPUT, "outputs");
	}
	else
	{
		std::string msg = string_format("FakeQuantWithMinMaxVars(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pFakeQuantWithMinMaxVars;
}
void* Create_FakeQuantWithMinMaxVarsGradient(std::string id, Json::Value pInputItem) {
	FakeQuantWithMinMaxVarsGradient* pFakeQuantWithMinMaxVarsGradient = nullptr;
	Scope* pScope = nullptr;
	Output* gradients = nullptr;
	Output* inputs = nullptr;
	Output* min = nullptr;
	Output* max = nullptr;
	FakeQuantWithMinMaxVarsGradient::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxVarsGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "gradients")
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
							gradients = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						gradients = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxVarsGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "inputs")
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
							inputs = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						inputs = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxVarsGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "min")
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
							min = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						min = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxVarsGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "max")
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
							max = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						max = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxVarsGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "FakeQuantWithMinMaxVarsGradient::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("num_bits_") != "")
				{
					attrs.num_bits_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("num_bits_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("FakeQuantWithMinMaxVarsGradient pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && gradients && inputs && min && max)
	{
		pFakeQuantWithMinMaxVarsGradient = new FakeQuantWithMinMaxVarsGradient(*pScope, *gradients, *inputs, *min, *max, attrs);
		ObjectInfo* pObj = AddObjectMap(pFakeQuantWithMinMaxVarsGradient, id, SYMBOL_FAKEQUANTWITHMINMAXVARSGRADIENT, "FakeQuantWithMinMaxVarsGradient", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pFakeQuantWithMinMaxVarsGradient->backprops_wrt_input, OUTPUT_TYPE_OUTPUT, "backprops_wrt_input");
			AddOutputInfo(pObj, &pFakeQuantWithMinMaxVarsGradient->backprop_wrt_min, OUTPUT_TYPE_OUTPUT, "backprop_wrt_min");
			AddOutputInfo(pObj, &pFakeQuantWithMinMaxVarsGradient->backprop_wrt_max, OUTPUT_TYPE_OUTPUT, "backprop_wrt_max");
		}
			
	}
	else
	{
		std::string msg = string_format("FakeQuantWithMinMaxVarsGradient(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pFakeQuantWithMinMaxVarsGradient;
}
void* Create_FakeQuantWithMinMaxVarsPerChannel(std::string id, Json::Value pInputItem) {
	FakeQuantWithMinMaxVarsPerChannel* pFakeQuantWithMinMaxVarsPerChannel = nullptr;
	Scope* pScope = nullptr;
	Output* inputs = nullptr;
	Output* min = nullptr;
	Output* max = nullptr;
	FakeQuantWithMinMaxVarsPerChannel::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxVarsPerChannel - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "inputs")
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
							inputs = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						inputs = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxVarsPerChannel - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "min")
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
							min = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						min = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxVarsPerChannel - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "max")
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
							max = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						max = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxVarsPerChannel - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "FakeQuantWithMinMaxVarsPerChannel::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("num_bits_") != "")
				{
					attrs.num_bits_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("num_bits_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("FakeQuantWithMinMaxVarsPerChannel pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && inputs && min && max)
	{
		pFakeQuantWithMinMaxVarsPerChannel = new FakeQuantWithMinMaxVarsPerChannel(*pScope, *inputs, *min, *max, attrs);
		ObjectInfo* pObj = AddObjectMap(pFakeQuantWithMinMaxVarsPerChannel, id, SYMBOL_FAKEQUANTWITHMINMAXVARSPERCHANNEL, "FakeQuantWithMinMaxVarsPerChannel", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pFakeQuantWithMinMaxVarsPerChannel->outputs, OUTPUT_TYPE_OUTPUT, "outputs");
	}
	else
	{
		std::string msg = string_format("FakeQuantWithMinMaxVarsPerChannel(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pFakeQuantWithMinMaxVarsPerChannel;
}
void* Create_FakeQuantWithMinMaxVarsPerChannelGradient(std::string id, Json::Value pInputItem) {
	FakeQuantWithMinMaxVarsPerChannelGradient* pFakeQuantWithMinMaxVarsPerChannelGradient = nullptr;
	Scope* pScope = nullptr;
	Output* gradients = nullptr;
	Output* inputs = nullptr;
	Output* min = nullptr;
	Output* max = nullptr;
	FakeQuantWithMinMaxVarsPerChannelGradient::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxVarsPerChannelGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "gradients")
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
							gradients = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						gradients = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxVarsPerChannelGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "inputs")
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
							inputs = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						inputs = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxVarsPerChannelGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "min")
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
							min = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						min = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxVarsPerChannelGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "max")
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
							max = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						max = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("FakeQuantWithMinMaxVarsPerChannelGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "FakeQuantWithMinMaxVarsPerChannelGradient::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("num_bits_") != "")
				{
					attrs.num_bits_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("num_bits_"));
				}
				if (attrParser.GetAttribute("narrow_range_") != "")
				{
					attrs.narrow_range_ = attrParser.ConvStrToBool(attrParser.GetAttribute("narrow_range_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("FakeQuantWithMinMaxVarsPerChannelGradient pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && gradients && inputs && min && max)
	{
		pFakeQuantWithMinMaxVarsPerChannelGradient = new FakeQuantWithMinMaxVarsPerChannelGradient(*pScope, *gradients,*inputs, *min, *max, attrs);
		ObjectInfo* pObj = AddObjectMap(pFakeQuantWithMinMaxVarsPerChannelGradient, id, SYMBOL_FAKEQUANTWITHMINMAXVARSPERCHANNELGRADIENT, "FakeQuantWithMinMaxVarsPerChannelGradient", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pFakeQuantWithMinMaxVarsPerChannelGradient->backprops_wrt_input, OUTPUT_TYPE_OUTPUT, "backprops_wrt_input");
			AddOutputInfo(pObj, &pFakeQuantWithMinMaxVarsPerChannelGradient->backprop_wrt_min, OUTPUT_TYPE_OUTPUT, "backprop_wrt_min");
			AddOutputInfo(pObj, &pFakeQuantWithMinMaxVarsPerChannelGradient->backprop_wrt_max, OUTPUT_TYPE_OUTPUT, "backprop_wrt_max");
		}
	}
	else
	{
		std::string msg = string_format("FakeQuantWithMinMaxVarsPerChannelGradient(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pFakeQuantWithMinMaxVarsPerChannelGradient;
}
void* Create_Fill(std::string id, Json::Value pInputItem) {
	Fill* pFill = nullptr;
	Scope* pScope = nullptr;
	Output* dims = nullptr;
	Output* value = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Fill - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dims")
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
							dims = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						dims = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Fill - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "value")
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
							value = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						value = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Fill - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("Fill pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && dims && value)
	{
		pFill = new Fill(*pScope, *dims, *value);
		ObjectInfo* pObj = AddObjectMap(pFill, id, SYMBOL_FILL, "Fill", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pFill->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("Fill(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pFill;
}
void* Create_Gather(std::string id, Json::Value pInputItem) {
	Gather* pGather = nullptr;
	Scope* pScope = nullptr;
	Output* params = nullptr;
	Output* indices = nullptr;
	Gather::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Gather - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "params")
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
							params = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						params = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Gather - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						indices = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Gather - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Gather::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("validate_indices_") != "")
				{
					attrs.validate_indices_ = attrParser.ConvStrToBool(attrParser.GetAttribute("validate_indices_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("Gather pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && params && indices)
	{
		pGather = new Gather(*pScope, *params, *indices, attrs);
		ObjectInfo* pObj = AddObjectMap(pGather, id, SYMBOL_GATHER, "Gather", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pGather->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("Gather(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pGather;
}
void* Create_GatherNd(std::string id, Json::Value pInputItem) {
	GatherNd* pGatherNd = nullptr;
	Scope* pScope = nullptr;
	Output* params = nullptr;
	Output* indices = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("GatherNd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "params")
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
							params = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						params = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("GatherNd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						indices = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("GatherNd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("GatherNd pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && params && indices)
	{
		pGatherNd = new GatherNd(*pScope, *params, *indices);
		ObjectInfo* pObj = AddObjectMap(pGatherNd, id, SYMBOL_GATHERND, "GatherNd", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pGatherNd->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("GatherNd(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pGatherNd;
}
void* Create_GatherV2(std::string id, Json::Value pInputItem) {
	GatherV2* pGatherV2 = nullptr;
	Scope* pScope = nullptr;
	Output* params = nullptr;
	Output* indices = nullptr;
	Output* axis = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("GatherV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "params")
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
							params = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						params = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("GatherV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						indices = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("GatherV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "axis")
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
							axis = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						axis = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("GatherV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("GatherV2 pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}
	if (pScope && params && indices && axis)
	{
		pGatherV2 = new GatherV2(*pScope, *params, *indices, *axis);
		ObjectInfo* pObj = AddObjectMap(pGatherV2, id, SYMBOL_GATHERV2, "GatherV2", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pGatherV2->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("GatherV2(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pGatherV2;
}
void* Create_Identity(std::string id, Json::Value pInputItem) {
	Identity* pIdentity = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Identity - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Identity - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("Identity pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput)
	{
		pIdentity = new Identity(*pScope, *pInput);
		ObjectInfo* pObj = AddObjectMap(pIdentity, id, SYMBOL_IDENTITY, "Identity", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pIdentity->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("Identity(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pIdentity;
}
void* Create_ImmutableConst(std::string id, Json::Value pInputItem) {
	ImmutableConst* pImmutableConst = nullptr;
	Scope* pScope = nullptr;
	DataType dtype;
	PartialTensorShape shape;
	std::string memory_region_name;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("ImmutableConst - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				dtype = GetDatatypeFromInitial(strPinInitial);
				if (dtype == DT_INVALID)
				{
					std::string msg = string_format("ImmutableConst - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
				}
			}
			else
			{
				std::string msg = string_format("ImmutableConst - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "shape")
		{
			if (strPinInterface == "PartialTensorShape")
			{
				shape = GetPartialShapeFromInitial(strPinInitial);
			}
			else
			{
				std::string msg = string_format("ImmutableConst - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "memory_region_name")
		{
			if (strPinInterface == "StringPiece")
			{
				memory_region_name = strPinInitial;
			}
			else
			{
				std::string msg = string_format("ImmutableConst - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("ImmutableConst pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope)
	{
		pImmutableConst = new ImmutableConst(*pScope, dtype, shape, memory_region_name);
		ObjectInfo* pObj = AddObjectMap(pImmutableConst, id, SYMBOL_IMMUTABLECONST, "ImmutableConst", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pImmutableConst->tensor, OUTPUT_TYPE_OUTPUT, "tensor");
	}
	else
	{
		std::string msg = string_format("ImmutableConst(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pImmutableConst;
}
void* Create_InvertPermutation(std::string id, Json::Value pInputItem) {
	InvertPermutation* pInvertPermutation = nullptr;
	Scope* pScope = nullptr;
	Output* x = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("InvertPermutation - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "x")
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
							x = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						x = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("InvertPermutation - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("InvertPermutation pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && x)
	{
		pInvertPermutation = new InvertPermutation(*pScope, *x);
		ObjectInfo* pObj = AddObjectMap(pInvertPermutation, id, SYMBOL_INVERTPERMUTATION, "InvertPermutation", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pInvertPermutation->y, OUTPUT_TYPE_OUTPUT, "y");
	}
	else
	{
		std::string msg = string_format("InvertPermutation(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pInvertPermutation;
}
void* Create_SetDiff1D(std::string id, Json::Value pInputItem) {
	SetDiff1D* pSetDiff1D = nullptr;
	Scope* pScope = nullptr;
	Output* x = nullptr;
	Output* y = nullptr;
	SetDiff1D::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("SetDiff1D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "x")
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
							x = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						x = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("SetDiff1D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "y")
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
							y = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						y = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("SetDiff1D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "SetDiff1D::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("out_idx_") != "")
				{
					attrs.out_idx_ = attrParser.ConvStrToDataType(attrParser.GetAttribute("out_idx_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("SetDiff1D pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && x && y)
	{
		pSetDiff1D = new SetDiff1D(*pScope, *x, *y, attrs);
		ObjectInfo* pObj = AddObjectMap(pSetDiff1D, id, SYMBOL_SETDIFF1D, "SetDiff1D", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSetDiff1D->out, OUTPUT_TYPE_OUTPUT, "out");
			AddOutputInfo(pObj, &pSetDiff1D->idx, OUTPUT_TYPE_OUTPUT, "idx");
		}
	}
	else
	{
		std::string msg = string_format("SetDiff1D(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pSetDiff1D;
}
void* Create_MatrixBandPart(std::string id, Json::Value pInputItem) {
	MatrixBandPart* pMatrixBandPart = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* num_lower = nullptr;
	Output* num_upper = nullptr;
	
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("MatrixBandPart - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("MatrixBandPart - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "num_lower")
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
							num_lower = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						num_lower = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("MatrixBandPart - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "num_upper")
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
							num_upper = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						num_upper = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("MatrixBandPart - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("MatrixBandPart pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput && num_lower && num_upper)
	{
		pMatrixBandPart = new MatrixBandPart(*pScope, *pInput, *num_lower, *num_upper);
		ObjectInfo* pObj = AddObjectMap(pMatrixBandPart, id, SYMBOL_MATRIXBANDPART, "MatrixBandPart", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pMatrixBandPart->band, OUTPUT_TYPE_OUTPUT, "band");
	}
	else
	{
		std::string msg = string_format("MatrixBandPart(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pMatrixBandPart;
}
void* Create_MatrixDiag(std::string id, Json::Value pInputItem) {
	MatrixDiag* pMatrixDiag = nullptr;
	Scope* pScope = nullptr;
	Output* diagonal = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("MatrixDiag - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "diagonal")
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
							diagonal = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						diagonal = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("MatrixDiag - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("MatrixDiag pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && diagonal)
	{
		pMatrixDiag = new MatrixDiag(*pScope, *diagonal);
		ObjectInfo* pObj = AddObjectMap(pMatrixDiag, id, SYMBOL_MATRIXDIAG, "MatrixDiag", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pMatrixDiag->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("MatrixDiag(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pMatrixDiag;
}
void* Create_MatrixDiagPart(std::string id, Json::Value pInputItem) {
	MatrixDiagPart* pMatrixDiagPart = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("MatrixDiagPart - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("MatrixDiagPart - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("MatrixDiagPart pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput)
	{
		pMatrixDiagPart = new MatrixDiagPart(*pScope, *pInput);
		ObjectInfo* pObj = AddObjectMap(pMatrixDiagPart, id, SYMBOL_MATRIXDIAGPART, "MatrixDiagPart", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pMatrixDiagPart->diagonal, OUTPUT_TYPE_OUTPUT, "diagonal");
	}
	else
	{
		std::string msg = string_format("MatrixDiagPart(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pMatrixDiagPart;
}
void* Create_MatrixSetDiag(std::string id, Json::Value pInputItem) {
	MatrixSetDiag* pMatrixSetDiag = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* diagonal = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("MatrixSetDiag - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("MatrixSetDiag - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "diagonal")
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
							diagonal = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						diagonal = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("MatrixSetDiag - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("MatrixSetDiag pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput && diagonal)
	{
		pMatrixSetDiag = new MatrixSetDiag(*pScope, *pInput, *diagonal);
		ObjectInfo* pObj = AddObjectMap(pMatrixSetDiag, id, SYMBOL_MATRIXSETDIAG, "MatrixSetDiag", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pMatrixSetDiag->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("MatrixSetDiag(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pMatrixSetDiag;
}
void* Create_MirrorPad(std::string id, Json::Value pInputItem) {
	MirrorPad* pMirrorPad = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* paddings = nullptr;
	std::string mode;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("MirrorPad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("MirrorPad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "paddings")
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
							paddings = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						paddings = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("MirrorPad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "mode")
		{
			if (strPinInterface == "StringPiece")
			{
				mode = strPinInitial;
			}
			else
			{
				std::string msg = string_format("MirrorPad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("MirrorPad pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput && paddings)
	{
		pMirrorPad = new MirrorPad(*pScope, *pInput, *paddings, mode);
		ObjectInfo* pObj = AddObjectMap(pMirrorPad, id, SYMBOL_MIRRORPAD, "MirrorPad", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pMirrorPad->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("MirrorPad(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pMirrorPad;
}
void* Create_OneHot(std::string id, Json::Value pInputItem) {
	OneHot* pOneHot = nullptr;
	Scope* pScope = nullptr;
	Output* indices = nullptr;
	Output* depth = nullptr;
	Output* on_value = nullptr;
	Output* off_value = nullptr;
	OneHot::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("OneHot - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						indices = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("OneHot - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "depth")
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
							depth = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						depth = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("OneHot - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "on_value")
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
							on_value = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						on_value = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("OneHot - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "off_value")
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
							off_value = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						off_value = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("OneHot - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "OneHot::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("axis_") != "")
				{
					attrs.axis_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("axis_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("OneHot pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && indices && depth && on_value && off_value)
	{
		pOneHot = new OneHot(*pScope, *indices, *depth, *on_value, *off_value, attrs);
		ObjectInfo* pObj = AddObjectMap(pOneHot, id, SYMBOL_ONEHOT, "OneHot", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pOneHot->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("OneHot(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pOneHot;
}
void* Create_OnesLike(std::string id, Json::Value pInputItem) {
	OnesLike* pOnesLike = nullptr;
	Scope* pScope = nullptr;
	Output* x = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("OnesLike - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "x")
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
							x = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						x = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("OnesLike - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("OnesLike pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && x)
	{
		pOnesLike = new OnesLike(*pScope, *x);
		ObjectInfo* pObj = AddObjectMap(pOnesLike, id, SYMBOL_ONESLIKE, "OnesLike", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pOnesLike->y, OUTPUT_TYPE_OUTPUT, "y");
	}
	else
	{
		std::string msg = string_format("OnesLike(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pOnesLike;
}
void* Create_Stack(std::string id, Json::Value pInputItem) {
	Stack* pStack = nullptr;
	Scope* pScope = nullptr;
	OutputList* values = nullptr;
	Stack::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Stack - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "values")
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
							values = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("Stack - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Stack::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("axis_") != "")
				{
					attrs.axis_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("axis_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("Stack pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && values)
	{
		pStack = new Stack(*pScope, *values, attrs);
		ObjectInfo* pObj = AddObjectMap(pStack, id, SYMBOL_STACK, "Stack", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pStack->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("Stack(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pStack;
}
void* Create_Pad(std::string id, Json::Value pInputItem) {
	Pad* pPad = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* paddings = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Pad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Pad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "paddings")
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
							paddings = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						paddings = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Pad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("Pad pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput && paddings)
	{
		pPad = new Pad(*pScope, *pInput, *paddings);
		ObjectInfo* pObj = AddObjectMap(pPad, id, SYMBOL_PAD, "Pad", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pPad->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("Pad(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pPad;
}
void* Create_PadV2(std::string id, Json::Value pInputItem) {
	PadV2* pPadV2 = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* paddings = nullptr;
	Output* constant_values = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("PadV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("PadV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "paddings")
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
							paddings = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						paddings = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("PadV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "constant_values")
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
							constant_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						constant_values = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("PadV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("PadV2 pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput && paddings && constant_values)
	{
		pPadV2 = new PadV2(*pScope, *pInput, *paddings, *constant_values);
		ObjectInfo* pObj = AddObjectMap(pPadV2, id, SYMBOL_PADV2, "PadV2", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pPadV2->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("PadV2(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pPadV2;
}
void* Create_ParallelConcat(std::string id, Json::Value pInputItem) {
	ParallelConcat* pParallelConcat = nullptr;
	Scope* pScope = nullptr;
	OutputList* values = nullptr;
	PartialTensorShape shape;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("ParallelConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "values")
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
							values = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("ParallelConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "shape")
		{
			if (strPinInterface == "PartialTensorShape")
			{
				shape = GetPartialShapeFromInitial(strPinInitial);
			}
			else
			{
				std::string msg = string_format("ParallelConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("ParallelConcat pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && values)
	{
		pParallelConcat = new ParallelConcat(*pScope, *values, shape);
		ObjectInfo* pObj = AddObjectMap(pParallelConcat, id, SYMBOL_PARALLELCONCAT, "ParallelConcat", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pParallelConcat->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("ParallelConcat(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pParallelConcat;
}
void* Create_Placeholder(std::string id, Json::Value pInputItem) {
	Placeholder* pPlaceholder = nullptr;
	Scope* pScope = nullptr;
	DataType dtype;
	Placeholder::Attrs attrs;


	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Placeholder - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				dtype = GetDatatypeFromInitial(strPinInitial);
				if (dtype == DT_INVALID)
				{
					std::string msg = string_format("Placeholder - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
				}
			}
			else
			{
				std::string msg = string_format("Placeholder - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Placeholder::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("shape_") != "")
				{
					attrs.shape_ = attrParser.ConvStrToPartialTensorShape(attrParser.GetAttribute("shape_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("Placeholder pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope)
	{
		pPlaceholder = new Placeholder(*pScope, dtype, attrs);
		ObjectInfo* pObj = AddObjectMap(pPlaceholder, id, SYMBOL_PLACEHOLDER, "Placeholder", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pPlaceholder->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("Placeholder(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pPlaceholder;
}
void* Create_PlaceholderWithDefault(std::string id, Json::Value pInputItem) {
	PlaceholderWithDefault* pPlaceholderWithDefault = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	PartialTensorShape shape;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("PlaceholderWithDefault - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("PlaceholderWithDefault - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "shape")
		{
			if (strPinInterface == "PartialTensorShape")
			{
				shape = GetPartialShapeFromInitial(strPinInitial);
			}
			else
			{
				std::string msg = string_format("PlaceholderWithDefault - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("PlaceholderWithDefault pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput)
	{
		pPlaceholderWithDefault = new PlaceholderWithDefault(*pScope, *pInput, shape);
		ObjectInfo* pObj = AddObjectMap(pPlaceholderWithDefault, id, SYMBOL_PLACEHOLDERWITHDEFAULT, "PlaceholderWithDefault", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pPlaceholderWithDefault->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("PlaceholderWithDefault(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pPlaceholderWithDefault;
}
void* Create_PreventGradient(std::string id, Json::Value pInputItem) {
	PreventGradient* pPreventGradient = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	PreventGradient::Attrs attrs;
	std::string message;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("PreventGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("PreventGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Stack::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("message_") != "")
				{
					message = attrParser.GetAttribute("message_");
					attrs.message_ = message;
				}
			}
		}
		else
		{
			std::string msg = string_format("PreventGradient pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput)
	{
		pPreventGradient = new PreventGradient(*pScope, *pInput, attrs);
		ObjectInfo* pObj = AddObjectMap(pPreventGradient, id, SYMBOL_PREVENTGRADIENT, "PreventGradient", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pPreventGradient->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("PreventGradient(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pPreventGradient;
}
void* Create_QuantizeAndDequantizeV2(std::string id, Json::Value pInputItem) {
	QuantizeAndDequantizeV2* pQuantizeAndDequantizeV2 = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* input_min = nullptr;
	Output* input_max = nullptr;
	QuantizeAndDequantizeV2::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("QuantizeAndDequantizeV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("QuantizeAndDequantizeV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "input_min")
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
							input_min = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						input_min = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("QuantizeAndDequantizeV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "input_max")
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
							input_max = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						input_max = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("QuantizeAndDequantizeV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QuantizeAndDequantizeV2::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("signed_input_") != "")
				{
					attrs.signed_input_ = attrParser.ConvStrToBool(attrParser.GetAttribute("signed_input_"));
				}
				if (attrParser.GetAttribute("num_bits_") != "")
				{
					attrs.num_bits_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("num_bits_"));
				}
				if (attrParser.GetAttribute("range_given_") != "")
				{
					attrs.range_given_ = attrParser.ConvStrToBool(attrParser.GetAttribute("range_given_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("QuantizeAndDequantizeV2 pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput && input_min && input_max)
	{
		pQuantizeAndDequantizeV2 = new QuantizeAndDequantizeV2(*pScope, *pInput, *input_min, *input_max, attrs);
		ObjectInfo* pObj = AddObjectMap(pQuantizeAndDequantizeV2, id, SYMBOL_QUANTIZEANDDEQUANTIZEV2, "QuantizeAndDequantizeV2", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pQuantizeAndDequantizeV2->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("QuantizeAndDequantizeV2(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pQuantizeAndDequantizeV2;
}
void* Create_QuantizeV2(std::string id, Json::Value pInputItem) {
	QuantizeV2* pQuantizeV2 = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* min_range = nullptr; 
	Output* max_range = nullptr;
	DataType T;
	QuantizeV2::Attrs attrs;
	std::string mode;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("QuantizeV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("QuantizeV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "min_range")
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
							min_range = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						min_range = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("QuantizeV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "max_range")
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
							max_range = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						max_range = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("QuantizeV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QuantizeV2::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("mode_") != "")
				{
					attrs.mode_ = attrParser.GetAttribute("mode_");
				}
			}
		}
		else if (strPinName == "T")
		{
			if (strPinInterface == "DataType")
			{
				T = GetDatatypeFromInitial(strPinInitial);
				if (T == DT_INVALID)
				{
					std::string msg = string_format("QuantizeV2 - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
				}
			}
			else
			{
				std::string msg = string_format("QuantizeV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("QuantizeV2 pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput && min_range && max_range)
	{
		pQuantizeV2 = new QuantizeV2(*pScope, *pInput, *min_range, *max_range, T, attrs);
		ObjectInfo* pObj = AddObjectMap(pQuantizeV2, id, SYMBOL_QUANTIZEV2, "QuantizeV2", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pQuantizeV2->output, OUTPUT_TYPE_OUTPUT, "output");
			AddOutputInfo(pObj, &pQuantizeV2->output_min, OUTPUT_TYPE_OUTPUT, "output_min");
			AddOutputInfo(pObj, &pQuantizeV2->output_max, OUTPUT_TYPE_OUTPUT, "output_max");
		}
	}
	else
	{
		std::string msg = string_format("QuantizeV2(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pQuantizeV2;
}
void* Create_QuantizedConcat(std::string id, Json::Value pInputItem) {
	QuantizedConcat* pQuantizedConcat = nullptr;
	Scope* pScope = nullptr;
	Output* concat_dim = nullptr;
	OutputList* values = nullptr;
	OutputList* input_mins = nullptr;
	OutputList* input_maxes = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("QuantizedConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "concat_dim")
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
							concat_dim = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						concat_dim = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("QuantizedConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "values")
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
							values = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("QuantizedConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "input_mins")
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
							input_mins = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("QuantizedConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "input_maxes")
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
							input_maxes = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("QuantizedConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("QuantizedConcat pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && concat_dim && values && input_mins && input_maxes)
	{
		pQuantizedConcat = new QuantizedConcat(*pScope, *concat_dim, *values, *input_mins, *input_maxes);
		ObjectInfo* pObj = AddObjectMap(pQuantizedConcat, id, SYMBOL_QUANTIZEDCONCAT, "QuantizedConcat", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pQuantizedConcat->output, OUTPUT_TYPE_OUTPUT, "output");
			AddOutputInfo(pObj, &pQuantizedConcat->output_min, OUTPUT_TYPE_OUTPUT, "output_min");
			AddOutputInfo(pObj, &pQuantizedConcat->output_max, OUTPUT_TYPE_OUTPUT, "output_max");
		}
	}
	else
	{
		std::string msg = string_format("QuantizedConcat(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pQuantizedConcat;
}
void* Create_QuantizedInstanceNorm(std::string id, Json::Value pInputItem) {
	QuantizedInstanceNorm* pQuantizedInstanceNorm = nullptr;
	Scope* pScope = nullptr;
	Output* x = nullptr;
	Output* x_min = nullptr;
	Output* x_max = nullptr;
	QuantizedInstanceNorm::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("QuantizedInstanceNorm - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "x")
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
							x = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						x = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("QuantizedInstanceNorm - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "x_min")
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
							x_min = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						x_min = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("QuantizedInstanceNorm - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "x_max")
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
							x_max = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						x_max = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("QuantizedInstanceNorm - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QuantizedInstanceNorm::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("output_range_given_") != "")
				{
					attrs.output_range_given_ = attrParser.ConvStrToBool(attrParser.GetAttribute("output_range_given_"));
				}
				if (attrParser.GetAttribute("given_y_min_") != "")
				{
					attrs.given_y_min_ = attrParser.ConvStrToFloat(attrParser.GetAttribute("given_y_min_"));
				}
				if (attrParser.GetAttribute("given_y_max_") != "")
				{
					attrs.given_y_max_ = attrParser.ConvStrToFloat(attrParser.GetAttribute("given_y_max_"));
				}
				if (attrParser.GetAttribute("variance_epsilon_") != "")
				{
					attrs.variance_epsilon_ = attrParser.ConvStrToFloat(attrParser.GetAttribute("variance_epsilon_"));
				}
				if (attrParser.GetAttribute("min_separation_") != "")
				{
					attrs.min_separation_ = attrParser.ConvStrToFloat(attrParser.GetAttribute("min_separation_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("QuantizedInstanceNorm pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && x && x_min && x_max)
	{
		pQuantizedInstanceNorm = new QuantizedInstanceNorm(*pScope, *x, *x_min, *x_max, attrs);
		ObjectInfo* pObj = AddObjectMap(pQuantizedInstanceNorm, id, SYMBOL_QUANTIZEDINSTANCENORM, "QuantizedInstanceNorm", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pQuantizedInstanceNorm->y, OUTPUT_TYPE_OUTPUT, "y");
			AddOutputInfo(pObj, &pQuantizedInstanceNorm->y_min, OUTPUT_TYPE_OUTPUT, "y_min");
			AddOutputInfo(pObj, &pQuantizedInstanceNorm->y_max, OUTPUT_TYPE_OUTPUT, "y_max");
		}
	}
	else
	{
		std::string msg = string_format("QuantizedInstanceNorm(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pQuantizedInstanceNorm;
}
void* Create_QuantizedReshape(std::string id, Json::Value pInputItem) {
	QuantizedReshape* pQuantizedReshape = nullptr;
	Scope* pScope = nullptr;
	Output* pTensor = nullptr;
	Output* shape = nullptr;
	Output* input_min = nullptr;
	Output* input_max = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("QuantizedReshape - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "tensor")
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
							pTensor = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pTensor = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("QuantizedReshape - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						shape = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("QuantizedReshape - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "input_min")
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
							input_min = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						input_min = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("QuantizedReshape - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "input_max")
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
							input_max = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						input_max = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("QuantizedReshape - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("QuantizedReshape pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pTensor && shape && input_min && input_max)
	{
		pQuantizedReshape = new QuantizedReshape(*pScope, *pTensor, *shape, *input_min, *input_max);
		ObjectInfo* pObj = AddObjectMap(pQuantizedReshape, id, SYMBOL_QUANTIZEDRESHAPE, "QuantizedReshape", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pQuantizedReshape->output, OUTPUT_TYPE_OUTPUT, "output");
			AddOutputInfo(pObj, &pQuantizedReshape->output_min, OUTPUT_TYPE_OUTPUT, "output_min");
			AddOutputInfo(pObj, &pQuantizedReshape->output_max, OUTPUT_TYPE_OUTPUT, "output_max");
		}
	}
	else
	{
		std::string msg = string_format("QuantizedReshape(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pQuantizedReshape;
}
void* Create_Rank(std::string id, Json::Value pInputItem) {
	Rank* pRank = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Rank - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Rank - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("Rank pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput)
	{
		pRank = new Rank(*pScope, *pInput);
		ObjectInfo* pObj = AddObjectMap(pRank, id, SYMBOL_RANK, "Rank", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pRank->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("Rank(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pRank;
}
void* Create_Reshape(std::string id, Json::Value pInputItem) {
	Reshape* pReshape = nullptr;
	Scope* pScope = nullptr;
	Output* pTensor = nullptr;
	Output* shape = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Reshape - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "tensor")
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
							pTensor = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pTensor = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Reshape - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						shape = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Reshape - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("Reshape pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pTensor && shape)
	{
		pReshape = new Reshape(*pScope, *pTensor, *shape);
		ObjectInfo* pObj = AddObjectMap(pReshape, id, SYMBOL_RESHAPE, "Reshape", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pReshape->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("Reshape(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pReshape;
}
void* Create_ResourceStridedSliceAssign(std::string id, Json::Value pInputItem) {
	ResourceStridedSliceAssign* pResourceStridedSliceAssign = nullptr;
	Scope* pScope = nullptr;
	Output* ref = nullptr;
	Output* begin = nullptr;
	Output* end = nullptr;
	Output* strides = nullptr;
	Output* value = nullptr;
	ResourceStridedSliceAssign::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("ResourceStridedSliceAssign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "ref")
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
							ref = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("ResourceStridedSliceAssign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "begin")
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
							begin = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						begin = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("ResourceStridedSliceAssign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "end")
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
							end = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						end = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("ResourceStridedSliceAssign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "strides")
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
							strides = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						strides = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("ResourceStridedSliceAssign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "value")
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
							value = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						value = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("ResourceStridedSliceAssign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceStridedSliceAssign::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("begin_mask_") != "")
				{
					attrs.begin_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("begin_mask_"));
				}
				if (attrParser.GetAttribute("end_mask_") != "")
				{
					attrs.end_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("end_mask_"));
				}
				if (attrParser.GetAttribute("ellipsis_mask_") != "")
				{
					attrs.ellipsis_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("ellipsis_mask_"));
				}
				if (attrParser.GetAttribute("new_axis_mask_") != "")
				{
					attrs.new_axis_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("new_axis_mask_"));
				}
				if (attrParser.GetAttribute("shrink_axis_mask_") != "")
				{
					attrs.shrink_axis_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("shrink_axis_mask_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("ResourceStridedSliceAssign pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && ref && begin && end && strides && value)
	{
		pResourceStridedSliceAssign = new ResourceStridedSliceAssign(*pScope, *ref, *begin, *end, *strides, *value, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceStridedSliceAssign, id, SYMBOL_RESOURCESTRIDEDSLICEASSIGN, "ResourceStridedSliceAssign", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceStridedSliceAssign->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("ResourceStridedSliceAssign(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pResourceStridedSliceAssign;
}
void* Create_ReverseSequence(std::string id, Json::Value pInputItem) {
	ReverseSequence* pReverseSequence = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* seq_lengths = nullptr;
	int64 seq_dim = 0;
	ReverseSequence::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("ResourceStridedSliceAssign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("ResourceStridedSliceAssign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "seq_lengths")
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
							seq_lengths = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						seq_lengths = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("ReverseSequence - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "seq_dim")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				if (strPinInitial == "")
				{
					seq_dim = 0;
				}
				else
				{
					seq_dim = stoll(strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("BatchToSpace - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ReverseSequence::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("batch_dim_") != "")
				{
					attrs.batch_dim_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("batch_dim_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("ReverseSequence pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput && seq_lengths)
	{
		pReverseSequence = new ReverseSequence(*pScope, *pInput, *seq_lengths, seq_dim, attrs);
		ObjectInfo* pObj = AddObjectMap(pReverseSequence, id, SYMBOL_REVERSESEQUENCE, "ReverseSequence", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pReverseSequence->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("ReverseSequence(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pReverseSequence;
}
void* Create_Reverse(std::string id, Json::Value pInputItem) {
	Reverse* pReverse = nullptr;
	Scope* pScope = nullptr;
	Output* pTensor = nullptr;
	Output* axis = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Reverse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "tensor")
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
							pTensor = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pTensor = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Reverse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "axis")
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
							axis = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						axis = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Reverse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("Reverse pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pTensor && axis)
	{
		pReverse = new Reverse(*pScope, *pTensor, *axis);
		ObjectInfo* pObj = AddObjectMap(pReverse, id, SYMBOL_REVERSE, "Reverse", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pReverse->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("Reverse(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pReverse;
}
void* Create_ScatterNd(std::string id, Json::Value pInputItem) {
	ScatterNd* pScatterNd = nullptr;
	Scope* pScope = nullptr;
	Output* indices = nullptr;
	Output* updates = nullptr;
	Output* shape = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("ScatterNd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						indices = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("ScatterNd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "updates")
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
							updates = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						updates = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("ScatterNd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						shape = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("ScatterNd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("ScatterNd pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && indices && updates && shape)
	{
		pScatterNd = new ScatterNd(*pScope, *indices, *updates, *shape);
		ObjectInfo* pObj = AddObjectMap(pScatterNd, id, SYMBOL_SCATTERND, "ScatterNd", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pScatterNd->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("ScatterNd(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pScatterNd;
}
void* Create_ScatterNdNonAliasingAdd(std::string id, Json::Value pInputItem) {
	ScatterNdNonAliasingAdd* pScatterNdNonAliasingAdd = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* indices = nullptr;
	Output* updates = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("ScatterNdNonAliasingAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pinput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("ScatterNdNonAliasingAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						indices = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("ScatterNdNonAliasingAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "updates")
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
							updates = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						updates = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("ScatterNdNonAliasingAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("ScatterNdNonAliasingAdd pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pinput && indices && updates)
	{
		pScatterNdNonAliasingAdd = new ScatterNdNonAliasingAdd(*pScope, *pinput, *indices, *updates);
		ObjectInfo* pObj = AddObjectMap(pScatterNdNonAliasingAdd, id, SYMBOL_SCATTERNDNONALIASINGADD, "ScatterNdNonAliasingAdd", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pScatterNdNonAliasingAdd->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("ScatterNdNonAliasingAdd(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pScatterNdNonAliasingAdd;
}
void* Create_Shape(std::string id, Json::Value pInputItem) {
	Shape* pShape = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Shape::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Shape - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Shape - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Shape::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("out_type_") != "")
				{
					attrs.out_type_ = attrParser.ConvStrToDataType(attrParser.GetAttribute("out_type_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("Shape pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput)
	{
		pShape = new Shape(*pScope, *pInput, attrs);
		ObjectInfo* pObj = AddObjectMap(pShape, id, SYMBOL_SHAPE, "Shape", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pShape->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("Shape(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pShape;
}
void* Create_ShapeN(std::string id, Json::Value pInputItem) {
	ShapeN* pShapeN = nullptr;
	Scope* pScope = nullptr;
	OutputList* pInput = nullptr;
	ShapeN::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("ShapeN - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "input")
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
							pInput = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("ShapeN - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ShapeN::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("out_type_") != "")
				{
					attrs.out_type_ = attrParser.ConvStrToDataType(attrParser.GetAttribute("out_type_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("ShapeN pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput)
	{
		pShapeN = new ShapeN(*pScope, *pInput, attrs);
		ObjectInfo* pObj = AddObjectMap(pShapeN, id, SYMBOL_SHAPEN, "ShapeN", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pShapeN->output, OUTPUT_TYPE_OUTPUTLIST, "output");
	}
	else
	{
		std::string msg = string_format("ShapeN(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pShapeN;
}
void* Create_Size(std::string id, Json::Value pInputItem) {
	Size* pSize = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Size::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Size - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Size - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Shape::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("out_type_") != "")
				{
					attrs.out_type_ = attrParser.ConvStrToDataType(attrParser.GetAttribute("out_type_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("Size pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput)
	{
		pSize = new Size(*pScope, *pInput, attrs);
		ObjectInfo* pObj = AddObjectMap(pSize, id, SYMBOL_SIZE, "Size", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSize->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("Size(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pSize;
}
void* Create_Slice(std::string id, Json::Value pInputItem) {
	Slice* pSlice = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* begin = nullptr;
	Output* size = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Slice - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Slice - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "begin")
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
							begin = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						begin = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Slice - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "size")
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
							size = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						size = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Slice - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("Slice pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput && begin && size)
	{
		pSlice = new Slice(*pScope, *pInput, *begin, *size);
		ObjectInfo* pObj = AddObjectMap(pSlice, id, SYMBOL_SLICE, "Slice", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSlice->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("Slice(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pSlice;
}
void* Create_SpaceToBatch(std::string id, Json::Value pInputItem) {
	SpaceToBatch* pSpaceToBatch = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* paddings = nullptr;
	int64 block_size;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("SpaceToBatch - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("SpaceToBatch - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "paddings")
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
							paddings = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						paddings = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("SpaceToBatch - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "block_size")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				if (strPinInitial == "")
				{
					block_size = 0;
				}
				else
				{
					block_size = stoll(strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("SpaceToBatch - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("SpaceToBatch pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput && paddings)
	{
		pSpaceToBatch = new SpaceToBatch(*pScope, *pInput, *paddings, block_size);
		ObjectInfo* pObj = AddObjectMap(pSpaceToBatch, id, SYMBOL_SPACETOBATCH, "SpaceToBatch", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSpaceToBatch->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("SpaceToBatch(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pSpaceToBatch;
}
void* Create_SpaceToBatchND(std::string id, Json::Value pInputItem) {
	SpaceToBatchND* pSpaceToBatchND = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* block_shape = nullptr;
	Output* paddings = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("SpaceToBatchND - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pinput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("SpaceToBatchND - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "block_shape")
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
							block_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						block_shape = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("SpaceToBatchND - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "paddings")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
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
							paddings = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						paddings = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("SpaceToBatchND - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("SpaceToBatchND pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pinput && block_shape && paddings)
	{
		pSpaceToBatchND = new SpaceToBatchND(*pScope, *pinput, *block_shape, *paddings);
		ObjectInfo* pObj = AddObjectMap(pSpaceToBatchND, id, SYMBOL_SPACETOBATCHND, "SpaceToBatchND", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSpaceToBatchND->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("SpaceToBatchND(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pSpaceToBatchND;
}
void* Create_SpaceToDepth(std::string id, Json::Value pInputItem) {
	SpaceToDepth* pSpaceToDepth = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	int64 block_size;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("SpaceToDepth - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("SpaceToDepth - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "block_size")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				if (strPinInitial == "")
				{
					block_size = 0;
				}
				else
				{
					block_size = stoll(strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("SpaceToDepth - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("SpaceToDepth pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput)
	{
		pSpaceToDepth = new SpaceToDepth(*pScope, *pInput, block_size);
		ObjectInfo* pObj = AddObjectMap(pSpaceToDepth, id, SYMBOL_SPACETODEPTH, "SpaceToDepth", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSpaceToDepth->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("SpaceToDepth(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pSpaceToDepth;
}
void* Create_Split(std::string id, Json::Value pInputItem) {
	Split* pSplit = nullptr;
	Scope* pScope = nullptr;
	Output* axis = nullptr;
	Output* value = nullptr;
	int64 num_split;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Split - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "axis")
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
							axis = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						axis = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Split - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "value")
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
							value = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						value = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Split - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "num_split")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				if (strPinInitial == "")
				{
					num_split = 0;
				}
				else
				{
					num_split = stoll(strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Split - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("Split pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && axis && value)
	{
		pSplit = new Split(*pScope, *axis, *value, num_split);
		ObjectInfo* pObj = AddObjectMap(pSplit, id, SYMBOL_SPLIT, "Split", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSplit->output, OUTPUT_TYPE_OUTPUTLIST, "output");
	}
	else
	{
		std::string msg = string_format("Split(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pSplit;
}
void* Create_SplitV(std::string id, Json::Value pInputItem) {
	SplitV* pSplitV = nullptr;
	Scope* pScope = nullptr;
	Output* value = nullptr;
	Output* size_splits = nullptr;
	Output* axis = nullptr;
	int64 num_split;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("SplitV - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "value")
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
							value = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						value = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("SplitV - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "size_splits")
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
							size_splits = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						size_splits = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("SplitV - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "axis")
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
							axis = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						axis = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("SplitV - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "num_split")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				if (strPinInitial == "")
				{
					num_split = 0;
				}
				else
				{
					num_split = stoll(strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("SplitV - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("SplitV pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && value && size_splits && axis)
	{
		pSplitV = new SplitV(*pScope, *value, *size_splits, *axis, num_split);
		ObjectInfo* pObj = AddObjectMap(pSplitV, id, SYMBOL_SPLITV, "SplitV", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSplitV->output, OUTPUT_TYPE_OUTPUTLIST, "output");
	}
	else
	{
		std::string msg = string_format("SplitV(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pSplitV;
}
void* Create_Squeeze(std::string id, Json::Value pInputItem) {
	Squeeze* pSqueeze = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Squeeze::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Squeeze - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Squeeze - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Squeeze::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("squeeze_dims_") != "")
				{
					std::vector<int> v_int;
					attrParser.ConvStrToArraySliceInt(attrParser.GetAttribute("squeeze_dims_"), v_int);
					gtl::ArraySlice<int> arrayInt(v_int);
					// r1.5 not support // attrs.squeeze_dims_ = arrayInt;
				}
			}
		}
		else
		{
			std::string msg = string_format("Squeeze pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput)
	{
		pSqueeze = new Squeeze(*pScope, *pInput, attrs);
		ObjectInfo* pObj = AddObjectMap(pSqueeze, id, SYMBOL_SQUEEZE, "Squeeze", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSqueeze->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("Squeeze(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pSqueeze;
}
void* Create_StopGradient(std::string id, Json::Value pInputItem) {
	StopGradient* pStopGradient = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("StopGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("StopGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("StopGradient pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput)
	{
		pStopGradient = new StopGradient(*pScope, *pInput);
		ObjectInfo* pObj = AddObjectMap(pStopGradient, id, SYMBOL_STOPGRADIENT, "StopGradient", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pStopGradient->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("StopGradient(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pStopGradient;
}
void* Create_StridedSlice(std::string id, Json::Value pInputItem) {
	StridedSlice* pStridedSlice = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* begin = nullptr;
	Output* end = nullptr;
	Output* strides = nullptr;
	StridedSlice::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("StridedSlice - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("StridedSlice - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "begin")
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
							begin = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						begin = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("StridedSlice - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "end")
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
							end = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						end = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("StridedSlice - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "strides")
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
							strides = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						strides = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("StridedSlice - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "StridedSlice::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("begin_mask_") != "")
				{
					attrs.begin_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("begin_mask_"));
				}
				if (attrParser.GetAttribute("end_mask_") != "")
				{
					attrs.end_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("end_mask_"));
				}
				if (attrParser.GetAttribute("ellipsis_mask_") != "")
				{
					attrs.ellipsis_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("ellipsis_mask_"));
				}
				if (attrParser.GetAttribute("new_axis_mask_") != "")
				{
					attrs.new_axis_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("new_axis_mask_"));
				}
				if (attrParser.GetAttribute("shrink_axis_mask_") != "")
				{
					attrs.shrink_axis_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("shrink_axis_mask_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("StridedSlice pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput && begin && end && strides)
	{
		pStridedSlice = new StridedSlice(*pScope, *pInput, *begin, *end, *strides, attrs);
		ObjectInfo* pObj = AddObjectMap(pStridedSlice, id, SYMBOL_STRIDEDSLICE, "StridedSlice", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pStridedSlice->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("StridedSlice(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pStridedSlice;
}
void* Create_StridedSliceAssign(std::string id, Json::Value pInputItem) {
	StridedSliceAssign* pStridedSliceAssign = nullptr;
	Scope* pScope = nullptr;
	Output* ref = nullptr;
	Output* begin = nullptr;
	Output* end = nullptr;
	Output* strides = nullptr;
	Output* value = nullptr;
	StridedSliceAssign::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("StridedSliceAssign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "ref")
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
							ref = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("StridedSliceAssign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "begin")
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
							begin = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						begin = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("StridedSliceAssign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "end")
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
							end = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						end = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("StridedSliceAssign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "strides")
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
							strides = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						strides = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("StridedSliceAssign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "value")
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
							value = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						value = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("StridedSliceAssign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "StridedSliceAssign::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("begin_mask_") != "")
				{
					attrs.begin_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("begin_mask_"));
				}
				if (attrParser.GetAttribute("end_mask_") != "")
				{
					attrs.end_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("end_mask_"));
				}
				if (attrParser.GetAttribute("ellipsis_mask_") != "")
				{
					attrs.ellipsis_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("ellipsis_mask_"));
				}
				if (attrParser.GetAttribute("new_axis_mask_") != "")
				{
					attrs.new_axis_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("new_axis_mask_"));
				}
				if (attrParser.GetAttribute("shrink_axis_mask_") != "")
				{
					attrs.shrink_axis_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("shrink_axis_mask_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("StridedSliceAssign pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && ref && begin && end && strides && value)
	{
		pStridedSliceAssign = new StridedSliceAssign(*pScope, *ref, *begin, *end, *strides, *value, attrs);
		ObjectInfo* pObj = AddObjectMap(pStridedSliceAssign, id, SYMBOL_STRIDEDSLICEASSIGN, "StridedSliceAssign", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pStridedSliceAssign->output_ref, OUTPUT_TYPE_OUTPUT, "output_ref");
	}
	else
	{
		std::string msg = string_format("StridedSliceAssign(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pStridedSliceAssign;
}
void* Create_StridedSliceGrad(std::string id, Json::Value pInputItem) {
	StridedSliceGrad* pStridedSliceGrad = nullptr;
	Scope* pScope = nullptr;
	Output* shape = nullptr;
	Output* begin = nullptr;
	Output* end = nullptr;
	Output* strides = nullptr;
	Output* dy = nullptr;
	StridedSliceGrad::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("StridedSliceGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						shape = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("StridedSliceGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "begin")
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
							begin = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						begin = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("StridedSliceGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "end")
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
							end = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						end = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("StridedSliceGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "strides")
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
							strides = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						strides = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("StridedSliceGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dy")
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
							dy = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						dy = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("StridedSliceGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "StridedSliceGrad::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("begin_mask_") != "")
				{
					attrs.begin_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("begin_mask_"));
				}
				if (attrParser.GetAttribute("end_mask_") != "")
				{
					attrs.end_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("end_mask_"));
				}
				if (attrParser.GetAttribute("ellipsis_mask_") != "")
				{
					attrs.ellipsis_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("ellipsis_mask_"));
				}
				if (attrParser.GetAttribute("new_axis_mask_") != "")
				{
					attrs.new_axis_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("new_axis_mask_"));
				}
				if (attrParser.GetAttribute("shrink_axis_mask_") != "")
				{
					attrs.shrink_axis_mask_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("shrink_axis_mask_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("StridedSliceGrad pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && shape && begin && end && strides && dy)
	{
		pStridedSliceGrad = new StridedSliceGrad(*pScope, *shape, *begin, *end, *strides, *dy, attrs);
		ObjectInfo* pObj = AddObjectMap(pStridedSliceGrad, id, SYMBOL_STRIDEDSLICEGRAD, "StridedSliceGrad", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pStridedSliceGrad->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("StridedSliceGrad(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pStridedSliceGrad;
}
void* Create_Tile(std::string id, Json::Value pInputItem) {
	Tile* pTile = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* multiples = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Tile - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pInput = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Tile - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "multiples")
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
							multiples = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						multiples = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Tile - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("Tile pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput && multiples)
	{
		pTile = new Tile(*pScope, *pInput, *multiples);
		ObjectInfo* pObj = AddObjectMap(pTile, id, SYMBOL_TILE, "Tile", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pTile->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("Tile(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pTile;
}
void* Create_Transpose(std::string id, Json::Value pInputItem) {
	Transpose* pTranspose = nullptr;
	Scope* pScope = nullptr;
	Output* x = nullptr;
	Output* perm = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Transpose - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "x")
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
							x = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						x = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Transpose - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "perm")
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
							perm = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						perm = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Transpose - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("Transpose pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && x && perm)
	{
		pTranspose = new Transpose(*pScope, *x, *perm);
		ObjectInfo* pObj = AddObjectMap(pTranspose, id, SYMBOL_TRANSPOSE, "Transpose", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pTranspose->y, OUTPUT_TYPE_OUTPUT, "y");
	}
	else
	{
		std::string msg = string_format("Transpose(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pTranspose;
}
void* Create_Unique(std::string id, Json::Value pInputItem) {
	Unique* pUnique = nullptr;
	Scope* pScope = nullptr;
	Output* x = nullptr;
	Unique::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Unique - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "x")
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
							x = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						x = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Unique - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Unique::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("out_idx_") != "")
				{
					attrs.out_idx_ = attrParser.ConvStrToDataType(attrParser.GetAttribute("out_idx_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("Unique pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && x)
	{
		pUnique = new Unique(*pScope, *x, attrs);
		ObjectInfo* pObj = AddObjectMap(pUnique, id, SYMBOL_UNIQUE, "Unique", pInputItem);
		if (pObj) {
			AddOutputInfo(pObj, &pUnique->y, OUTPUT_TYPE_OUTPUT, "y");
			AddOutputInfo(pObj, &pUnique->idx, OUTPUT_TYPE_OUTPUT, "idx");
		}
	}
	else
	{
		std::string msg = string_format("Unique(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pUnique;
}
void* Create_UniqueWithCounts(std::string id, Json::Value pInputItem) {
	UniqueWithCounts* pUniqueWithCounts = nullptr;
	Scope* pScope = nullptr;
	Output* x = nullptr;
	UniqueWithCounts::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("UniqueWithCounts - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "x")
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
							x = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						x = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("UniqueWithCounts - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Unique::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("out_idx_") != "")
				{
					attrs.out_idx_ = attrParser.ConvStrToDataType(attrParser.GetAttribute("out_idx_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("UniqueWithCounts pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && x)
	{
		pUniqueWithCounts = new UniqueWithCounts(*pScope, *x, attrs);
		ObjectInfo* pObj = AddObjectMap(pUniqueWithCounts, id, SYMBOL_UNIQUEWITHCOUNTS, "UniqueWithCounts", pInputItem);
		if (pObj) {
			AddOutputInfo(pObj, &pUniqueWithCounts->y, OUTPUT_TYPE_OUTPUT, "y");
			AddOutputInfo(pObj, &pUniqueWithCounts->idx, OUTPUT_TYPE_OUTPUT, "idx");
			AddOutputInfo(pObj, &pUniqueWithCounts->count, OUTPUT_TYPE_OUTPUT, "count");
		}
	}
	else
	{
		std::string msg = string_format("UniqueWithCounts(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pUniqueWithCounts;
}
void* Create_Unstack(std::string id, Json::Value pInputItem) {
	Unstack* pUnstack = nullptr;
	Scope* pScope = nullptr;
	Output* value = nullptr;
	int64 num;
	Unstack::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Unstack - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "value")
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
							value = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						value = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Unstack - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "num")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Int64")
			{
				if (strPinInitial == "")
				{
					num = 0;
				}
				else
				{
					num = stoll(strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("BatchToSpace - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Unstack::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("axis_") != "")
				{
					attrs.axis_ = attrParser.ConvStrToInt64(attrParser.GetAttribute("axis_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("Unstack pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && value)
	{
		pUnstack = new Unstack(*pScope, *value, num, attrs);
		ObjectInfo* pObj = AddObjectMap(pUnstack, id, SYMBOL_UNSTACK, "Unstack", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pUnstack->output, OUTPUT_TYPE_OUTPUTLIST, "output");
	}
	else
	{
		std::string msg = string_format("Unstack(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pUnstack;
}
void* Create_Where(std::string id, Json::Value pInputItem) {
	Where* pWhere = nullptr;
	Scope* pScope = nullptr;
	Output* condition = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Where - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "condition")
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
							condition = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						condition = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Where - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("Where pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && condition)
	{
		pWhere = new Where(*pScope, *condition);
		ObjectInfo* pObj = AddObjectMap(pWhere, id, SYMBOL_WHERE, "Where", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pWhere->index, OUTPUT_TYPE_OUTPUT, "index");
	}
	else
	{
		std::string msg = string_format("Where(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pWhere;
}
void* Create_ZerosLike(std::string id, Json::Value pInputItem) {
	ZerosLike* pZerosLike = nullptr;
	Scope* pScope = nullptr;
	Output* x = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strPinDataType = ItemValue.get("pin-datatype", "").asString();
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("ZerosLike - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "x")
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
							x = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						x = (Output*)Create_StrToOutput(*m_pScope, strPinDataType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("ZerosLike - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("ZerosLike pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && x)
	{
		pZerosLike = new ZerosLike(*pScope, *x);
		ObjectInfo* pObj = AddObjectMap(pZerosLike, id, SYMBOL_ZEROSLIKE, "ZerosLike", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pZerosLike->y, OUTPUT_TYPE_OUTPUT, "y");
	}
	else
	{
		std::string msg = string_format("ZerosLike(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pZerosLike;
}
