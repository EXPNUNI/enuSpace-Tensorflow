#include "stdafx.h"
#include "tf_core.h"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "include/json/json.h"

#include "GlobalHeader.h"
#include "tensorflow.h"
#include "utility_functions.h"
#include "enuSpaceToTensorflow.h"


void* Create_ClientSession(std::string id, Json::Value pInputItem) {
	Scope* pScope = nullptr;
	ClientSession* pSession = nullptr;
	ObjectInfo* pSessionObj = nullptr;
	FetchInfo* pFetchInfo = nullptr;

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
			if (strPinInterface == "Scope")
			{
				if (m_pScope)
				{
					ClientSession* pCreateSession = new ClientSession(*m_pScope);
					pSession = pCreateSession;

					pSessionObj = AddObjectMap(pSession, id, SYMBOL_CLIENTSESSION, "ClientSession", pInputItem);
					pSessionObj->pScope = m_pScope;		// Scope 객체 설정 (실행시 Scope가 ok 상태인지 체크후 실행

					pFetchInfo = AddRunObjectMap(pSessionObj);
				}
				else
				{
					std::string msg = string_format("warning : ClientSession - %s(%s) scope information missed.", id.c_str(), strPinName.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : ClientSession - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "fetch_outputs")
		{
			if (strPinInterface == "std::vector(tensorflow::Output)")
			{
				if (pFetchInfo)
				{
					ObjectInfo* pfetchObj = LookupFromObjectMap(strInSymbolId);
					if (pfetchObj)
					{
						OutputInfo* pOutputObj = LookupFromOutputMap(pfetchObj, strInSymbolPinName);
						if (pOutputObj)
						{
							if (pOutputObj->type == OUTPUT_TYPE_OUTPUT)
							{
								pFetchInfo->fetch_object.push_back(pfetchObj);
								pFetchInfo->fetch_outputs.push_back(*(Output*)pOutputObj->pOutput);
								pFetchInfo->pin_names.push_back(strInSymbolPinName);
							}
							else
							{
								std::string msg = string_format("warning : ClientSession - %s(%s) transfer information missed (lookup type is not output).", id.c_str(), strPinName.c_str());
								PrintMessage(msg);
							}
						}
						else
						{
							std::string msg = string_format("warning : ClientSession - %s(%s) transfer information missed (lookup output map).", id.c_str(), strPinName.c_str());
							PrintMessage(msg);
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ClientSession - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pSession == nullptr)
	{
		std::string msg = string_format("error : ClientSession(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}

	return pSession;
}

void* Create_Input(std::string id, Json::Value pInputItem) {
	Output* pOutput = nullptr;
	Input* pInput = nullptr;
	Tensor* pTensor = nullptr;

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

		if (strPinName == "input")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strInSymbolName == "#Tensor" && strInSymbolPinInterface == "Tensor" && strPinInterface == "Tensor")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					if (pObj->type == SYMBOL_TENSOR)
					{
						pTensor = (Tensor*)pObj->pObject;
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Input - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Input pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pTensor)
	{
		pInput = new Input(*pTensor);
		ObjectInfo* pObj = AddObjectMap(pInput, id, SYMBOL_INPUT, "Input", pInputItem);
	}
	else
	{
		std::string msg = string_format("error : Input(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pInput;
}

void* Create_Input_Initializer(std::string id, Json::Value pInputItem) {
	Input* pInput_Initializer = nullptr;
	Scope* pScope = nullptr;
	return pInput_Initializer;
}

void* Create_InputList(std::string id, Json::Value pInputItem) {
	InputList* pInputList = nullptr;
	Scope* pScope = nullptr;
	return pInputList;
}

void* Create_Operation(std::string id, Json::Value pInputItem) {
	Operation* pOperation = nullptr;
	Scope* pScope = nullptr;
	return pOperation;
}

void* Create_Output(std::string id, Json::Value pInputItem) {
	Output* pOutput = nullptr;
	Scope* pScope = nullptr;
	return pOutput;
}

void* Create_Scope(std::string id, Json::Value pInputItem) {

	Scope* pRoot = new Scope(Scope::NewRootScope());
	m_pScope = pRoot;
	AddObjectMap(pRoot, id, SYMBOL_SCOPE, "Scope", pInputItem);
	return pRoot;
}

void* Create_Status(std::string id, Json::Value pInputItem) {
	Status* pStatus = nullptr;
	Scope* pScope = nullptr;
	return pStatus;
}

void* Create_Tensor(std::string id, Json::Value pInputItem) {
	Tensor* pTensor = nullptr;

	tensorflow::TensorShape shape;
	tensorflow::DataType dtype = DT_DOUBLE;

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

		if (strPinName == "shape")
		{
			if (strPinInterface == "TensorShape")
			{
				std::vector<int64> arrayslice;
				std::vector<int64> arraydims;
				GetArrayDimsFromShape(strPinInitial, arraydims, arrayslice);
				gtl::ArraySlice< int64 > arraySlice(arraydims);
				shape = TensorShape(arraySlice);
			}
			else
			{
				std::string msg = string_format("warning : Tensor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				if (strPinInitial == "double")
					dtype = DT_DOUBLE;
				else if (strPinInitial == "float")
					dtype = DT_FLOAT;
				else if (strPinInitial == "int")
					dtype = DT_INT32;
				else if (strPinInitial == "bool")
					dtype = DT_BOOL;
				else if (strPinInitial == "string")
					dtype = DT_STRING;
				else
				{
					std::string msg = string_format("warning : Tensor - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : Tensor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Tensor pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	pTensor = new Tensor(dtype, shape);

	AddObjectMap(pTensor, id, SYMBOL_TENSOR, "Tensor", pInputItem);
	return pTensor;
}


void* Create_Input_ex(std::string id, Json::Value pInputItem)
{
	Input* pInput = nullptr;
	tensorflow::TensorShape shape;
	tensorflow::DataType dtype;

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

		if (strPinName == "shape")
		{
			if (strPinInterface == "TensorShape")
			{
				std::vector<int64> arrayslice;
				std::vector<int64> arraydims;
				GetArrayDimsFromShape(strPinInitial, arraydims, arrayslice);
				gtl::ArraySlice< int64 > arraySlice(arraydims);
				shape = TensorShape(arraySlice);
			}
			else
			{
				std::string msg = string_format("warning : Input_ex - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				if (strPinInitial == "double")
					dtype = DT_DOUBLE;
				else if (strPinInitial == "float")
					dtype = DT_FLOAT;
				else if (strPinInitial == "int")
					dtype = DT_INT32;
				else if (strPinInitial == "bool")
					dtype = DT_BOOL;
				else if (strPinInitial == "string")
					dtype = DT_STRING;
				else
				{
					std::string msg = string_format("warning : Input_ex - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : Input_ex pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	Tensor tensor(dtype, shape);

	pInput = new Input(tensor);
	ObjectInfo* pObj = AddObjectMap(pInput, id, SYMBOL_INPUT_EX, "Input", pInputItem);

	return pInput;
}