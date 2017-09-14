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
	std::unordered_map<Output, Input::Initializer, OutputHash>* pFeedType = nullptr;

	std::string  device = "/cpu:0";      // device set interface

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();                        // val
		std::string strPinType = ItemValue.get("pin-type", "").asString();                        // double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();                  // 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();               // ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();                  // ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();         // ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();   // ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();               // tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();                     // [2][2]

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
					pSessionObj->pScope = m_pScope;      // Scope 객체 설정 (실행시 Scope가 ok 상태인지 체크후 실행

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
		else if (strPinName == "inputs")
		{
			if (strPinInterface == "FeedType")
			{
				if (pFetchInfo)
				{
					ObjectInfo* pfetchObj = LookupFromObjectMap(strInSymbolId);
					if (pfetchObj)
					{
						if (pfetchObj->type == SYMBOL_FEEDTYPE)
						{
							pFeedType = (std::unordered_map<Output, Input::Initializer, OutputHash>*)pfetchObj->pObject;
							pFetchInfo->pFeedType = pFeedType;						
						}
						else
						{
							std::string msg = string_format("warning : ClientSession - %s(%s) transfer information missed (lookup output map).", id.c_str(), strPinName.c_str());
							PrintMessage(msg);
						}
					}
				}
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
								pFetchInfo->output.fetch_object.push_back(pfetchObj);
								pFetchInfo->output.fetch_outputs.push_back(*(Output*)pOutputObj->pOutput);
								pFetchInfo->output.pin_names.push_back(strInSymbolPinName);
							}
							else if (pOutputObj->type == OUTPUT_TYPE_OUTPUTLIST)
							{
								pFetchInfo->output_list.fetch_object.push_back(pfetchObj);
								pFetchInfo->output_list.fetch_outputs.push_back(*(OutputList*)pOutputObj->pOutput);
								pFetchInfo->output_list.pin_names.push_back(strInSymbolPinName);
							}
							else
							{
								std::string msg = string_format("warning : ClientSession - %s(%s) transfer information missed (lookup type is not output).", id.c_str(), strPinName.c_str());
								PrintMessage(msg);
							}
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
		else if (strPinName == "run_outputs")
		{
			if (strPinInterface == "std::vector(tensorflow::Operation)")
			{
				if (pFetchInfo)
				{
					ObjectInfo* pfetchObj = LookupFromObjectMap(strInSymbolId);
					if (pfetchObj)
					{
						OutputInfo* pOutputObj = LookupFromOutputMap(pfetchObj, strInSymbolPinName);
						if (pOutputObj)
						{
							if (pOutputObj->type == OUTPUT_TYPE_OPERATION)
							{
								pFetchInfo->output.run_outputs.push_back(*(Operation*)pOutputObj->pOutput);
								pFetchInfo->output.pin_names.push_back(strInSymbolPinName);
							}
							else if (pOutputObj->type == OUTPUT_TYPE_OUTPUT)
							{
								std::string msg = string_format("warning : ClientSession - %s(%s) not support output object.", id.c_str(), strPinName.c_str());
								PrintMessage(msg);
							}
							else if (pOutputObj->type == OUTPUT_TYPE_OUTPUTLIST)
							{
								std::string msg = string_format("warning : ClientSession - %s(%s) not support operationlist object.", id.c_str(), strPinName.c_str());
								PrintMessage(msg);
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
		// device set interface
		else if (strPinName == "device")
		{
			if (strPinInterface == "device" && !strPinInitial.empty())
			{
				device = strPinInitial;
			}
		}
	}

	if (pSession == nullptr)
	{
		std::string msg = string_format("error : ClientSession(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	else
	{
		// device set interface
		GraphDef def;
		m_pScope->ToGraphDef(&def);

		for (int i = 0; i < def.node_size(); ++i)
		{
			auto node = def.mutable_node(i);
			if (node->device().empty())
			{
				node->set_device(device);
			}
		}
	}

	return pSession;
}

void* Create_Input_Initializer(std::string id, Json::Value pInputItem) {
	Input::Initializer* pInput_Initializer = nullptr;
	Scope* pScope = nullptr;
	Tensor* ptensor = nullptr;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "t")
		{
			if (strPinInterface == "Tensor")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					if (pObj->type == SYMBOL_TENSOR)
					{
						ptensor = (Tensor*)pObj->pObject;						
					}
				}
				else
				{
					if (!strPinInitial.empty())
						ptensor = Create_StrToTensor(strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : Initializer - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Initializer pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (ptensor)
	{
		pInput_Initializer = new Input::Initializer(*ptensor);
		AddObjectMap(pInput_Initializer, id, SYMBOL_INPUT_INITIALIZER, "output", pInputItem);
	}

	return pInput_Initializer;
}

void* Create_InputList(std::string id, Json::Value pInputItem) {
	Scope* pScope = nullptr;
	OutputList* pInputList = nullptr;
	OutputList outputlist;
	int iCheck_Pintype = 0;

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

		if (strPinName == "outputlist")
		{
			if (strInSymbolPinInterface == "Output" || strInSymbolPinInterface == "OutputList")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->type == OUTPUT_TYPE_OUTPUT)
						{
							outputlist.push_back(*((Output*)pOutputObj->pOutput));
						}
						else if (pOutputObj->type == OUTPUT_TYPE_OUTPUTLIST)
						{
							OutputList* outputcopylist = (OutputList*)pOutputObj->pOutput;
							outputlist.reserve(outputlist.size() + outputcopylist->size());
							outputlist.insert(outputlist.end(), outputcopylist->begin(), outputcopylist->end());
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : InputList - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : InputList pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (!outputlist.empty())
	{
		pInputList = new OutputList(outputlist);
		ObjectInfo* pObj = AddObjectMap(pInputList, id, SYMBOL_INPUTLIST, "output", pInputItem);
 		if (pObj)
 			AddOutputInfo(pObj, pInputList, OUTPUT_TYPE_OUTPUTLIST, "output");
	}
	else
	{
		std::string msg = string_format("error : InputList(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}

	return pInputList;
}

void* Create_Operation(std::string id, Json::Value pInputItem) {
	Operation* pOperation = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	int n_count = 0;

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

		if (strPinName == "operation")
		{
			if (strPinInterface == "Operation")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->type == OUTPUT_TYPE_OUTPUT)
						{
							pInput = (Output*)pOutputObj->pOutput;
							n_count++;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Operation - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Operation pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pInput && n_count == 1)
	{
		pOperation = new Operation(pInput->node());
		ObjectInfo* pObj = AddObjectMap(pOperation, id, SYMBOL_OPERATION, "output", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, pOperation, OUTPUT_TYPE_OPERATION, "output");
	}
	else if (n_count > 1)
	{
		std::string msg = string_format("error : Operation(%s) more than one transfer is not supported.", id.c_str());
		PrintMessage(msg);
	}
	else
	{
		std::string msg = string_format("error : Operation(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	
	return pOperation;
}

void* Create_Output(std::string id, Json::Value pInputItem) {
	Output* pOutput = nullptr;
	Scope* pScope = nullptr;
	Input* pInput = nullptr;

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
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pInput = (Input*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Output - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Output pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pInput)
	{
		pOutput = new Output(pInput->node());
		ObjectInfo* pObj = AddObjectMap(pOutput, id, SYMBOL_OUTPUT, "Output", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, pOutput, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("error : Output(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}

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
	std::string strinitvalues;
	std::string strdatatype;
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

		if (strPinName == "dtype")
		{
			if (!strPinInitial.empty())
			{
				dtype = GetDatatypeFromInitial(strPinInitial);
				if (dtype == DT_INVALID)
				{
					std::string msg = string_format("warning : Tensor - %s(%s) unknown dtype", id.c_str(), strPinName.c_str());
					PrintMessage(msg);
				}
				strdatatype = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : Tensor - %s(%s) dtype is not initialized", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "initvalues")
		{
			if (!strPinInitial.empty())
				strinitvalues = strPinInitial;
		}
		else
		{
			std::string msg = string_format("warning : Tensor pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (dtype != DT_INVALID)
	{
		pTensor = Create_StrToTensor(strdatatype, "", strinitvalues);
		AddObjectMap(pTensor, id, SYMBOL_TENSOR, "Tensor", pInputItem);
	}

	return pTensor;
}


void* Create_Input(std::string id, Json::Value pInputItem)
{
	Input* pInput = nullptr;
	tensorflow::TensorShape shape;
	tensorflow::DataType dtype;
	std::string strinitvalues;
	std::string strdatatype;
	Tensor* pTensor = nullptr;
	Output* pOutput = nullptr;

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

		if (strPinName == "dtype")
		{
			if (!strPinInitial.empty())
			{
				dtype = GetDatatypeFromInitial(strPinInitial);
				if (dtype == DT_INVALID)
				{
					std::string msg = string_format("warning : Input_ex - %s(%s) unknown dtype", id.c_str(), strPinName.c_str());
					PrintMessage(msg);
				}
				strdatatype = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : Input_ex - %s(%s) dtype is not initialized", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "initvalues")
		{
			if(strInSymbolPinInterface == "Tensor")
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
			else if (!strPinInitial.empty())
			{
				strinitvalues = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : Input - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Input_ex pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pTensor)
	{
		pInput = new Input(*pTensor);
		pOutput = new Output(pInput->node());
		ObjectInfo* pObj = AddObjectMap(pOutput, id, SYMBOL_INPUT, "output", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, pOutput, OUTPUT_TYPE_OUTPUT, "output");
	}
	else if(strinitvalues != "" && dtype)
	{
		pTensor = Create_StrToTensor(strdatatype, "", strinitvalues);
		pInput = new Input(*pTensor);
		pOutput = new Output(pInput->node());
		ObjectInfo* pObj = AddObjectMap(pOutput, id, SYMBOL_INPUT, "output", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, pOutput, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("error : Input(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}

	return pInput;
}

void* Create_FeedType(std::string id, Json::Value pInputItem)
{
	ClientSession::FeedType* pFeedType;
	Output* pinput = nullptr;
	Input::Initializer* initializer = nullptr;
	OutputHash* outputhash = nullptr;

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
			if (strPinInterface == "Input")
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
				std::string msg = string_format("warning : FeedType - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "initializer")
		{
			if (strPinInterface == "Input::Initializer")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					if (pObj->type == SYMBOL_INPUT_INITIALIZER)
						initializer = (Input::Initializer*)pObj->pObject;
					else
					{
						std::string msg = string_format("warning : FeedType - %s(%s) not support another type(input initializer).", id.c_str(), strPinName.c_str());
						PrintMessage(msg);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : FeedType - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "outputhash")
		{

		}
		else
		{
			std::string msg = string_format("warning : FeedType pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pinput && initializer)
	{
		// pFeedType = new ClientSession::FeedType(*pinput, *initializer, outputhash);
		//pFeedType->insert(ClientSession::FeedType::value_type(*pinput, *initializer));

		ClientSession::FeedType* pFeedType;
		pFeedType = new ClientSession::FeedType();

		pFeedType->insert({ *pinput, *initializer });

		AddObjectMap(pFeedType, id, SYMBOL_FEEDTYPE, "output", pInputItem);
	}
	return pFeedType;
}

void* Create_Const(std::string id, Json::Value pInputItem)
{
	Scope* pScope = nullptr;
	Output* pOutput = new Output();
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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : Scope, 연결 핀 : Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Const - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "val")
		{
			if (strInSymbolPinName == "" && strPinInterface == "Input::Initializer")
			{
				std::vector<int64> array_slice;
				std::vector<int64> arraydims;
				GetArrayDimsFromShape(strPinShape, arraydims, array_slice);

				if (strPinType == "double")
				{
					std::vector<double> arrayvals;
					GetDoubleVectorFromInitial(strPinInitial, arrayvals);

					gtl::ArraySlice< int64 > arraySlice(arraydims);
					pTensor = new Tensor(DT_DOUBLE, TensorShape(arraySlice));

					int i = 0;
					for (std::vector<double>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
					{
						pTensor->flat<double>()(i) = *it;
						i++;
					}
					arraySlice.clear();
					arrayvals.clear();
				}
				else if (strPinType == "float")
				{
					std::vector<float> arrayvals;
					GetFloatVectorFromInitial(strPinInitial, arrayvals);

					gtl::ArraySlice< int64 > arraySlice(arraydims);
					pTensor = new Tensor(DT_FLOAT, TensorShape(arraySlice));

					int i = 0;
					for (std::vector<float>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
					{
						pTensor->flat<float>()(i) = *it;
						i++;
					}
					arraySlice.clear();
					arrayvals.clear();
				}
				else if (strPinType == "int")
				{
					std::vector<int> arrayvals;
					GetIntVectorFromInitial(strPinInitial, arrayvals);

					gtl::ArraySlice< int64 > arraySlice(arraydims);
					pTensor = new Tensor(DT_INT32, TensorShape(arraySlice));

					int i = 0;
					for (std::vector<int>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
					{
						pTensor->flat<int>()(i) = *it;
						i++;
					}
					arraySlice.clear();
					arrayvals.clear();
				}
				else if (strPinType == "bool")
				{
					std::vector<bool> arrayvals;
					GetBoolVectorFromInitial(strPinInitial, arrayvals);

					gtl::ArraySlice< int64 > arraySlice(arraydims);
					pTensor = new Tensor(DT_BOOL, TensorShape(arraySlice));

					int i = 0;
					for (std::vector<bool>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
					{
						pTensor->flat<bool>()(i) = *it;
						i++;
					}
					arraySlice.clear();
					arrayvals.clear();
				}
				else if (strPinType == "string")
				{
					std::vector<std::string> arrayvals;
					GetStringVectorFromInitial(strPinInitial, arrayvals);

					gtl::ArraySlice< int64 > arraySlice(arraydims);
					pTensor = new Tensor(DT_STRING, TensorShape(arraySlice));

					int i = 0;
					for (std::vector<std::string>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
					{
						pTensor->flat<std::string>()(i) = *it;
						i++;
					}
					arraySlice.clear();
					arrayvals.clear();
				}
				else
				{
					std::string msg = string_format("warning : Const - %s(val-initvalue) transfer information missed.", id.c_str());
					PrintMessage(msg);
				}

				array_slice.clear();
				arraydims.clear();
			}
		}
	}
	if (pScope == nullptr)
	{
		std::string msg = string_format("warning : Const - %s(scope) transfer information missed.", id.c_str());
		PrintMessage(msg);
	}
	if (pTensor == nullptr)
	{
		std::string msg = string_format("warning : Const - %s(val) transfer information missed.", id.c_str());
		PrintMessage(msg);
	}

	if (pScope && pTensor)
	{
		*pOutput = Const(*pScope, *pTensor);
		ObjectInfo* pObj = AddObjectMap(pOutput, id, SYMBOL_CONST, "Const", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, pOutput, OUTPUT_TYPE_OUTPUT, "output");
		// pObj->pOutput = pOutput;
	}
	else
	{
		std::string msg = string_format("error : Const(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}

	if (pTensor)
		delete pTensor;

	return pOutput;
}