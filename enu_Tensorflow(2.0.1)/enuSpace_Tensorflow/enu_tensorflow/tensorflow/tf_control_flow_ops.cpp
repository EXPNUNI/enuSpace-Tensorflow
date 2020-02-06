#include "stdafx.h"
#include "tf_control_flow_ops.h"

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

void* Create_Abort(std::string id, Json::Value pInputItem) {
	Abort* pAbort = nullptr;
	Scope* pScope = nullptr;
	Abort::Attrs attrs;

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
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Abort - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Abort::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("error_msg_") != "")
				{
					attrs.error_msg_ = attrParser.GetAttribute("error_msg_");
				}
				if (attrParser.GetAttribute("exit_without_error_") != "")
				{
					attrs.ExitWithoutError(attrParser.ConvStrToBool(attrParser.GetAttribute("exit_without_error_")));
				}
			}
		}
		else
		{
			std::string msg = string_format("Abort pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope)
	{
		pAbort = new Abort(*pScope, attrs);
		ObjectInfo* pObj = AddObjectMap(pAbort, id, SYMBOL_ABORT, "Abort", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pAbort->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("Abort(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pAbort;
}

void* Create_ControlTrigger(std::string id, Json::Value pInputItem) {
	ControlTrigger* pControlTrigger = nullptr;
	Scope* pScope = nullptr;

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
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("ControlTrigger - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("ControlTrigger pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope)
	{
		pControlTrigger = new ControlTrigger(*pScope);
		ObjectInfo* pObj = AddObjectMap(pControlTrigger, id, SYMBOL_CONTROLTRIGGER, "ControlTrigger", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pControlTrigger->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("ControlTrigger : Abort(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pControlTrigger;
} 

void* Create_LoopCond(std::string id, Json::Value pInputItem) {
	LoopCond* pLoopCond = nullptr;
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
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("LoopCond - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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

			}
			else
			{
				std::string msg = string_format("LoopCond - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("LoopCond pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pInput)
	{
		pLoopCond = new LoopCond(*pScope, *pInput);
		ObjectInfo* pObj = AddObjectMap(pLoopCond, id, SYMBOL_LOOPCOND, "LoopCond", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pLoopCond->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("LoopCond(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pLoopCond;
}

void* Create_Merge(std::string id, Json::Value pInputItem) {
	Merge* pMerge = nullptr;
	Scope* pScope = nullptr;
	OutputList* inputs = nullptr;

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
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Merge - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
							inputs = (OutputList*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("Merge - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("Merge pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && inputs)
	{
		pMerge = new Merge(*pScope, *inputs);
		ObjectInfo* pObj = AddObjectMap(pMerge, id, SYMBOL_MERGE, "Merge", pInputItem);
		if (pObj) {
			AddOutputInfo(pObj, &pMerge->output, OUTPUT_TYPE_OUTPUT, "output");
			AddOutputInfo(pObj, &pMerge->value_index, OUTPUT_TYPE_OUTPUT, "value_index");
		}
	}
	else
	{
		std::string msg = string_format("Merge(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pMerge;
}

void* Create_NextIteration(std::string id, Json::Value pInputItem) {
	NextIteration* pNextIteration = nullptr;
	Scope* pScope = nullptr;
	Output* data = nullptr;

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
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("NextIteration - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "data")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							data = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("NextIteration - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("NextIteration pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && data)
	{
		pNextIteration = new NextIteration(*pScope, *data);
		ObjectInfo* pObj = AddObjectMap(pNextIteration, id, SYMBOL_NEXTITERATION, "NextIteration", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pNextIteration->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("NextIteration(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pNextIteration;
}

void* Create_RefNextIteration(std::string id, Json::Value pInputItem) {
	RefNextIteration* pRefNextIteration = nullptr;
	Scope* pScope = nullptr;
	Output* data = nullptr;

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
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("RefNextIteration - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "data")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							data = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("RefNextIteration - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("RefNextIteration pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && data)
	{
		pRefNextIteration = new RefNextIteration(*pScope, *data);
		ObjectInfo* pObj = AddObjectMap(pRefNextIteration, id, SYMBOL_REFNEXTITERATION, "RefNextIteration", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pRefNextIteration->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("RefNextIteration(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pRefNextIteration;
}

void* Create_RefSelect(std::string id, Json::Value pInputItem) {
	RefSelect* pRefSelect = nullptr;
	Scope* pScope = nullptr;
	Output* index = nullptr;
	OutputList* inputs = nullptr;

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
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("RefSelect - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "index")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							index = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("RefSelect - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
							inputs = (OutputList*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("RefSelect - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("RefSelect pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && index && inputs)
	{
		pRefSelect = new RefSelect(*pScope, *index, *inputs);
		ObjectInfo* pObj = AddObjectMap(pRefSelect, id, SYMBOL_REFSELECT, "RefSelect", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pRefSelect->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("RefSelect(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pRefSelect;
}

void* Create_RefSwitch(std::string id, Json::Value pInputItem) {
	RefSwitch* pRefSwitch = nullptr;
	Scope* pScope = nullptr;
	Output* data = nullptr;
	Output* pred = nullptr;

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
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("RefSwitch - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "data")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							data = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("RefSwitch - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "pred")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pred = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("RefSwitch - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("RefSwitch pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && data && pred)
	{
		pRefSwitch = new RefSwitch(*pScope, *data, *pred);
		ObjectInfo* pObj = AddObjectMap(pRefSwitch, id, SYMBOL_REFSWITCH, "RefSwitch", pInputItem);
		if (pObj) {
			AddOutputInfo(pObj, &pRefSwitch->output_false, OUTPUT_TYPE_OUTPUT, "output_false");
			AddOutputInfo(pObj, &pRefSwitch->output_true, OUTPUT_TYPE_OUTPUT, "output_true");
		}
	}
	else
	{
		std::string msg = string_format("RefSwitch(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pRefSwitch;
}

void* Create_Switch(std::string id, Json::Value pInputItem) {
	Switch* pSwitch = nullptr;
	Scope* pScope = nullptr;
	Output* data = nullptr;
	Output* pred = nullptr;

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
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("Switch - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "data")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							data = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						data = (Output*)Create_StrToOutput(*m_pScope, "", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Switch - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "pred")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pred = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pred = (Output*)Create_StrToOutput(*m_pScope, "DT_BOOL", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("Switch - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("Switch pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && data && pred)
	{
		pSwitch = new Switch(*pScope, *data, *pred);
		ObjectInfo* pObj = AddObjectMap(pSwitch, id, SYMBOL_SWITCH, "Switch", pInputItem);
		if (pObj) {
			AddOutputInfo(pObj, &pSwitch->output_false, OUTPUT_TYPE_OUTPUT, "output_false");
			AddOutputInfo(pObj, &pSwitch->output_true, OUTPUT_TYPE_OUTPUT, "output_true");
		}
	}
	else
	{
		std::string msg = string_format("Switch(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pSwitch;
}