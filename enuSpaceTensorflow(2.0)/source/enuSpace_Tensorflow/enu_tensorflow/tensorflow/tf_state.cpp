#include "stdafx.h"
#include "tf_state.h"


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


void* Create_Assign(std::string id, Json::Value pInputItem) {
	Assign* pAssign = nullptr;
	Scope* pScope = nullptr;
	Output* ref = nullptr;
	Output* value = nullptr;
	Assign::Attrs attrs;

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
				std::string msg = string_format("warning : Assign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
				std::string msg = string_format("warning : Assign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
			}
			else
			{
				std::string msg = string_format("warning : Assign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Assign::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if(attrParser.GetAttribute("validate_shape_")!="")
					attrs =attrs.ValidateShape(attrParser.GetValue_bool("validate_shape_"));
				if (attrParser.GetAttribute("use_locking_") != "")
					attrs =attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : Assign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && ref && value)
	{
		pAssign = new Assign(*pScope, *ref, *value, attrs);
		ObjectInfo* pObj = AddObjectMap(pAssign, id, SYMBOL_ASSIGN, "Assign", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAssign->output_ref, OUTPUT_TYPE_OUTPUT, "output_ref");
		}
	}
	else
	{
		std::string msg = string_format("error : Assign(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}

	return pAssign;
}

void* Create_AssignAdd(std::string id, Json::Value pInputItem) {
	AssignAdd* pAssignAdd = nullptr;
	Scope* pScope = nullptr;
	Output* ref = nullptr;
	Output* value = nullptr;
	AssignAdd::Attrs attrs;

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
				std::string msg = string_format("warning : AssignAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
				std::string msg = string_format("warning : AssignAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
			}
			else
			{
				std::string msg = string_format("warning : AssignAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "AssignAdd::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")
					attrs =attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));
			
			}
		}
		else
		{
			std::string msg = string_format("warning : AssignAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && ref && value)
	{
		pAssignAdd = new AssignAdd(*pScope, *ref, *value, attrs);
		ObjectInfo* pObj = AddObjectMap(pAssignAdd, id, SYMBOL_ASSIGNADD, "AssignAdd", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAssignAdd->output_ref, OUTPUT_TYPE_OUTPUT, "output_ref");
		}
	}
	else
	{
		std::string msg = string_format("error : AssignAdd(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAssignAdd;
}

void* Create_AssignSub(std::string id, Json::Value pInputItem) {
	AssignSub* pAssignSub = nullptr;
	Scope* pScope = nullptr;
	Output* ref = nullptr;
	Output* value = nullptr;
	AssignSub::Attrs attrs;

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
				std::string msg = string_format("warning : AssignSub - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
				std::string msg = string_format("warning : AssignSub - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
			}
			else
			{
				std::string msg = string_format("warning : AssignSub - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "AssignSub::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")
					attrs =attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));

			}
		}
		else
		{
			std::string msg = string_format("warning : AssignSub - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && ref && value)
	{
		pAssignSub = new AssignSub(*pScope, *ref, *value, attrs);
		ObjectInfo* pObj = AddObjectMap(pAssignSub, id, SYMBOL_ASSIGNSUB, "AssignSub", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAssignSub->output_ref, OUTPUT_TYPE_OUTPUT, "output_ref");
		}
	}
	else
	{
		std::string msg = string_format("error : AssignSub(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAssignSub;
}

void* Create_CountUpTo(std::string id, Json::Value pInputItem) {
	CountUpTo* pCountUpTo = nullptr;
	Scope* pScope = nullptr;
	Output* ref = nullptr;
	int64 limit =0;

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
				std::string msg = string_format("warning : CountUpTo - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
				std::string msg = string_format("warning : CountUpTo - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "limit")
		{
			if (strPinInterface == "int64")
			{
				limit = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : CountUpTo - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : CountUpTo - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && ref)
	{
		pCountUpTo = new CountUpTo(*pScope, *ref, limit);
		ObjectInfo* pObj = AddObjectMap(pCountUpTo, id, SYMBOL_COUNTUPTO, "CountUpTo", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pCountUpTo->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : CountUpTo(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pCountUpTo;
}

void* Create_DestroyTemporaryVariable(std::string id, Json::Value pInputItem) {
	DestroyTemporaryVariable* pDestroyTemporaryVariable = nullptr;
	Scope* pScope = nullptr;
	Output* ref = nullptr;
	StringPiece var_name;
	std::string temp1 ="";
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
				std::string msg = string_format("warning : DestroyTemporaryVariable - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
				std::string msg = string_format("warning : DestroyTemporaryVariable - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "var_name")
		{
			if (strPinInterface == "StringPiece")
			{
				temp1 = strPinInitial;
				var_name = temp1;
			}
			else
			{
				std::string msg = string_format("warning : DestroyTemporaryVariable - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : DestroyTemporaryVariable - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && ref)
	{
		
		pDestroyTemporaryVariable = new DestroyTemporaryVariable(*pScope, *ref, var_name);
		ObjectInfo* pObj = AddObjectMap(pDestroyTemporaryVariable, id, SYMBOL_DESTROYTEMPORARYVARIABLE, "DestroyTemporaryVariable", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pDestroyTemporaryVariable->value, OUTPUT_TYPE_OUTPUT, "value");
		}
	}
	else
	{
		std::string msg = string_format("error : DestroyTemporaryVariable(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pDestroyTemporaryVariable;
}

void* Create_IsVariableInitialized(std::string id, Json::Value pInputItem) {
	IsVariableInitialized* pIsVariableInitialized = nullptr;
	Scope* pScope = nullptr;
	Output* ref = new Output();
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
				std::string msg = string_format("warning : IsVariableInitialized - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
				std::string msg = string_format("warning : IsVariableInitialized - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : IsVariableInitialized - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && ref)
	{
		pIsVariableInitialized = new IsVariableInitialized(*pScope, *ref);
		ObjectInfo* pObj = AddObjectMap(pIsVariableInitialized, id, SYMBOL_ISVARIABLEINITIALIZED, "IsVariableInitialized", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pIsVariableInitialized->is_initialized, OUTPUT_TYPE_OUTPUT, "is_initialized");
		}
	}
	else
	{
		std::string msg = string_format("error : IsVariableInitialized(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pIsVariableInitialized;
}

void* Create_ScatterAdd(std::string id, Json::Value pInputItem) {
	ScatterAdd* pScatterAdd = nullptr;
	Scope* pScope = nullptr;
	Output* ref = nullptr;
	Output* indices = nullptr;
	Output* updates = nullptr;
	ScatterAdd::Attrs attrs;

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
				std::string msg = string_format("warning : ScatterAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
				std::string msg = string_format("warning : ScatterAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : ScatterAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
			}
			else
			{
				std::string msg = string_format("warning : ScatterAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ScatterAdd::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")
					attrs =attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));

			}
		}
		else
		{
			std::string msg = string_format("warning : ScatterAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && ref && indices && updates)
	{
		pScatterAdd = new ScatterAdd(*pScope, *ref ,*indices,*updates, attrs);
		ObjectInfo* pObj = AddObjectMap(pScatterAdd, id, SYMBOL_SCATTERADD, "ScatterAdd", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pScatterAdd->output_ref, OUTPUT_TYPE_OUTPUT, "output_ref");
		}
	}
	else
	{
		std::string msg = string_format("error : ScatterAdd(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pScatterAdd;
}

void* Create_ScatterDiv(std::string id, Json::Value pInputItem) {
	ScatterDiv* pScatterDiv = nullptr;
	Scope* pScope = nullptr;
	Output* ref = nullptr;
	Output* indices = nullptr;
	Output* updates = nullptr;
	ScatterDiv::Attrs attrs;

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
				std::string msg = string_format("warning : ScatterDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
				std::string msg = string_format("warning : ScatterDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : ScatterDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
			}
			else
			{
				std::string msg = string_format("warning : ScatterDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ScatterDiv::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")
					attrs =attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));

			}
		}
		else
		{
			std::string msg = string_format("warning : ScatterDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && ref && indices && updates)
	{
		pScatterDiv = new ScatterDiv(*pScope, *ref, *indices, *updates, attrs);
		ObjectInfo* pObj = AddObjectMap(pScatterDiv, id, SYMBOL_SCATTERDIV, "ScatterDiv", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pScatterDiv->output_ref, OUTPUT_TYPE_OUTPUT, "output_ref");
		}
	}
	else
	{
		std::string msg = string_format("error : ScatterDiv(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pScatterDiv;
}

void* Create_ScatterMul(std::string id, Json::Value pInputItem) {
	ScatterMul* pScatterMul = nullptr;
	Scope* pScope = nullptr;
	Output* ref = nullptr;
	Output* indices = nullptr;
	Output* updates = nullptr;
	ScatterMul::Attrs attrs;

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
				std::string msg = string_format("warning : ScatterMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
				std::string msg = string_format("warning : ScatterMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : ScatterMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
			}
			else
			{
				std::string msg = string_format("warning : ScatterMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ScatterMul::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")
					attrs = attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));

			}
		}
		else
		{
			std::string msg = string_format("warning : ScatterMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && ref && indices && updates)
	{
		pScatterMul = new ScatterMul(*pScope, *ref, *indices, *updates, attrs);
		ObjectInfo* pObj = AddObjectMap(pScatterMul, id, SYMBOL_SCATTERMUL, "ScatterMul", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pScatterMul->output_ref, OUTPUT_TYPE_OUTPUT, "output_ref");
		}
	}
	else
	{
		std::string msg = string_format("error : ScatterMul(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pScatterMul;
}

void* Create_ScatterNdAdd(std::string id, Json::Value pInputItem) {
	ScatterNdAdd* pScatterNdAdd = nullptr;
	Scope* pScope = nullptr;
	Output* ref = nullptr;
	Output* indices = nullptr;
	Output* updates = nullptr;
	ScatterNdAdd::Attrs attrs;

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
				std::string msg = string_format("warning : ScatterNdAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
				std::string msg = string_format("warning : ScatterNdAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : ScatterNdAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
			}
			else
			{
				std::string msg = string_format("warning : ScatterNdAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ScatterNdAdd::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")
					attrs = attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));

			}
		}
		else
		{
			std::string msg = string_format("warning : ScatterNdAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && ref && indices && updates)
	{
		pScatterNdAdd = new ScatterNdAdd(*pScope, *ref, *indices, *updates, attrs);
		ObjectInfo* pObj = AddObjectMap(pScatterNdAdd, id, SYMBOL_SCATTERNDADD, "ScatterNdAdd", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pScatterNdAdd->output_ref, OUTPUT_TYPE_OUTPUT, "output_ref");
		}
	}
	else
	{
		std::string msg = string_format("error : ScatterNdAdd(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pScatterNdAdd;
}

void* Create_ScatterNdSub(std::string id, Json::Value pInputItem) {
	ScatterNdSub* pScatterNdSub = nullptr;
	Scope* pScope = nullptr;
	Output* ref = nullptr;
	Output* indices = nullptr;
	Output* updates = nullptr;
	ScatterNdSub::Attrs attrs;

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
				std::string msg = string_format("warning : ScatterNdSub - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
				std::string msg = string_format("warning : ScatterNdSub - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : ScatterNdSub - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
			}
			else
			{
				std::string msg = string_format("warning : ScatterNdSub - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ScatterNdSub::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")
					attrs = attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));

			}
		}
		else
		{
			std::string msg = string_format("warning : ScatterNdSub - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && ref && indices && updates)
	{
		pScatterNdSub = new ScatterNdSub(*pScope, *ref, *indices, *updates, attrs);
		ObjectInfo* pObj = AddObjectMap(pScatterNdSub, id, SYMBOL_SCATTERNDSUB, "ScatterNdSub", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pScatterNdSub->output_ref, OUTPUT_TYPE_OUTPUT, "output_ref");
		}
	}
	else
	{
		std::string msg = string_format("error : ScatterNdSub(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pScatterNdSub;
}

void* Create_ScatterNdUpdate(std::string id, Json::Value pInputItem) {
	ScatterNdUpdate* pScatterNdUpdate = nullptr;
	Scope* pScope = nullptr;
	Output* ref = nullptr;
	Output* indices = nullptr;
	Output* updates = nullptr;
	ScatterNdUpdate::Attrs attrs;

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
				std::string msg = string_format("warning : ScatterNdUpdate - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
				std::string msg = string_format("warning : ScatterNdUpdate - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : ScatterNdUpdate - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
			}
			else
			{
				std::string msg = string_format("warning : ScatterNdUpdate - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ScatterNdUpdate::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")
					attrs =attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));

			}
		}
		else
		{
			std::string msg = string_format("warning : ScatterNdUpdate - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && ref && indices && updates)
	{
		pScatterNdUpdate = new ScatterNdUpdate(*pScope, *ref, *indices, *updates, attrs);
		ObjectInfo* pObj = AddObjectMap(pScatterNdUpdate, id, SYMBOL_SCATTERNDUPDATE, "ScatterNdUpdate", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pScatterNdUpdate->output_ref, OUTPUT_TYPE_OUTPUT, "output_ref");
		}
	}
	else
	{
		std::string msg = string_format("error : ScatterNdUpdate(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pScatterNdUpdate;
}

void* Create_ScatterSub(std::string id, Json::Value pInputItem) {
	ScatterSub* pScatterSub = nullptr;
	Scope* pScope = nullptr;
	Output* ref = nullptr;
	Output* indices = nullptr;
	Output* updates = nullptr;
	ScatterSub::Attrs attrs;

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
				std::string msg = string_format("warning : ScatterSub - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
				std::string msg = string_format("warning : ScatterSub - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : ScatterSub - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
			}
			else
			{
				std::string msg = string_format("warning : ScatterSub - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ScatterSub::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")
					attrs = attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));

			}
		}
		else
		{
			std::string msg = string_format("warning : ScatterSub - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && ref && indices && updates)
	{
		pScatterSub = new ScatterSub(*pScope, *ref, *indices, *updates, attrs);
		ObjectInfo* pObj = AddObjectMap(pScatterSub, id, SYMBOL_SCATTERSUB, "ScatterSub", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pScatterSub->output_ref, OUTPUT_TYPE_OUTPUT, "output_ref");
		}
	}
	else
	{
		std::string msg = string_format("error : ScatterSub(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pScatterSub;
}

void* Create_ScatterUpdate(std::string id, Json::Value pInputItem) {
	ScatterUpdate* pScatterUpdate = nullptr;
	Scope* pScope = nullptr;
	Output* ref = nullptr;
	Output* indices = nullptr;
	Output* updates = nullptr;
	ScatterUpdate::Attrs attrs;

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
				std::string msg = string_format("warning : ScatterUpdate - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
				std::string msg = string_format("warning : ScatterUpdate - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : ScatterUpdate - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
			}
			else
			{
				std::string msg = string_format("warning : ScatterUpdate - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ScatterUpdate::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")
					attrs=attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));

			}
		}
		else
		{
			std::string msg = string_format("warning : ScatterUpdate - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && ref && indices && updates)
	{
		pScatterUpdate = new ScatterUpdate(*pScope, *ref, *indices, *updates, attrs);
		ObjectInfo* pObj = AddObjectMap(pScatterUpdate, id, SYMBOL_SCATTERUPDATE, "ScatterUpdate", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pScatterUpdate->output_ref, OUTPUT_TYPE_OUTPUT, "output_ref");
		}
	}
	else
	{
		std::string msg = string_format("error : ScatterUpdate(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pScatterUpdate;
}

void* Create_TemporaryVariable(std::string id, Json::Value pInputItem) {
	TemporaryVariable* pTemporaryVariable = nullptr;
	Scope* pScope = nullptr;
	PartialTensorShape shape;
	DataType dtype;
	TemporaryVariable::Attrs attrs;
	StringPiece strTemp;
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
				std::string msg = string_format("warning : TemporaryVariable - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "shape")
		{
			if (strPinInterface == "PartialTensorShape")
			{
				if (strPinInitial !="")
				{
					shape = GetPartialShapeFromInitial(strPinInitial);
				}
				
			}
			else
			{
				std::string msg = string_format("warning : TemporaryVariable - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				dtype = GetDatatypeFromInitial(strPinInitial);
				if(dtype==DT_INVALID)
				{
					std::string msg = string_format("warning : TemporaryVariable - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : TemporaryVariable - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "TemporaryVariable::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("var_name_") !="")
				{
					strTemp = attrParser.GetAttribute("var_name_");
					attrs = attrs.VarName(strTemp);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : TemporaryVariable - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope)
	{
		pTemporaryVariable = new TemporaryVariable(*pScope, shape,dtype, attrs);
		ObjectInfo* pObj = AddObjectMap(pTemporaryVariable, id, SYMBOL_TEMPORARYVARIABLE, "TemporaryVariable", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pTemporaryVariable->ref, OUTPUT_TYPE_OUTPUT, "ref");
		}
	}
	else
	{
		std::string msg = string_format("error : TemporaryVariable(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pTemporaryVariable;
}


void* Create_Variable(std::string id, Json::Value pInputItem) {
	Scope* pScope = nullptr;
	tensorflow::ops::Variable* pOutput = nullptr;
	tensorflow::PartialTensorShape shape;
	tensorflow::DataType dtype = DT_INVALID;
	tensorflow::ops::Variable::Attrs attrs;
	std::string temp1, temp2;

	std::string strDataType;
	std::string strDataShape;
	std::string strInitPinType;
	std::string strInitPinShape;
	std::string strInitPinInitial;

	bool bInterfaceObj = false;
	ObjectInfo* pInterfaceObj = nullptr;

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
				std::string msg = string_format("warning : Variable - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "shape")
		{
			if (strPinInterface == "PartialTensorShape")
			{
				strDataShape = strPinInitial;
				//if (strPinInitial != "")
				//{
				//	shape = GetPartialShapeFromInitial(strPinInitial);
				//}
			}
			else
			{
				std::string msg = string_format("warning : Variable - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				strDataType = strPinInitial;
				//dtype = GetDatatypeFromInitial(strPinInitial);
				//if (dtype == DT_INVALID)
				//{
				//	std::string msg = string_format("warning : Variable - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
				//	PrintMessage(msg);
				//}
			}
			else
			{
				std::string msg = string_format("warning : Variable - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Variable::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("container_") != "")
				{
					temp1 = attrParser.GetAttribute("container_");
					attrs = attrs.Container(temp1);
				}
				if (attrParser.GetAttribute("SharedName_") != "")
				{
					temp2 = attrParser.GetAttribute("SharedName_");
					attrs = attrs.Container(temp2);
				}
			}
		}
		else if (strPinName == "initvalues")
		{
			strInitPinType = strPinType;
			strInitPinShape = strPinShape;
			strInitPinInitial = strPinInitial;
			if (strPinInterface == "Input" && !strInSymbolId.empty())
			{
				// 초기화 루틴은 ClientSession 생성부에서 처리 수행함.
				bInterfaceObj = true;
				pInterfaceObj = LookupFromObjectMap(strInSymbolId);
			}
		}
		else
		{
			std::string msg = string_format("warning : Variable pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{	
		if (bInterfaceObj)
		{
			if (pInterfaceObj)
			{
				if (pInterfaceObj->type == SYMBOL_CONST)
				{
					// 연결된 Const의 val값의 정보를 통하여 생성할 Varaible객체의 타입 및 Shape를 취득함.
					int iSize = (int)pInterfaceObj->param.size();
					for (int subindex = 0; subindex < iSize; ++subindex)
					{
						Json::Value ItemValue = pInterfaceObj->param[subindex];

						std::string strSubPinName = ItemValue.get("pin-name", "").asString();
						if (strSubPinName == "val")
						{
							// double, float, int, bool, string
							std::string strSubPinType = ItemValue.get("pin-type", "").asString();

							// [2][2]
							std::string strSubPinShape = ItemValue.get("pin-shape", "").asString();

							std::vector<int64> array_slice;
							std::vector<int64> arraydims;
							GetArrayDimsFromShape(strSubPinShape, arraydims, array_slice);

							if (strSubPinType == "double")
								dtype = DT_DOUBLE;
							else if (strSubPinType == "float")
								dtype = DT_FLOAT;
							else if (strSubPinType == "int")
								dtype = DT_INT32;
							else if (strSubPinType == "bool")
								dtype = DT_BOOL;
							else if (strSubPinType == "string")
								dtype = DT_STRING;
							else
								dtype = DT_INVALID;

							if (dtype != DT_INVALID)
							{
								PartialTensorShape PartialTS(arraydims);
								pOutput = new Variable(*pScope, PartialTS, dtype, attrs);
								PartialTS.Clear();
							}
							else
							{
								std::string msg = string_format("error : Variable(%s) Object create failed. input const object(%s).", id.c_str(), strSubPinType.c_str());
								PrintMessage(msg);
							}
							break;
						}
					}
				}
				else
				{
					std::string msg = string_format("error : Variable(%s) Object create failed. not support object. use const object.", id.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("error : Variable(%s) Object create failed. interface object(%s).", id.c_str(), strDataType.c_str());
				PrintMessage(msg);
			}
		}
		// initvalues = {1.0f, 2.0f}, datatype = DT_FLOAT, shape = [10] or null 
		else if (strInitPinType == "string" && !strDataType.empty())
		{
			if (strInitPinInitial.empty() == false)
			{
				dtype = GetDatatypeFromInitial(strDataType);
				shape = GetPartialShapeFromInitial(strDataShape);
				pOutput = new Variable(*pScope, shape, dtype, attrs);
			}
			else
			{
				std::string msg = string_format("waring : Variable init values empty - (%s). default value set 0", id.c_str());
				PrintMessage(msg);

				dtype = GetDatatypeFromInitial(strDataType);
				shape = GetPartialShapeFromInitial(strDataShape);
				pOutput = new Variable(*pScope, shape, dtype, attrs);
			}
		}
		// initvalues = 1.0f;2.0f, datatype = "", strInitPinType = "string, int, float, bool, double", strInitPinShape = [10][10]
		else if (strDataType.empty())
		{
			if (strInitPinType != "")
			{
				if (strInitPinType == "double")
					dtype = DT_DOUBLE;
				else if (strInitPinType == "float")
					dtype = DT_FLOAT;
				else if (strInitPinType == "int")
					dtype = DT_INT32;
				else if (strInitPinType == "bool")
					dtype = DT_BOOL;
				else if (strInitPinType == "string")
					dtype = DT_STRING;
			}

			if (dtype != DT_INVALID)
			{
				shape = GetPartialShapeFromInitial(strInitPinShape);
				pOutput = new Variable(*pScope, shape, dtype, attrs);
			}
			else
			{
				std::string msg = string_format("error : Variable(%s) Object create failed.", id.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("error : Variable(%s) Object create failed. dtype(%s).", id.c_str(), strDataType.c_str());
			PrintMessage(msg);
		}

		if (pOutput)
		{
			ObjectInfo* pObj = AddObjectMap(pOutput, id, SYMBOL_VARIABLE, "Variable", pInputItem);
			if (pObj)
				AddOutputInfo(pObj, &pOutput->ref, OUTPUT_TYPE_OUTPUT, "ref");
		}
	}
	else
	{
		std::string msg = string_format("error : Variable(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pOutput;
}

