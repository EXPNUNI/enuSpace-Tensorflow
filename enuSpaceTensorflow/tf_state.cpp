#include "stdafx.h"
#include "tf_state.h"


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
					attrs.ValidateShape(attrParser.GetValue_bool("validate_shape_"));
				if (attrParser.GetAttribute("use_locking_") != "")
					attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));
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
					attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));
			
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
					attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));

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
				var_name = strPinInitial;
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
					attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));

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
					attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));

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
					attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));

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
					attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));

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
					attrs.use_locking_ = attrParser.GetValue_bool("use_locking_");

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
					attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));

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
					attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));

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
					attrs.UseLocking(attrParser.GetValue_bool("use_locking_"));

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
	std::string temp;

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
					temp = attrParser.GetValue_StringPiece("var_name_");
					attrs.var_name_ = temp;
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
	tensorflow::DataType dtype = DT_DOUBLE;
	tensorflow::ops::Variable::Attrs attrs;


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
				if (strPinInitial != "")
				{
					shape = GetPartialShapeFromInitial(strPinInitial);
				}

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
				dtype = GetDatatypeFromInitial(strPinInitial);
				if (dtype == DT_INVALID)
				{
					std::string msg = string_format("warning : Variable - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
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
					attrs.Container(attrParser.GetValue_StringPiece("container_"));
				if (attrParser.GetAttribute("SharedName_") != "")
					attrs.Container(attrParser.GetValue_StringPiece("SharedName_"));
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
		
		pOutput = new Variable(*pScope, shape, dtype, attrs);
		ObjectInfo* pObj = AddObjectMap(pOutput, id, SYMBOL_VARIABLE, "Variable", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pOutput->ref, OUTPUT_TYPE_OUTPUT, "ref");
	}
	else
	{
		std::string msg = string_format("error : Variable(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pOutput;
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

void* Create_Const_ex(std::string id, Json::Value pInputItem)
{
	Scope* pScope = nullptr;
	Output* pOutput = new Output();
	Tensor* pTensor = nullptr;
	DataType dtype = DT_DOUBLE;
	std::string strVal = "";
	std::string strShape = "";
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
				std::string msg = string_format("warning : Const_ex - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				dtype = GetDatatypeFromInitial(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : Const_ex - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "val")
		{
			if (strInSymbolPinName == "" && strPinInterface == "Input::Initializer")
			{
				strVal = strPinInitial;
				strShape = strPinShape;
			}
			else
			{
				std::string msg = string_format("warning : Const_ex - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (strVal != "" && strShape !="")
	{
		std::vector<int64> array_slice;
		std::vector<int64> arraydims;
		GetArrayDimsFromShape(strShape, arraydims, array_slice);

		if (dtype == DT_DOUBLE)
		{
			std::vector<double> arrayvals;
			GetDoubleVectorFromInitial(strVal, arrayvals);

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
		else if (dtype == DT_FLOAT)
		{
			std::vector<float> arrayvals;
			GetFloatVectorFromInitial(strVal, arrayvals);

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
		else if (dtype == DT_INT32)
		{
			std::vector<int> arrayvals;
			GetIntVectorFromInitial(strVal, arrayvals);

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
		else if (dtype == DT_INT64)
		{
			std::vector<int64> arrayvals;
			GetInt64VectorFromInitial(strVal, arrayvals);

			gtl::ArraySlice< int64 > arraySlice(arraydims);
			pTensor = new Tensor(DT_INT64, TensorShape(arraySlice));

			int i = 0;
			for (std::vector<int64>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
			{
				pTensor->flat<int64>()(i) = *it;
				i++;
			}
			arraySlice.clear();
			arrayvals.clear();
		}
		else if (dtype == DT_BOOL)
		{
			std::vector<bool> arrayvals;
			GetBoolVectorFromInitial(strVal, arrayvals);

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
		else if (dtype == DT_STRING)
		{
			std::vector<std::string> arrayvals;
			GetStringVectorFromInitial(strVal, arrayvals);

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
			std::string msg = string_format("warning : Const_ex - %s(val-initvalue) transfer information missed.", id.c_str());
			PrintMessage(msg);
		}

		array_slice.clear();
		arraydims.clear();
	}

	if (pScope == nullptr)
	{
		std::string msg = string_format("warning : Const_ex - %s(scope) transfer information missed.", id.c_str());
		PrintMessage(msg);
	}
	if (pTensor == nullptr)
	{
		std::string msg = string_format("warning : Const_ex - %s(val) transfer information missed.", id.c_str());
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
		std::string msg = string_format("error : Const_ex(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}

	if (pTensor)
		delete pTensor;

	return pOutput;
}