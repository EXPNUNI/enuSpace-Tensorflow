#include "stdafx.h"
#include "tf_math.h"


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

void* Create_Abs(std::string id, Json::Value pInputItem) {
	Abs* pAbs = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Abs - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if(!strPinInitial.empty())
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType,"", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : Abs - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Abs pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pAbs = new Abs(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pAbs, id, SYMBOL_ABS, "Abs", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAbs->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Abs(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAbs;
}

void* Create_Acos(std::string id, Json::Value pInputItem) {
	Acos* pAcos = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Acos - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : Acos - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Acos pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pAcos = new Acos(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pAcos, id, SYMBOL_ACOS, "Acos", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAcos->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Acos(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAcos;
}

void* Create_Add(std::string id, Json::Value pInputItem) {
	Scope* pScope = nullptr;
	Output* pX = nullptr;
	Output* pY = nullptr;
	Add* pAdd = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Add - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pX = (Output*)pOutputObj->pOutput;
						}
					}
					// pX = pObj->pOutput;
				}
				else
				{
					if (!strPinInitial.empty())
						pX = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : Add - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pY = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pY = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : Add - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Add pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pX && pY)
	{
		pAdd = new Add(*pScope, *pX, *pY);
		ObjectInfo* pObj = AddObjectMap(pAdd, id, SYMBOL_ADD, "Add", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAdd->z, OUTPUT_TYPE_OUTPUT, "z");
		}
	}
	else
	{
		std::string msg = string_format("error : Add(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAdd;
}

void* Create_AddN(std::string id, Json::Value pInputItem) {
	AddN* pAddN = nullptr;
	Scope* pScope = nullptr;
	OutputList* pinputs = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : AddN - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pinputs = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pinputs = (OutputList*)Create_StrToOutputList(*m_pScope,strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : AddN - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : AddN pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinputs)
	{
		if (pScope->ok())
		{
			pAddN = new AddN(*pScope, *pinputs);
			ObjectInfo* pObj = AddObjectMap(pAddN, id, SYMBOL_ADDN, "AddN", pInputItem);
			if (pObj)
			{
				AddOutputInfo(pObj, &pAddN->sum, OUTPUT_TYPE_OUTPUT, "sum");
			}

			if (pScope->ok() == false)
			{
				std::string msg = string_format("error : Create AddN(%s) Object create failed. (Scope fail)", id.c_str());
				PrintMessage(msg);
			}
		}
	}
	else
	{
		std::string msg = string_format("error : AddN(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAddN;
}

void* Create_All(std::string id, Json::Value pInputItem) {
	All* pAll = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* paxis = nullptr;
	All::Attrs attrs;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : All - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pinput = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : All - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							paxis = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						paxis = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : All - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "All::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("keep_dims_") != "") attrs = attrs.KeepDims(attrParser.ConvStrToBool(attrParser.GetAttribute("keep_dims_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : All pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput && paxis)
	{
		if (pScope->ok())
		{
			pAll = new All(*pScope, *pinput, *paxis, attrs);
			ObjectInfo* pObj = AddObjectMap(pAll, id, SYMBOL_ALL, "All", pInputItem);
			if (pObj)
			{
				AddOutputInfo(pObj, &pAll->output, OUTPUT_TYPE_OUTPUT, "output");
			}

			if (pScope->ok() == false)
			{
				std::string msg = string_format("error : Create All(%s) Object create failed. (Scope fail)", id.c_str());
				PrintMessage(msg);
			}
		}
	}
	else
	{
		std::string msg = string_format("error : All(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAll;
}

void* Create_Any(std::string id, Json::Value pInputItem) {
	Any* pAny = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* paxis = nullptr;
	Any::Attrs attrs;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Any - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				else
				{
					if (!strPinInitial.empty())
					{
						pinput = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Any - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							paxis = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						paxis = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : Any - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Any::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("keep_dims_") != "") attrs = attrs.KeepDims(attrParser.ConvStrToBool(attrParser.GetAttribute("keep_dims_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : Any pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput && paxis)
	{
		pAny = new Any(*pScope, *pinput, *paxis, attrs);
		ObjectInfo* pObj = AddObjectMap(pAny, id, SYMBOL_ANY, "Any", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAny->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : Any(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAny;
}

void* Create_ApproximateEqual(std::string id, Json::Value pInputItem) {
	ApproximateEqual* pApproximateEqual = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;
	ApproximateEqual::Attrs attrs;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ApproximateEqual - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApproximateEqual - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApproximateEqual - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ApproximateEqual::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("tolerance_") != "") attrs = attrs.Tolerance(attrParser.ConvStrToFloat(attrParser.GetAttribute("tolerance_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ApproximateEqual pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pApproximateEqual = new ApproximateEqual(*pScope, *px, *py, attrs);
		ObjectInfo* pObj = AddObjectMap(pApproximateEqual, id, SYMBOL_APPROXIMATEEQUAL, "ApproximateEqual", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pApproximateEqual->z, OUTPUT_TYPE_OUTPUT, "z");
		}
	}
	else
	{
		std::string msg = string_format("error : ApproximateEqual(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pApproximateEqual;
}

void* Create_ArgMax(std::string id, Json::Value pInputItem) {
	ArgMax* pArgMax = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* pdimension = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ArgMax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				else
				{
					if (!strPinInitial.empty())
					{
						pinput = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ArgMax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "dimension")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pdimension = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pdimension = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ArgMax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : ArgMax pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput && pdimension)
	{
		pArgMax = new ArgMax(*pScope, *pinput, *pdimension);
		ObjectInfo* pObj = AddObjectMap(pArgMax, id, SYMBOL_ARGMAX, "ArgMax", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pArgMax->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : ArgMax(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pArgMax;
}

void* Create_ArgMin(std::string id, Json::Value pInputItem) {
	ArgMin* pArgMin = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* pdimension = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ArgMin - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				else
				{
					if (!strPinInitial.empty())
					{
						pinput = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ArgMin - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "dimension")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pdimension = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pdimension = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ArgMin - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : ArgMin pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput && pdimension)
	{
		pArgMin = new ArgMin(*pScope, *pinput, *pdimension);
		ObjectInfo* pObj = AddObjectMap(pArgMin, id, SYMBOL_ARGMIN, "ArgMin", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pArgMin->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : ArgMin(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pArgMin;
}

void* Create_Asin(std::string id, Json::Value pInputItem) {
	Asin* pAsin = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Asin - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Asin - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Asin pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pAsin = new Asin(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pAsin, id, SYMBOL_ASIN, "Asin", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAsin->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Asin(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAsin;
}

void* Create_Atan(std::string id, Json::Value pInputItem) {
	Atan* pAtan = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : pAtan - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : pAtan - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : pAtan pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pAtan = new Atan(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pAtan, id, SYMBOL_ATAN, "Atan", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAtan->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : pAtan(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAtan;
}

void* Create_Atan2(std::string id, Json::Value pInputItem) {
	Atan2* pAtan2 = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : pAtan2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : pAtan2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : pAtan2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : pAtan2 pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pAtan2 = new Atan2(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pAtan2, id, SYMBOL_ATAN2, "pAtan2", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAtan2->z, OUTPUT_TYPE_OUTPUT, "z");
		}
	}
	else
	{
		std::string msg = string_format("error : pAtan2(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAtan2;
}

void* Create_BatchMatMul(std::string id, Json::Value pInputItem) {
	BatchMatMul* pBatchMatMul = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;
	BatchMatMul::Attrs attrs;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : BatchMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : BatchMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : BatchMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "BatchMatMul::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("adj_x_") != "") attrs = attrs.AdjX(attrParser.ConvStrToBool(attrParser.GetAttribute("adj_x_")));
				if (attrParser.GetAttribute("adj_y_") != "") attrs = attrs.AdjY(attrParser.ConvStrToBool(attrParser.GetAttribute("adj_y_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : BatchMatMul pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pBatchMatMul = new BatchMatMul(*pScope, *px, *py, attrs);
		ObjectInfo* pObj = AddObjectMap(pBatchMatMul, id, SYMBOL_BATCHMATMUL, "BatchMatMul", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pBatchMatMul->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : BatchMatMul(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pBatchMatMul;
}

void* Create_Betainc(std::string id, Json::Value pInputItem) {
	Betainc* pBetainc = nullptr;
	Scope* pScope = nullptr;
	Output* pa = nullptr;
	Output* pb = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Betainc - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pa = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pa = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Betainc - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pb = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pb = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Betainc - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Betainc - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Betainc pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && pa && pb)
	{
		pBetainc = new Betainc(*pScope,  *pa, *pb, *px);
		ObjectInfo* pObj = AddObjectMap(pBetainc, id, SYMBOL_BETAINC, "Betainc", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pBetainc->z, OUTPUT_TYPE_OUTPUT, "z");
		}
	}
	else
	{
		std::string msg = string_format("error : Betainc(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pBetainc;
}

void* Create_Bincount(std::string id, Json::Value pInputItem) {
	Bincount* pBincount = nullptr;
	Scope* pScope = nullptr;
	Output* parr = nullptr;
	Output* psize = nullptr;
	Output* pweights = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Bincount - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "arr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							parr = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						parr = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Bincount - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							psize = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						psize = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Bincount - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "weights")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pweights = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pweights = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Bincount - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Bincount pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && parr && psize && pweights)
	{
		pBincount = new Bincount(*pScope, *parr, *psize, *pweights);
		ObjectInfo* pObj = AddObjectMap(pBincount, id, SYMBOL_BINCOUNT, "Bincount", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pBincount->bins, OUTPUT_TYPE_OUTPUT, "bins");
		}
	}
	else
	{
		std::string msg = string_format("error : Bincount(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pBincount;
}

void* Create_Bucketize(std::string id, Json::Value pInputItem) {
	Bucketize* pBucketize = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	std::vector<float> v_boundaries;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Bucketize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				else
				{
					if (!strPinInitial.empty())
					{
						pinput = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Bucketize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "boundaries")
		{
			if (strPinInterface == "gtl::ArraySlice<float>")
			{
				GetFloatVectorFromInitial(strPinInitial, v_boundaries);
			}
			else
			{
				std::string msg = string_format("warning : Bucketize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Bucketize pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput)
	{
		gtl::ArraySlice<float> boundaries(v_boundaries);

		pBucketize = new Bucketize(*pScope, *pinput, boundaries);
		ObjectInfo* pObj = AddObjectMap(pBucketize, id, SYMBOL_BUCKETIZE, "Bucketize", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pBucketize->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		// JIWOO boundaries.clear();
	}
	else
	{
		std::string msg = string_format("error : Bucketize(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_boundaries.clear();
	return pBucketize;
}

void* Create_Cast(std::string id, Json::Value pInputItem) {
	Cast* pCast = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	DataType DstT;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Cast - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Cast - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "DstT")
		{
			if (strPinInterface == "DataType")
			{
				if (!(DstT = GetDatatypeFromInitial(strPinInitial)))
				{
					std::string msg = string_format("warning : Cast - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : Cast - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Cast pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pCast = new Cast(*pScope, *px, DstT);
		ObjectInfo* pObj = AddObjectMap(pCast, id, SYMBOL_CAST, "Cast", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pCast->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Cast(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pCast;
}

void* Create_Ceil(std::string id, Json::Value pInputItem) {
	Ceil* pCeil = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Ceil - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Ceil - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Ceil pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pCeil = new Ceil(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pCeil, id, SYMBOL_CEIL, "Ceil", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pCeil->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Ceil(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pCeil;
}

void* Create_Complex(std::string id, Json::Value pInputItem) {
	Complex* pComplex = nullptr;
	Scope* pScope = nullptr;
	Output* preal = nullptr;
	Output* pimag = nullptr;
	Complex::Attrs attrs;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Complex - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "real")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							preal = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						preal = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Complex - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "imag")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pimag = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pimag = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Complex - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Complex::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("Tout_") != "") attrs = attrs.Tout(attrParser.ConvStrToDataType(attrParser.GetAttribute("Tout_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : Complex pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && preal && pimag)
	{
		pComplex = new Complex(*pScope, *preal, *pimag, attrs);
		ObjectInfo* pObj = AddObjectMap(pComplex, id, SYMBOL_COMPLEX, "Complex", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pComplex->out, OUTPUT_TYPE_OUTPUT, "out");
		}
	}
	else
	{
		std::string msg = string_format("error : Complex(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pComplex;
}

void* Create_ComplexAbs(std::string id, Json::Value pInputItem) {
	ComplexAbs* pComplexAbs = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	ComplexAbs::Attrs attrs;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ComplexAbs - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ComplexAbs - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ComplexAbs::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("Tout_") != "") attrs = attrs.Tout(attrParser.ConvStrToDataType(attrParser.GetAttribute("Tout_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ComplexAbs pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pComplexAbs = new ComplexAbs(*pScope, *px, attrs);
		ObjectInfo* pObj = AddObjectMap(pComplexAbs, id, SYMBOL_COMPLEXABS, "ComplexAbs", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pComplexAbs->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : ComplexAbs(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pComplexAbs;
}

void* Create_Conj(std::string id, Json::Value pInputItem) {
	Conj* pConj = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : pConj - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				else
				{
					if (!strPinInitial.empty())
					{
						pinput = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : pConj - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : pConj pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput)
	{
		pConj = new Conj(*pScope, *pinput);
		ObjectInfo* pObj = AddObjectMap(pConj, id, SYMBOL_CONJ, "Conj", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pConj->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : pConj(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pConj;
}

void* Create_Cos(std::string id, Json::Value pInputItem) {
	Cos* pCos = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Cos - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Cos - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Cos pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pCos = new Cos(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pCos, id, SYMBOL_COS, "Cos", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pCos->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Cos(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pCos;
}
void* Create_Cosh(std::string id, Json::Value pInputItem) {
	Cosh* pCosh = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Cosh - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Cosh - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Cosh pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pCosh = new Cosh(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pCosh, id, SYMBOL_COSH, "Cosh", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pCosh->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Cosh(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pCosh;
}

void* Create_Cross(std::string id, Json::Value pInputItem) {
	Cross* pCross = nullptr;
	Scope* pScope = nullptr;
	Output* pa = nullptr;
	Output* pb = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Cross - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pa = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pa = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Cross - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pb = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pb = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Cross - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Cross pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pa && pb)
	{
		pCross = new Cross(*pScope, *pa, *pb);
		ObjectInfo* pObj = AddObjectMap(pCross, id, SYMBOL_CROSS, "Cross", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pCross->product, OUTPUT_TYPE_OUTPUT, "product");
		}
	}
	else
	{
		std::string msg = string_format("error : Cross(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pCross;
}

void* Create_Cumprod(std::string id, Json::Value pInputItem) {
	Cumprod* pCumprod = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* paxis = nullptr;
	Cumprod::Attrs attrs;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Cumprod - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Cumprod - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							paxis = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						paxis = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Cumprod - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Cumprod::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("exclusive_") != "") attrs = attrs.Exclusive(attrParser.ConvStrToBool(attrParser.GetAttribute("exclusive_")));
				if (attrParser.GetAttribute("reverse_") != "") attrs = attrs.Reverse(attrParser.ConvStrToBool(attrParser.GetAttribute("reverse_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : Cumprod pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && paxis)
	{
		pCumprod = new Cumprod(*pScope, *px, *paxis, attrs);
		ObjectInfo* pObj = AddObjectMap(pCumprod, id, SYMBOL_CUMPROD, "Cumprod", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pCumprod->out, OUTPUT_TYPE_OUTPUT, "out");
		}
	}
	else
	{
		std::string msg = string_format("error : Cumprod(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pCumprod;
}

void* Create_Cumsum(std::string id, Json::Value pInputItem) {
	Cumsum* pCumsum = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* paxis = nullptr;
	Cumsum::Attrs attrs;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Cumsum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Cumsum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							paxis = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						paxis = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Cumsum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Cumsum::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("exclusive_") != "") attrs = attrs.Exclusive(attrParser.ConvStrToBool(attrParser.GetAttribute("exclusive_")));
				if (attrParser.GetAttribute("reverse_") != "") attrs = attrs.Reverse(attrParser.ConvStrToBool(attrParser.GetAttribute("reverse_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : Cumsum pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && paxis)
	{
		pCumsum = new Cumsum(*pScope, *px, *paxis, attrs);
		ObjectInfo* pObj = AddObjectMap(pCumsum, id, SYMBOL_CUMSUM, "Cumsum", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pCumsum->out, OUTPUT_TYPE_OUTPUT, "out");
		}
	}
	else
	{
		std::string msg = string_format("error : Cumsum(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pCumsum;
}

void* Create_Digamma(std::string id, Json::Value pInputItem) {
	Digamma* pDigamma = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Digamma - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Digamma - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Digamma pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pDigamma = new Digamma(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pDigamma, id, SYMBOL_DIGAMMA, "Digamma", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pDigamma->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Digamma(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pDigamma;
}

void* Create_Div(std::string id, Json::Value pInputItem) {
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;
	Div* pDiv = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Div - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Div - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Div - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Div pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pDiv = new Div(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pDiv, id, SYMBOL_DIV, "Div", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pDiv->z, OUTPUT_TYPE_OUTPUT, "z");
	}
	else
	{
		std::string msg = string_format("error : Div(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pDiv;
}

void* Create_Equal(std::string id, Json::Value pInputItem) {
	Equal* pEqual = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Equal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Equal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Equal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Equal pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pEqual = new Equal(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pEqual, id, SYMBOL_EQUAL, "Equal", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pEqual->z, OUTPUT_TYPE_OUTPUT, "z");
		}
	}
	else
	{
		std::string msg = string_format("error : Equal(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pEqual;
}

void* Create_Erf(std::string id, Json::Value pInputItem) {
	Erf* pErf = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Erf - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Erf - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Erf pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pErf = new Erf(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pErf, id, SYMBOL_ERF, "Erf", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pErf->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Erf(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pErf;
}

void* Create_Erfc(std::string id, Json::Value pInputItem) {
	Erfc* pErfc = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Erfc - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Erfc - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Erfc pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pErfc = new Erfc(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pErfc, id, SYMBOL_ERFC, "Erfc", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pErfc->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Erfc(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pErfc;
}

void* Create_Exp(std::string id, Json::Value pInputItem) {
	Exp* pExp = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Exp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Exp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Exp pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pExp = new Exp(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pExp, id, SYMBOL_EXP, "Exp", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pExp->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Exp(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pExp;
}

void* Create_Expm1(std::string id, Json::Value pInputItem) {
	Expm1* pExpm1 = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Expm1 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Expm1 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Expm1 pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pExpm1 = new Expm1(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pExpm1, id, SYMBOL_EXPM1, "Expm1", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pExpm1->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Expm1(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pExpm1;
}

void* Create_Floor(std::string id, Json::Value pInputItem) {
	Floor* pFloor = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Floor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Floor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Floor pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pFloor = new Floor(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pFloor, id, SYMBOL_FLOOR, "Floor", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pFloor->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Floor(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pFloor;
}

void* Create_FloorDiv(std::string id, Json::Value pInputItem) {
	FloorDiv* pFloorDiv = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : FloorDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : FloorDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : FloorDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : FloorDiv pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pFloorDiv = new FloorDiv(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pFloorDiv, id, SYMBOL_FLOORDIV, "FloorDiv", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pFloorDiv->z, OUTPUT_TYPE_OUTPUT, "z");
		}
	}
	else
	{
		std::string msg = string_format("error : FloorDiv(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pFloorDiv;
}

void* Create_FloorMod(std::string id, Json::Value pInputItem) {
	FloorMod* pFloorMod = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py= nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : FloorMod - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : FloorMod - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : FloorMod - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : FloorMod pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pFloorMod = new FloorMod(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pFloorMod, id, SYMBOL_FLOORMOD, "FloorMod", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pFloorMod->z, OUTPUT_TYPE_OUTPUT, "z");
		}
	}
	else
	{
		std::string msg = string_format("error : FloorMod(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pFloorMod;
}

void* Create_Greater(std::string id, Json::Value pInputItem) {
	Greater* pGreater = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Greater - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Greater - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Greater - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Greater pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pGreater = new Greater(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pGreater, id, SYMBOL_GREATER, "Greater", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pGreater->z, OUTPUT_TYPE_OUTPUT, "z");
		}
	}
	else
	{
		std::string msg = string_format("error : Greater(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pGreater;
}

void* Create_GreaterEqual(std::string id, Json::Value pInputItem) {
	GreaterEqual* pGreaterEqual = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : GreaterEqual - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : GreaterEqual - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : GreaterEqual - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : GreaterEqual pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pGreaterEqual = new GreaterEqual(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pGreaterEqual, id, SYMBOL_GREATEREQUAL, "GreaterEqual", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pGreaterEqual->z, OUTPUT_TYPE_OUTPUT, "z");
		}
	}
	else
	{
		std::string msg = string_format("error : GreaterEqual(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pGreaterEqual;
}

void* Create_Igamma(std::string id, Json::Value pInputItem) {
	Igamma* pIgamma = nullptr;
	Scope* pScope = nullptr;
	Output* pa = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Igamma - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pa = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pa = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Igamma - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Igamma - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Igamma pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && pa)
	{
		pIgamma = new Igamma(*pScope, *pa, *px);
		ObjectInfo* pObj = AddObjectMap(pIgamma, id, SYMBOL_IGAMMA, "Igamma", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pIgamma->z, OUTPUT_TYPE_OUTPUT, "z");
		}
	}
	else
	{
		std::string msg = string_format("error : pIgamma(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pIgamma;
}

void* Create_Igammac(std::string id, Json::Value pInputItem) {
	Igammac* pIgammac = nullptr;
	Scope* pScope = nullptr;
	Output* pa = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : pIgammac - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pa = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pa = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : pIgammac - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : pIgammac - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : pIgammac pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && px)
	{
		pIgammac = new Igammac(*pScope, *pa, *px);
		ObjectInfo* pObj = AddObjectMap(pIgammac, id, SYMBOL_IGAMMAC, "pIgammac", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pIgammac->z, OUTPUT_TYPE_OUTPUT, "z");
		}
	}
	else
	{
		std::string msg = string_format("error : pIgammac(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pIgammac;
}

void* Create_Imag(std::string id, Json::Value pInputItem) {
	Imag* pImag = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Imag::Attrs attrs;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Imag - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				else
				{
					if (!strPinInitial.empty())
					{
						pinput = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Imag - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Imag::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("Tout_") != "") attrs = attrs.Tout(attrParser.ConvStrToDataType(attrParser.GetAttribute("Tout_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : Imag pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput)
	{
		pImag = new Imag(*pScope, *pinput, attrs);
		ObjectInfo* pObj = AddObjectMap(pImag, id, SYMBOL_IMAG, "Imag", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pImag->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : Imag(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pImag;
}
void* Create_IsFinite(std::string id, Json::Value pInputItem) {
	IsFinite* pIsFinite = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : IsFinite - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : IsFinite - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : IsFinite pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pIsFinite = new IsFinite(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pIsFinite, id, SYMBOL_ISFINITE, "IsFinite", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pIsFinite->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : IsFinite(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pIsFinite;
}
void* Create_IsInf(std::string id, Json::Value pInputItem) {
	IsInf* pIsInf = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : IsInf - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : IsInf - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : IsInf pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pIsInf = new IsInf(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pIsInf, id, SYMBOL_ISINF, "IsInf", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pIsInf->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : IsInf(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pIsInf;
}

void* Create_IsNan(std::string id, Json::Value pInputItem) {
	IsNan* pIsNan = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : IsNan - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : IsNan - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : IsNan pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pIsNan = new IsNan(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pIsNan, id, SYMBOL_ISNAN, "IsNan", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pIsNan->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : IsNan(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pIsNan;
}

void* Create_Less(std::string id, Json::Value pInputItem) {
	Less* pLess = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Less - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Less - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Less - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Less pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pLess = new Less(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pLess, id, SYMBOL_LESS, "Less", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pLess->z, OUTPUT_TYPE_OUTPUT, "z");
		}
	}
	else
	{
		std::string msg = string_format("error : Less(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pLess;
}

void* Create_LessEqual(std::string id, Json::Value pInputItem) {
	LessEqual* pLessEqual = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : LessEqual - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : LessEqual - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : LessEqual - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : LessEqual pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pLessEqual = new LessEqual(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pLessEqual, id, SYMBOL_LESSEQUAL, "LessEqual", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pLessEqual->z, OUTPUT_TYPE_OUTPUT, "z");
		}
	}
	else
	{
		std::string msg = string_format("error : LessEqual(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pLessEqual;
}

void* Create_Lgamma(std::string id, Json::Value pInputItem) {
	Lgamma* pLgamma = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Lgamma - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Lgamma - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Lgamma pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pLgamma = new Lgamma(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pLgamma, id, SYMBOL_LGAMMA, "Lgamma", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pLgamma->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Lgamma(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pLgamma;
}

void* Create_LinSpace(std::string id, Json::Value pInputItem) {
	LinSpace* pLinSpace = nullptr;
	Scope* pScope = nullptr;
	Output* pstart = nullptr;
	Output* pstop = nullptr;
	Output* pnum = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : LinSpace - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "start")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pstart = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pstart = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "",strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : LinSpace - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "stop")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pstop = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pstop = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : LinSpace - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pnum = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pnum = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : LinSpace - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : LinSpace pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pstart && pstop && pnum)
	{
		pLinSpace = new LinSpace(*pScope, *pstart, *pstop, *pnum);
		ObjectInfo* pObj = AddObjectMap(pLinSpace, id, SYMBOL_LINSPACE, "LinSpace", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pLinSpace->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : LinSpace(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pLinSpace;
}

void* Create_Log(std::string id, Json::Value pInputItem) {
	Log* pLog = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Log - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Log - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Log pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pLog = new Log(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pLog, id, SYMBOL_LOG, "Log", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pLog->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Log(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pLog;
}

void* Create_Log1p(std::string id, Json::Value pInputItem) {
	Log1p* pLog1p = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Log1p - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Log1p - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Log1p pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pLog1p = new Log1p(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pLog1p, id, SYMBOL_LOG1P, "Log1p", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pLog1p->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Log1p(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pLog1p;
}

void* Create_LogicalAnd(std::string id, Json::Value pInputItem) {
	LogicalAnd* pLogicalAnd = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : LogicalAnd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : LogicalAnd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : LogicalAnd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : LogicalAnd pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pLogicalAnd = new LogicalAnd(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pLogicalAnd, id, SYMBOL_LOGICALAND, "LogicalAnd", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pLogicalAnd->z, OUTPUT_TYPE_OUTPUT, "z");
		}
	}
	else
	{
		std::string msg = string_format("error : LogicalAnd(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pLogicalAnd;
}

void* Create_LogicalNot(std::string id, Json::Value pInputItem) {
	LogicalNot* pLogicalNot = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : LogicalNot - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : LogicalNot - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : LogicalNot pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pLogicalNot = new LogicalNot(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pLogicalNot, id, SYMBOL_LOGICALNOT, "LogicalNot", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pLogicalNot->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : LogicalNot(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pLogicalNot;
}

void* Create_LogicalOr(std::string id, Json::Value pInputItem) {
	LogicalOr* pLogicalOr = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : LogicalOr - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : LogicalOr - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : LogicalOr - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : LogicalOr pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pLogicalOr = new LogicalOr(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pLogicalOr, id, SYMBOL_LOGICALOR, "LogicalOr", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pLogicalOr->z, OUTPUT_TYPE_OUTPUT, "z");
		}
	}
	else
	{
		std::string msg = string_format("error : LogicalOr(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pLogicalOr;
}

void* Create_MatMul(std::string id, Json::Value pInputItem) {
	MatMul* pMatMul = nullptr;
	Scope* pScope = nullptr;
	Output* pa = nullptr;
	Output* pb = nullptr;
	MatMul::Attrs attrs;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : MatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pa = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pa = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : MatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pb = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pb = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : MatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "MatMul::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("transpose_a_") != "") attrs = attrs.TransposeA(attrParser.ConvStrToBool(attrParser.GetAttribute("transpose_a_")));
				if (attrParser.GetAttribute("transpose_b_") != "") attrs = attrs.TransposeB(attrParser.ConvStrToBool(attrParser.GetAttribute("transpose_b_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : MatMul pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pa && pb)
	{
		pMatMul = new MatMul(*pScope, *pa, *pb, attrs);
		ObjectInfo* pObj = AddObjectMap(pMatMul, id, SYMBOL_MATMUL, "MatMul", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pMatMul->product, OUTPUT_TYPE_OUTPUT, "product");
	}
	else
	{
		std::string msg = string_format("error : MatMul(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pMatMul;
}

void* Create_Max(std::string id, Json::Value pInputItem) {
	Max* pMax = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* paxis = nullptr;
	Max::Attrs attrs;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Max - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				else
				{
					if (!strPinInitial.empty())
					{
						pinput = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Max - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							paxis = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						paxis = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Max - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Max::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if(attrParser.GetAttribute("keep_dims_") != "") attrs = attrs.KeepDims(attrParser.ConvStrToBool(attrParser.GetAttribute("keep_dims_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : Max pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput && paxis)
	{
		pMax = new Max(*pScope, *pinput, *paxis, attrs);
		ObjectInfo* pObj = AddObjectMap(pMax, id, SYMBOL_MAX, "Max", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pMax->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("error : Max(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pMax;
}

void* Create_Maximum(std::string id, Json::Value pInputItem) {
	Maximum* pMaximum = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Maximum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Maximum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Maximum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Maximum pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pMaximum = new Maximum(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pMaximum, id, SYMBOL_MAXIMUM, "Maximum", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pMaximum->z, OUTPUT_TYPE_OUTPUT, "z");
	}
	else
	{
		std::string msg = string_format("error : Maximum(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pMaximum;
}

void* Create_Mean(std::string id, Json::Value pInputItem) {
	Mean* pMean = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* paxis = nullptr;
	Mean::Attrs attrs;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Mean - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				else
				{
					if (!strPinInitial.empty())
					{
						pinput = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Mean - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							paxis = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						paxis = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Mean - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Mean::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("keep_dims_") != "") attrs = attrs.KeepDims(attrParser.ConvStrToBool(attrParser.GetAttribute("keep_dims_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : Mean pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput && paxis)
	{
		pMean = new Mean(*pScope, *pinput, *paxis, attrs);

		ObjectInfo* pObj = AddObjectMap(pMean, id, SYMBOL_MEAN, "Mean", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pMean->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("error : Mean(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pMean;
}

void* Create_Min(std::string id, Json::Value pInputItem) {
	Min* pMin = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* paxis = nullptr;
	Min::Attrs attrs;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Max - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				else
				{
					if (!strPinInitial.empty())
					{
						pinput = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Min - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							paxis = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						paxis = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Min - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Min::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("keep_dims_") != "") attrs = attrs.KeepDims(attrParser.ConvStrToBool(attrParser.GetAttribute("keep_dims_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : Min pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput && paxis)
	{
		pMin = new Min(*pScope, *pinput, *paxis, attrs);
		ObjectInfo* pObj = AddObjectMap(pMin, id, SYMBOL_MIN, "Min", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pMin->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("error : Min(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pMin;
}

void* Create_Minimum(std::string id, Json::Value pInputItem) {
	Minimum* pMinimum = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Minimum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Minimum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Minimum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Minimum pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pMinimum = new Minimum(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pMinimum, id, SYMBOL_MINIMUM, "Minimum", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pMinimum->z, OUTPUT_TYPE_OUTPUT, "z");
	}
	else
	{
		std::string msg = string_format("error : Minimum(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pMinimum;
}

void* Create_Mod(std::string id, Json::Value pInputItem) {
	Mod* pMod = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Mod - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Mod - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Mod - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Mod pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pMod = new Mod(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pMod, id, SYMBOL_MOD, "Mod", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pMod->z, OUTPUT_TYPE_OUTPUT, "z");
	}
	else
	{
		std::string msg = string_format("error : Mod(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pMod;
}

void* Create_Multiply(std::string id, Json::Value pInputItem) {
	Multiply* pMultiply = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Multiply - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Multiply - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Multiply - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Multiply pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pMultiply = new Multiply(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pMultiply, id, SYMBOL_MULTIPLY, "Multiply", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pMultiply->z, OUTPUT_TYPE_OUTPUT, "z");
	}
	else
	{
		std::string msg = string_format("error : Multiply(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pMultiply;;
}

void* Create_Negate(std::string id, Json::Value pInputItem) {
	Negate* pNegate = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Negate - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Negate - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Negate pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pNegate = new Negate(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pNegate, id, SYMBOL_NEGATE, "Negate", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pNegate->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Negate(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pNegate;
}

void* Create_NotEqual(std::string id, Json::Value pInputItem) {
	NotEqual* pNotEqual = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : NotEqual - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : NotEqual - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : NotEqual - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : NotEqual pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pNotEqual = new NotEqual(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pNotEqual, id, SYMBOL_NOTEQUAL, "NotEqual", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pNotEqual->z, OUTPUT_TYPE_OUTPUT, "z");
	}
	else
	{
		std::string msg = string_format("error : NotEqual(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pNotEqual;
}

void* Create_Polygamma(std::string id, Json::Value pInputItem) {
	Polygamma* pPolygamma = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Polygamma - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Polygamma - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Polygamma - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Polygamma pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pPolygamma = new Polygamma(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pPolygamma, id, SYMBOL_POLYGAMMA, "Polygamma", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pPolygamma->z, OUTPUT_TYPE_OUTPUT, "z");
	}
	else
	{
		std::string msg = string_format("error : Polygamma(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pPolygamma;
}

void* Create_Pow(std::string id, Json::Value pInputItem) {
	Pow* pPow = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Pow - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Pow - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Pow - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Pow pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pPow = new Pow(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pPow, id, SYMBOL_POW, "Pow", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pPow->z, OUTPUT_TYPE_OUTPUT, "z");
	}
	else
	{
		std::string msg = string_format("error : Pow(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pPow;
}

void* Create_Prod(std::string id, Json::Value pInputItem) {
	Prod* pProd = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* paxis = nullptr;
	Prod::Attrs attrs;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Prod - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				else
				{
					if (!strPinInitial.empty())
					{
						pinput = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Prod - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							paxis = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						paxis = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Prod - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Prod::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("keep_dims_") != "") attrs = attrs.KeepDims(attrParser.ConvStrToBool(attrParser.GetAttribute("keep_dims_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : Prod pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput && paxis)
	{
		pProd = new Prod(*pScope, *pinput, *paxis, attrs);
		ObjectInfo* pObj = AddObjectMap(pProd, id, SYMBOL_PROD, "Prod", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pProd->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("error : Prod(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pProd;
}

void* Create_QuantizeDownAndShrinkRange(std::string id, Json::Value pInputItem) {
	QuantizeDownAndShrinkRange* pQuantizeDownAndShrinkRange = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* pinput_min = nullptr;
	Output* pinput_max = nullptr;
	DataType out_type;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : QuantizeDownAndShrinkRange - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				else
				{
					if (!strPinInitial.empty())
					{
						pinput = (Output*)Create_StrToOutput(*m_pScope,"DT_QINT16", "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizeDownAndShrinkRange - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pinput_min = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pinput_min = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizeDownAndShrinkRange - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pinput_max = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pinput_max = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizeDownAndShrinkRange - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "out_type")
		{
			if (strPinInterface == "DataType")
			{
				if (!(out_type = GetDatatypeFromInitial(strPinInitial)))
				{
					std::string msg = string_format("warning : QuantizeDownAndShrinkRange - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizeDownAndShrinkRange - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : QuantizeDownAndShrinkRange pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput && pinput_min && pinput_max)
	{
		pQuantizeDownAndShrinkRange = new QuantizeDownAndShrinkRange(*pScope, *pinput, *pinput_min, *pinput_max, out_type);
		ObjectInfo* pObj = AddObjectMap(pQuantizeDownAndShrinkRange, id, SYMBOL_QUANTIZEDOWNANDSHRINKRANGE, "QuantizeDownAndShrinkRange", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pQuantizeDownAndShrinkRange->output, OUTPUT_TYPE_OUTPUT, "output");
			AddOutputInfo(pObj, &pQuantizeDownAndShrinkRange->output_min, OUTPUT_TYPE_OUTPUT, "output_min");
			AddOutputInfo(pObj, &pQuantizeDownAndShrinkRange->output_max, OUTPUT_TYPE_OUTPUT, "output_max");
		}
	}
	else
	{
		std::string msg = string_format("error : QuantizeDownAndShrinkRange(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pQuantizeDownAndShrinkRange;
}

void* Create_QuantizedMatMul(std::string id, Json::Value pInputItem) {
	QuantizedMatMul* pQuantizedMatMul = nullptr;
	Scope* pScope = nullptr;
	Output* pa = nullptr;
	Output* pb = nullptr;
	Output* pmin_a = nullptr;
	Output* pmax_a = nullptr;
	Output* pmin_b = nullptr;
	Output* pmax_b = nullptr;
	QuantizedMatMul::Attrs attrs;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : QuantizedMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pa = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pa = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizedMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pb = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pb = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizedMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "min_a")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pmin_a = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pmin_a = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizedMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "max_a")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pmax_a = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pmax_a = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizedMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "min_b")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pmin_b = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pmin_b = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizedMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "max_b")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pmax_b = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pmax_b = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizedMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QuantizedMatMul::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("Toutput_") != "") attrs = attrs.Toutput(attrParser.ConvStrToDataType(attrParser.GetAttribute("Toutput_")));
				if (attrParser.GetAttribute("transpose_a_") != "") attrs = attrs.TransposeA(attrParser.ConvStrToBool(attrParser.GetAttribute("transpose_a_")));
				if (attrParser.GetAttribute("transpose_b_") != "") attrs = attrs.TransposeB(attrParser.ConvStrToBool(attrParser.GetAttribute("transpose_b_")));
				if (attrParser.GetAttribute("Tactivation_") != "") attrs = attrs.Tactivation(attrParser.ConvStrToDataType(attrParser.GetAttribute("Tactivation_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : QuantizedMatMul pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pa && pb && pmin_a && pmax_a && pmin_b && pmax_b)
	{
		pQuantizedMatMul = new QuantizedMatMul(*pScope, *pa, *pb, *pmin_a, *pmax_a, *pmin_b, *pmax_b, attrs);
		ObjectInfo* pObj = AddObjectMap(pQuantizedMatMul, id, SYMBOL_QUANTIZEDMATMUL, "QuantizedMatMul", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pQuantizedMatMul->out, OUTPUT_TYPE_OUTPUT, "out");
			AddOutputInfo(pObj, &pQuantizedMatMul->min_out, OUTPUT_TYPE_OUTPUT, "min_out");
			AddOutputInfo(pObj, &pQuantizedMatMul->max_out, OUTPUT_TYPE_OUTPUT, "max_out");
		}
	}
	else
	{
		std::string msg = string_format("error : QuantizedMatMul(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pQuantizedMatMul;
}

void* Create_QuantizedMul(std::string id, Json::Value pInputItem) {
	QuantizedMul* pQuantizedMul = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;
	Output* pmin_x = nullptr;
	Output* pmax_x = nullptr;
	Output* pmin_y = nullptr;
	Output* pmax_y = nullptr;
	QuantizedMul::Attrs attrs;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : QuantizedMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizedMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizedMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "min_x")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pmin_x = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pmin_x = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizedMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "max_x")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pmax_x = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pmax_x = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizedMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "min_y")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pmin_y = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pmin_y = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizedMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "max_y")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pmax_y = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pmax_y = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizedMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QuantizedMul::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("Toutput_") != "") attrs = attrs.Toutput(attrParser.ConvStrToDataType(attrParser.GetAttribute("Toutput_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : QuantizedMatMul pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py && pmin_x && pmax_x && pmin_y && pmax_y)
	{
		pQuantizedMul = new QuantizedMul(*pScope, *px, *py, *pmin_x, *pmax_x, *pmin_y, *pmax_y, attrs);
		ObjectInfo* pObj = AddObjectMap(pQuantizedMul, id, SYMBOL_QUANTIZEDMUL, "QuantizedMul", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pQuantizedMul->z, OUTPUT_TYPE_OUTPUT, "z");
			AddOutputInfo(pObj, &pQuantizedMul->min_z, OUTPUT_TYPE_OUTPUT, "min_z");
			AddOutputInfo(pObj, &pQuantizedMul->max_z, OUTPUT_TYPE_OUTPUT, "max_z");
		}
	}
	else
	{
		std::string msg = string_format("error : QuantizedMul(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pQuantizedMul;
}

void* Create_Range(std::string id, Json::Value pInputItem) {
	Range* pRange = nullptr;
	Scope* pScope = nullptr;
	Output* pstart = nullptr;
	Output* plimit = nullptr;
	Output* pdelta = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Range - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "start")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pstart = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pstart = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Range - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "limit")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							plimit = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						plimit = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Range - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "delta")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pdelta = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pdelta = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Range - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Range pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pstart && plimit && pdelta )
	{
		pRange = new Range(*pScope, *pstart, *plimit, *pdelta);
		ObjectInfo* pObj = AddObjectMap(pRange, id, SYMBOL_RANGE, "Range", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pRange->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : Range(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRange;
}

void* Create_Real(std::string id, Json::Value pInputItem) {
	Real* pReal = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Real::Attrs attrs;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Real - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				else
				{
					if (!strPinInitial.empty())
					{
						pinput = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Real - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Real::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("Tout_") != "") attrs = attrs.Tout(attrParser.ConvStrToDataType(attrParser.GetAttribute("Tout_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : Real pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput)
	{
		pReal = new Real(*pScope, *pinput, attrs);
		ObjectInfo* pObj = AddObjectMap(pReal, id, SYMBOL_REAL, "Real", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pReal->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : Real(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pReal;
}

void* Create_RealDiv(std::string id, Json::Value pInputItem) {
	RealDiv* pRealDiv = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : RealDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : RealDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : RealDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : RealDiv pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pRealDiv = new RealDiv(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pRealDiv, id, SYMBOL_REALDIV, "RealDiv", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pRealDiv->z, OUTPUT_TYPE_OUTPUT, "z");
	}
	else
	{
		std::string msg = string_format("error : RealDiv(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRealDiv;
}

void* Create_Reciprocal(std::string id, Json::Value pInputItem) {
	Reciprocal* pReciprocal = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Reciprocal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Reciprocal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Reciprocal pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pReciprocal = new Reciprocal(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pReciprocal, id, SYMBOL_RECIPROCAL, "Reciprocal", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pReciprocal->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Reciprocal(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pReciprocal;
}

void* Create_RequantizationRange(std::string id, Json::Value pInputItem) {
	RequantizationRange* pRequantizationRange = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* pinput_min = nullptr;
	Output* pinput_max = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : RequantizationRange - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				else
				{
					if (!strPinInitial.empty())
					{
						pinput = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : RequantizationRange - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pinput_min = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pinput_min = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : RequantizationRange - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pinput_max = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pinput_max = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : RequantizationRange - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : RequantizationRange pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput && pinput_min && pinput_max)
	{
		pRequantizationRange = new RequantizationRange(*pScope, *pinput, *pinput_min, *pinput_max);
		ObjectInfo* pObj = AddObjectMap(pRequantizationRange, id, SYMBOL_REQUANTIZATIONRANGE, "RequantizationRange", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pRequantizationRange->output_min, OUTPUT_TYPE_OUTPUT, "output_min");
			AddOutputInfo(pObj, &pRequantizationRange->output_max, OUTPUT_TYPE_OUTPUT, "output_max");
		}
	}
	else
	{
		std::string msg = string_format("error : RequantizationRange(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRequantizationRange;
}

void* Create_Requantize(std::string id, Json::Value pInputItem) {
	Requantize* pRequantize = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* pinput_min = nullptr;
	Output* pinput_max = nullptr;
	Output* prequested_output_min = nullptr;
	Output* prequested_output_max = nullptr;
	DataType out_type;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Requantize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				else
				{
					if (!strPinInitial.empty())
					{
						pinput = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Requantize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pinput_min = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pinput_min = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Requantize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pinput_max = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pinput_max = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Requantize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "requested_output_min")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							prequested_output_min = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						prequested_output_min = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Requantize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "requested_output_max")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							prequested_output_max = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						prequested_output_max = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Requantize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "out_type")
		{
			if (strPinInterface == "DataType")
			{
				if (!(out_type = GetDatatypeFromInitial(strPinInitial)))
				{
					std::string msg = string_format("warning : Requantize - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : Requantize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Requantize pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput && pinput_min && pinput_max && prequested_output_min && prequested_output_max)
	{
		pRequantize = new Requantize(*pScope, *pinput, *pinput_min, *pinput_max, *prequested_output_min, *prequested_output_max, out_type);
		ObjectInfo* pObj = AddObjectMap(pRequantize, id, SYMBOL_REQUANTIZE, "Requantize", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pRequantize->output, OUTPUT_TYPE_OUTPUT, "output");
			AddOutputInfo(pObj, &pRequantize->output_min, OUTPUT_TYPE_OUTPUT, "output_min");
			AddOutputInfo(pObj, &pRequantize->output_max, OUTPUT_TYPE_OUTPUT, "output_max");
		}
	}
	else
	{
		std::string msg = string_format("error : Requantize(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRequantize;
}

void* Create_Rint(std::string id, Json::Value pInputItem) {
	Rint* pRint = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Rint - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Rint - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Rint pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pRint = new Rint(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pRint, id, SYMBOL_RINT, "Rint", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pRint->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Rint(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRint;
}

void* Create_Round(std::string id, Json::Value pInputItem) {
	Round* pRound = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Round - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Round - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Round pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pRound = new Round(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pRound, id, SYMBOL_ROUND, "Round", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pRound->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Round(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRound;
}

void* Create_Rsqrt(std::string id, Json::Value pInputItem) {
	Rsqrt* pRsqrt = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Rsqrt - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Rsqrt - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Rsqrt pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pRsqrt = new Rsqrt(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pRsqrt, id, SYMBOL_RSQRT, "Rsqrt", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pRsqrt->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Rsqrt(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRsqrt;
}

void* Create_SegmentMax(std::string id, Json::Value pInputItem) {
	SegmentMax* pSegmentMax = nullptr;
	Scope* pScope = nullptr;
	Output* pdata = nullptr;
	Output* psegment_ids = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SegmentMax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pdata = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pdata = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SegmentMax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "segment_ids")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							psegment_ids = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						psegment_ids = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SegmentMax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SegmentMax pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pdata && psegment_ids)
	{
		pSegmentMax = new SegmentMax(*pScope, *pdata, *psegment_ids);
		ObjectInfo* pObj = AddObjectMap(pSegmentMax, id, SYMBOL_SEGMENTMAX, "SegmentMax", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSegmentMax->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : SegmentMax(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSegmentMax;
}

void* Create_SegmentMean(std::string id, Json::Value pInputItem) {
	SegmentMean* pSegmentMean = nullptr;
	Scope* pScope = nullptr;
	Output* pdata = nullptr;
	Output* psegment_ids = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SegmentMean - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pdata = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pdata = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SegmentMean - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "segment_ids")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							psegment_ids = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						psegment_ids = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SegmentMean - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SegmentMean pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pdata && psegment_ids)
	{
		pSegmentMean = new SegmentMean(*pScope, *pdata, *psegment_ids);
		ObjectInfo* pObj = AddObjectMap(pSegmentMean, id, SYMBOL_SEGMENTMEAN, "SegmentMean", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSegmentMean->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : SegmentMean(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSegmentMean;
}

void* Create_SegmentMin(std::string id, Json::Value pInputItem) {
	SegmentMin* pSegmentMin = nullptr;
	Scope* pScope = nullptr;
	Output* pdata = nullptr;
	Output* psegment_ids = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SegmentMin - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pdata = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pdata = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SegmentMin - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "segment_ids")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							psegment_ids = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						psegment_ids = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SegmentMin - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SegmentMin pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pdata && psegment_ids)
	{
		pSegmentMin = new SegmentMin(*pScope, *pdata, *psegment_ids);
		ObjectInfo* pObj = AddObjectMap(pSegmentMin, id, SYMBOL_SEGMENTMIN, "SegmentMin", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSegmentMin->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : SegmentMin(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSegmentMin;
}

void* Create_SegmentProd(std::string id, Json::Value pInputItem) {
	SegmentProd* pSegmentProd = nullptr;
	Scope* pScope = nullptr;
	Output* pdata = nullptr;
	Output* psegment_ids = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SegmentProd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pdata = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pdata = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SegmentProd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "segment_ids")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							psegment_ids = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						psegment_ids = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SegmentProd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SegmentProd pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pdata && psegment_ids)
	{
		pSegmentProd = new SegmentProd(*pScope, *pdata, *psegment_ids);
		ObjectInfo* pObj = AddObjectMap(pSegmentProd, id, SYMBOL_SEGMENTPROD, "SegmentProd", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSegmentProd->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : SegmentProd(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSegmentProd;
}

void* Create_SegmentSum(std::string id, Json::Value pInputItem) {
	SegmentSum* pSegmentSum = nullptr;
	Scope* pScope = nullptr;
	Output* pdata = nullptr;
	Output* psegment_ids = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SegmentSum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pdata = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pdata = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SegmentSum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "segment_ids")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							psegment_ids = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						psegment_ids = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SegmentSum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SegmentSum pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pdata && psegment_ids)
	{
		pSegmentSum = new SegmentSum(*pScope, *pdata, *psegment_ids);
		ObjectInfo* pObj = AddObjectMap(pSegmentSum, id, SYMBOL_SEGMENTSUM, "SegmentSum", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSegmentSum->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : SegmentSum(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSegmentSum;
}

void* Create_Sigmoid(std::string id, Json::Value pInputItem) {
	Sigmoid* pSigmoid = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Sigmoid - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Sigmoid - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Sigmoid pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pSigmoid = new Sigmoid(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pSigmoid, id, SYMBOL_SIGMOID, "Sigmoid", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSigmoid->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Sigmoid(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSigmoid;
}

void* Create_Sign(std::string id, Json::Value pInputItem) {
	Sign* pSign = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Sign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Sign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Sign pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pSign = new Sign(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pSign, id, SYMBOL_SIGN, "Sign", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSign->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Sign(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSign;
}

void* Create_Sin(std::string id, Json::Value pInputItem) {
	Sin* pSin = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Sin - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Sin - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Sin pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pSin = new Sin(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pSin, id, SYMBOL_SIN, "Sin", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSin->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Sin(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSin;
}
void* Create_Sinh(std::string id, Json::Value pInputItem) {
	Sinh* pSinh = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Sinh - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Sinh - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Sin pinh name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pSinh = new Sinh(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pSinh, id, SYMBOL_SINH, "Sinh", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSinh->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Sinh(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSinh;
}
void* Create_SparseMatMul(std::string id, Json::Value pInputItem) {
	SparseMatMul* pSparseMatMul = nullptr;
	Scope* pScope = nullptr;
	Output* pa = nullptr;
	Output* pb = nullptr;
	SparseMatMul::Attrs attrs;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SparseMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pa = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pa = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pb = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pb = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseMatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "SparseMatMul::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("transpose_a_") != "") attrs = attrs.TransposeA(attrParser.ConvStrToBool(attrParser.GetAttribute("transpose_a_")));
				if (attrParser.GetAttribute("transpose_b_") != "") attrs = attrs.TransposeB(attrParser.ConvStrToBool(attrParser.GetAttribute("transpose_b_")));
				if (attrParser.GetAttribute("a_is_sparse_") != "") attrs = attrs.AIsSparse(attrParser.ConvStrToBool(attrParser.GetAttribute("a_is_sparse_")));
				if (attrParser.GetAttribute("b_is_sparse_") != "") attrs = attrs.BIsSparse(attrParser.ConvStrToBool(attrParser.GetAttribute("b_is_sparse_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : QuantizedMatMul pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pa && pb)
	{
		pSparseMatMul = new SparseMatMul(*pScope, *pa, *pb, attrs);
		ObjectInfo* pObj = AddObjectMap(pSparseMatMul, id, SYMBOL_SPARSEMATMUL, "SparseMatMul", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseMatMul->product, OUTPUT_TYPE_OUTPUT, "product");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseMatMul(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseMatMul;
}

void* Create_SparseSegmentMean(std::string id, Json::Value pInputItem) {
	SparseSegmentMean* pSparseSegmentMean = nullptr;
	Scope* pScope = nullptr;
	Output* pdata = nullptr;
	Output* pindices = nullptr;
	Output* psegment_ids = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentMean - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pdata = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pdata = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentMean - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pindices = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pindices = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentMean - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "segment_ids")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							psegment_ids = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						psegment_ids = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentMean - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseSegmentMean pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pdata && pindices && psegment_ids)
	{
		pSparseSegmentMean = new SparseSegmentMean(*pScope, *pdata, *pindices, *psegment_ids);
		ObjectInfo* pObj = AddObjectMap(pSparseSegmentMean, id, SYMBOL_SPARSESEGMENTMEAN, "SparseSegmentMean", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseSegmentMean->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseSegmentMean(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseSegmentMean;
}

void* Create_SparseSegmentMeanGrad(std::string id, Json::Value pInputItem) {
	SparseSegmentMeanGrad* pSparseSegmentMeanGrad = nullptr;
	Scope* pScope = nullptr;
	Output* pgrad = nullptr;
	Output* pindices = nullptr;
	Output* psegment_ids = nullptr;
	Output* poutput_dim0 = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentMeanGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "grad")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pgrad = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pgrad = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentMeanGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pindices = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pindices = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentMeanGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "segment_ids")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							psegment_ids = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						psegment_ids = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentMeanGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "output_dim0")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							poutput_dim0 = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						poutput_dim0 = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentMeanGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseSegmentMeanGrad pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pgrad && pindices && psegment_ids && poutput_dim0)
	{
		pSparseSegmentMeanGrad = new SparseSegmentMeanGrad(*pScope, *pgrad, *pindices, *psegment_ids, *poutput_dim0);
		ObjectInfo* pObj = AddObjectMap(pSparseSegmentMeanGrad, id, SYMBOL_SPARSESEGMENTMEANGRAD, "SparseSegmentMeanGrad", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseSegmentMeanGrad->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseSegmentMeanGrad(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseSegmentMeanGrad;
}

void* Create_SparseSegmentSqrtN(std::string id, Json::Value pInputItem) {
	SparseSegmentSqrtN* pSparseSegmentSqrtN = nullptr;
	Scope* pScope = nullptr;
	Output* pdata = nullptr;
	Output* pindices = nullptr;
	Output* psegment_ids = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentSqrtN - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pdata = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pdata = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentSqrtN - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pindices = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pindices = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentSqrtN - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "segment_ids")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							psegment_ids = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						psegment_ids = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentSqrtN - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseSegmentSqrtN pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pdata && pindices && psegment_ids)
	{
		pSparseSegmentSqrtN = new SparseSegmentSqrtN(*pScope, *pdata, *pindices, *psegment_ids);
		ObjectInfo* pObj = AddObjectMap(pSparseSegmentSqrtN, id, SYMBOL_SPARSESEGMENTSQRTN, "SparseSegmentSqrtN", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseSegmentSqrtN->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseSegmentSqrtN(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseSegmentSqrtN;
}

void* Create_SparseSegmentSqrtNGrad(std::string id, Json::Value pInputItem) {
	SparseSegmentSqrtNGrad* pSparseSegmentSqrtNGrad = nullptr;
	Scope* pScope = nullptr;
	Output* pgrad = nullptr;
	Output* pindices = nullptr;
	Output* psegment_ids = nullptr;
	Output* poutput_dim0 = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentSqrtNGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "grad")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pgrad = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pgrad = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentSqrtNGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pindices = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pindices = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentSqrtNGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "segment_ids")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							psegment_ids = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						psegment_ids = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentSqrtNGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "output_dim0")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							poutput_dim0 = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						poutput_dim0 = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentSqrtNGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseSegmentSqrtNGrad pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pgrad && pindices && psegment_ids && poutput_dim0)
	{
		pSparseSegmentSqrtNGrad = new SparseSegmentSqrtNGrad(*pScope, *pgrad, *pindices, *psegment_ids, *poutput_dim0);
		ObjectInfo* pObj = AddObjectMap(pSparseSegmentSqrtNGrad, id, SYMBOL_SPARSESEGMENTSQRTNGRAD, "SparseSegmentSqrtNGrad", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseSegmentSqrtNGrad->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseSegmentSqrtNGrad(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseSegmentSqrtNGrad;
}

void* Create_SparseSegmentSum(std::string id, Json::Value pInputItem) {
	SparseSegmentSum* pSparseSegmentSum = nullptr;
	Scope* pScope = nullptr;
	Output* pdata = nullptr;
	Output* pindices = nullptr;
	Output* psegment_ids = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentSum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pdata = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pdata = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentSum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pindices = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pindices = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentSum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "segment_ids")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							psegment_ids = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						psegment_ids = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSegmentSum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseSegmentSum pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pdata && pindices && psegment_ids)
	{
		pSparseSegmentSum = new SparseSegmentSum(*pScope, *pdata, *pindices, *psegment_ids);
		ObjectInfo* pObj = AddObjectMap(pSparseSegmentSum, id, SYMBOL_SPARSESEGMENTSUM, "SparseSegmentSum", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseSegmentSum->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseSegmentSum(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseSegmentSum;
}

void* Create_Sqrt(std::string id, Json::Value pInputItem) {
	Sqrt* pSqrt = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Sqrt - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Sqrt - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Sqrt pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pSqrt = new Sqrt(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pSqrt, id, SYMBOL_SQRT, "Sqrt", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSqrt->y, OUTPUT_TYPE_OUTPUT, "y");
		}
	}
	else
	{
		std::string msg = string_format("error : Sqrt(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSqrt;
}

void* Create_Square(std::string id, Json::Value pInputItem) {
	Square* pSquare = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Square - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Square - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Square pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pSquare = new Square(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pSquare, id, SYMBOL_SQUARE, "Square", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSquare->y, OUTPUT_TYPE_OUTPUT, "y");
	}
	else
	{
		std::string msg = string_format("error : Square(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSquare;
}

void* Create_SquaredDifference(std::string id, Json::Value pInputItem) {
	SquaredDifference* pSquaredDifference = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SquaredDifference - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SquaredDifference - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SquaredDifference - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SquaredDifference pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pSquaredDifference = new SquaredDifference(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pSquaredDifference, id, SYMBOL_POW, "SquaredDifference", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSquaredDifference->z, OUTPUT_TYPE_OUTPUT, "z");
	}
	else
	{
		std::string msg = string_format("error : SquaredDifference(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSquaredDifference;
}

void* Create_Subtract(std::string id, Json::Value pInputItem) {
	Subtract* pSubtract = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Subtract - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Subtract - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Subtract - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Subtract pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pSubtract = new Subtract(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pSubtract, id, SYMBOL_SUBTRACT, "Subtract", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSubtract->z, OUTPUT_TYPE_OUTPUT, "z");
	}
	else
	{
		std::string msg = string_format("error : Subtract(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSubtract;
}

void* Create_Sum(std::string id, Json::Value pInputItem) {
	Sum* pSum = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* paxis = nullptr;
	Sum::Attrs attrs;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Sum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				else
				{
					if (!strPinInitial.empty())
					{
						pinput = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Sum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							paxis = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						paxis = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : All - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Sum::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("keep_dims_") != "") attrs = attrs.KeepDims(attrParser.ConvStrToBool(attrParser.GetAttribute("keep_dims_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : Sum pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput && paxis)
	{
		pSum = new Sum(*pScope, *pinput, *paxis, attrs);
		ObjectInfo* pObj = AddObjectMap(pSum, id, SYMBOL_SUM, "Sum", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSum->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : Sum(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSum;
}

void* Create_Tan(std::string id, Json::Value pInputItem) {
	Tan* pTan = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Tan - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Tan - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Tan pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pTan = new Tan(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pTan, id, SYMBOL_TAN, "Tan", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pTan->y, OUTPUT_TYPE_OUTPUT, "y");
	}
	else
	{
		std::string msg = string_format("error : Tan(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pTan;
}

void* Create_Tanh(std::string id, Json::Value pInputItem) {
	Tanh* pTanh = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Tanh - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Tanh - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Tanh pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px)
	{
		pTanh = new Tanh(*pScope, *px);
		ObjectInfo* pObj = AddObjectMap(pTanh, id, SYMBOL_TANH, "Tanh", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pTanh->y, OUTPUT_TYPE_OUTPUT, "y");
	}
	else
	{
		std::string msg = string_format("error : Tanh(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pTanh;
}

void* Create_TruncateDiv(std::string id, Json::Value pInputItem) {
	TruncateDiv* pTruncateDiv = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : TruncateDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TruncateDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TruncateDiv - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : TruncateDiv pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pTruncateDiv = new TruncateDiv(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pTruncateDiv, id, SYMBOL_TRUNCATEDIV, "TruncateDiv", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pTruncateDiv->z, OUTPUT_TYPE_OUTPUT, "z");
	}
	else
	{
		std::string msg = string_format("error : TruncateDiv(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pTruncateDiv;
}

void* Create_TruncateMod(std::string id, Json::Value pInputItem) {
	TruncateMod* pTruncateMod = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : TruncateMod - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TruncateMod - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TruncateMod - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : TruncateMod pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && py)
	{
		pTruncateMod = new TruncateMod(*pScope, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pTruncateMod, id, SYMBOL_TRUNCATEMOD, "TruncateMod", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pTruncateMod->z, OUTPUT_TYPE_OUTPUT, "z");
	}
	else
	{
		std::string msg = string_format("error : TruncateMod(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pTruncateMod;
}

void* Create_UnsortedSegmentMax(std::string id, Json::Value pInputItem) {
	UnsortedSegmentMax* pUnsortedSegmentMax = nullptr;
	Scope* pScope = nullptr;
	Output* pdata = nullptr;
	Output* psegment_ids = nullptr;
	Output* pnum_segments = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : UnsortedSegmentMax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pdata = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pdata = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : UnsortedSegmentMax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "segment_ids")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							psegment_ids = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						psegment_ids = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : UnsortedSegmentMax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_segments")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pnum_segments = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pnum_segments = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : UnsortedSegmentMax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : UnsortedSegmentMax pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pdata && psegment_ids && pnum_segments)
	{
		pUnsortedSegmentMax = new UnsortedSegmentMax(*pScope, *pdata, *psegment_ids, *pnum_segments);
		ObjectInfo* pObj = AddObjectMap(pUnsortedSegmentMax, id, SYMBOL_UNSORTEDSEGMENTMAX, "UnsortedSegmentMax", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pUnsortedSegmentMax->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : UnsortedSegmentMax(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pUnsortedSegmentMax;
}

void* Create_UnsortedSegmentSum(std::string id, Json::Value pInputItem) {
	UnsortedSegmentSum* pUnsortedSegmentSum = nullptr;
	Scope* pScope = nullptr;
	Output* pdata = nullptr;
	Output* psegment_ids = nullptr;
	Output* pnum_segments = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : UnsortedSegmentSum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pdata = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pdata = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : UnsortedSegmentSum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "segment_ids")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							psegment_ids = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						psegment_ids = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : UnsortedSegmentSum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_segments")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pnum_segments = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pnum_segments = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : UnsortedSegmentSum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : UnsortedSegmentSum pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pdata && psegment_ids && pnum_segments)
	{
		pUnsortedSegmentSum = new UnsortedSegmentSum(*pScope, *pdata, *psegment_ids, *pnum_segments);
		ObjectInfo* pObj = AddObjectMap(pUnsortedSegmentSum, id, SYMBOL_UNSORTEDSEGMENTSUM, "UnsortedSegmentSum", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pUnsortedSegmentSum->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : UnsortedSegmentSum(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pUnsortedSegmentSum;
}

void* Create_Where3(std::string id, Json::Value pInputItem) {
	Where3* pWhere3 = nullptr;
	Scope* pScope = nullptr;
	Output* pcondition = nullptr;
	Output* px = nullptr;
	Output* py = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Where3 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pcondition = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pcondition = (Output*)Create_StrToOutput(*m_pScope,"DT_BOOL", "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Where3 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Where3 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							py = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						py = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Where3 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Where3 pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pcondition && px && py)
	{
		pWhere3 = new Where3(*pScope, *pcondition, *px, *py);
		ObjectInfo* pObj = AddObjectMap(pWhere3, id, SYMBOL_WHERE3, "Where3", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pWhere3->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("error : Where3(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pWhere3;
}

void* Create_Zeta(std::string id, Json::Value pInputItem) {
	Zeta* pZeta = nullptr;
	Scope* pScope = nullptr;
	Output* px = nullptr;
	Output* pq = nullptr;

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

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Zeta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							px = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						px = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Zeta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "q")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pq = (Output*)pOutputObj->pOutput;
						}
					}

					// pY = pObj->pOutput;
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pq = (Output*)Create_StrToOutput(*m_pScope,strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Zeta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Zeta pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && px && pq)
	{
		pZeta = new Zeta(*pScope, *px, *pq);
		ObjectInfo* pObj = AddObjectMap(pZeta, id, SYMBOL_ZETA, "Zeta", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pZeta->z, OUTPUT_TYPE_OUTPUT, "z");
	}
	else
	{
		std::string msg = string_format("error : Zeta(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pZeta;
}
