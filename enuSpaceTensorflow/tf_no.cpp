#include "stdafx.h"
#include "tf_no.h"


#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "include/json/json.h"

#include "GlobalHeader.h"
#include "tensorflow.h"
#include "utility_functions.h"
#include "enuSpaceToTensorflow.h"

void* Create_NoOp(std::string id, Json::Value pInputItem) {
	NoOp* pNoOp = nullptr;
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
			if (strPinInterface == "tensorflow::Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : NoOp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : NoOp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope)
	{
		*pNoOp = NoOp(*pScope);
		ObjectInfo* pObj = AddObjectMap(pNoOp, id, SYMBOL_NOOP, "NoOp", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pNoOp->operation, OUTPUT_TYPE_OPERATION, "operation");
			
		}
	}
	else
	{
		std::string msg = string_format("error : NoOp(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pNoOp;
}