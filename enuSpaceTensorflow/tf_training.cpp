#include "stdafx.h"
#include "tf_training.h"

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
// #include "gradient_descent_optimizer.h"
// #include "optimizer.h"

void* Create_ApplyAdadelta(std::string id, Json::Value pInputItem) {
	ApplyAdadelta* pApplyAdadelta = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* accum_update = nullptr;
	Output* lr = nullptr;
	Output* rho = nullptr;
	Output* epsilon = nullptr;
	Output* grad = nullptr;
	ApplyAdadelta::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum_update")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum_update = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "rho")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							rho = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "epsilon")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							epsilon = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ApplyAdadelta::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") !="")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ApplyAdadelta pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && var && accum &&accum_update &&lr &&rho &&epsilon &&grad)
	{
		pApplyAdadelta = new ApplyAdadelta(*pScope, *var, *accum,*accum_update,*lr,*rho,*epsilon,*grad, attrs);
		ObjectInfo* pObj = AddObjectMap(pApplyAdadelta, id, SYMBOL_APPLYADADELTA, "ApplyAdadelta", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pApplyAdadelta->out, OUTPUT_TYPE_OUTPUT, "out");
	}
	else
	{
		std::string msg = string_format("error : ApplyAdadelta(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pApplyAdadelta;
}

void* Create_ApplyAdagrad(std::string id, Json::Value pInputItem) {
	ApplyAdagrad* pApplyAdagrad = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* lr = nullptr;
	Output* grad = nullptr;
	ApplyAdagrad::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ApplyAdagrad::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ApplyAdagrad pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && var && accum &&lr &&grad)
	{
		pApplyAdagrad = new ApplyAdagrad(*pScope, *var, *accum, *lr, *grad, attrs);
		ObjectInfo* pObj = AddObjectMap(pApplyAdagrad, id, SYMBOL_APPLYADAGRAD, "ApplyAdagrad", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pApplyAdagrad->out, OUTPUT_TYPE_OUTPUT, "out");
	}
	else
	{
		std::string msg = string_format("error : ApplyAdagrad(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pApplyAdagrad;
}

void* Create_ApplyAdagradDA(std::string id, Json::Value pInputItem) {
	ApplyAdagradDA* pApplyAdagradDA = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* gradient_accumulator = nullptr;
	Output* gradient_squared_accumulator = nullptr;
	Output* grad = nullptr;
	Output* lr = nullptr;
	Output* l1 = nullptr;
	Output* l2 = nullptr;
	Output* global_step = nullptr;
	ApplyAdagradDA::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "gradient_accumulator")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							gradient_accumulator = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "gradient_squared_accumulator")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							gradient_squared_accumulator = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l1")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l1 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l2")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l2 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "global_step")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							global_step = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ApplyAdagradDA::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ApplyAdagradDA pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && var && gradient_accumulator &&gradient_squared_accumulator &&grad &&lr &&l1 &&l2 &&global_step)
	{
		pApplyAdagradDA = new ApplyAdagradDA(*pScope, *var, *gradient_accumulator, *gradient_squared_accumulator, *grad, *lr, *l1, *l2, *global_step, attrs);
		ObjectInfo* pObj = AddObjectMap(pApplyAdagradDA, id, SYMBOL_APPLYADAGRADDA, "ApplyAdagradDA", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pApplyAdagradDA->out, OUTPUT_TYPE_OUTPUT, "out");
	}
	else
	{
		std::string msg = string_format("error : ApplyAdagradDA(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pApplyAdagradDA;
}

void* Create_ApplyAdam(std::string id, Json::Value pInputItem) {
	ApplyAdam* pApplyAdam = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* m = nullptr;
	Output* v = nullptr;
	Output* beta1_power = nullptr;
	Output* beta2_power = nullptr;
	Output* lr = nullptr;
	Output* beta1 = nullptr;
	Output* beta2 = nullptr;
	Output* epsilon = nullptr;
	Output* grad = nullptr;
	ApplyAdam::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "m")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							m = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "v")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							v = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "beta1_power")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							beta1_power = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "beta2_power")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							beta2_power = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "beta1")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							beta1 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "beta2")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							beta2 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "epsilon")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							epsilon = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ApplyAdam::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
				if (attrParser.GetAttribute("use_nesterov_") != "")attrs.UseNesterov(attrParser.ConvStrToBool(attrParser.GetAttribute("use_nesterov_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ApplyAdam pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && var && m &&v &&beta1_power &&beta2_power &&lr &&beta1 &&beta2 &&epsilon &&grad)
	{
		pApplyAdam = new ApplyAdam(*pScope, *var, *m, *v, *beta1_power, *beta2_power, *lr, *beta1, *beta2, *epsilon, *grad, attrs);
		ObjectInfo* pObj = AddObjectMap(pApplyAdam, id, SYMBOL_APPLYADAM, "ApplyAdam", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pApplyAdam->out, OUTPUT_TYPE_OUTPUT, "out");
	}
	else
	{
		std::string msg = string_format("error : ApplyAdam(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pApplyAdam;
}

void* Create_ApplyCenteredRMSProp(std::string id, Json::Value pInputItem) {
	ApplyCenteredRMSProp* pApplyCenteredRMSProp = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* mg = nullptr;
	Output* ms = nullptr;
	Output* mom = nullptr;
	Output* lr = nullptr;
	Output* rho = nullptr;
	Output* momentum = nullptr;
	Output* epsilon = nullptr;
	Output* grad = nullptr;
	ApplyCenteredRMSProp::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "mg")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							mg = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "ms")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							ms = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "mom")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							mom = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "rho")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							rho = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "momentum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							momentum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "epsilon")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							epsilon = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ApplyAdam::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ApplyCenteredRMSProp pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && var && mg &&ms &&mom &&lr &&rho &&momentum &&epsilon&&grad)
	{
		pApplyCenteredRMSProp = new ApplyCenteredRMSProp(*pScope, *var, *mg, *ms, *mom, *lr, *rho, *momentum, *epsilon, *grad, attrs);
		ObjectInfo* pObj = AddObjectMap(pApplyCenteredRMSProp, id, SYMBOL_APPLYCENTEREDRMSPROP, "ApplyCenteredRMSProp", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pApplyCenteredRMSProp->out, OUTPUT_TYPE_OUTPUT, "out");
	}
	else
	{
		std::string msg = string_format("error : ApplyCenteredRMSProp(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pApplyCenteredRMSProp;
}

void* Create_ApplyFtrl(std::string id, Json::Value pInputItem) {
	ApplyFtrl* pApplyFtrl = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* linear = nullptr;
	Output* grad = nullptr;
	Output* lr = nullptr;
	Output* l1 = nullptr;
	Output* l2 = nullptr;
	Output* lr_power = nullptr;
	ApplyFtrl::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "linear")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							linear = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l1")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l1 = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l2")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l2 = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr_power")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr_power = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ApplyFtrl::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ApplyFtrl pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && var && accum &&linear &&grad &&lr &&l1 &&l2 &&lr_power)
	{
		pApplyFtrl = new ApplyFtrl(*pScope, *var, *accum, *linear, *grad, *lr, *l1, *l2, *lr_power, attrs);
		ObjectInfo* pObj = AddObjectMap(pApplyFtrl, id, SYMBOL_APPLYFTRL, "ApplyFtrl", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pApplyFtrl->out, OUTPUT_TYPE_OUTPUT, "out");
	}
	else
	{
		std::string msg = string_format("error : ApplyFtrl(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pApplyFtrl;
}


void* Create_ApplyGradientDescent(std::string id, Json::Value pInputItem)
{
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* alpha = nullptr;
	Output* delta = nullptr;
	ApplyGradientDescent::Attrs attrs;
	ApplyGradientDescent* pGradientDescent = nullptr;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ApplyGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "alpha")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							alpha = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							delta = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ApplyGradientDescent::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ApplyGradientDescent pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && var && alpha && delta)
	{
		pGradientDescent = new ApplyGradientDescent(*pScope, *var, *alpha, *delta, attrs);
		ObjectInfo* pObj = AddObjectMap(pGradientDescent, id, SYMBOL_APPLYGRADIENTDESCENT, "ApplyGradientDescent", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pGradientDescent->out, OUTPUT_TYPE_OUTPUT, "out");
	}
	else
	{
		std::string msg = string_format("error : ApplyGradientDescent(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pGradientDescent;
}

void* Create_ApplyMomentum(std::string id, Json::Value pInputItem) {
	ApplyMomentum* pApplyMomentum = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* lr = nullptr;
	Output* grad = nullptr;
	Output* momentum = nullptr;
	ApplyMomentum::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "momentum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							momentum = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : ApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ApplyMomentum::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
				if (attrParser.GetAttribute("use_nesterov_") != "")attrs.UseNesterov(attrParser.ConvStrToBool(attrParser.GetAttribute("use_nesterov_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ApplyMomentum pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && var && accum && lr &&grad &&momentum)
	{
		pApplyMomentum = new ApplyMomentum(*pScope, *var, *accum, *lr,*grad, *momentum, attrs);
		ObjectInfo* pObj = AddObjectMap(pApplyMomentum, id, SYMBOL_APPLYMOMENTUM, "ApplyMomentum", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pApplyMomentum->out, OUTPUT_TYPE_OUTPUT, "out");
	}
	else
	{
		std::string msg = string_format("error : ApplyMomentum(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pApplyMomentum;
}

void* Create_ApplyProximalAdagrad(std::string id, Json::Value pInputItem) {
	ApplyProximalAdagrad* pApplyProximalAdagrad = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* lr = nullptr;
	Output* l1 = nullptr;
	Output* l2 = nullptr;
	Output* grad = nullptr;
	ApplyProximalAdagrad::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l1")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l1 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l2")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l2 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ApplyProximalAdagrad::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ApplyProximalAdagrad pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && var && accum && lr &&l1 &&l2 &&grad)
	{
		pApplyProximalAdagrad = new ApplyProximalAdagrad(*pScope, *var, *accum, *lr, *l1,*l2,*grad, attrs);
		ObjectInfo* pObj = AddObjectMap(pApplyProximalAdagrad, id, SYMBOL_APPLYPROXIMALADAGRAD, "ApplyProximalAdagrad", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pApplyProximalAdagrad->out, OUTPUT_TYPE_OUTPUT, "out");
	}
	else
	{
		std::string msg = string_format("error : ApplyProximalAdagrad(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pApplyProximalAdagrad;
}

void* Create_ApplyProximalGradientDescent(std::string id, Json::Value pInputItem) {
	ApplyProximalGradientDescent* pApplyProximalGradientDescent = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* alpha = nullptr;
	Output* l1 = nullptr;
	Output* l2 = nullptr;
	Output* delta = nullptr;
	ApplyProximalGradientDescent::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "alpha")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							alpha = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l1")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l1 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l2")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l2 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							delta = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ApplyProximalGradientDescent::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ApplyProximalGradientDescent pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && var && alpha &&l1 &&l2 &&delta)
	{
		pApplyProximalGradientDescent = new ApplyProximalGradientDescent(*pScope, *var, *alpha, *l1, *l2, *delta, attrs);
		ObjectInfo* pObj = AddObjectMap(pApplyProximalGradientDescent, id, SYMBOL_APPLYPROXIMALGRADIENTDESCENT, "ApplyProximalGradientDescent", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pApplyProximalGradientDescent->out, OUTPUT_TYPE_OUTPUT, "out");
	}
	else
	{
		std::string msg = string_format("error : ApplyProximalGradientDescent(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pApplyProximalGradientDescent;
}

void* Create_ApplyRMSProp(std::string id, Json::Value pInputItem) {
	ApplyRMSProp* pApplyRMSProp = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* ms = nullptr;
	Output* mom = nullptr;
	Output* lr = nullptr;
	Output* rho = nullptr;
	Output* momentum = nullptr;
	Output* epsilon = nullptr;
	Output* grad = nullptr;
	std::string msgParam = string_format("#%s", id.c_str());

	ApplyRMSProp::Attrs attrs;

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
				std::string msg = string_format("warning : ApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "ms")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							ms = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "mom")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							mom = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "rho")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							rho = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "momentum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							momentum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "epsilon")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							epsilon = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ApplyRMSProp::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ApplyRMSProp pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && var && ms &&mom &&lr &&momentum &&rho &&epsilon &&grad)
	{
		pApplyRMSProp = new ApplyRMSProp(*pScope, *var, *ms, *mom, *lr, *rho,*momentum,*epsilon,*grad, attrs);
		ObjectInfo* pObj = AddObjectMap(pApplyRMSProp, id, SYMBOL_APPLYRMSPROP, "ApplyRMSProp", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pApplyRMSProp->out, OUTPUT_TYPE_OUTPUT, "out");
	}
	else
	{
		std::string msg = string_format("error : ApplyRMSProp(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pApplyRMSProp;
}

void* Create_ResourceApplyAdadelta(std::string id, Json::Value pInputItem) {
	ResourceApplyAdadelta* pResourceApplyAdadelta = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* accum_update = nullptr;
	Output* lr = nullptr;
	Output* rho = nullptr;
	Output* epsilon = nullptr;
	Output* grad = nullptr;
	ResourceApplyAdadelta::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum_update")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum_update = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "rho")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							rho = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "epsilon")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							epsilon = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceApplyAdadelta::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceApplyAdadelta pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && var && accum &&accum_update &&lr &&rho &&epsilon &&grad)
	{
		pResourceApplyAdadelta = new ResourceApplyAdadelta(*pScope, *var, *accum, *accum_update, *lr, *rho,*epsilon, *grad, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceApplyAdadelta, id, SYMBOL_RESOURCEAPPLYADADELTA, "ResourceApplyAdadelta", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceApplyAdadelta->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceApplyAdadelta(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceApplyAdadelta;
}

void* Create_ResourceApplyAdagrad(std::string id, Json::Value pInputItem) {
	ResourceApplyAdagrad* pResourceApplyAdagrad = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* lr = nullptr;
	Output* grad = nullptr;
	ResourceApplyAdagrad::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceApplyAdagrad::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceApplyAdagrad pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && var && accum  &&lr &&grad)
	{
		pResourceApplyAdagrad = new ResourceApplyAdagrad(*pScope, *var, *accum, *lr, *grad, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceApplyAdagrad, id, SYMBOL_RESOURCEAPPLYADAGRAD, "ResourceApplyAdagrad", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceApplyAdagrad->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceApplyAdagrad(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceApplyAdagrad;
}

void* Create_ResourceApplyAdagradDA(std::string id, Json::Value pInputItem) {
	ResourceApplyAdagradDA* pResourceApplyAdagradDA = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* gradient_accumulator = nullptr;
	Output* gradient_squared_accumulator = nullptr;
	Output* grad = nullptr;
	Output* lr = nullptr;
	Output* l1 = nullptr;
	Output* l2 = nullptr;
	Output* global_step = nullptr;
	ResourceApplyAdagradDA::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "gradient_accumulator")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							gradient_accumulator = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "gradient_squared_accumulator")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							gradient_squared_accumulator = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l1")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l1 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l2")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l2 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "global_step")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							global_step = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceApplyAdagradDA::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceApplyAdagradDA pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && gradient_accumulator && gradient_squared_accumulator  &&grad &&lr  && l1  &&l2 &&global_step)
	{
		pResourceApplyAdagradDA = new ResourceApplyAdagradDA(*pScope,*var, *gradient_accumulator, *gradient_squared_accumulator, *grad, *lr,*l1,*l2,*global_step, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceApplyAdagradDA, id, SYMBOL_RESOURCEAPPLYADAGRADDA, "ResourceApplyAdagradDA", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceApplyAdagradDA->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceApplyAdagradDA(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceApplyAdagradDA;
}

void* Create_ResourceApplyAdam(std::string id, Json::Value pInputItem) {
	ResourceApplyAdam* pResourceApplyAdam = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* m = nullptr;
	Output* v = nullptr;
	Output* beta1_power = nullptr;
	Output* beta2_power = nullptr;
	Output* lr = nullptr;
	Output* beta1 = nullptr;
	Output* beta2 = nullptr;
	Output* epsilon = nullptr;
	Output* grad = nullptr;
	ResourceApplyAdam::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "m")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							m = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "v")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							v = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "beta1_power")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							beta1_power = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "beta2_power")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							beta2_power = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "beta1")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							beta1 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "beta2")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							beta2 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "epsilon")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							epsilon = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyAdamResourceApplyAdam - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceApplyAdam::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
				if (attrParser.GetAttribute("use_nesterov_") != "")attrs.UseNesterov(attrParser.ConvStrToBool(attrParser.GetAttribute("use_nesterov_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceApplyAdam pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && m && v  &&beta1_power &&beta2_power  && lr  &&beta1 &&beta2 &&epsilon &&grad)
	{
		pResourceApplyAdam = new ResourceApplyAdam(*pScope, *var, *m, *v, *beta1_power, *beta2_power, *lr, *beta1, *beta1,*epsilon,*grad, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceApplyAdam, id, SYMBOL_RESOURCEAPPLYADAM, "ResourceApplyAdam", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceApplyAdam->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceApplyAdam(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceApplyAdam;
}

void* Create_ResourceApplyCenteredRMSProp(std::string id, Json::Value pInputItem) {
	ResourceApplyCenteredRMSProp* pResourceApplyCenteredRMSProp = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* mg = nullptr;
	Output* ms = nullptr;
	Output* mom = nullptr;
	Output* lr = nullptr;
	Output* rho = nullptr;
	Output* momentum = nullptr;
	Output* epsilon = nullptr;
	Output* grad = nullptr;
	ResourceApplyCenteredRMSProp::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "mg")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							mg = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "ms")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							ms = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "mom")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							mom = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "rho")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							rho = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "momentum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							momentum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "epsilon")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							epsilon = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceApplyCenteredRMSProp::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceApplyCenteredRMSProp pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && mg && ms  &&mom &&lr &&rho &&momentum &&epsilon &&grad)
	{
		pResourceApplyCenteredRMSProp = new ResourceApplyCenteredRMSProp(*pScope, *var, *mg, *ms, *mom, *lr, *rho, *momentum, *epsilon, *grad, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceApplyCenteredRMSProp, id, SYMBOL_RESOURCEAPPLYCENTEREDRMSPROP, "ResourceApplyCenteredRMSProp", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceApplyCenteredRMSProp->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceApplyCenteredRMSProp(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceApplyCenteredRMSProp;
}

void* Create_ResourceApplyFtrl(std::string id, Json::Value pInputItem) {
	ResourceApplyFtrl* pResourceApplyFtrl = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* linear = nullptr;
	Output* grad = nullptr;
	Output* lr = nullptr;
	Output* l1 = nullptr;
	Output* l2 = nullptr;
	Output* lr_power = nullptr;
	ResourceApplyFtrl::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "linear")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							linear = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l1")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l1 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l2")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l2 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr_power")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr_power = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceApplyFtrl::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceApplyFtrl pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && accum && linear  &&grad &&lr &&l1 &&l2 &&lr_power)
	{
		pResourceApplyFtrl = new ResourceApplyFtrl(*pScope, *var, *accum, *linear, *grad,*lr, *l1, *l2, *lr_power, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceApplyFtrl, id, SYMBOL_RESOURCEAPPLYFTRL, "ResourceApplyFtrl", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceApplyFtrl->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceApplyFtrl(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceApplyFtrl;
}

void* Create_ResourceApplyGradientDescent(std::string id, Json::Value pInputItem) {
	ResourceApplyGradientDescent* pResourceApplyGradientDescent = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* alpha = nullptr;
	Output* delta = nullptr;
	ResourceApplyGradientDescent::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "alpha")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							alpha = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							delta = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceApplyGradientDescent::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceApplyGradientDescent pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && alpha &&delta)
	{
		pResourceApplyGradientDescent = new ResourceApplyGradientDescent(*pScope, *var, *alpha, *delta, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceApplyGradientDescent, id, SYMBOL_RESOURCEAPPLYGRADIENTDESCENT, "ResourceApplyGradientDescent", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceApplyGradientDescent->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceApplyGradientDescent(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceApplyGradientDescent;
}

void* Create_ResourceApplyMomentum(std::string id, Json::Value pInputItem) {
	ResourceApplyMomentum* pResourceApplyMomentum = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* lr = nullptr;
	Output* grad = nullptr;
	Output* momentum = nullptr;
	ResourceApplyMomentum::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "momentum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							momentum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceApplyMomentum::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
				if (attrParser.GetAttribute("use_nesterov_") != "")attrs.UseNesterov(attrParser.ConvStrToBool(attrParser.GetAttribute("use_nesterov_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceApplyMomentum pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && accum &&lr &&grad &&momentum )
	{
		pResourceApplyMomentum = new ResourceApplyMomentum(*pScope, *var, *accum, *lr,*grad,*momentum, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceApplyMomentum, id, SYMBOL_RESOURCEAPPLYMOMENTUM, "ResourceApplyMomentum", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceApplyMomentum->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceApplyMomentum(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceApplyMomentum;
}

void* Create_ResourceApplyProximalAdagrad(std::string id, Json::Value pInputItem) {
	ResourceApplyProximalAdagrad* pResourceApplyProximalAdagrad = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* lr = nullptr;
	Output* l1 = nullptr;
	Output* l2 = nullptr;
	Output* grad = nullptr;
	ResourceApplyProximalAdagrad::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l1")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l1 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l2")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l2 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceApplyProximalAdagrad::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceApplyProximalAdagrad pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && accum &&lr &&l1 &&l2 &&grad )
	{
		pResourceApplyProximalAdagrad = new ResourceApplyProximalAdagrad(*pScope, *var, *accum, *lr, *l1, *l2, *grad, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceApplyProximalAdagrad, id, SYMBOL_RESOURCEAPPLYPROXIMALADAGRAD, "ResourceApplyProximalAdagrad", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceApplyProximalAdagrad->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceApplyProximalAdagrad(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceApplyProximalAdagrad;
}

void* Create_ResourceApplyProximalGradientDescent(std::string id, Json::Value pInputItem) {
	ResourceApplyProximalGradientDescent* pResourceApplyProximalGradientDescent = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* alpha = nullptr;
	Output* l1 = nullptr;
	Output* l2 = nullptr;
	Output* delta = nullptr;
	ResourceApplyProximalGradientDescent::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "alpha")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							alpha = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l1")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l1 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l2")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l2 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							delta = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceApplyProximalGradientDescent::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceApplyProximalGradientDescent pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && alpha && l1 &&l2 &&delta)
	{
		pResourceApplyProximalGradientDescent = new ResourceApplyProximalGradientDescent(*pScope, *var, *alpha, *l1, *l2, *delta, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceApplyProximalGradientDescent, id, SYMBOL_RESOURCEAPPLYPROXIMALGRADIENTDESCENT, "ResourceApplyProximalGradientDescent", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceApplyProximalGradientDescent->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceApplyProximalGradientDescent(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceApplyProximalGradientDescent;
}

void* Create_ResourceApplyRMSProp(std::string id, Json::Value pInputItem) {
	ResourceApplyRMSProp* pResourceApplyRMSProp = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* ms = nullptr;
	Output* mom = nullptr;
	Output* lr = nullptr;
	Output* rho = nullptr;
	Output* momentum = nullptr;
	Output* epsilon = nullptr;
	Output* grad = nullptr;
	ResourceApplyRMSProp::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "ms")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							ms = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "mom")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							mom = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "rho")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							rho = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "momentum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							momentum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "epsilon")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							epsilon = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceApplyRMSProp::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceApplyRMSProp pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && ms && mom &&lr && rho &&momentum &&epsilon &&grad)
	{
		pResourceApplyRMSProp = new ResourceApplyRMSProp(*pScope, *var, *ms, *mom, *lr, *rho,*momentum,*epsilon,*grad, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceApplyRMSProp, id, SYMBOL_RESOURCEAPPLYRMSPROP, "ResourceApplyRMSProp", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceApplyRMSProp->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceApplyRMSProp(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceApplyRMSProp;
}

void* Create_ResourceSparseApplyAdadelta(std::string id, Json::Value pInputItem) {
	ResourceSparseApplyAdadelta* pResourceSparseApplyAdadelta = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* accum_update = nullptr;
	Output* lr = nullptr;
	Output* rho = nullptr;
	Output* epsilon = nullptr;
	Output* grad = nullptr;
	Output* indices = nullptr;
	ResourceSparseApplyAdadelta::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum_update")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum_update = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "rho")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							rho = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "epsilon")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							epsilon = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
				std::string msg = string_format("warning : ResourceSparseApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceSparseApplyAdadelta::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceSparseApplyAdadelta pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && accum && accum_update &&lr && rho &&epsilon &&grad && indices)
	{
		pResourceSparseApplyAdadelta = new ResourceSparseApplyAdadelta(*pScope, *var, *accum, *accum_update, *lr, *rho, *epsilon, *grad,*indices, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceSparseApplyAdadelta, id, SYMBOL_RESOURCESPARSEAPPLYADADELTA, "ResourceSparseApplyAdadelta", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceSparseApplyAdadelta->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceSparseApplyAdadelta(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceSparseApplyAdadelta;
}

void* Create_ResourceSparseApplyAdagrad(std::string id, Json::Value pInputItem) {
	ResourceSparseApplyAdagrad* pResourceSparseApplyAdagrad = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* lr = nullptr;
	Output* grad = nullptr;
	Output* indices = nullptr;
	ResourceSparseApplyAdagrad::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
				std::string msg = string_format("warning : ResourceSparseApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceSparseApplyAdagrad::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceSparseApplyAdagrad pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && accum  &&lr &&grad && indices)
	{
		pResourceSparseApplyAdagrad = new ResourceSparseApplyAdagrad(*pScope, *var, *accum, *lr, *grad, *indices, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceSparseApplyAdagrad, id, SYMBOL_RESOURCESPARSEAPPLYADAGRAD, "ResourceSparseApplyAdagrad", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceSparseApplyAdagrad->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceSparseApplyAdagrad(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceSparseApplyAdagrad;
}

void* Create_ResourceSparseApplyAdagradDA(std::string id, Json::Value pInputItem) {
	ResourceSparseApplyAdagradDA* pResourceSparseApplyAdagradDA = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* gradient_accumulator = nullptr;
	Output* gradient_squared_accumulator = nullptr;
	Output* grad = nullptr;
	Output* indices = nullptr;
	Output* lr = nullptr;
	Output* l1 = nullptr;
	Output* l2 = nullptr;
	Output* global_step = nullptr;
	ResourceSparseApplyAdagradDA::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "gradient_accumulator")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							gradient_accumulator = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "gradient_squared_accumulator")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							gradient_squared_accumulator = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
				std::string msg = string_format("warning : ResourceSparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l1")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l1 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l2")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l2 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "global_step")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							global_step = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceSparseApplyAdagrad::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceSparseApplyAdagrad pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && gradient_accumulator  &&gradient_squared_accumulator &&grad && indices && lr && l1 && l2 && global_step)
	{
		pResourceSparseApplyAdagradDA = new ResourceSparseApplyAdagradDA(*pScope, *var, *gradient_accumulator, *gradient_squared_accumulator, *grad, *indices,*lr,*l1,*l2,*global_step, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceSparseApplyAdagradDA, id, SYMBOL_RESOURCESPARSEAPPLYADAGRADDA, "ResourceSparseApplyAdagradDA", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceSparseApplyAdagradDA->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceSparseApplyAdagrad(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceSparseApplyAdagradDA;
}

void* Create_ResourceSparseApplyCenteredRMSProp(std::string id, Json::Value pInputItem) {
	ResourceSparseApplyCenteredRMSProp* pResourceSparseApplyCenteredRMSProp = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* mg = nullptr;
	Output* ms = nullptr;
	Output* mom = nullptr;
	Output* lr = nullptr;
	Output* rho = nullptr;
	Output* momentum = nullptr;
	Output* epsilon = nullptr;
	Output* grad = nullptr;
	Output* indices = nullptr;
	ResourceSparseApplyCenteredRMSProp::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "mg")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							mg = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "ms")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							ms = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "mom")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							mom = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "rho")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							rho = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "momentum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							momentum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "epsilon")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							epsilon = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
				std::string msg = string_format("warning : ResourceSparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceSparseApplyCenteredRMSProp::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceSparseApplyCenteredRMSProp pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && mg &&ms &&mom && lr && rho && momentum && epsilon && grad && indices)
	{
		pResourceSparseApplyCenteredRMSProp = new ResourceSparseApplyCenteredRMSProp(*pScope, *var, *mg, *ms, *mom, *lr, *rho, *momentum, *epsilon, *grad,*indices, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceSparseApplyCenteredRMSProp, id, SYMBOL_RESOURCESPARSEAPPLYCENTEREDRMSPROP, "ResourceSparseApplyCenteredRMSProp", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceSparseApplyCenteredRMSProp->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceSparseApplyCenteredRMSProp(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceSparseApplyCenteredRMSProp;
}

void* Create_ResourceSparseApplyFtrl(std::string id, Json::Value pInputItem) {
	ResourceSparseApplyFtrl* pResourceSparseApplyFtrl = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* linear = nullptr;
	Output* grad = nullptr;
	Output* indices = nullptr;
	Output* lr = nullptr;
	Output* l1 = nullptr;
	Output* l2 = nullptr;
	Output* lr_power = nullptr;
	ResourceSparseApplyFtrl::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "linear")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							linear = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
				std::string msg = string_format("warning : ResourceSparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l1")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l1 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l2")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l2 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr_power")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr_power = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceSparseApplyFtrl::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceSparseApplyFtrl pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && accum &&linear &&grad &&indices && lr && l1 && l2 && lr_power)
	{
		pResourceSparseApplyFtrl = new ResourceSparseApplyFtrl(*pScope, *var, *accum, *linear, *grad, *indices, *lr, *l1, *l2, *lr_power, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceSparseApplyFtrl, id, SYMBOL_RESOURCESPARSEAPPLYFTRL, "ResourceSparseApplyFtrl", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceSparseApplyFtrl->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceSparseApplyFtrl(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceSparseApplyFtrl;
}

void* Create_ResourceSparseApplyMomentum(std::string id, Json::Value pInputItem) {
	ResourceSparseApplyMomentum* pResourceSparseApplyMomentum = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* lr = nullptr;
	Output* grad = nullptr;
	Output* indices = nullptr;
	Output* momentum = nullptr;
	ResourceSparseApplyMomentum::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
				std::string msg = string_format("warning : ResourceSparseApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "momentum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							momentum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceSparseApplyMomentum::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
				if (attrParser.GetAttribute("use_nesterov_") != "")attrs.UseNesterov(attrParser.ConvStrToBool(attrParser.GetAttribute("use_nesterov_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceSparseApplyMomentum pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && accum &&lr &&grad &&indices && momentum)
	{
		pResourceSparseApplyMomentum = new ResourceSparseApplyMomentum(*pScope, *var, *accum, *lr, *grad, *indices, *momentum, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceSparseApplyMomentum, id, SYMBOL_RESOURCESPARSEAPPLYMOMENTUM, "ResourceSparseApplyMomentum", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceSparseApplyMomentum->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceSparseApplyMomentum(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceSparseApplyMomentum;
}

void* Create_ResourceSparseApplyProximalAdagrad(std::string id, Json::Value pInputItem) {
	ResourceSparseApplyProximalAdagrad* pResourceSparseApplyProximalAdagrad = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* lr = nullptr;
	Output* l1 = nullptr;
	Output* l2 = nullptr;
	Output* grad = nullptr;
	Output* indices = nullptr;
	ResourceSparseApplyProximalAdagrad::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l1")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l1 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l2")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l2 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
				std::string msg = string_format("warning : ResourceSparseApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceSparseApplyProximalAdagrad::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceSparseApplyProximalAdagrad pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && accum &&lr &&l1 &&l2 &&grad && indices)
	{
		pResourceSparseApplyProximalAdagrad = new ResourceSparseApplyProximalAdagrad(*pScope, *var, *accum, *lr, *l1,*l2,*grad, *indices, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceSparseApplyProximalAdagrad, id, SYMBOL_RESOURCESPARSEAPPLYPROXIMALADAGRAD, "ResourceSparseApplyProximalAdagrad", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceSparseApplyProximalAdagrad->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceSparseApplyProximalAdagrad(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceSparseApplyProximalAdagrad;
}

void* Create_ResourceSparseApplyProximalGradientDescent(std::string id, Json::Value pInputItem) {
	ResourceSparseApplyProximalGradientDescent* pResourceSparseApplyProximalGradientDescent = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* alpha = nullptr;
	Output* l1 = nullptr;
	Output* l2 = nullptr;
	Output* grad = nullptr;
	Output* indices = nullptr;
	ResourceSparseApplyProximalGradientDescent::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "alpha")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							alpha = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l1")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l1 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l2")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l2 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
				std::string msg = string_format("warning : ResourceSparseApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceSparseApplyProximalGradientDescent::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceSparseApplyProximalGradientDescent pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && alpha && l1 &&l2 &&grad &&indices)
	{
		pResourceSparseApplyProximalGradientDescent = new ResourceSparseApplyProximalGradientDescent(*pScope, *var, *alpha, *l1, *l2, *grad,*indices, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceSparseApplyProximalGradientDescent, id, SYMBOL_RESOURCESPARSEAPPLYPROXIMALGRADIENTDESCENT, "ResourceSparseApplyProximalGradientDescent", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceSparseApplyProximalGradientDescent->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceSparseApplyProximalGradientDescent(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceSparseApplyProximalGradientDescent;
}

void* Create_ResourceSparseApplyRMSProp(std::string id, Json::Value pInputItem) {
	ResourceSparseApplyRMSProp* pResourceSparseApplyRMSProp = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* ms = nullptr;
	Output* mom = nullptr;
	Output* lr = nullptr;
	Output* rho = nullptr;
	Output* momentum = nullptr;
	Output* epsilon = nullptr;
	Output* grad = nullptr;
	Output* indices = nullptr;
	ResourceSparseApplyRMSProp::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "ms")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							ms = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "mom")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							mom = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "rho")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							rho = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "momentum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							momentum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "epsilon")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							epsilon = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResourceSparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
				std::string msg = string_format("warning : ResourceSparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResourceSparseApplyRMSProp::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResourceSparseApplyRMSProp pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && ms && mom &&lr && rho &&momentum &&epsilon &&grad)
	{
		pResourceSparseApplyRMSProp = new ResourceSparseApplyRMSProp(*pScope, *var, *ms, *mom, *lr, *rho, *momentum, *epsilon, *grad,*indices, attrs);
		ObjectInfo* pObj = AddObjectMap(pResourceSparseApplyRMSProp, id, SYMBOL_RESOURCESPARSEAPPLYRMSPROP, "ResourceSparseApplyRMSProp", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pResourceSparseApplyRMSProp->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : ResourceSparseApplyRMSProp(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pResourceSparseApplyRMSProp;
}

void* Create_SparseApplyAdadelta(std::string id, Json::Value pInputItem) {
	SparseApplyAdadelta* pSparseApplyAdadelta = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* accum_update = nullptr;
	Output* lr = nullptr;
	Output* rho = nullptr;
	Output* epsilon = nullptr;
	Output* grad = nullptr;
	Output* indices = nullptr;
	SparseApplyAdadelta::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum_update")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum_update = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "rho")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							rho = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "epsilon")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							epsilon = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
				std::string msg = string_format("warning : SparseApplyAdadelta - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "SparseApplyAdadelta::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseApplyAdadelta pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && accum && accum_update &&lr && rho &&epsilon &&grad && indices)
	{
		pSparseApplyAdadelta = new SparseApplyAdadelta(*pScope, *var, *accum, *accum_update, *lr, *rho, *epsilon, *grad, *indices, attrs);
		ObjectInfo* pObj = AddObjectMap(pSparseApplyAdadelta, id, SYMBOL_SPARSEAPPLYADADELTA, "SparseApplyAdadelta", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSparseApplyAdadelta->out, OUTPUT_TYPE_OUTPUT, "out");
	}
	else
	{
		std::string msg = string_format("error : SparseApplyAdadelta(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pSparseApplyAdadelta;
}

void* Create_SparseApplyAdagrad(std::string id, Json::Value pInputItem) {
	SparseApplyAdagrad* pSparseApplyAdagrad = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* lr = nullptr;
	Output* grad = nullptr;
	Output* indices = nullptr;
	SparseApplyAdagrad::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
				std::string msg = string_format("warning : SparseApplyAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "SparseApplyAdagrad::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseApplyAdagrad pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && accum  &&lr &&grad && indices)
	{
		pSparseApplyAdagrad = new SparseApplyAdagrad(*pScope, *var, *accum, *lr, *grad, *indices, attrs);
		ObjectInfo* pObj = AddObjectMap(pSparseApplyAdagrad, id, SYMBOL_SPARSEAPPLYADAGRAD, "SparseApplyAdagrad", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSparseApplyAdagrad->out, OUTPUT_TYPE_OUTPUT, "out");
	}
	else
	{
		std::string msg = string_format("error : SparseApplyAdagrad(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pSparseApplyAdagrad;
}

void* Create_SparseApplyAdagradDA(std::string id, Json::Value pInputItem) {
	SparseApplyAdagradDA* pSparseApplyAdagradDA = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* gradient_accumulator = nullptr;
	Output* gradient_squared_accumulator = nullptr;
	Output* grad = nullptr;
	Output* indices = nullptr;
	Output* lr = nullptr;
	Output* l1 = nullptr;
	Output* l2 = nullptr;
	Output* global_step = nullptr;
	SparseApplyAdagradDA::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "gradient_accumulator")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							gradient_accumulator = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "gradient_squared_accumulator")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							gradient_squared_accumulator = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
				std::string msg = string_format("warning : SparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l1")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l1 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l2")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l2 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "global_step")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							global_step = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyAdagradDA - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "SparseApplyAdagradDA::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseApplyAdagradDA pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && gradient_accumulator  &&gradient_squared_accumulator &&grad && indices && lr && l1 && l2 && global_step)
	{
		pSparseApplyAdagradDA = new SparseApplyAdagradDA(*pScope, *var, *gradient_accumulator, *gradient_squared_accumulator, *grad, *indices, *lr, *l1, *l2, *global_step, attrs);
		ObjectInfo* pObj = AddObjectMap(pSparseApplyAdagradDA, id, SYMBOL_SPARSEAPPLYADAGRADDA, "SparseApplyAdagradDA", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSparseApplyAdagradDA->out, OUTPUT_TYPE_OUTPUT, "out");
	}
	else
	{
		std::string msg = string_format("error : SparseApplyAdagradDA(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pSparseApplyAdagradDA;
}

void* Create_SparseApplyCenteredRMSProp(std::string id, Json::Value pInputItem) {
	SparseApplyCenteredRMSProp* pSparseApplyCenteredRMSProp = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* mg = nullptr;
	Output* ms = nullptr;
	Output* mom = nullptr;
	Output* lr = nullptr;
	Output* rho = nullptr;
	Output* momentum = nullptr;
	Output* epsilon = nullptr;
	Output* grad = nullptr;
	Output* indices = nullptr;
	SparseApplyCenteredRMSProp::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "mg")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							mg = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "ms")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							ms = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "mom")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							mom = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "rho")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							rho = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "momentum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							momentum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "epsilon")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							epsilon = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
				std::string msg = string_format("warning : SparseApplyCenteredRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "SparseApplyCenteredRMSProp::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseApplyCenteredRMSProp pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && mg &&ms &&mom && lr && rho && momentum && epsilon && grad && indices)
	{
		pSparseApplyCenteredRMSProp = new SparseApplyCenteredRMSProp(*pScope, *var, *mg, *ms, *mom, *lr, *rho, *momentum, *epsilon, *grad, *indices, attrs);
		ObjectInfo* pObj = AddObjectMap(pSparseApplyCenteredRMSProp, id, SYMBOL_SPARSEAPPLYCENTEREDRMSPROP, "SparseApplyCenteredRMSProp", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSparseApplyCenteredRMSProp->out, OUTPUT_TYPE_OPERATION, "out");
	}
	else
	{
		std::string msg = string_format("error : SparseApplyCenteredRMSProp(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pSparseApplyCenteredRMSProp;
}

void* Create_SparseApplyFtrl(std::string id, Json::Value pInputItem) {
	SparseApplyFtrl* pSparseApplyFtrl = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* linear = nullptr;
	Output* grad = nullptr;
	Output* indices = nullptr;
	Output* lr = nullptr;
	Output* l1 = nullptr;
	Output* l2 = nullptr;
	Output* lr_power = nullptr;
	SparseApplyFtrl::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "linear")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							linear = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
				std::string msg = string_format("warning : SparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l1")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l1 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l2")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l2 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr_power")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr_power = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyFtrl - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "SparseApplyFtrl::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseApplyFtrl pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && accum &&linear &&grad &&indices && lr && l1 && l2 && lr_power)
	{
		pSparseApplyFtrl = new SparseApplyFtrl(*pScope, *var, *accum, *linear, *grad, *indices, *lr, *l1, *l2, *lr_power, attrs);
		ObjectInfo* pObj = AddObjectMap(pSparseApplyFtrl, id, SYMBOL_SPARSEAPPLYFTRL, "SparseApplyFtrl", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSparseApplyFtrl->out, OUTPUT_TYPE_OUTPUT, "out");
	}
	else
	{
		std::string msg = string_format("error : SparseApplyFtrl(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pSparseApplyFtrl;
}

void* Create_SparseApplyMomentum(std::string id, Json::Value pInputItem) {
	SparseApplyMomentum* pSparseApplyMomentum = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* lr = nullptr;
	Output* grad = nullptr;
	Output* indices = nullptr;
	Output* momentum = nullptr;
	SparseApplyMomentum::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
				std::string msg = string_format("warning : SparseApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "momentum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							momentum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyMomentum - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "SparseApplyMomentum::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
				if (attrParser.GetAttribute("use_nesterov_") != "")attrs.UseNesterov(attrParser.ConvStrToBool(attrParser.GetAttribute("use_nesterov_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseApplyMomentum pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && accum &&lr &&grad &&indices && momentum)
	{
		pSparseApplyMomentum = new SparseApplyMomentum(*pScope, *var, *accum, *lr, *grad, *indices, *momentum, attrs);
		ObjectInfo* pObj = AddObjectMap(pSparseApplyMomentum, id, SYMBOL_SPARSEAPPLYMOMENTUM, "SparseApplyMomentum", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSparseApplyMomentum->out, OUTPUT_TYPE_OUTPUT, "out");
	}
	else
	{
		std::string msg = string_format("error : SparseApplyMomentum(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pSparseApplyMomentum;
}

void* Create_SparseApplyProximalAdagrad(std::string id, Json::Value pInputItem) {
	SparseApplyProximalAdagrad* pSparseApplyProximalAdagrad = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* accum = nullptr;
	Output* lr = nullptr;
	Output* l1 = nullptr;
	Output* l2 = nullptr;
	Output* grad = nullptr;
	Output* indices = nullptr;
	SparseApplyProximalAdagrad::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "accum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							accum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l1")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l1 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l2")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l2 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
				std::string msg = string_format("warning : SparseApplyProximalAdagrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "SparseApplyProximalAdagrad::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseApplyProximalAdagrad pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && accum &&lr &&l1 &&l2 &&grad && indices)
	{
		pSparseApplyProximalAdagrad = new SparseApplyProximalAdagrad(*pScope, *var, *accum, *lr, *l1, *l2, *grad, *indices, attrs);
		ObjectInfo* pObj = AddObjectMap(pSparseApplyProximalAdagrad, id, SYMBOL_SPARSEAPPLYPROXIMALADAGRAD, "SparseApplyProximalAdagrad", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSparseApplyProximalAdagrad->out, OUTPUT_TYPE_OUTPUT, "out");
	}
	else
	{
		std::string msg = string_format("error : SparseApplyProximalAdagrad(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pSparseApplyProximalAdagrad;
}

void* Create_SparseApplyProximalGradientDescent(std::string id, Json::Value pInputItem) {
	SparseApplyProximalGradientDescent* pSparseApplyProximalGradientDescent = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* alpha = nullptr;
	Output* l1 = nullptr;
	Output* l2 = nullptr;
	Output* grad = nullptr;
	Output* indices = nullptr;
	SparseApplyProximalGradientDescent::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "alpha")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							alpha = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l1")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l1 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "l2")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							l2 = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
				std::string msg = string_format("warning : SparseApplyProximalGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "SparseApplyProximalGradientDescent::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseApplyProximalGradientDescent pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && alpha && l1 &&l2 &&grad &&indices)
	{
		pSparseApplyProximalGradientDescent = new SparseApplyProximalGradientDescent(*pScope, *var, *alpha, *l1, *l2, *grad, *indices, attrs);
		ObjectInfo* pObj = AddObjectMap(pSparseApplyProximalGradientDescent, id, SYMBOL_SPARSEAPPLYPROXIMALGRADIENTDESCENT, "SparseApplyProximalGradientDescent", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSparseApplyProximalGradientDescent->out, OUTPUT_TYPE_OUTPUT, "out");
	}
	else
	{
		std::string msg = string_format("error : SparseApplyProximalGradientDescent(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pSparseApplyProximalGradientDescent;
}

void* Create_SparseApplyRMSProp(std::string id, Json::Value pInputItem) {
	SparseApplyRMSProp* pSparseApplyRMSProp = nullptr;
	Scope* pScope = nullptr;
	Output* var = nullptr;
	Output* ms = nullptr;
	Output* mom = nullptr;
	Output* lr = nullptr;
	Output* rho = nullptr;
	Output* momentum = nullptr;
	Output* epsilon = nullptr;
	Output* grad = nullptr;
	Output* indices = nullptr;
	SparseApplyRMSProp::Attrs attrs;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							var = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "ms")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							ms = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "mom")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							mom = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "lr")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							lr = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "rho")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							rho = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "momentum")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							momentum = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "epsilon")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							epsilon = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
				std::string msg = string_format("warning : SparseApplyRMSProp - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "SparseApplyRMSProp::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_locking_") != "")attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseApplyRMSProp pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope&&var && ms && mom &&lr && rho &&momentum &&epsilon &&grad)
	{
		pSparseApplyRMSProp = new SparseApplyRMSProp(*pScope, *var, *ms, *mom, *lr, *rho, *momentum, *epsilon, *grad, *indices, attrs);
		ObjectInfo* pObj = AddObjectMap(pSparseApplyRMSProp, id, SYMBOL_SPARSEAPPLYRMSPROP, "SparseApplyRMSProp", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSparseApplyRMSProp->out, OUTPUT_TYPE_OUTPUT, "out");
	}
	else
	{
		std::string msg = string_format("error : SparseApplyRMSProp(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pSparseApplyRMSProp;
}

void* Create_GradientDescentOptimizer(std::string id, Json::Value pInputItem)
{
	/*
	Scope* pScope = nullptr;
	Output* loss = nullptr;
	float learning_rate = 0;
	OutputList* train = new OutputList();
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
				std::string msg = string_format("warning : GradientDescentOptimizer - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "learning_rate")
		{
			if (strPinInterface == "float")
			{

				learning_rate = stof(strPinInitial);
			
				
			}
			else
			{
				std::string msg = string_format("warning : GradientDescentOptimizer - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "loss")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							loss = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : GradientDescentOptimizer - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
	}

	if (pScope&&loss)
	{
		*train = GradientDescentOptimizer(learning_rate).Minimize(*pScope, { *loss });

		ObjectInfo* pObj = AddObjectMap(train, id, SYMBOL_GRADIENTDESENTOPTIMIZER, "GradientDescentOptimizer", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, train, OUTPUT_TYPE_OUTPUTLIST, "train");
	}
	else
	{
		std::string msg = string_format("error : GradientDescentOptimizer(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	*/
	return nullptr;
}
