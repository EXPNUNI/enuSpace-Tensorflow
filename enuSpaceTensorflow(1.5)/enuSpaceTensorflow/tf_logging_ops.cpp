#include "stdafx.h"
#include "tf_logging_ops.h"

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



void* Create_Assert(std::string id, Json::Value pInputItem) {
	Assert* pAssert = nullptr;
	Scope* pScope = nullptr;
	Output *pcondition = nullptr;
	OutputList *pdata = nullptr;
	Assert::Attrs attrs;
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
				std::string msg = string_format("warning : Assert - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
			}
			else
			{
				std::string msg = string_format("warning : Assert - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "data")
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
							pdata = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Assert - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Assert::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("summarize_") != "") attrs = attrs.Summarize(attrParser.ConvStrToInt64(attrParser.GetAttribute("summarize_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : Assert pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && pcondition && pdata)
	{
		pAssert = new Assert(*pScope, *pcondition, *pdata, attrs);
		ObjectInfo* pObj = AddObjectMap(pAssert, id, SYMBOL_ASSERT, "Assert", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAssert->operation, OUTPUT_TYPE_OPERATION, "operation");
		}
	}
	else
	{
		std::string msg = string_format("error : Assert(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pAssert;
}

void* Create_HistogramSummary(std::string id, Json::Value pInputItem) {
	HistogramSummary* pHistogramSummary = nullptr;
	Scope* pScope = nullptr;
	Output *ptag = nullptr;
	Output *pvalues = nullptr;
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
				std::string msg = string_format("warning : HistogramSummary - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "tag")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							ptag = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : HistogramSummary - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							pvalues = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : HistogramSummary - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else
		{
			std::string msg = string_format("warning : HistogramSummary pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && ptag && pvalues)
	{
		pHistogramSummary = new HistogramSummary(*pScope, *ptag, *pvalues);
		ObjectInfo* pObj = AddObjectMap(pHistogramSummary, id, SYMBOL_HISTOGRAMSUMMARY, "HistogramSummary", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pHistogramSummary->summary, OUTPUT_TYPE_OUTPUT, "summary");
		}
	}
	else
	{
		std::string msg = string_format("error : HistogramSummary(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pHistogramSummary;
}

void* Create_MergeSummary(std::string id, Json::Value pInputItem) {
	MergeSummary* pMergeSummary = nullptr;
	Scope* pScope = nullptr;
	OutputList *pinputs = nullptr;
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
				std::string msg = string_format("warning : MergeSummary - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
			}
			else
			{
				std::string msg = string_format("warning : MergeSummary - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else
		{
			std::string msg = string_format("warning : MergeSummary pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && pinputs)
	{
		pMergeSummary = new MergeSummary(*pScope, *pinputs);
		ObjectInfo* pObj = AddObjectMap(pMergeSummary, id, SYMBOL_MERGESUMMARY, "MergeSummary", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pMergeSummary->summary, OUTPUT_TYPE_OUTPUT, "summary");
		}
	}
	else
	{
		std::string msg = string_format("error : MergeSummary(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pMergeSummary;
}

void* Create_Print(std::string id, Json::Value pInputItem) {
	Print* pPrint = nullptr;
	Scope* pScope = nullptr;
	Output *pinput = nullptr;
	OutputList *pdata = nullptr;
	Print::Attrs attrs;
	std::string message_;
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
				std::string msg = string_format("warning : Print - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
			}
			else
			{
				std::string msg = string_format("warning : Print - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "data")
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
							pdata = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Print - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Assert::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("message_") != "")
				{
					message_ = attrParser.GetAttribute("message_");
					attrs = attrs.Message(message_);
				}
				if (attrParser.GetAttribute("first_n_") != "") attrs = attrs.FirstN(attrParser.ConvStrToInt64(attrParser.GetAttribute("first_n_")));
				if (attrParser.GetAttribute("summarize_") != "") attrs = attrs.Summarize(attrParser.ConvStrToInt64(attrParser.GetAttribute("summarize_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : Print pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && pinput && pdata)
	{
		pPrint = new Print(*pScope, *pinput, *pdata, attrs);
		ObjectInfo* pObj = AddObjectMap(pPrint, id, SYMBOL_PRINT, "Print", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pPrint->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : Print(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pPrint;
}

void* Create_ScalarSummary(std::string id, Json::Value pInputItem) {
	ScalarSummary* pScalarSummary = nullptr;
	Scope* pScope = nullptr;
	Output *ptag = nullptr;
	Output *pvalues = nullptr;
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
				std::string msg = string_format("warning : ScalarSummary - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "tags")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							ptag = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ScalarSummary - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
							pvalues = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ScalarSummary - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else
		{
			std::string msg = string_format("warning : ScalarSummary pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && ptag && pvalues)
	{
		pScalarSummary = new ScalarSummary(*pScope, *ptag, *pvalues);
		ObjectInfo* pObj = AddObjectMap(pScalarSummary, id, SYMBOL_SCALARSUMMARY, "ScalarSummary", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pScalarSummary->summary, OUTPUT_TYPE_OUTPUT, "summary");
		}
	}
	else
	{
		std::string msg = string_format("error : ScalarSummary(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pScalarSummary;
}

void* Create_TensorSummary(std::string id, Json::Value pInputItem) {
	TensorSummary* pTensorSummary = nullptr;
	Scope* pScope = nullptr;
	Output *ptensor = nullptr;
	TensorSummary::Attrs attrs;
	std::string description_;
	std::string display_name_;
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
				std::string msg = string_format("warning : TensorSummary - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
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
			}
			else
			{
				std::string msg = string_format("warning : TensorSummary - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "TensorSummary::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("description_") != "")
				{
					description_ = attrParser.GetAttribute("description_");
					attrs = attrs.Description(description_);
				}
				if (attrParser.GetAttribute("labels_") != "")
				{
					std::vector<std::string> sVec;
					if (GetStringVectorFromInitial(attrParser.GetAttribute("labels_"), sVec))
					{
						gtl::ArraySlice<string> sVecSlice(sVec);
						attrs = attrs.Labels(sVecSlice);
					}
				}
				if (attrParser.GetAttribute("display_name_") != "")
				{
					display_name_ = attrParser.GetAttribute("display_name_");
					attrs = attrs.DisplayName(display_name_);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : TensorSummary pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && ptensor)
	{
		pTensorSummary = new TensorSummary(*pScope, *ptensor, attrs);
		ObjectInfo* pObj = AddObjectMap(pTensorSummary, id, SYMBOL_TENSORSUMMARY, "TensorSummary", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pTensorSummary->summary, OUTPUT_TYPE_OUTPUT, "summary");
		}
	}
	else
	{
		std::string msg = string_format("error : TensorSummary(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	return pTensorSummary;
}