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

void* Create_ApplyAdadelta(std::string id, Json::Value pInputItem) {
	ApplyAdadelta* pApplyAdadelta = nullptr;
	Scope* pScope = nullptr;
	return pApplyAdadelta;
}

void* Create_ApplyAdagrad(std::string id, Json::Value pInputItem) {
	ApplyAdagrad* pApplyAdagrad = nullptr;
	Scope* pScope = nullptr;
	return pApplyAdagrad;
}

void* Create_ApplyAdagradDA(std::string id, Json::Value pInputItem) {
	ApplyAdagradDA* pApplyAdagradDA = nullptr;
	Scope* pScope = nullptr;
	return pApplyAdagradDA;
}

void* Create_ApplyAdam(std::string id, Json::Value pInputItem) {
	ApplyAdam* pApplyAdam = nullptr;
	Scope* pScope = nullptr;
	return pApplyAdam;
}

void* Create_ApplyCenteredRMSProp(std::string id, Json::Value pInputItem) {
	ApplyCenteredRMSProp* pApplyCenteredRMSProp = nullptr;
	Scope* pScope = nullptr;
	return pApplyCenteredRMSProp;
}

void* Create_ApplyFtrl(std::string id, Json::Value pInputItem) {
	ApplyFtrl* pApplyFtrl = nullptr;
	Scope* pScope = nullptr;
	return pApplyFtrl;
}


void* Create_ApplyGradientDescent(std::string id, Json::Value pInputItem)
{
	Scope* pScope = nullptr;
	Output* pVar = nullptr;
	Output* pAlpha = nullptr;
	Output* pDelta = nullptr;
	ApplyGradientDescent::Attrs attrs;
	ApplyGradientDescent* pGradientDescent = nullptr;

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
				PrintMessage(msg);
			}
		}
		else if (strPinName == "var")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, "output");
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pVar = (Output*)pOutputObj->pOutput;
						}
					}

					// pVar = pObj->pOutput;
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "alpha")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, "output");
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pAlpha = (Output*)pOutputObj->pOutput;
						}
					}
					// pAlpha = pObj->pOutput;
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, "output");
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pDelta = (Output*)pOutputObj->pOutput;
						}
					}

					// pDelta = pObj->pOutput;
				}
			}
			else
			{
				std::string msg = string_format("warning : ApplyGradientDescent - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			CAttributeParser attrParser(strPinInterface, strPinInitial);
			attrs.UseLocking(attrParser.ConvStrToBool(attrParser.GetAttribute("use_locking_")));
		}
		else
		{
			std::string msg = string_format("warning : ApplyGradientDescent pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pVar && pAlpha && pDelta)
	{
		pGradientDescent = new ApplyGradientDescent(*pScope, *pVar, *pAlpha, *pDelta, attrs);
		ObjectInfo* pObj = AddObjectMap(pGradientDescent, id, SYMBOL_APPLYGRADIENTDESCENT, "ApplyGradientDescent", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pGradientDescent->out, OUTPUT_TYPE_OUTPUT, "output");
		//	pObj->pOutput = &pGradientDescent->out;
	}
	else
	{
		std::string msg = string_format("error : ApplyGradientDescent(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pGradientDescent;
}

void* Create_ApplyMomentum(std::string id, Json::Value pInputItem) {
	ApplyMomentum* pApplyMomentum = nullptr;
	Scope* pScope = nullptr;
	return pApplyMomentum;
}

void* Create_ApplyProximalAdagrad(std::string id, Json::Value pInputItem) {
	ApplyProximalAdagrad* pApplyProximalAdagrad = nullptr;
	Scope* pScope = nullptr;
	return pApplyProximalAdagrad;
}

void* Create_ApplyProximalGradientDescent(std::string id, Json::Value pInputItem) {
	ApplyProximalGradientDescent* pApplyProximalGradientDescent = nullptr;
	Scope* pScope = nullptr;
	return pApplyProximalGradientDescent;
}

void* Create_ApplyRMSProp(std::string id, Json::Value pInputItem) {
	ApplyRMSProp* pApplyRMSProp = nullptr;
	Scope* pScope = nullptr;
	return pApplyRMSProp;
}

void* Create_ResourceApplyAdadelta(std::string id, Json::Value pInputItem) {
	ResourceApplyAdadelta* pResourceApplyAdadelta = nullptr;
	Scope* pScope = nullptr;
	return pResourceApplyAdadelta;
}

void* Create_ResourceApplyAdagrad(std::string id, Json::Value pInputItem) {
	ResourceApplyAdagrad* pResourceApplyAdagrad = nullptr;
	Scope* pScope = nullptr;
	return pResourceApplyAdagrad;
}

void* Create_ResourceApplyAdagradDA(std::string id, Json::Value pInputItem) {
	ResourceApplyAdagradDA* pResourceApplyAdagradDA = nullptr;
	Scope* pScope = nullptr;
	return pResourceApplyAdagradDA;
}

void* Create_ResourceApplyAdam(std::string id, Json::Value pInputItem) {
	ResourceApplyAdam* pResourceApplyAdam = nullptr;
	Scope* pScope = nullptr;
	return pResourceApplyAdam;
}

void* Create_ResourceApplyCenteredRMSProp(std::string id, Json::Value pInputItem) {
	ResourceApplyCenteredRMSProp* pResourceApplyCenteredRMSProp = nullptr;
	Scope* pScope = nullptr;
	return pResourceApplyCenteredRMSProp;
}

void* Create_ResourceApplyFtrl(std::string id, Json::Value pInputItem) {
	ResourceApplyFtrl* pResourceApplyFtrl = nullptr;
	Scope* pScope = nullptr;
	return pResourceApplyFtrl;
}

void* Create_ResourceApplyGradientDescent(std::string id, Json::Value pInputItem) {
	ResourceApplyGradientDescent* pResourceApplyGradientDescent = nullptr;
	Scope* pScope = nullptr;
	return pResourceApplyGradientDescent;
}

void* Create_ResourceApplyMomentum(std::string id, Json::Value pInputItem) {
	ResourceApplyMomentum* pResourceApplyMomentum = nullptr;
	Scope* pScope = nullptr;
	return pResourceApplyMomentum;
}

void* Create_ResourceApplyProximalAdagrad(std::string id, Json::Value pInputItem) {
	ResourceApplyProximalAdagrad* pResourceApplyProximalAdagrad = nullptr;
	Scope* pScope = nullptr;
	return pResourceApplyProximalAdagrad;
}

void* Create_ResourceApplyProximalGradientDescent(std::string id, Json::Value pInputItem) {
	ResourceApplyProximalGradientDescent* pResourceApplyProximalGradientDescent = nullptr;
	Scope* pScope = nullptr;
	return pResourceApplyProximalGradientDescent;
}

void* Create_ResourceApplyRMSProp(std::string id, Json::Value pInputItem) {
	ResourceApplyRMSProp* pResourceApplyRMSProp = nullptr;
	Scope* pScope = nullptr;
	return pResourceApplyRMSProp;
}

void* Create_ResourceSparseApplyAdadelta(std::string id, Json::Value pInputItem) {
	ResourceSparseApplyAdadelta* pResourceSparseApplyAdadelta = nullptr;
	Scope* pScope = nullptr;
	return pResourceSparseApplyAdadelta;
}

void* Create_ResourceSparseApplyAdagrad(std::string id, Json::Value pInputItem) {
	ResourceSparseApplyAdagrad* pResourceSparseApplyAdagrad = nullptr;
	Scope* pScope = nullptr;
	return pResourceSparseApplyAdagrad;
}

void* Create_ResourceSparseApplyAdagradDA(std::string id, Json::Value pInputItem) {
	ResourceSparseApplyAdagradDA* pResourceSparseApplyAdagradDA = nullptr;
	Scope* pScope = nullptr;
	return pResourceSparseApplyAdagradDA;
}

void* Create_ResourceSparseApplyCenteredRMSProp(std::string id, Json::Value pInputItem) {
	ResourceSparseApplyCenteredRMSProp* pResourceSparseApplyCenteredRMSProp = nullptr;
	Scope* pScope = nullptr;
	return pResourceSparseApplyCenteredRMSProp;
}

void* Create_ResourceSparseApplyFtrl(std::string id, Json::Value pInputItem) {
	ResourceSparseApplyFtrl* pResourceSparseApplyFtrl = nullptr;
	Scope* pScope = nullptr;
	return pResourceSparseApplyFtrl;
}

void* Create_ResourceSparseApplyMomentum(std::string id, Json::Value pInputItem) {
	ResourceSparseApplyMomentum* pResourceSparseApplyMomentum = nullptr;
	Scope* pScope = nullptr;
	return pResourceSparseApplyMomentum;
}

void* Create_ResourceSparseApplyProximalAdagrad(std::string id, Json::Value pInputItem) {
	ResourceSparseApplyProximalAdagrad* pResourceSparseApplyProximalAdagrad = nullptr;
	Scope* pScope = nullptr;
	return pResourceSparseApplyProximalAdagrad;
}

void* Create_ResourceSparseApplyProximalGradientDescent(std::string id, Json::Value pInputItem) {
	ResourceSparseApplyProximalGradientDescent* pResourceSparseApplyProximalGradientDescent = nullptr;
	Scope* pScope = nullptr;
	return pResourceSparseApplyProximalGradientDescent;
}

void* Create_ResourceSparseApplyRMSProp(std::string id, Json::Value pInputItem) {
	ResourceSparseApplyRMSProp* pResourceSparseApplyRMSProp = nullptr;
	Scope* pScope = nullptr;
	return pResourceSparseApplyRMSProp;
}

void* Create_SparseApplyAdadelta(std::string id, Json::Value pInputItem) {
	SparseApplyAdadelta* pSparseApplyAdadelta = nullptr;
	Scope* pScope = nullptr;
	return pSparseApplyAdadelta;
}

void* Create_SparseApplyAdagrad(std::string id, Json::Value pInputItem) {
	SparseApplyAdagrad* pSparseApplyAdagrad = nullptr;
	Scope* pScope = nullptr;
	return pSparseApplyAdagrad;
}

void* Create_SparseApplyAdagradDA(std::string id, Json::Value pInputItem) {
	SparseApplyAdagradDA* pSparseApplyAdagradDA = nullptr;
	Scope* pScope = nullptr;
	return pSparseApplyAdagradDA;
}

void* Create_SparseApplyCenteredRMSProp(std::string id, Json::Value pInputItem) {
	SparseApplyCenteredRMSProp* pSparseApplyCenteredRMSProp = nullptr;
	Scope* pScope = nullptr;
	return pSparseApplyCenteredRMSProp;
}

void* Create_SparseApplyFtrl(std::string id, Json::Value pInputItem) {
	SparseApplyFtrl* pSparseApplyFtrl = nullptr;
	Scope* pScope = nullptr;
	return pSparseApplyFtrl;
}

void* Create_SparseApplyMomentum(std::string id, Json::Value pInputItem) {
	SparseApplyMomentum* pSparseApplyMomentum = nullptr;
	Scope* pScope = nullptr;
	return pSparseApplyMomentum;
}

void* Create_SparseApplyProximalAdagrad(std::string id, Json::Value pInputItem) {
	SparseApplyProximalAdagrad* pSparseApplyProximalAdagrad = nullptr;
	Scope* pScope = nullptr;
	return pSparseApplyProximalAdagrad;
}

void* Create_SparseApplyProximalGradientDescent(std::string id, Json::Value pInputItem) {
	SparseApplyProximalGradientDescent* pSparseApplyProximalGradientDescent = nullptr;
	Scope* pScope = nullptr;
	return pSparseApplyProximalGradientDescent;
}

void* Create_SparseApplyRMSProp(std::string id, Json::Value pInputItem) {
	SparseApplyRMSProp* pSparseApplyRMSProp = nullptr;
	Scope* pScope = nullptr;
	return pSparseApplyRMSProp;
}
