#include "stdafx.h"
#include "tf_random.h"

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

void* Create_Multinomial(std::string id, Json::Value pInputItem) {
	Multinomial* pMultinomial = nullptr;
	Scope* pScope = nullptr;
	return pMultinomial;
}

void* Create_ParameterizedTruncatedNormal(std::string id, Json::Value pInputItem) {
	ParameterizedTruncatedNormal* pParameterizedTruncatedNormal = nullptr;
	Scope* pScope = nullptr;
	return pParameterizedTruncatedNormal;
}

void* Create_RandomGamma(std::string id, Json::Value pInputItem) {
	RandomGamma* pRandomGamma = nullptr;
	Scope* pScope = nullptr;
	return pRandomGamma;
}

void* Create_RandomNormal(std::string id, Json::Value pInputItem) {
	RandomNormal* pRandomNormal = nullptr;
	Scope* pScope = nullptr;
	Input* pShape = nullptr;
	tensorflow::DataType dtype = DT_DOUBLE;
	tensorflow::ops::RandomNormal::Attrs attrs;

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
				std::string msg = string_format("warning : RandomNormal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "shape")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					if (pObj->type == SYMBOL_INPUT || pObj->type == SYMBOL_INPUT_EX)
					{
						pShape = (Input*)pObj->pObject;		// SYMBOL_INPUT은 자체가 Input 임.
					}
					else
					{
						std::string msg = string_format("warning : RandomNormal - %s(%s) Could not set Input object.", id.c_str(), strPinName.c_str());
						PrintMessage(msg);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : RandomNormal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				//else if (strPinInitial == "bool")
				//	dtype = DT_BOOL;
				//else if (strPinInitial == "string")
				//	dtype = DT_STRING;
				else
				{
					std::string msg = string_format("warning : RandomNormal - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : RandomNormal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ops::RandomNormal::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				attrs.Seed(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed_")));
				attrs.Seed2(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed2_")));
			}
			else
			{
				std::string msg = string_format("warning : RandomNormal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : RandomNormal pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pShape)
	{
		std::vector<int64> arrayslice;

		Tensor orgtensor = pShape->tensor();
		int idim = orgtensor.dims();

		arrayslice.push_back(idim);

		gtl::ArraySlice< int64 > arraySlice(arrayslice);
		TensorShape shape = TensorShape(arraySlice);

		Tensor tensor(DT_INT32, shape);

		for (int i = 0; i < idim; i++)
		{
			int64 idim = orgtensor.dim_size(i);
			tensor.flat<int>()(i) = idim;
		}

		Input input(tensor);
		pRandomNormal = new RandomNormal(*pScope, input, dtype, attrs);

		//pRandomNormal = new RandomNormal(*pScope, *pShape, dtype, attrs);

		ObjectInfo* pObj = AddObjectMap(pRandomNormal, id, SYMBOL_RANDOMNORMAL, "RandomNormal", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pRandomNormal->output, OUTPUT_TYPE_OUTPUT, "output");
			// pObj->pOutput = &pRandomNormal->output;
	}
	else
	{
		std::string msg = string_format("error : RandomNormal(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRandomNormal;
}

void* Create_RandomPoisson(std::string id, Json::Value pInputItem) {
	RandomPoisson* pRandomPoisson = nullptr;
	Scope* pScope = nullptr;
	return pRandomPoisson;
}

void* Create_RandomShuffle(std::string id, Json::Value pInputItem) {
	RandomShuffle* pRandomShuffle = nullptr;
	Scope* pScope = nullptr;
	return pRandomShuffle;
}

void* Create_RandomUniform(std::string id, Json::Value pInputItem) {
	RandomUniform* pRandomUniform = nullptr;
	Scope* pScope = nullptr;
	return pRandomUniform;
}

void* Create_RandomUniformInt(std::string id, Json::Value pInputItem) {
	RandomUniformInt* pRandomUniformInt = nullptr;
	Scope* pScope = nullptr;
	return pRandomUniformInt;
}

void* Create_TruncatedNormal(std::string id, Json::Value pInputItem) {
	TruncatedNormal* pTruncatedNormal = nullptr;
	Scope* pScope = nullptr;
	return pTruncatedNormal;
}


void* Create_RandomNormal_ex(std::string id, Json::Value pInputItem)
{
	RandomNormal* pRandomNormal = nullptr;

	Scope* pScope = nullptr;
	tensorflow::TensorShape shape;
	tensorflow::DataType dtype = DT_DOUBLE;
	tensorflow::ops::RandomNormal::Attrs attrs;

	std::string str_shape;

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
				std::string msg = string_format("warning : RandomNormal_ex - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "shape")
		{
			if (strPinInterface == "TensorShape")
			{
				str_shape = strPinInitial;
				//std::vector<int64> arraydims;
				//GetArrayDimsFromShape(strPinInitial, arraydims);
				//gtl::ArraySlice< int64 > arraySlice(arraydims);
				//shape = TensorShape(arraySlice);
			}
			else
			{
				std::string msg = string_format("warning : RandomNormal_ex - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				//else if (strPinInitial == "bool")
				//	dtype = DT_BOOL;
				//else if (strPinInitial == "string")
				//	dtype = DT_STRING;
				else
				{
					std::string msg = string_format("warning : RandomNormal_ex - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : RandomNormal_ex - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ops::RandomNormal::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				attrs.Seed(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed_")));
				attrs.Seed2(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed2_")));
			}
			else
			{
				std::string msg = string_format("warning : RandomNormal_ex - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : RandomNormal_ex pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		std::vector<int64> arraydims;
		std::vector<int64> arrayslice;
		GetArrayDimsFromShape(str_shape, arraydims, arrayslice);
		gtl::ArraySlice< int64 > arraySlice(arrayslice);
		shape = TensorShape(arraySlice);

		Tensor tensor(DT_INT32, shape);

		int i = 0;
		for (std::vector<int64>::iterator it = arraydims.begin(); it != arraydims.end(); it++)
		{
			int idim = *it;
			tensor.flat<int>()(i) = idim;
			i++;
		}

		Input input(tensor);
		pRandomNormal = new RandomNormal(*pScope, input, dtype, attrs);

		ObjectInfo* pObj = AddObjectMap(pRandomNormal, id, SYMBOL_RANDOMNORMAL_EX, "RandomNormal_ex", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pRandomNormal->output, OUTPUT_TYPE_OUTPUT, "output");
		//	pObj->pOutput = &pRandomNormal->output;
	}
	else
	{
		std::string msg = string_format("error : RandomNormal_ex(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRandomNormal;
}