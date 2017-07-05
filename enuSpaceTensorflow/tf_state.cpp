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


void* Create_Assign(std::string id, Json::Value pInputItem) {
	Assign* pAssign = nullptr;
	Scope* pScope = nullptr;
	return pAssign;
}

void* Create_AssignAdd(std::string id, Json::Value pInputItem) {
	AssignAdd* pAssignAdd = nullptr;
	Scope* pScope = nullptr;
	return pAssignAdd;
}

void* Create_AssignSub(std::string id, Json::Value pInputItem) {
	AssignSub* pAssignSub = nullptr;
	Scope* pScope = nullptr;
	return pAssignSub;
}

void* Create_CountUpTo(std::string id, Json::Value pInputItem) {
	CountUpTo* pCountUpTo = nullptr;
	Scope* pScope = nullptr;
	return pCountUpTo;
}

void* Create_DestroyTemporaryVariable(std::string id, Json::Value pInputItem) {
	DestroyTemporaryVariable* pDestroyTemporaryVariable = nullptr;
	Scope* pScope = nullptr;
	return pDestroyTemporaryVariable;
}

void* Create_IsVariableInitialized(std::string id, Json::Value pInputItem) {
	IsVariableInitialized* pIsVariableInitialized = nullptr;
	Scope* pScope = nullptr;
	return pIsVariableInitialized;
}

void* Create_ScatterAdd(std::string id, Json::Value pInputItem) {
	ScatterAdd* pScatterAdd = nullptr;
	Scope* pScope = nullptr;
	return pScatterAdd;
}

void* Create_ScatterDiv(std::string id, Json::Value pInputItem) {
	ScatterDiv* pScatterDiv = nullptr;
	Scope* pScope = nullptr;
	return pScatterDiv;
}

void* Create_ScatterMul(std::string id, Json::Value pInputItem) {
	ScatterMul* pScatterMul = nullptr;
	Scope* pScope = nullptr;
	return pScatterMul;
}

void* Create_ScatterNdAdd(std::string id, Json::Value pInputItem) {
	ScatterNdAdd* pScatterNdAdd = nullptr;
	Scope* pScope = nullptr;
	return pScatterNdAdd;
}

void* Create_ScatterNdSub(std::string id, Json::Value pInputItem) {
	ScatterNdSub* pScatterNdSub = nullptr;
	Scope* pScope = nullptr;
	return pScatterNdSub;
}

void* Create_ScatterNdUpdate(std::string id, Json::Value pInputItem) {
	ScatterNdUpdate* pScatterNdUpdate = nullptr;
	Scope* pScope = nullptr;
	return pScatterNdUpdate;
}

void* Create_ScatterSub(std::string id, Json::Value pInputItem) {
	ScatterSub* pScatterSub = nullptr;
	Scope* pScope = nullptr;
	return pScatterSub;
}

void* Create_ScatterUpdate(std::string id, Json::Value pInputItem) {
	ScatterUpdate* pScatterUpdate = nullptr;
	Scope* pScope = nullptr;
	return pScatterUpdate;
}

void* Create_TemporaryVariable(std::string id, Json::Value pInputItem) {
	TemporaryVariable* pTemporaryVariable = nullptr;
	Scope* pScope = nullptr;
	return pTemporaryVariable;
}

void* Create_Variable(std::string id, Json::Value pInputItem) {
	Scope* pScope = nullptr;
	tensorflow::ops::Variable* pOutput = nullptr;
	tensorflow::TensorShape shape;
	tensorflow::DataType dtype = DT_DOUBLE;
	tensorflow::ops::Variable::Attrs attrs;

	tensorflow::Input* pInput = nullptr;

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
				std::string msg = string_format("warning : Variable - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "shape")
		{
			if (strPinInterface == "TensorShape")
			{
				std::vector<int64> arrayslice;
				std::vector<int64> arraydims;
				GetArrayDimsFromShape(strPinInitial, arraydims, arrayslice);
				gtl::ArraySlice< int64 > arraySlice(arraydims);
				shape = TensorShape(arraySlice);
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
				if (strPinInitial == "double")
					dtype = DT_DOUBLE;
				else if (strPinInitial == "float")
					dtype = DT_FLOAT;
				else if (strPinInitial == "int")
					dtype = DT_INT32;
				else if (strPinInitial == "bool")
					dtype = DT_BOOL;
				else if (strPinInitial == "string")
					dtype = DT_STRING;
				else
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
			if (strPinInterface == "ops::Variable::Attrs")
			{
				std::string attr;
				std::string value;
				int iType = 0;
				for (std::string::size_type i = 0; i < strPinInitial.size(); i++)
				{
					if (strPinInitial[i] == '=')
					{
						iType = 1;
					}
					else if (strPinInitial[i] == ';')
					{
						iType = 0;
						if (attr == "container_")
							attrs.Container(value);
						else if (attr == "shared_name")
							attrs.SharedName(value);

						attr = "";
						value = "";
					}
					else
					{
						if (iType == 0)
							attr = attr + strPinInitial[i];
						else
							value = value + strPinInitial[i];
					}
				}
				if (attr.length() > 0)
				{
					if (attr == "container_")
						attrs.Container(value);
					else if (attr == "shared_name_")
						attrs.SharedName(value);
				}
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
			AddOutputInfo(pObj, &pOutput->ref, OUTPUT_TYPE_OUTPUT, "output");
			//	pObj->pOutput = &pOutput->ref;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
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
					GetDoubleVectorFormInitial(strPinInitial, arrayvals);

					gtl::ArraySlice< int64 > arraySlice(arraydims);
					pTensor = new Tensor(DT_DOUBLE, TensorShape(arraySlice));

					int i = 0;
					for (std::vector<double>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
					{
						pTensor->flat<double>()(i) = *it;
						i++;
					}
				}
				else if (strPinType == "float")
				{
					std::vector<float> arrayvals;
					GetFloatVectorFormInitial(strPinInitial, arrayvals);

					gtl::ArraySlice< int64 > arraySlice(arraydims);
					pTensor = new Tensor(DT_FLOAT, TensorShape(arraySlice));

					int i = 0;
					for (std::vector<float>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
					{
						pTensor->flat<float>()(i) = *it;
						i++;
					}
				}
				else if (strPinType == "int")
				{
					std::vector<int> arrayvals;
					GetIntVectorFormInitial(strPinInitial, arrayvals);

					gtl::ArraySlice< int64 > arraySlice(arraydims);
					pTensor = new Tensor(DT_INT32, TensorShape(arraySlice));

					int i = 0;
					for (std::vector<int>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
					{
						pTensor->flat<int>()(i) = *it;
						i++;
					}
				}
				else if (strPinType == "bool")
				{
					std::vector<bool> arrayvals;
					GetBoolVectorFormInitial(strPinInitial, arrayvals);

					gtl::ArraySlice< int64 > arraySlice(arraydims);
					pTensor = new Tensor(DT_BOOL, TensorShape(arraySlice));

					int i = 0;
					for (std::vector<bool>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
					{
						pTensor->flat<bool>()(i) = *it;
						i++;
					}
				}
				else if (strPinType == "string")
				{
					std::vector<std::string> arrayvals;
					GetStringVectorFormInitial(strPinInitial, arrayvals);

					gtl::ArraySlice< int64 > arraySlice(arraydims);
					pTensor = new Tensor(DT_STRING, TensorShape(arraySlice));

					int i = 0;
					for (std::vector<std::string>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
					{
						pTensor->flat<std::string>()(i) = *it;
						i++;
					}
				}
				else
				{
					std::string msg = string_format("warning : Const - %s(val-initvalue) transfer information missed.", id.c_str());
					PrintMessage(msg);
				}
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