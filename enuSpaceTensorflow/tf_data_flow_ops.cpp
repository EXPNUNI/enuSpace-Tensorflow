#include "stdafx.h"
#include "tf_data_flow_ops.h"

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

void* Create_AccumulatorApplyGradient(std::string id, Json::Value pInputItem) {
	AccumulatorApplyGradient* pAccumulatorApplyGradient = nullptr;
	Scope* pScope = nullptr;
	Output *phandle = nullptr;
	Output *plocal_step = nullptr;
	Output *pgradient = nullptr;
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
				std::string msg = string_format("warning : AccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "local_step")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							plocal_step = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "gradient")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pgradient = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pAccumulatorApplyGradient = new AccumulatorApplyGradient(*pScope, *phandle, *plocal_step, *pgradient);
		ObjectInfo* pObj = AddObjectMap(pAccumulatorApplyGradient, id, SYMBOL_ACCUMULATORAPPLYGRADIENT, "AccumulatorApplyGradient", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAccumulatorApplyGradient->operation, OUTPUT_TYPE_OPERATION, "operation");
		}
	}
	else
	{
		std::string msg = string_format("error : AccumulatorApplyGradient(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAccumulatorApplyGradient;
}

void* Create_AccumulatorNumAccumulated(std::string id, Json::Value pInputItem) {
	AccumulatorNumAccumulated* pAccumulatorNumAccumulated = nullptr;
	Scope* pScope = nullptr;
	Output *phandle = nullptr;
	Output *pnew_global_step = nullptr;
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
				std::string msg = string_format("warning : AccumulatorSetGlobalStep - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AccumulatorNumAccumulated - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pAccumulatorNumAccumulated = new AccumulatorNumAccumulated(*pScope, *phandle);
		ObjectInfo* pObj = AddObjectMap(pAccumulatorNumAccumulated, id, SYMBOL_ACCUMULATORNUMACCUMULATED, "AccumulatorNumAccumulated", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAccumulatorNumAccumulated->num_accumulated, OUTPUT_TYPE_OUTPUT, "num_accumulated");
		}
	}
	else
	{
		std::string msg = string_format("error : AccumulatorNumAccumulated(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAccumulatorNumAccumulated;
}

void* Create_AccumulatorSetGlobalStep(std::string id, Json::Value pInputItem) {
	AccumulatorSetGlobalStep* pAccumulatorSetGlobalStep = nullptr;
	Scope* pScope = nullptr;
	Output *phandle = nullptr;
	Output *pnew_global_step = nullptr;
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
				std::string msg = string_format("warning : AccumulatorSetGlobalStep - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AccumulatorSetGlobalStep - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "new_global_step")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pnew_global_step = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AccumulatorSetGlobalStep - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pAccumulatorSetGlobalStep = new AccumulatorSetGlobalStep(*pScope, *phandle, *pnew_global_step);
		ObjectInfo* pObj = AddObjectMap(pAccumulatorSetGlobalStep, id, SYMBOL_ACCUMULATORSETGLOBALSTEP, "AccumulatorSetGlobalStep", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAccumulatorSetGlobalStep->operation, OUTPUT_TYPE_OPERATION, "operation");
		}
	}
	else
	{
		std::string msg = string_format("error : AccumulatorSetGlobalStep(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAccumulatorSetGlobalStep;
}

void* Create_AccumulatorTakeGradient(std::string id, Json::Value pInputItem) {
	AccumulatorTakeGradient* pAccumulatorTakeGradient = nullptr;
	Scope* pScope = nullptr;
	Output *phandle = nullptr;
	Output *pnum_required = nullptr;
	DataType dtype;
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
				std::string msg = string_format("warning : AccumulatorTakeGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AccumulatorTakeGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_required")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pnum_required = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AccumulatorTakeGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				//   dtype = DT_BOOL;
				//else if (strPinInitial == "string")
				//   dtype = DT_STRING;
				else
				{
					std::string msg = string_format("warning : AccumulatorTakeGradient - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : AccumulatorTakeGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pAccumulatorTakeGradient = new AccumulatorTakeGradient(*pScope, *phandle, *pnum_required, dtype);
		ObjectInfo* pObj = AddObjectMap(pAccumulatorTakeGradient, id, SYMBOL_ACCUMULATORTAKEGRADIENT, "AccumulatorTakeGradient", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAccumulatorTakeGradient->average, OUTPUT_TYPE_OUTPUT, "average");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseAccumulatorTakeGradient(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}

	return pAccumulatorTakeGradient;
}

void* Create_Barrier(std::string id, Json::Value pInputItem) {
	Barrier* pBarrier = nullptr;
	Scope* pScope = nullptr;
	DataTypeSlice component_types;
	Barrier::Attrs attrs;
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
				std::string msg = string_format("warning : Barrier - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "component_types")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				component_types = GetDatatypeSliceFromInital(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : Barrier - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Barrier::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				//attrParser.ConvStrToArraySliceTensorshape(attrParser.GetAttribute("shapes_"));
				if (attrParser.GetAttribute("shapes_") != "") attrs.Shapes(attrParser.GetValue_arraySliceTensorshape("shapes_"));
				if (attrParser.GetAttribute("capacity_") != "") attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "") attrs.Container(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("container_")));
				if (attrParser.GetAttribute("shared_name_") != "") attrs.SharedName(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("shared_name_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : Barrier pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pBarrier = new Barrier(*pScope, component_types, attrs);
		ObjectInfo* pObj = AddObjectMap(pBarrier, id, SYMBOL_BARRIER, "Barrier", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pBarrier->handle, OUTPUT_TYPE_OUTPUT, "handle");
	}
	else
	{
		std::string msg = string_format("error : Barrier(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pBarrier;
}

void* Create_BarrierClose(std::string id, Json::Value pInputItem) {
	BarrierClose* pBarrierClose = nullptr;
	Scope* pScope = nullptr;
	Output *phandle = nullptr;
	BarrierClose::Attrs attrs;
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
				std::string msg = string_format("warning : BarrierClose - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : BarrierClose - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "BarrierClose::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("CancelPendingEnqueues") != "") attrs.CancelPendingEnqueues(attrParser.ConvStrToBool(attrParser.GetAttribute("CancelPendingEnqueues")));
			}
		}
		else
		{
			std::string msg = string_format("warning : BarrierClose pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}

	}

	if (pScope)
	{
		pBarrierClose = new BarrierClose(*pScope, *phandle, attrs);
		ObjectInfo* pObj = AddObjectMap(pBarrierClose, id, SYMBOL_BARRIERREADYSIZE, "BarrierClose", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pBarrierClose->operation, OUTPUT_TYPE_OPERATION, "operation");
		}
	}
	else
	{
		std::string msg = string_format("error : BarrierClose(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pBarrierClose;
}

void* Create_BarrierIncompleteSize(std::string id, Json::Value pInputItem) {
	BarrierIncompleteSize* pBarrierIncompleteSize = nullptr;
	Scope* pScope = nullptr;
	Output *phandle = nullptr;
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
				std::string msg = string_format("warning : BarrierIncompleteSize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : BarrierIncompleteSize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pBarrierIncompleteSize = new BarrierIncompleteSize(*pScope, *phandle);
		ObjectInfo* pObj = AddObjectMap(pBarrierIncompleteSize, id, SYMBOL_BARRIERREADYSIZE, "BarrierIncompleteSize", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pBarrierIncompleteSize->size, OUTPUT_TYPE_OUTPUT, "size");
		}
	}
	else
	{
		std::string msg = string_format("error : BarrierIncompleteSize(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pBarrierIncompleteSize;
}

void* Create_BarrierInsertMany(std::string id, Json::Value pInputItem) {
	BarrierInsertMany* pBarrierInsertMany = nullptr;
	Scope* pScope = nullptr;
	Output *phandle = nullptr;
	Output *pkeys = nullptr;
	Output *pvalues = nullptr;
	int64 component_index = 0;
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
				std::string msg = string_format("warning : BarrierInsertMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : BarrierInsertMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "keys")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pkeys = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : BarrierInsertMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
				std::string msg = string_format("warning : BarrierInsertMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "component_index")
		{
			if (strPinInterface == "Input")
			{
				component_index = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : BarrierInsertMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}

	}

	if (pScope)
	{
		pBarrierInsertMany = new BarrierInsertMany(*pScope, *phandle, *pkeys, *pvalues, component_index);
		ObjectInfo* pObj = AddObjectMap(pBarrierInsertMany, id, SYMBOL_BARRIERINSERTMANY, "BarrierInsertMany", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pBarrierInsertMany->operation, OUTPUT_TYPE_OPERATION, "operation");
		}
	}
	else
	{
		std::string msg = string_format("error : BarrierInsertMany(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pBarrierInsertMany;
}

void* Create_BarrierReadySize(std::string id, Json::Value pInputItem) {
	BarrierReadySize* pBarrierReadySize = nullptr;
	Scope* pScope = nullptr;
	Output *phandle = nullptr;
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
				std::string msg = string_format("warning : BarrierReadySize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : BarrierReadySize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pBarrierReadySize = new BarrierReadySize(*pScope, *phandle);
		ObjectInfo* pObj = AddObjectMap(pBarrierReadySize, id, SYMBOL_BARRIERREADYSIZE, "BarrierReadySize", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pBarrierReadySize->size, OUTPUT_TYPE_OUTPUT, "size");
		}
	}
	else
	{
		std::string msg = string_format("error : BarrierReadySize(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pBarrierReadySize;
}

void* Create_BarrierTakeMany(std::string id, Json::Value pInputItem) {
	BarrierTakeMany* pBarrierTakeMany = nullptr;
	Scope* pScope = nullptr;
	Output *phandle = nullptr;
	Output *pnum_elements = nullptr;
	DataTypeSlice component_types;
	BarrierTakeMany::Attrs attrs;
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
				std::string msg = string_format("warning : BarrierTakeMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : BarrierTakeMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_elements")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pnum_elements = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : BarrierTakeMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "component_types")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				component_types = GetDatatypeSliceFromInital(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : BarrierTakeMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "BarrierTakeMany::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("allow_small_batch_") != "") attrs.AllowSmallBatch(attrParser.ConvStrToBool(attrParser.GetAttribute("allow_small_batch_")));
				if (attrParser.GetAttribute("wait_for_incomplete_") != "") attrs.WaitForIncomplete(attrParser.ConvStrToBool(attrParser.GetAttribute("wait_for_incomplete_")));
				if (attrParser.GetAttribute("timeout_ms_") != "") attrs.TimeoutMs(attrParser.ConvStrToInt64(attrParser.GetAttribute("timeout_ms_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : BarrierTakeMany pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pBarrierTakeMany = new BarrierTakeMany(*pScope, *phandle, *pnum_elements, component_types, attrs);
		ObjectInfo* pObj = AddObjectMap(pBarrierTakeMany, id, SYMBOL_BARRIERTAKEMANY, "BarrierTakeMany", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pBarrierTakeMany->indices, OUTPUT_TYPE_OUTPUT, "indices");
			AddOutputInfo(pObj, &pBarrierTakeMany->keys, OUTPUT_TYPE_OUTPUT, "keys");
			AddOutputInfo(pObj, &pBarrierTakeMany->values, OUTPUT_TYPE_OUTPUTLIST, "values");
		}
	}
	else
	{
		std::string msg = string_format("error : BarrierTakeMany(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pBarrierTakeMany;
}

void* Create_ConditionalAccumulator(std::string id, Json::Value pInputItem) {
	ConditionalAccumulator* pConditionalAccumulator = nullptr;
	Scope* pScope = nullptr;
	DataType dtype;
	PartialTensorShape shape;
	ConditionalAccumulator::Attrs attrs;
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
				std::string msg = string_format("warning : ConditionalAccumulator - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "shapes")
		{
			if (strPinInterface == "PartialTensorShape")
			{
				shape = GetPartialShapeFromInital(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : ConditionalAccumulator - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				//   dtype = DT_BOOL;
				//else if (strPinInitial == "string")
				//   dtype = DT_STRING;
				else
				{
					std::string msg = string_format("warning : ConditionalAccumulator - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : ConditionalAccumulator - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "PriorityQueue::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("container_") != "") attrs.Container(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("container_")));
				if (attrParser.GetAttribute("shared_name_") != "") attrs.SharedName(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("shared_name_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ConditionalAccumulator pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pConditionalAccumulator = new ConditionalAccumulator(*pScope, dtype, shape, attrs);
		ObjectInfo* pObj = AddObjectMap(pConditionalAccumulator, id, SYMBOL_CONDITIONALACCUMULATOR, "ConditionalAccumulator", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pConditionalAccumulator->handle, OUTPUT_TYPE_OUTPUT, "handle");
	}
	else
	{
		std::string msg = string_format("error : ConditionalAccumulator(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pConditionalAccumulator;
}

void* Create_DeleteSessionTensor(std::string id, Json::Value pInputItem) {
	DeleteSessionTensor* pDeleteSessionTensor = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
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
				std::string msg = string_format("warning : DeleteSessionTensor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DeleteSessionTensor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pDeleteSessionTensor = new DeleteSessionTensor(*pScope, *phandle);
		ObjectInfo* pObj = AddObjectMap(pDeleteSessionTensor, id, SYMBOL_DELETESESSIONTENSOR, "DeleteSessionTensor", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pDeleteSessionTensor->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : DeleteSessionTensor(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pDeleteSessionTensor;
}

void* Create_DynamicPartition(std::string id, Json::Value pInputItem) {
	DynamicPartition* pDynamicPartition = nullptr;
	Scope* pScope = nullptr;
	Output *pdata = nullptr;
	Output *ppartitions = nullptr;
	int64 num_partitions = 0;
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
				std::string msg = string_format("warning : DynamicPartition - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
			}
			else
			{
				std::string msg = string_format("warning : DynamicPartition - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "partitions")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							ppartitions = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DynamicPartition - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_partitions")
		{
			if (strPinInterface == "int64")
			{
				num_partitions = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : DynamicPartition - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pDynamicPartition = new DynamicPartition(*pScope, *pdata, *ppartitions, num_partitions);
		ObjectInfo* pObj = AddObjectMap(pDynamicPartition, id, SYMBOL_DYNAMICPARTITION, "DynamicPartition", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pDynamicPartition->outputs, OUTPUT_TYPE_OUTPUTLIST, "outputs");
	}
	else
	{
		std::string msg = string_format("error : DynamicPartition(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pDynamicPartition;
}

void* Create_DynamicStitch(std::string id, Json::Value pInputItem) {
	DynamicStitch* pDynamicStitch = nullptr;
	Scope* pScope = nullptr;
	OutputList *pindices = nullptr;
	OutputList *pdata = nullptr;
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
				std::string msg = string_format("warning : DynamicStitch - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pindices = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DynamicStitch - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pdata = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DynamicStitch - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pDynamicStitch = new DynamicStitch(*pScope, *pindices, *pdata);
		ObjectInfo* pObj = AddObjectMap(pDynamicStitch, id, SYMBOL_DYNAMICSTITCH, "DynamicStitch", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pDynamicStitch->merged, OUTPUT_TYPE_OUTPUT, "merged");
	}
	else
	{
		std::string msg = string_format("error : DynamicStitch(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pDynamicStitch;
}

void* Create_FIFOQueue(std::string id, Json::Value pInputItem) {
	FIFOQueue* pFIFOQueue = nullptr;
	Scope* pScope = nullptr;
	DataTypeSlice component_types;
	FIFOQueue::Attrs attrs;
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
				std::string msg = string_format("warning : FIFOQueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "component_types")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				component_types = GetDatatypeSliceFromInital(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : FIFOQueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "FIFOQueue::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				//attrParser.ConvStrToArraySliceTensorshape(attrParser.GetAttribute("shapes_"));
				if (attrParser.GetAttribute("shapes_") != "") attrs.Shapes(attrParser.GetValue_arraySliceTensorshape("shapes_"));
				if (attrParser.GetAttribute("capacity_") != "") attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "") attrs.Container(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("container_")));
				if (attrParser.GetAttribute("shared_name_") != "") attrs.SharedName(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("shared_name_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : FIFOQueue pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pFIFOQueue = new FIFOQueue(*pScope, component_types, attrs);
		ObjectInfo* pObj = AddObjectMap(pFIFOQueue, id, SYMBOL_FIFOQUEUE, "FIFOQueue", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pFIFOQueue->handle, OUTPUT_TYPE_OUTPUT, "handle");
	}
	else
	{
		std::string msg = string_format("error : FIFOQueue(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pFIFOQueue;
}

void* Create_GetSessionHandle(std::string id, Json::Value pInputItem) {
	GetSessionHandle* pGetSessionHandle = nullptr;
	Scope* pScope = nullptr;
	Output* pvalue = nullptr;
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
				std::string msg = string_format("warning : GetSessionHandle - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pvalue = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : GetSessionHandle - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pGetSessionHandle = new GetSessionHandle(*pScope, *pvalue);
		ObjectInfo* pObj = AddObjectMap(pGetSessionHandle, id, SYMBOL_GETSESSIONHANDLE, "GetSessionHandle", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pGetSessionHandle->handle, OUTPUT_TYPE_OUTPUT, "handle");
		}
	}
	else
	{
		std::string msg = string_format("error : GetSessionHandle(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pGetSessionHandle;
}

void* Create_GetSessionHandleV2(std::string id, Json::Value pInputItem) {
	GetSessionHandleV2* pGetSessionHandleV2 = nullptr;
	Scope* pScope = nullptr;
	Output* pvalue = nullptr;
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
				std::string msg = string_format("warning : GetSessionHandleV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pvalue = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : GetSessionHandleV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pGetSessionHandleV2 = new GetSessionHandleV2(*pScope, *pvalue);
		ObjectInfo* pObj = AddObjectMap(pGetSessionHandleV2, id, SYMBOL_GETSESSIONHANDLEV2, "GetSessionHandleV2", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pGetSessionHandleV2->handle, OUTPUT_TYPE_OUTPUT, "handle");
		}
	}
	else
	{
		std::string msg = string_format("error : GetSessionHandleV2(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pGetSessionHandleV2;
}

void* Create_GetSessionTensor(std::string id, Json::Value pInputItem) {
	GetSessionTensor* pGetSessionTensor = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
	DataType dtype;
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
				std::string msg = string_format("warning : GetSessionTensor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : GetSessionTensor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				//   dtype = DT_BOOL;
				//else if (strPinInitial == "string")
				//   dtype = DT_STRING;
				else
				{
					std::string msg = string_format("warning : GetSessionTensor - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : GetSessionTensor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pGetSessionTensor = new GetSessionTensor(*pScope, *phandle, dtype);
		ObjectInfo* pObj = AddObjectMap(pGetSessionTensor, id, SYMBOL_GETSESSIONTENSOR, "GetSessionTensor", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pGetSessionTensor->value, OUTPUT_TYPE_OUTPUT, "value");
		}
	}
	else
	{
		std::string msg = string_format("error : GetSessionTensor(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pGetSessionTensor;
}

void* Create_PaddingFIFOQueue(std::string id, Json::Value pInputItem) {
	PaddingFIFOQueue* pPaddingFIFOQueue = nullptr;
	Scope* pScope = nullptr;
	DataTypeSlice component_types;
	PaddingFIFOQueue::Attrs attrs;
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
				std::string msg = string_format("warning : PaddingFIFOQueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "component_types")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				component_types = GetDatatypeSliceFromInital(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : PaddingFIFOQueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "PaddingFIFOQueue::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				//attrParser.ConvStrToArraySliceTensorshape(attrParser.GetAttribute("shapes_"));
				if(attrParser.GetAttribute("shapes_") != "") attrs.Shapes(attrParser.GetValue_arraySliceTensorshape("shapes_"));
				if (attrParser.GetAttribute("capacity_") != "") attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "") attrs.Container(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("container_")));
				if (attrParser.GetAttribute("shared_name_") != "") attrs.SharedName(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("shared_name_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : PaddingFIFOQueue pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pPaddingFIFOQueue = new PaddingFIFOQueue(*pScope, component_types, attrs);
		ObjectInfo* pObj = AddObjectMap(pPaddingFIFOQueue, id, SYMBOL_PADDINGFIFOQUEUE, "PaddingFIFOQueue", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pPaddingFIFOQueue->handle, OUTPUT_TYPE_OUTPUT, "handle");
	}
	else
	{
		std::string msg = string_format("error : PaddingFIFOQueue(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pPaddingFIFOQueue;
}

void* Create_PriorityQueue(std::string id, Json::Value pInputItem) {
	PriorityQueue* pPriorityQueue = nullptr;
	Scope* pScope = nullptr;
	gtl::ArraySlice<PartialTensorShape> shapes;
	PriorityQueue::Attrs attrs;
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
				std::string msg = string_format("warning : PriorityQueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "shapes")
		{
			if (strPinInterface == "gtl::ArraySlice<PartialTensorShape>")
			{
				shapes = GetArrayShapeFromInital(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : PriorityQueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "PriorityQueue::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("component_types_") != "") attrs.ComponentTypes(attrParser.ConvStrToDataTypeSlice("component_types_"));
				if (attrParser.GetAttribute("capacity_") != "") attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "") attrs.Container(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("container_")));
				if (attrParser.GetAttribute("shared_name_") != "") attrs.SharedName(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("shared_name_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : PriorityQueue pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pPriorityQueue = new PriorityQueue(*pScope, shapes, attrs);
		ObjectInfo* pObj = AddObjectMap(pPriorityQueue, id, SYMBOL_PRIORITYQUEUE, "PriorityQueue", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pPriorityQueue->handle, OUTPUT_TYPE_OUTPUT, "handle");
	}
	else
	{
		std::string msg = string_format("error : PriorityQueue(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pPriorityQueue;
}

void* Create_QueueClose(std::string id, Json::Value pInputItem) {
	QueueClose* pQueueClose = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
	QueueClose::Attrs attrs;
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
				std::string msg = string_format("warning : QueueClose - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QueueClose - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QueueClose::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("cancel_pending_enqueues_") != "") attrs.CancelPendingEnqueues(attrParser.ConvStrToInt64(attrParser.GetAttribute("cancel_pending_enqueues_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : QueueClose pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pQueueClose = new QueueClose(*pScope, *phandle, attrs);
		ObjectInfo* pObj = AddObjectMap(pQueueClose, id, SYMBOL_QUEUECLOSE, "QueueClose", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pQueueClose->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : QueueClose(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}

	return pQueueClose;
}

void* Create_QueueDequeue(std::string id, Json::Value pInputItem) {
	QueueDequeue* pQueueDequeue = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
	DataTypeSlice component_types;
	QueueDequeue::Attrs attrs;
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
				std::string msg = string_format("warning : QueueDequeue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QueueDequeue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "component_types")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				component_types = GetDatatypeSliceFromInital(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : QueueDequeue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QueueDequeue::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("timeout_ms_") != "") attrs.TimeoutMs(attrParser.ConvStrToInt64(attrParser.GetAttribute("timeout_ms_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : QueueDequeue pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pQueueDequeue = new QueueDequeue(*pScope, *phandle, component_types, attrs);
		ObjectInfo* pObj = AddObjectMap(pQueueDequeue, id, SYMBOL_QUEUEDEQUEUE, "QueueDequeue", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pQueueDequeue->components, OUTPUT_TYPE_OUTPUTLIST, "components");
	}
	else
	{
		std::string msg = string_format("error : QueueDequeue(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pQueueDequeue;
}

void* Create_QueueDequeueMany(std::string id, Json::Value pInputItem) {
	QueueDequeueMany* pQueueDequeueMany = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
	Output* pn = nullptr;
	DataTypeSlice component_types;
	QueueDequeueMany::Attrs attrs;
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
				std::string msg = string_format("warning : QueueDequeueMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QueueDequeueMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "n")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QueueDequeueMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "component_types")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				component_types = GetDatatypeSliceFromInital(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : QueueDequeueMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QueueDequeueMany::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("timeout_ms_") != "") attrs.TimeoutMs(attrParser.ConvStrToInt64(attrParser.GetAttribute("timeout_ms_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : QueueDequeueMany pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pQueueDequeueMany = new QueueDequeueMany(*pScope, *phandle, *pn, component_types, attrs);
		ObjectInfo* pObj = AddObjectMap(pQueueDequeueMany, id, SYMBOL_QUEUEDEQUEUEMANY, "QueueDequeueMany", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pQueueDequeueMany->components, OUTPUT_TYPE_OUTPUTLIST, "components");
	}
	else
	{
		std::string msg = string_format("error : QueueDequeueMany(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pQueueDequeueMany;
}

void* Create_QueueDequeueUpTo(std::string id, Json::Value pInputItem) {
	QueueDequeueUpTo* pQueueDequeueUpTo = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
	Output *pn = nullptr;
	DataTypeSlice component_types;
	QueueDequeueUpTo::Attrs attrs;
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
				std::string msg = string_format("warning : QueueDequeueUpTo - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QueueDequeueUpTo - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "n")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QueueDequeueUpTo - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "component_types")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				component_types = GetDatatypeSliceFromInital(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : QueueDequeueUpTo - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QueueDequeueUpTo::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("timeout_ms_") != "") attrs.TimeoutMs(attrParser.ConvStrToInt64(attrParser.GetAttribute("timeout_ms_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : QueueDequeueUpTo pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pQueueDequeueUpTo = new QueueDequeueUpTo(*pScope, *phandle, *pn, component_types, attrs);
		ObjectInfo* pObj = AddObjectMap(pQueueDequeueUpTo, id, SYMBOL_QUEUEDEQUEUEUPTO, "QueueDequeueUpTo", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pQueueDequeueUpTo->components, OUTPUT_TYPE_OUTPUTLIST, "components");
	}
	else
	{
		std::string msg = string_format("error : QueueDequeueUpTo(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pQueueDequeueUpTo;
}

void* Create_QueueEnqueue(std::string id, Json::Value pInputItem) {
	QueueEnqueue* pQueueEnqueue = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
	OutputList* components = nullptr;
	QueueEnqueue::Attrs attrs;
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
				std::string msg = string_format("warning : QueueEnqueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QueueEnqueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "components")
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
							components = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QueueEnqueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QueueEnqueue::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("timeout_ms_") != "") attrs.TimeoutMs(attrParser.ConvStrToInt64(attrParser.GetAttribute("timeout_ms_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : QueueEnqueue pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pQueueEnqueue = new QueueEnqueue(*pScope, *phandle, *components, attrs);
		ObjectInfo* pObj = AddObjectMap(pQueueEnqueue, id, SYMBOL_QUEUEENQUEUE, "QueueEnqueue", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pQueueEnqueue->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : QueueEnqueue(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pQueueEnqueue;
}

void* Create_QueueEnqueueMany(std::string id, Json::Value pInputItem) {
	QueueEnqueueMany* pQueueEnqueueMany = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
	OutputList* components = nullptr;
	QueueEnqueueMany::Attrs attrs;
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
				std::string msg = string_format("warning : QueueEnqueueMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QueueEnqueueMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "components")
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
							components = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QueueEnqueueMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QueueEnqueueMany::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("timeout_ms_") != "") attrs.TimeoutMs(attrParser.ConvStrToInt64(attrParser.GetAttribute("timeout_ms_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : QueueEnqueueMany pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pQueueEnqueueMany = new QueueEnqueueMany(*pScope, *phandle, *components, attrs);
		ObjectInfo* pObj = AddObjectMap(pQueueEnqueueMany, id, SYMBOL_QUEUEENQUEUEMANY, "QueueEnqueueMany", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pQueueEnqueueMany->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : QueueEnqueueMany(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pQueueEnqueueMany;
}

void* Create_QueueSize(std::string id, Json::Value pInputItem) {
	QueueSize* pQueueSize = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = new Output();
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
				std::string msg = string_format("warning : QueueSize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QueueSize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pQueueSize = new QueueSize(*pScope, *phandle);
		ObjectInfo* pObj = AddObjectMap(pQueueSize, id, SYMBOL_QUEUESIZE, "QueueSize", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pQueueSize->size, OUTPUT_TYPE_OUTPUT, "size");
	}
	else
	{
		std::string msg = string_format("error : QueueSize(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pQueueSize;
}

void* Create_RandomShuffleQueue(std::string id, Json::Value pInputItem) {
	RandomShuffleQueue* pRandomShuffleQueue = nullptr;
	Scope* pScope = nullptr;
	DataTypeSlice component_types;
	RandomShuffleQueue::Attrs attrs;
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
				std::string msg = string_format("warning : RandomShuffleQueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "component_types")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				component_types = GetDatatypeSliceFromInital(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : RandomShuffleQueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "RandomShuffleQueue::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("shapes_") != "") attrs.Shapes(attrParser.GetValue_arraySliceTensorshape("shapes_"));
				if (attrParser.GetAttribute("capacity_") != "") attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("min_after_dequeue_") != "") attrs.MinAfterDequeue(attrParser.ConvStrToInt64(attrParser.GetAttribute("min_after_dequeue_")));
				if (attrParser.GetAttribute("seed_") != "") attrs.Seed(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed_")));
				if (attrParser.GetAttribute("seed2_") != "") attrs.Seed2(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed2_")));
				if (attrParser.GetAttribute("container_") != "") attrs.Container(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("container_")));
				if (attrParser.GetAttribute("shared_name_") != "") attrs.SharedName(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("shared_name_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : RandomShuffleQueue pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pRandomShuffleQueue = new RandomShuffleQueue(*pScope, component_types, attrs);
		ObjectInfo* pObj = AddObjectMap(pRandomShuffleQueue, id, SYMBOL_RANDOMSHUFFLEQUEUE, "RandomShuffleQueue", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pRandomShuffleQueue->handle, OUTPUT_TYPE_OUTPUT, "handle");
	}
	else
	{
		std::string msg = string_format("error : RandomShuffleQueue(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRandomShuffleQueue;
}

void* Create_RecordInput(std::string id, Json::Value pInputItem) {
	RecordInput* pRecordInput = nullptr;
	Scope* pScope = nullptr;
	StringPiece file_pattern;
	RecordInput::Attrs attrs;
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
				std::string msg = string_format("warning : RecordInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "file_pattern")
		{
			if (strPinInterface == "StringPiece")
			{
				file_pattern = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : RecordInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "RecordInput::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("file_random_seed_") != "") attrs.FileRandomSeed(attrParser.ConvStrToInt64(attrParser.GetAttribute("file_random_seed_")));
				if (attrParser.GetAttribute("file_shuffle_shift_ratio_") != "") attrs.FileShuffleShiftRatio(attrParser.ConvStrToFloat(attrParser.GetAttribute("file_shuffle_shift_ratio_")));
				if (attrParser.GetAttribute("file_buffer_size_") != "") attrs.FileBufferSize(attrParser.ConvStrToInt64(attrParser.GetAttribute("file_buffer_size_")));
				if (attrParser.GetAttribute("file_parallelism_") != "") attrs.FileParallelism(attrParser.ConvStrToInt64(attrParser.GetAttribute("file_parallelism_")));
				if (attrParser.GetAttribute("batch_size_") != "") attrs.BatchSize(attrParser.ConvStrToInt64(attrParser.GetAttribute("batch_size_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : RecordInput pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pRecordInput = new RecordInput(*pScope, file_pattern, attrs);
		ObjectInfo* pObj = AddObjectMap(pRecordInput, id, SYMBOL_RECORDINPUT, "RecordInput", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pRecordInput->records, OUTPUT_TYPE_OUTPUT, "records");
	}
	else
	{
		std::string msg = string_format("error : RecordInput(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRecordInput;
}

void* Create_SparseAccumulatorApplyGradient(std::string id, Json::Value pInputItem) {
	SparseAccumulatorApplyGradient* pSparseAccumulatorApplyGradient = nullptr;
	Scope* pScope = nullptr;
	Output *phandle = nullptr;
	Output *plocal_step = nullptr;
	Output *pgradient_indices = nullptr;
	Output *pgradient_values = nullptr;
	Output *pgradient_shape = nullptr;
	bool has_known_shape = false;
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
				std::string msg = string_format("warning : SparseAccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseAccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "local_step")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							plocal_step = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseAccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "gradient_indices")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pgradient_indices = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseAccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "gradient_values")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pgradient_values = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseAccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "gradient_values")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pgradient_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseAccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "has_known_shape")
		{
			if (strPinInterface == "bool")
			{
				if(strPinName == "true" || strPinName == "TRUE") has_known_shape = true;
				else has_known_shape = false;
			}
			else
			{
				std::string msg = string_format("warning : SparseAccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pSparseAccumulatorApplyGradient = new SparseAccumulatorApplyGradient(*pScope, *phandle, *plocal_step, *pgradient_indices, *pgradient_values, *pgradient_shape, has_known_shape);
		ObjectInfo* pObj = AddObjectMap(pSparseAccumulatorApplyGradient, id, SYMBOL_SPARSEACCUMULATORAPPLYGRADIENT, "SparseAccumulatorApplyGradient", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSparseAccumulatorApplyGradient->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : SparseAccumulatorApplyGradient(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseAccumulatorApplyGradient;
}

void* Create_SparseAccumulatorTakeGradient(std::string id, Json::Value pInputItem) {
	SparseAccumulatorTakeGradient* pSparseAccumulatorTakeGradient = nullptr;
	Scope* pScope = nullptr;
	Output *phandle = nullptr;
	Output *pnum_required = nullptr;
	DataType dtype;
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
				std::string msg = string_format("warning : SparseAccumulatorTakeGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseAccumulatorTakeGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_required")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pnum_required = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseAccumulatorTakeGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				//   dtype = DT_BOOL;
				//else if (strPinInitial == "string")
				//   dtype = DT_STRING;
				else
				{
					std::string msg = string_format("warning : SparseAccumulatorTakeGradient - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseAccumulatorTakeGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pSparseAccumulatorTakeGradient = new SparseAccumulatorTakeGradient(*pScope, *phandle, *pnum_required, dtype);
		ObjectInfo* pObj = AddObjectMap(pSparseAccumulatorTakeGradient, id, SYMBOL_SPARSEACCUMULATORTAKEGRADIENT, "SparseAccumulatorTakeGradient", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseAccumulatorTakeGradient->indices, OUTPUT_TYPE_OUTPUT, "indices");
			AddOutputInfo(pObj, &pSparseAccumulatorTakeGradient->indices, OUTPUT_TYPE_OUTPUT, "values");
			AddOutputInfo(pObj, &pSparseAccumulatorTakeGradient->indices, OUTPUT_TYPE_OUTPUT, "shape");
		}
	}
	else
	{
		std::string msg = string_format("error : SparseAccumulatorTakeGradient(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseAccumulatorTakeGradient;
}

void* Create_SparseConditionalAccumulator(std::string id, Json::Value pInputItem) {
	SparseConditionalAccumulator* pSparseConditionalAccumulator = nullptr;
	Scope* pScope = nullptr;
	PartialTensorShape shape;
	DataType dtype;
	SparseConditionalAccumulator::Attrs attrs;
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
				std::string msg = string_format("warning : SparseConditionalAccumulator - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "shape")
		{
			if (strPinInterface == "PartialTensorShape")
			{
				shape = GetPartialShapeFromInital(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : SparseConditionalAccumulator - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				//   dtype = DT_BOOL;
				//else if (strPinInitial == "string")
				//   dtype = DT_STRING;
				else
				{
					std::string msg = string_format("warning : SparseConditionalAccumulator - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseConditionalAccumulator - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "SparseConditionalAccumulator::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("container_") != "") attrs.Container(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("container_")));
				if (attrParser.GetAttribute("shared_name_") != "") attrs.SharedName(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("shared_name_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseConditionalAccumulator pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pSparseConditionalAccumulator = new SparseConditionalAccumulator(*pScope, dtype, shape, attrs);
		ObjectInfo* pObj = AddObjectMap(pSparseConditionalAccumulator, id, SYMBOL_SPARSECONDITIONALACCUMULATOR, "SparseConditionalAccumulator", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSparseConditionalAccumulator->handle, OUTPUT_TYPE_OUTPUT, "handle");
	}
	else
	{
		std::string msg = string_format("error : pSparseConditionalAccumulator(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseConditionalAccumulator;
}

void* Create_Stage(std::string id, Json::Value pInputItem) {
	Stage* pStage = nullptr;
	Scope* pScope = nullptr;
	OutputList* pvalues = nullptr;
	Stage::Attrs attrs;
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
				std::string msg = string_format("warning : Stage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pvalues = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Stage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Stage::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("container_") != "") attrs.Container(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("container_")));
				if (attrParser.GetAttribute("shared_name_") != "") attrs.SharedName(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("shared_name_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : Stage pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pStage = new Stage(*pScope, *pvalues, attrs);
		ObjectInfo* pObj = AddObjectMap(pStage, id, SYMBOL_STAGE, "Stage", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pStage->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : Stage(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pStage;
}

void* Create_TensorArray(std::string id, Json::Value pInputItem) {
	TensorArray* pTensorArray = nullptr;
	Scope* pScope = nullptr;
	Output* psize = nullptr;
	DataType dtype;
	TensorArray::Attrs attrs;
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
				std::string msg = string_format("warning : TensorArray - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
			}
			else
			{
				std::string msg = string_format("warning : TensorArray - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				//   dtype = DT_BOOL;
				//else if (strPinInitial == "string")
				//   dtype = DT_STRING;
				else
				{
					std::string msg = string_format("warning : TensorArray - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArray - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "TensorArray::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("element_shape_") != "") attrs.ElementShape(attrParser.GetValue_PartialTensorShape("element_shape_"));
				if (attrParser.GetAttribute("dynamic_size_") != "") attrs.DynamicSize(attrParser.ConvStrToBool(attrParser.GetAttribute("dynamic_size_")));
				if (attrParser.GetAttribute("clear_after_read_") != "") attrs.ClearAfterRead(attrParser.ConvStrToBool(attrParser.GetAttribute("clear_after_read_")));
				if (attrParser.GetAttribute("tensor_array_name_") != "") attrs.TensorArrayName(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("tensor_array_name_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : TensorArray pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pTensorArray = new TensorArray(*pScope, *psize, dtype, attrs);
		ObjectInfo* pObj = AddObjectMap(pTensorArray, id, SYMBOL_TENSORARRAY, "TensorArray", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pTensorArray->handle, OUTPUT_TYPE_OUTPUT, "handle");
			AddOutputInfo(pObj, &pTensorArray->flow, OUTPUT_TYPE_OUTPUT, "flow");
		}
	}
	else
	{
		std::string msg = string_format("error : TensorArray(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pTensorArray;
}

void* Create_TensorArrayClose(std::string id, Json::Value pInputItem) {
	TensorArrayClose* pTensorArrayClose = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
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
				std::string msg = string_format("warning : TensorArrayClose - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayClose - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pTensorArrayClose = new TensorArrayClose(*pScope, *phandle);
		ObjectInfo* pObj = AddObjectMap(pTensorArrayClose, id, SYMBOL_TENSORARRAYCLOSE, "TensorArrayClose", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pTensorArrayClose->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("error : TensorArrayClose(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pTensorArrayClose;
}

void* Create_TensorArrayConcat(std::string id, Json::Value pInputItem) {
	TensorArrayConcat* pTensorArrayConcat = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
	Output* pflow_in = nullptr;
	DataType dtype;
	TensorArrayConcat::Attrs attrs;
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
				std::string msg = string_format("warning : TensorArrayConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "flow_in")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pflow_in = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				//   dtype = DT_BOOL;
				//else if (strPinInitial == "string")
				//   dtype = DT_STRING;
				else
				{
					std::string msg = string_format("warning : TensorArrayConcat - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "TensorArrayConcat::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("element_shape_except0_") != "") attrs.ElementShapeExcept0(attrParser.GetValue_PartialTensorShape("element_shape_except0_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : TensorArrayConcat pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pTensorArrayConcat = new TensorArrayConcat(*pScope, *phandle, *pflow_in, dtype, attrs);
		ObjectInfo* pObj = AddObjectMap(pTensorArrayConcat, id, SYMBOL_TENSORARRAYCONCAT, "TensorArrayConcat", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pTensorArrayConcat->value, OUTPUT_TYPE_OUTPUT, "value");
			AddOutputInfo(pObj, &pTensorArrayConcat->lengths, OUTPUT_TYPE_OUTPUT, "lengths");
		}
	}
	else
	{
		std::string msg = string_format("error : TensorArrayConcat(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pTensorArrayConcat;
}

void* Create_TensorArrayGather(std::string id, Json::Value pInputItem) {
	TensorArrayGather* pTensorArrayGather = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
	Output* pindices = nullptr;
	Output* pflow_in = nullptr;
	DataType dtype;
	TensorArrayGather::Attrs attrs;
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
				std::string msg = string_format("warning : TensorArrayGather - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayGather - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayGather - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "flow_in")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pflow_in = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayGather - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				//   dtype = DT_BOOL;
				//else if (strPinInitial == "string")
				//   dtype = DT_STRING;
				else
				{
					std::string msg = string_format("warning : TensorArrayGather - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayGather - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "TensorArrayGather::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("element_shape_") != "") attrs.ElementShape(attrParser.GetValue_PartialTensorShape("element_shape_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : TensorArrayGather pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pTensorArrayGather = new TensorArrayGather(*pScope, *phandle, *pindices, *pflow_in, dtype, attrs);
		ObjectInfo* pObj = AddObjectMap(pTensorArrayGather, id, SYMBOL_TENSORARRAYGATHER, "TensorArrayGather", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pTensorArrayGather->value, OUTPUT_TYPE_OUTPUT, "value");
		}
	}
	else
	{
		std::string msg = string_format("error : TensorArrayGather(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pTensorArrayGather;
}

void* Create_TensorArrayGrad(std::string id, Json::Value pInputItem) {
	TensorArrayGrad* pTensorArrayGrad = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
	Output* pflow_in = nullptr;
	StringPiece source;
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
				std::string msg = string_format("warning : TensorArrayGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "flow_in")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pflow_in = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "source")
		{
			if (strPinInterface == "StringPiece")
			{
				source = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pTensorArrayGrad = new TensorArrayGrad(*pScope, *phandle,*pflow_in, source);
		ObjectInfo* pObj = AddObjectMap(pTensorArrayGrad, id, SYMBOL_TENSORARRAYGRAD, "TensorArrayGrad", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pTensorArrayGrad->grad_handle, OUTPUT_TYPE_OUTPUT, "grad_handle");
			AddOutputInfo(pObj, &pTensorArrayGrad->flow_out, OUTPUT_TYPE_OUTPUT, "flow_out");
		}
	}
	else
	{
		std::string msg = string_format("error : TensorArrayGrad(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pTensorArrayGrad;
}

void* Create_TensorArrayRead(std::string id, Json::Value pInputItem) {
	TensorArrayRead* pTensorArrayRead = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
	Output* pindex = nullptr;
	Output* pflow_in = nullptr;
	DataType dtype;
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
				std::string msg = string_format("warning : TensorArrayRead - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayRead - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pindex = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayRead - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "flow_in")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pflow_in = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayRead - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				//   dtype = DT_BOOL;
				//else if (strPinInitial == "string")
				//   dtype = DT_STRING;
				else
				{
					std::string msg = string_format("warning : TensorArrayRead - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayRead - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pTensorArrayRead = new TensorArrayRead(*pScope, *phandle, *pindex, *pflow_in, dtype);
		ObjectInfo* pObj = AddObjectMap(pTensorArrayRead, id, SYMBOL_TENSORARRAYREAD, "TensorArrayRead", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pTensorArrayRead->value, OUTPUT_TYPE_OUTPUT, "value");
		}
	}
	else
	{
		std::string msg = string_format("error : TensorArrayRead(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pTensorArrayRead;
}

void* Create_TensorArrayScatter(std::string id, Json::Value pInputItem) {
	TensorArrayScatter* pTensorArrayScatter = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
	Output* pindices = nullptr;
	Output* pvalue = nullptr;
	Output* pflow_in = nullptr;
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
				std::string msg = string_format("warning : TensorArrayScatter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayScatter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayScatter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pvalue = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayScatter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "flow_in")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pflow_in = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayScatter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pTensorArrayScatter = new TensorArrayScatter(*pScope, *phandle, *pindices, *pvalue, *pflow_in);
		ObjectInfo* pObj = AddObjectMap(pTensorArrayScatter, id, SYMBOL_TENSORARRAYSCATTER, "TensorArrayScatter", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pTensorArrayScatter->flow_out, OUTPUT_TYPE_OUTPUT, "flow_out");
		}
	}
	else
	{
		std::string msg = string_format("error : TensorArrayScatter(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pTensorArrayScatter;
}

void* Create_TensorArraySize(std::string id, Json::Value pInputItem) {
	TensorArraySize* pTensorArraySize = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
	Output* pflow_in = nullptr;
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
				std::string msg = string_format("warning : TensorArraySize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArraySize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "flow_in")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pflow_in = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArraySize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pTensorArraySize = new TensorArraySize(*pScope, *phandle, *pflow_in);
		ObjectInfo* pObj = AddObjectMap(pTensorArraySize, id, SYMBOL_TENSORARRAYSIZE, "TensorArraySize", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pTensorArraySize->size, OUTPUT_TYPE_OUTPUT, "size");
		}
	}
	else
	{
		std::string msg = string_format("error : TensorArraySize(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pTensorArraySize;
}

void* Create_TensorArraySplit(std::string id, Json::Value pInputItem) {
	TensorArraySplit* pTensorArraySplit = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
	Output* pvalue = nullptr;
	Output* plengths = nullptr;
	Output* pflow_in = nullptr;
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
				std::string msg = string_format("warning : TensorArraySplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArraySplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "lengths")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							plengths = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArraySplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pvalue = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArraySplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "flow_in")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pflow_in = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArraySplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pTensorArraySplit = new TensorArraySplit(*pScope, *phandle, *pvalue,*plengths, *pflow_in);
		ObjectInfo* pObj = AddObjectMap(pTensorArraySplit, id, SYMBOL_TENSORARRAYSPLIT, "TensorArraySplit", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pTensorArraySplit->flow_out, OUTPUT_TYPE_OUTPUT, "flow_out");
		}
	}
	else
	{
		std::string msg = string_format("error : TensorArraySplit(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pTensorArraySplit;
}

void* Create_TensorArrayWrite(std::string id, Json::Value pInputItem) {
	TensorArrayWrite* pTensorArrayWrite = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
	Output* pindex = nullptr;
	Output* pvalue = nullptr;
	Output* pflow_in = nullptr;
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
				std::string msg = string_format("warning : TensorArraySplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							phandle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayWrite - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							pindex = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayWrite - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pvalue = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayWrite - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "flow_in")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pflow_in = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TensorArrayWrite - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
	}

	if (pScope)
	{
		pTensorArrayWrite = new TensorArrayWrite(*pScope, *phandle, *pindex, *pvalue, *pflow_in);
		ObjectInfo* pObj = AddObjectMap(pTensorArrayWrite, id, SYMBOL_TENSORARRAYWRITE, "TensorArrayWrite", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pTensorArrayWrite->flow_out, OUTPUT_TYPE_OUTPUT, "flow_out");
		}
	}
	else
	{
		std::string msg = string_format("error : TensorArrayWrite(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pTensorArrayWrite;
}

void* Create_Unstage(std::string id, Json::Value pInputItem) {
	Unstage* pUnstage = nullptr;
	Scope* pScope = nullptr;
	DataTypeSlice dtypes;
	Unstage::Attrs attrs;
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
				std::string msg = string_format("warning : Unstage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				dtypes = GetDatatypeSliceFromInital(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : Unstage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Unstage::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("container_") != "") attrs.Container(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("container_")));
				if (attrParser.GetAttribute("shared_name_") != "") attrs.SharedName(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("shared_name_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : Unstage pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pUnstage = new Unstage(*pScope, dtypes, attrs);
		ObjectInfo* pObj = AddObjectMap(pUnstage, id, SYMBOL_UNSTAGE, "Unstage", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pUnstage->values, OUTPUT_TYPE_OUTPUTLIST, "values");
	}
	else
	{
		std::string msg = string_format("error : Unstage(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pUnstage;
}
