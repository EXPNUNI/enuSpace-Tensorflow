#include "stdafx.h"
#include "tf_data_flow_ops.h"

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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("AccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("AccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						plocal_step = (Output*)Create_StrToOutput(*m_pScope, "DT_INT64", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("AccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("AccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("AccumulatorApplyGradient pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && plocal_step && pgradient)
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
		std::string msg = string_format("AccumulatorApplyGradient(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("AccumulatorSetGlobalStep - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("AccumulatorNumAccumulated - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("AccumulatorNumAccumulated pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle)
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
		std::string msg = string_format("AccumulatorNumAccumulated(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("AccumulatorSetGlobalStep - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("AccumulatorSetGlobalStep - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pnew_global_step = (Output*)Create_StrToOutput(*m_pScope, "DT_INT64", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("AccumulatorSetGlobalStep - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("AccumulatorSetGlobalStep pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && pnew_global_step)
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
		std::string msg = string_format("AccumulatorSetGlobalStep(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("AccumulatorTakeGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("AccumulatorTakeGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pnum_required = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("AccumulatorTakeGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				if (!(dtype = GetDatatypeFromInitial(strPinInitial)))
				{
					std::string msg = string_format("AccumulatorTakeGradient - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
				}
			}
			else
			{
				std::string msg = string_format("AccumulatorTakeGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("AccumulatorTakeGradient pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && pnum_required)
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
		std::string msg = string_format("AccumulatorTakeGradient(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pAccumulatorTakeGradient;
}

void* Create_Barrier(std::string id, Json::Value pInputItem) {
	Barrier* pBarrier = nullptr;
	Scope* pScope = nullptr;
	std::vector<tensorflow::DataType> vDT;
	Barrier::Attrs attrs;
	std::vector<PartialTensorShape> v_PTS;
	std::string container_ = "";
	std::string sharedname_ = "";
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("Barrier - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "component_types")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if(!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("Barrier - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Barrier::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				//attrParser.ConvStrToArraySliceTensorshape(attrParser.GetAttribute("shapes_"));
				if (attrParser.GetAttribute("shapes_") != "")
				{
					if (attrParser.GetValue_arraySliceTensorshape("shapes_", v_PTS))
					{
						attrs = attrs.Shapes(v_PTS);
					}
				}
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					sharedname_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = sharedname_;
				}
			}
		}
		else
		{
			std::string msg = string_format("Barrier pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && vDT.size() > 0)
	{
		DataTypeSlice component_types(vDT);
		pBarrier = new Barrier(*pScope, component_types, attrs);
		vDT.clear();
		ObjectInfo* pObj = AddObjectMap(pBarrier, id, SYMBOL_BARRIER, "Barrier", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pBarrier->handle, OUTPUT_TYPE_OUTPUT, "handle");
	}
	else
	{
		std::string msg = string_format("Barrier(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("BarrierClose - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("BarrierClose - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "BarrierClose::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("CancelPendingEnqueues") != "") attrs = attrs.CancelPendingEnqueues(attrParser.ConvStrToBool(attrParser.GetAttribute("CancelPendingEnqueues")));
			}
		}
		else
		{
			std::string msg = string_format("BarrierClose pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}

	}

	if (pScope && phandle)
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
		std::string msg = string_format("BarrierClose(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("BarrierIncompleteSize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("BarrierIncompleteSize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("BarrierIncompleteSize pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle)
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
		std::string msg = string_format("BarrierIncompleteSize(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("BarrierInsertMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("BarrierInsertMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pkeys = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("BarrierInsertMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pkeys = (Output*)Create_StrToOutput(*m_pScope, "", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("BarrierInsertMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "component_index")
		{
			if (strPinInterface == "Input")
			{
				if(strPinInitial != "")
					component_index = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("BarrierInsertMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("BarrierInsertMany pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && pkeys && pvalues)
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
		std::string msg = string_format("BarrierInsertMany(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("BarrierReadySize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("BarrierReadySize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("BarrierReadySize pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle)
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
		std::string msg = string_format("BarrierReadySize(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pBarrierReadySize;
}

void* Create_BarrierTakeMany(std::string id, Json::Value pInputItem) {
	BarrierTakeMany* pBarrierTakeMany = nullptr;
	Scope* pScope = nullptr;
	Output *phandle = nullptr;
	Output *pnum_elements = nullptr;
	std::vector<tensorflow::DataType> vDT;
	BarrierTakeMany::Attrs attrs;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("BarrierTakeMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("BarrierTakeMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
					{
						pnum_elements = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("BarrierTakeMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "component_types")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("BarrierTakeMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "BarrierTakeMany::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("allow_small_batch_") != "") attrs = attrs.AllowSmallBatch(attrParser.ConvStrToBool(attrParser.GetAttribute("allow_small_batch_")));
				if (attrParser.GetAttribute("wait_for_incomplete_") != "") attrs = attrs.WaitForIncomplete(attrParser.ConvStrToBool(attrParser.GetAttribute("wait_for_incomplete_")));
				if (attrParser.GetAttribute("timeout_ms_") != "") attrs = attrs.TimeoutMs(attrParser.ConvStrToInt64(attrParser.GetAttribute("timeout_ms_")));
			}
		}
		else
		{
			std::string msg = string_format("BarrierTakeMany pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && pnum_elements && vDT.size() > 0)
	{
		DataTypeSlice component_types(vDT);
		pBarrierTakeMany = new BarrierTakeMany(*pScope, *phandle, *pnum_elements, component_types, attrs);
		vDT.clear();
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
		std::string msg = string_format("BarrierTakeMany(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("ConditionalAccumulator - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "shape")
		{
			if (strPinInterface == "PartialTensorShape")
			{
				shape = GetPartialShapeFromInitial(strPinInitial);
			}
			else
			{
				std::string msg = string_format("ConditionalAccumulator - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				if (!(dtype = GetDatatypeFromInitial(strPinInitial)))
				{
					std::string msg = string_format("ConditionalAccumulator - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
				}
			}
			else
			{
				std::string msg = string_format("ConditionalAccumulator - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "PriorityQueue::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs = attrs.Container(container_);
				}
				if (attrParser.GetAttribute("shared_name_") != "") 
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs = attrs.SharedName(shared_name_);
				}
			}
		}
		else
		{
			std::string msg = string_format("ConditionalAccumulator pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
		std::string msg = string_format("ConditionalAccumulator(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("DeleteSessionTensor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("DeleteSessionTensor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("DeleteSessionTensor pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle)
	{
		pDeleteSessionTensor = new DeleteSessionTensor(*pScope, *phandle);
		ObjectInfo* pObj = AddObjectMap(pDeleteSessionTensor, id, SYMBOL_DELETESESSIONTENSOR, "DeleteSessionTensor", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pDeleteSessionTensor->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("DeleteSessionTensor(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("DynamicPartition - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
						pdata = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("DynamicPartition - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
					{
						ppartitions = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("DynamicPartition - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "num_partitions")
		{
			if (strPinInterface == "int64")
			{
				if(!strPinInitial.empty())
					num_partitions = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("DynamicPartition - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("DynamicPartition pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pdata && ppartitions)
	{
		if (pScope->ok())
		{
			pDynamicPartition = new DynamicPartition(*pScope, *pdata, *ppartitions, num_partitions);
			ObjectInfo* pObj = AddObjectMap(pDynamicPartition, id, SYMBOL_DYNAMICPARTITION, "DynamicPartition", pInputItem);
			if (pObj)
			{
				AddOutputInfo(pObj, &pDynamicPartition->outputs, OUTPUT_TYPE_OUTPUTLIST, "outputs");
			}

			if (pScope->ok() == false)
			{
				std::string msg = string_format("Create DynamicPartition(%s) Object create failed. (Scope fail)", id.c_str());
				SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
			}
		}
	}
	else
	{
		std::string msg = string_format("DynamicPartition(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("DynamicStitch - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "indices")
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
							pindices = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pindices = (OutputList*)Create_StrToOutputList(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("DynamicStitch - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pdata = (OutputList*)Create_StrToOutputList(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("DynamicStitch - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("DynamicStitch pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pindices && pdata)
	{
		pDynamicStitch = new DynamicStitch(*pScope, *pindices, *pdata);
		ObjectInfo* pObj = AddObjectMap(pDynamicStitch, id, SYMBOL_DYNAMICSTITCH, "DynamicStitch", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pDynamicStitch->merged, OUTPUT_TYPE_OUTPUT, "merged");
	}
	else
	{
		std::string msg = string_format("DynamicStitch(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pDynamicStitch;
}

void* Create_FIFOQueue(std::string id, Json::Value pInputItem) {
	FIFOQueue* pFIFOQueue = nullptr;
	Scope* pScope = nullptr;
	DataTypeSlice* pcomponent_types = nullptr;
	std::vector<tensorflow::DataType> vDT;
	FIFOQueue::Attrs attrs;
	std::vector<PartialTensorShape> v_PTS;
	std::string container_ = "";
	std::string sharedname_ = "";
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("FIFOQueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "component_types")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pcomponent_types = (DataTypeSlice*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if(!strPinInitial.empty())
						GetDatatypeSliceFromInitial(strPinInitial, vDT);
				}
			}
			else
			{
				std::string msg = string_format("FIFOQueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "FIFOQueue::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				//attrParser.ConvStrToArraySliceTensorshape(attrParser.GetAttribute("shapes_"));
				if (attrParser.GetAttribute("shapes_") != "")
				{
					if (attrParser.GetValue_arraySliceTensorshape("shapes_", v_PTS))
					{
						attrs = attrs.Shapes(v_PTS);
					}
				}
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					sharedname_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = sharedname_;
				}
			}
		}
		else
		{
			std::string msg = string_format("FIFOQueue pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && (pcomponent_types || vDT.size() > 0))
	{
		DataTypeSlice component_types(vDT);
		if (pcomponent_types)
		{
			pFIFOQueue = new FIFOQueue(*pScope, *pcomponent_types, attrs);
		}
		else
		{
			pFIFOQueue = new FIFOQueue(*pScope, component_types, attrs);
		}
		vDT.clear();
		ObjectInfo* pObj = AddObjectMap(pFIFOQueue, id, SYMBOL_FIFOQUEUE, "FIFOQueue", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pFIFOQueue->handle, OUTPUT_TYPE_OPERATION, "handle");
	}
	else
	{
		std::string msg = string_format("FIFOQueue(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("GetSessionHandle - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("GetSessionHandle - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("GetSessionHandle pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pvalue)
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
		std::string msg = string_format("GetSessionHandle(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("GetSessionHandleV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("GetSessionHandleV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("GetSessionHandleV2 pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pvalue)
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
		std::string msg = string_format("GetSessionHandleV2(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("GetSessionTensor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("GetSessionTensor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				if (!(dtype = GetDatatypeFromInitial(strPinInitial)))
				{
					std::string msg = string_format("GetSessionTensor - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
				}
			}
			else
			{
				std::string msg = string_format("GetSessionTensor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("GetSessionTensor pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle)
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
		std::string msg = string_format("GetSessionTensor(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pGetSessionTensor;
}

void* Create_MapClear(std::string id, Json::Value pInputItem) {
	MapClear* pMapClear = nullptr;
	Scope* pScope = nullptr;
	std::vector<DataType> vDT;
	int iSize = (int)pInputItem.size();
	MapClear::Attrs attrs;
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("MapClear - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial( strPinInitial,vDT);
			}
			else
			{
				std::string msg = string_format("MapClear - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "MapClear::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("memory_limit_") != "") attrs = attrs.MemoryLimit(attrParser.ConvStrToInt64(attrParser.GetAttribute("memory_limit_")));
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = shared_name_;
				}
			}
		}
		else
		{
			std::string msg = string_format("MapClear pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && vDT.size()>0)
	{
		DataTypeSlice dtypes(vDT);
		pMapClear = new MapClear(*pScope, dtypes, attrs);
		ObjectInfo* pObj = AddObjectMap(pMapClear, id, SYMBOL_MAPCLEAR, "MapClear", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pMapClear->operation, OUTPUT_TYPE_OPERATION, "operation");
		}
	}
	else
	{
		std::string msg = string_format("MapClear(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pMapClear;
}
void* Create_MapIncompleteSize(std::string id, Json::Value pInputItem) {
	MapIncompleteSize* pMapIncompleteSize = nullptr;
	Scope* pScope = nullptr;
	std::vector<DataType> vDT;
	int iSize = (int)pInputItem.size();
	MapIncompleteSize::Attrs attrs;
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("MapIncompleteSize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("MapIncompleteSize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "MapIncompleteSize::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("memory_limit_") != "") attrs = attrs.MemoryLimit(attrParser.ConvStrToInt64(attrParser.GetAttribute("memory_limit_")));
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = shared_name_;
				}
			}
		}
		else
		{
			std::string msg = string_format("MapIncompleteSize pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && vDT.size() > 0)
	{
		DataTypeSlice dtypes(vDT);
		pMapIncompleteSize = new MapIncompleteSize(*pScope, dtypes, attrs);
		ObjectInfo* pObj = AddObjectMap(pMapIncompleteSize, id, SYMBOL_MAPINCOMPLETESIZE, "MapIncompleteSize", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pMapIncompleteSize->size, OUTPUT_TYPE_OUTPUT, "size");
		}
	}
	else
	{
		std::string msg = string_format("MapIncompleteSize(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pMapIncompleteSize;
}
void* Create_MapPeek(std::string id, Json::Value pInputItem) {
	MapPeek* pMapPeek = nullptr;
	Scope* pScope = nullptr;
	Output* key = nullptr;
	Output* indices = nullptr;
	std::vector<DataType> vDT;
	int iSize = (int)pInputItem.size();
	MapPeek::Attrs attrs;
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("MapPeek - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "key")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							key = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						key = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("MapPeek - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						indices = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("MapPeek - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("MapPeek - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "MapPeek::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("memory_limit_") != "") attrs = attrs.MemoryLimit(attrParser.ConvStrToInt64(attrParser.GetAttribute("memory_limit_")));
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = shared_name_;
				}
			}
		}
		else
		{
			std::string msg = string_format("MapPeek pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && key && indices && vDT.size() > 0)
	{
		DataTypeSlice dtypes(vDT);
		pMapPeek = new MapPeek(*pScope, *key, *indices, dtypes, attrs);
		ObjectInfo* pObj = AddObjectMap(pMapPeek, id, SYMBOL_MAPPEEK, "MapPeek", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pMapPeek->values, OUTPUT_TYPE_OUTPUTLIST, "values");
		}
	}
	else
	{
		std::string msg = string_format("MapPeek(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pMapPeek;
}
void* Create_MapSize(std::string id, Json::Value pInputItem) {
	MapSize* pMapSize = nullptr;
	Scope* pScope = nullptr;
	std::vector<DataType> vDT;
	int iSize = (int)pInputItem.size();
	MapSize::Attrs attrs;
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("MapSize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("MapSize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "MapSize::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("memory_limit_") != "") attrs = attrs.MemoryLimit(attrParser.ConvStrToInt64(attrParser.GetAttribute("memory_limit_")));
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = shared_name_;
				}
			}
		}
		else
		{
			std::string msg = string_format("MapSize pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && vDT.size() > 0)
	{
		DataTypeSlice dtypes(vDT);
		pMapSize = new MapSize(*pScope, dtypes, attrs);
		ObjectInfo* pObj = AddObjectMap(pMapSize, id, SYMBOL_MAPSIZE, "MapSize", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pMapSize->size, OUTPUT_TYPE_OUTPUT, "size");
		}
	}
	else
	{
		std::string msg = string_format("MapSize(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pMapSize;
}
void* Create_MapStage(std::string id, Json::Value pInputItem) {
	MapStage* pMapStage = nullptr;
	Scope* pScope = nullptr;
	Output* key = nullptr;
	Output* indices = nullptr;
	OutputList* values = nullptr;
	std::vector<DataType> vDT;
	int iSize = (int)pInputItem.size();
	MapStage::Attrs attrs;
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("MapStage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "key")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							key = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						key = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("MapStage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						indices = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("MapStage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "values")
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
							values = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						values = (OutputList*)Create_StrToOutput(*m_pScope, "", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("MapStage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("MapStage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "MapStage::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("memory_limit_") != "") attrs = attrs.MemoryLimit(attrParser.ConvStrToInt64(attrParser.GetAttribute("memory_limit_")));
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = shared_name_;
				}
			}
		}
		else
		{
			std::string msg = string_format("MapStage pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && key && indices && values && vDT.size() > 0)
	{
		DataTypeSlice dtypes(vDT);
		pMapStage = new MapStage(*pScope, *key, *indices, *values, dtypes, attrs);
		ObjectInfo* pObj = AddObjectMap(pMapStage, id, SYMBOL_MAPSTAGE, "MapStage", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pMapStage->operation, OUTPUT_TYPE_OPERATION, "operation");
		}
	}
	else
	{
		std::string msg = string_format("MapStage(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pMapStage;
}
void* Create_MapUnstage(std::string id, Json::Value pInputItem) {
	MapUnstage* pMapUnstage = nullptr;
	Scope* pScope = nullptr;
	Output* key = nullptr;
	Output* indices = nullptr;
	std::vector<DataType> vDT;
	int iSize = (int)pInputItem.size();
	MapUnstage::Attrs attrs;
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("MapUnstage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "key")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							key = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						key = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("MapUnstage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						indices = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("MapUnstage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("MapUnstage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "MapUnstage::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("memory_limit_") != "") attrs = attrs.MemoryLimit(attrParser.ConvStrToInt64(attrParser.GetAttribute("memory_limit_")));
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = shared_name_;
				}
			}
		}
		else
		{
			std::string msg = string_format("MapUnstage pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && key && indices && vDT.size() > 0)
	{
		DataTypeSlice dtypes(vDT);
		pMapUnstage = new MapUnstage(*pScope, *key, *indices, dtypes, attrs);
		ObjectInfo* pObj = AddObjectMap(pMapUnstage, id, SYMBOL_MAPUNSTAGE, "MapUnstage", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pMapUnstage->values, OUTPUT_TYPE_OUTPUTLIST, "values");
		}
	}
	else
	{
		std::string msg = string_format("MapUnstage(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pMapUnstage;
}
void* Create_MapUnstageNoKey(std::string id, Json::Value pInputItem) {
	MapUnstageNoKey* pMapUnstageNoKey = nullptr;
	Scope* pScope = nullptr;
	Output* key = nullptr;
	Output* indices = nullptr;
	std::vector<DataType> vDT;
	int iSize = (int)pInputItem.size();
	MapUnstageNoKey::Attrs attrs;
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("MapUnstageNoKey - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						indices = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("MapUnstageNoKey - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("MapUnstageNoKey - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "MapUnstageNoKey::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("memory_limit_") != "") attrs = attrs.MemoryLimit(attrParser.ConvStrToInt64(attrParser.GetAttribute("memory_limit_")));
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = shared_name_;
				}
			}
		}
		else
		{
			std::string msg = string_format("MapUnstageNoKey pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && indices && vDT.size() > 0)
	{
		DataTypeSlice dtypes(vDT);
		pMapUnstageNoKey = new MapUnstageNoKey(*pScope, *indices, dtypes, attrs);
		ObjectInfo* pObj = AddObjectMap(pMapUnstageNoKey, id, SYMBOL_MAPUNSTAGE, "MapUnstage", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pMapUnstageNoKey->key, OUTPUT_TYPE_OUTPUT, "key");
			AddOutputInfo(pObj, &pMapUnstageNoKey->values, OUTPUT_TYPE_OUTPUTLIST, "values");
		}
	}
	else
	{
		std::string msg = string_format("MapUnstage(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pMapUnstageNoKey;
}
void* Create_OrderedMapClear(std::string id, Json::Value pInputItem) {
	OrderedMapClear* pOrderedMapClear = nullptr;
	Scope* pScope = nullptr;
	std::vector<DataType> vDT;
	int iSize = (int)pInputItem.size();
	OrderedMapClear::Attrs attrs;
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("OrderedMapClear - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("OrderedMapClear - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "OrderedMapClear::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("memory_limit_") != "") attrs = attrs.MemoryLimit(attrParser.ConvStrToInt64(attrParser.GetAttribute("memory_limit_")));
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = shared_name_;
				}
			}
		}
		else
		{
			std::string msg = string_format("OrderedMapClear pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && vDT.size()>0)
	{
		DataTypeSlice dtypes(vDT);
		pOrderedMapClear = new OrderedMapClear(*pScope, dtypes, attrs);
		ObjectInfo* pObj = AddObjectMap(pOrderedMapClear, id, SYMBOL_ORDEREDMAPCLEAR, "OrderedMapClear", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pOrderedMapClear->operation, OUTPUT_TYPE_OPERATION, "operation");
		}
	}
	else
	{
		std::string msg = string_format("OrderedMapClear(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pOrderedMapClear;
}
void* Create_OrderedMapIncompleteSize(std::string id, Json::Value pInputItem) {
	OrderedMapIncompleteSize* pOrderedMapIncompleteSize = nullptr;
	Scope* pScope = nullptr;
	std::vector<DataType> vDT;
	int iSize = (int)pInputItem.size();
	OrderedMapIncompleteSize::Attrs attrs;
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("OrderedMapIncompleteSize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("OrderedMapIncompleteSize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "OrderedMapIncompleteSize::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("memory_limit_") != "") attrs = attrs.MemoryLimit(attrParser.ConvStrToInt64(attrParser.GetAttribute("memory_limit_")));
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = shared_name_;
				}
			}
		}
		else
		{
			std::string msg = string_format("OrderedMapIncompleteSize pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && vDT.size() > 0)
	{
		DataTypeSlice dtypes(vDT);
		pOrderedMapIncompleteSize = new OrderedMapIncompleteSize(*pScope, dtypes, attrs);
		ObjectInfo* pObj = AddObjectMap(pOrderedMapIncompleteSize, id, SYMBOL_ORDEREDMAPINCOMPLETESIZE, "OrderedMapIncompleteSize", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pOrderedMapIncompleteSize->size, OUTPUT_TYPE_OUTPUT, "size");
		}
	}
	else
	{
		std::string msg = string_format("OrderedMapIncompleteSize(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pOrderedMapIncompleteSize;
}
void* Create_OrderedMapPeek(std::string id, Json::Value pInputItem) {
	OrderedMapPeek* pOrderedMapPeek = nullptr;
	Scope* pScope = nullptr;
	Output* key = nullptr;
	Output* indices = nullptr;
	std::vector<DataType> vDT;
	int iSize = (int)pInputItem.size();
	OrderedMapPeek::Attrs attrs;
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("OrderedMapPeek - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "key")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							key = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						key = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("OrderedMapPeek - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						indices = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("OrderedMapPeek - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("OrderedMapPeek - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "OrderedMapPeek::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("memory_limit_") != "") attrs = attrs.MemoryLimit(attrParser.ConvStrToInt64(attrParser.GetAttribute("memory_limit_")));
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = shared_name_;
				}
			}
		}
		else
		{
			std::string msg = string_format("OrderedMapPeek pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && key && indices && vDT.size() > 0)
	{
		DataTypeSlice dtypes(vDT);
		pOrderedMapPeek = new OrderedMapPeek(*pScope, *key, *indices, dtypes, attrs);
		ObjectInfo* pObj = AddObjectMap(pOrderedMapPeek, id, SYMBOL_ORDEREDMAPPEEK, "OrderedMapPeek", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pOrderedMapPeek->values, OUTPUT_TYPE_OUTPUTLIST, "values");
		}
	}
	else
	{
		std::string msg = string_format("OrderedMapPeek(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pOrderedMapPeek;
}
void* Create_OrderedMapSize(std::string id, Json::Value pInputItem) {
	OrderedMapSize* pOrderedMapSize = nullptr;
	Scope* pScope = nullptr;
	std::vector<DataType> vDT;
	int iSize = (int)pInputItem.size();
	OrderedMapSize::Attrs attrs;
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("OrderedMapSize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("OrderedMapSize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "OrderedMapSize::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("memory_limit_") != "") attrs = attrs.MemoryLimit(attrParser.ConvStrToInt64(attrParser.GetAttribute("memory_limit_")));
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = shared_name_;
				}
			}
		}
		else
		{
			std::string msg = string_format("OrderedMapSize pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && vDT.size() > 0)
	{
		DataTypeSlice dtypes(vDT);
		pOrderedMapSize = new OrderedMapSize(*pScope, dtypes, attrs);
		ObjectInfo* pObj = AddObjectMap(pOrderedMapSize, id, SYMBOL_ORDEREDMAPSIZE, "OrderedMapSize", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pOrderedMapSize->size, OUTPUT_TYPE_OUTPUT, "size");
		}
	}
	else
	{
		std::string msg = string_format("OrderedMapSize(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pOrderedMapSize;
}
void* Create_OrderedMapStage(std::string id, Json::Value pInputItem) {
	OrderedMapStage* pOrderedMapStage = nullptr;
	Scope* pScope = nullptr;
	Output* key = nullptr;
	Output* indices = nullptr;
	OutputList* values = nullptr;
	std::vector<DataType> vDT;
	int iSize = (int)pInputItem.size();
	OrderedMapStage::Attrs attrs;
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("OrderedMapStage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "key")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							key = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						key = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("OrderedMapStage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						indices = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("OrderedMapStage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "values")
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
							values = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						values = (OutputList*)Create_StrToOutput(*m_pScope, "", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("OrderedMapStage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("OrderedMapStage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "OrderedMapStage::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("memory_limit_") != "") attrs = attrs.MemoryLimit(attrParser.ConvStrToInt64(attrParser.GetAttribute("memory_limit_")));
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = shared_name_;
				}
			}
		}
		else
		{
			std::string msg = string_format("OrderedMapStage pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && key && indices && values && vDT.size() > 0)
	{
		DataTypeSlice dtypes(vDT);
		pOrderedMapStage = new OrderedMapStage(*pScope, *key, *indices, *values, dtypes, attrs);
		ObjectInfo* pObj = AddObjectMap(pOrderedMapStage, id, SYMBOL_ORDEREDMAPSTAGE, "OrderedMapStage", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pOrderedMapStage->operation, OUTPUT_TYPE_OPERATION, "operation");
		}
	}
	else
	{
		std::string msg = string_format("OrderedMapStage(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pOrderedMapStage;
}
void* Create_OrderedMapUnstage(std::string id, Json::Value pInputItem) {
	OrderedMapUnstage* pOrderedMapUnstage = nullptr;
	Scope* pScope = nullptr;
	Output* key = nullptr;
	Output* indices = nullptr;
	std::vector<DataType> vDT;
	int iSize = (int)pInputItem.size();
	OrderedMapUnstage::Attrs attrs;
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("OrderedMapUnstage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "key")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							key = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						key = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("OrderedMapUnstage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						indices = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("OrderedMapUnstage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("OrderedMapUnstage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "OrderedMapUnstage::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("memory_limit_") != "") attrs = attrs.MemoryLimit(attrParser.ConvStrToInt64(attrParser.GetAttribute("memory_limit_")));
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = shared_name_;
				}
			}
		}
		else
		{
			std::string msg = string_format("OrderedMapUnstage pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && key && indices && vDT.size() > 0)
	{
		DataTypeSlice dtypes(vDT);
		pOrderedMapUnstage = new OrderedMapUnstage(*pScope, *key, *indices, dtypes, attrs);
		ObjectInfo* pObj = AddObjectMap(pOrderedMapUnstage, id, SYMBOL_ORDEREDMAPUNSTAGE, "OrderedMapUnstage", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pOrderedMapUnstage->values, OUTPUT_TYPE_OUTPUTLIST, "values");
		}
	}
	else
	{
		std::string msg = string_format("OrderedMapUnstage(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pOrderedMapUnstage;
}
void* Create_OrderedMapUnstageNoKey(std::string id, Json::Value pInputItem) {
	OrderedMapUnstageNoKey* pOrderedMapUnstageNoKey = nullptr;
	Scope* pScope = nullptr;
	Output* key = nullptr;
	Output* indices = nullptr;
	std::vector<DataType> vDT;
	int iSize = (int)pInputItem.size();
	OrderedMapUnstageNoKey::Attrs attrs;
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("OrderedMapUnstageNoKey - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						indices = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("OrderedMapUnstageNoKey - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("OrderedMapUnstageNoKey - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "OrderedMapUnstageNoKey::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("memory_limit_") != "") attrs = attrs.MemoryLimit(attrParser.ConvStrToInt64(attrParser.GetAttribute("memory_limit_")));
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = shared_name_;
				}
			}
		}
		else
		{
			std::string msg = string_format("OrderedMapUnstageNoKey pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && indices && vDT.size() > 0)
	{
		DataTypeSlice dtypes(vDT);
		pOrderedMapUnstageNoKey = new OrderedMapUnstageNoKey(*pScope, *indices, dtypes, attrs);
		ObjectInfo* pObj = AddObjectMap(pOrderedMapUnstageNoKey, id, SYMBOL_ORDEREDMAPUNSTAGE, "MapUnstage", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pOrderedMapUnstageNoKey->key, OUTPUT_TYPE_OUTPUT, "key");
			AddOutputInfo(pObj, &pOrderedMapUnstageNoKey->values, OUTPUT_TYPE_OUTPUTLIST, "values");
		}
	}
	else
	{
		std::string msg = string_format("MapUnstage(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pOrderedMapUnstageNoKey;
}

void* Create_PaddingFIFOQueue(std::string id, Json::Value pInputItem) {
	PaddingFIFOQueue* pPaddingFIFOQueue = nullptr;
	Scope* pScope = nullptr;
	PaddingFIFOQueue::Attrs attrs;
	std::vector<tensorflow::DataType> vDT;
	std::vector<PartialTensorShape> v_PTS;
	std::string container_ = "";
	std::string sharedname_ = "";
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("PaddingFIFOQueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "component_types")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("PaddingFIFOQueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "PaddingFIFOQueue::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				//attrParser.ConvStrToArraySliceTensorshape(attrParser.GetAttribute("shapes_"));
				if (attrParser.GetAttribute("shapes_") != "")
				{
					if (attrParser.GetValue_arraySliceTensorshape("shapes_", v_PTS))
					{
						attrs = attrs.Shapes(v_PTS);
					}
				}
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					sharedname_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = sharedname_;
				}
			}
		}
		else
		{
			std::string msg = string_format("PaddingFIFOQueue pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && vDT.size() > 0)
	{
		DataTypeSlice component_types(vDT);
		pPaddingFIFOQueue = new PaddingFIFOQueue(*pScope, component_types, attrs);
		vDT.clear();
		ObjectInfo* pObj = AddObjectMap(pPaddingFIFOQueue, id, SYMBOL_PADDINGFIFOQUEUE, "PaddingFIFOQueue", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pPaddingFIFOQueue->handle, OUTPUT_TYPE_OUTPUT, "handle");
	}
	else
	{
		std::string msg = string_format("PaddingFIFOQueue(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pPaddingFIFOQueue;
}

void* Create_PriorityQueue(std::string id, Json::Value pInputItem) {
	PriorityQueue* pPriorityQueue = nullptr;
	Scope* pScope = nullptr;
	PriorityQueue::Attrs attrs;
	std::string container_ = "";
	std::string sharedname_ = "";
	std::vector<DataType> v_DT;
	std::vector<PartialTensorShape> vec_PTS;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("PriorityQueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "shapes")
		{
			if (strPinInterface == "gtl::ArraySlice<PartialTensorShape>")
			{
				 GetArrayShapeFromInitial(strPinInitial, vec_PTS);
			}
			else
			{
				std::string msg = string_format("PriorityQueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "PriorityQueue::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("component_types_") != "")
				{
					
					attrParser.ConvStrToDataTypeSlice(attrParser.GetAttribute("component_types_"),v_DT);
					attrs = attrs.ComponentTypes(v_DT);
				}
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					sharedname_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = sharedname_;
				}
			}
		}
		else
		{
			std::string msg = string_format("PriorityQueue pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope)
	{
		gtl::ArraySlice<PartialTensorShape> shapes(vec_PTS);
		pPriorityQueue = new PriorityQueue(*pScope, shapes, attrs);
		ObjectInfo* pObj = AddObjectMap(pPriorityQueue, id, SYMBOL_PRIORITYQUEUE, "PriorityQueue", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pPriorityQueue->handle, OUTPUT_TYPE_OUTPUT, "handle");
	}
	else
	{
		std::string msg = string_format("PriorityQueue(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("QueueClose - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("QueueClose - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QueueClose::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("cancel_pending_enqueues_") != "") attrs = attrs.CancelPendingEnqueues(attrParser.ConvStrToBool(attrParser.GetAttribute("cancel_pending_enqueues_")));
			}
		}
		else
		{
			std::string msg = string_format("QueueClose pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle)
	{
		pQueueClose = new QueueClose(*pScope, *phandle, attrs);
		ObjectInfo* pObj = AddObjectMap(pQueueClose, id, SYMBOL_QUEUECLOSE, "QueueClose", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pQueueClose->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("QueueClose(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}

	return pQueueClose;
}

void* Create_QueueDequeue(std::string id, Json::Value pInputItem) {
	QueueDequeue* pQueueDequeue = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
	std::vector<tensorflow::DataType> vDT;
	QueueDequeue::Attrs attrs;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("QueueDequeue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("QueueDequeue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "component_types")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("QueueDequeue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QueueDequeue::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("timeout_ms_") != "") attrs = attrs.TimeoutMs(attrParser.ConvStrToInt64(attrParser.GetAttribute("timeout_ms_")));
			}
		}
		else
		{
			std::string msg = string_format("QueueDequeue pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && vDT.size() > 0)
	{
		DataTypeSlice component_types(vDT);
		pQueueDequeue = new QueueDequeue(*pScope, *phandle, component_types, attrs);
		vDT.clear();
		ObjectInfo* pObj = AddObjectMap(pQueueDequeue, id, SYMBOL_QUEUEDEQUEUE, "QueueDequeue", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pQueueDequeue->components, OUTPUT_TYPE_OUTPUTLIST, "components");
	}
	else
	{
		std::string msg = string_format("QueueDequeue(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pQueueDequeue;
}

void* Create_QueueDequeueMany(std::string id, Json::Value pInputItem) {
	QueueDequeueMany* pQueueDequeueMany = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
	Output* pn = nullptr;
	std::vector<tensorflow::DataType> vDT;
	QueueDequeueMany::Attrs attrs;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("QueueDequeueMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("QueueDequeueMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
							pn = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pn = (Output*)Create_StrToOutput(*m_pScope, "DT_INT32", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("QueueDequeueMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "component_types")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("QueueDequeueMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QueueDequeueMany::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("timeout_ms_") != "") attrs = attrs.TimeoutMs(attrParser.ConvStrToInt64(attrParser.GetAttribute("timeout_ms_")));
			}
		}
		else
		{
			std::string msg = string_format("QueueDequeueMany pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && pn && vDT.size() > 0)
	{
		DataTypeSlice component_types(vDT);
		pQueueDequeueMany = new QueueDequeueMany(*pScope, *phandle, *pn, component_types, attrs);
		vDT.clear();
		ObjectInfo* pObj = AddObjectMap(pQueueDequeueMany, id, SYMBOL_QUEUEDEQUEUEMANY, "QueueDequeueMany", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pQueueDequeueMany->components, OUTPUT_TYPE_OUTPUTLIST, "components");
	}
	else
	{
		std::string msg = string_format("QueueDequeueMany(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pQueueDequeueMany;
}

void* Create_QueueDequeueUpTo(std::string id, Json::Value pInputItem) {
	QueueDequeueUpTo* pQueueDequeueUpTo = nullptr;
	Scope* pScope = nullptr;
	Output* phandle = nullptr;
	Output *pn = nullptr;
	std::vector<tensorflow::DataType> vDT;
	QueueDequeueUpTo::Attrs attrs;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("QueueDequeueUpTo - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("QueueDequeueUpTo - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
							pn = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pn = (Output*)Create_StrToOutput(*m_pScope, "DT_INT32", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("QueueDequeueUpTo - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "component_types")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("QueueDequeueUpTo - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QueueDequeueUpTo::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("timeout_ms_") != "") attrs = attrs.TimeoutMs(attrParser.ConvStrToInt64(attrParser.GetAttribute("timeout_ms_")));
			}
		}
		else
		{
			std::string msg = string_format("QueueDequeueUpTo pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && pn && vDT.size() > 0)
	{
		DataTypeSlice component_types(vDT);
		pQueueDequeueUpTo = new QueueDequeueUpTo(*pScope, *phandle, *pn, component_types, attrs);
		vDT.clear();
		ObjectInfo* pObj = AddObjectMap(pQueueDequeueUpTo, id, SYMBOL_QUEUEDEQUEUEUPTO, "QueueDequeueUpTo", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pQueueDequeueUpTo->components, OUTPUT_TYPE_OUTPUTLIST, "components");
	}
	else
	{
		std::string msg = string_format("QueueDequeueUpTo(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("QueueEnqueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("QueueEnqueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("QueueEnqueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QueueEnqueue::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("timeout_ms_") != "") attrs = attrs.TimeoutMs(attrParser.ConvStrToInt64(attrParser.GetAttribute("timeout_ms_")));
			}
		}
		else
		{
			std::string msg = string_format("QueueEnqueue pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && components)
	{
		pQueueEnqueue = new QueueEnqueue(*pScope, *phandle, *components, attrs);
		ObjectInfo* pObj = AddObjectMap(pQueueEnqueue, id, SYMBOL_QUEUEENQUEUE, "QueueEnqueue", pInputItem);
		if (pObj)
		{
			
//			Node* test = pQueueEnqueue->operation.node();
//			Output* OUT1 = new Output(test);
			AddOutputInfo(pObj, &pQueueEnqueue->operation, OUTPUT_TYPE_OPERATION, "operation");
		}
			//AddOutputInfo(pObj, &pQueueEnqueue->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("QueueEnqueue(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("QueueEnqueueMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("QueueEnqueueMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("QueueEnqueueMany - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QueueEnqueueMany::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("timeout_ms_") != "") attrs = attrs.TimeoutMs(attrParser.ConvStrToInt64(attrParser.GetAttribute("timeout_ms_")));
			}
		}
		else
		{
			std::string msg = string_format("QueueEnqueueMany pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && components)
	{
		pQueueEnqueueMany = new QueueEnqueueMany(*pScope, *phandle, *components, attrs);
		ObjectInfo* pObj = AddObjectMap(pQueueEnqueueMany, id, SYMBOL_QUEUEENQUEUEMANY, "QueueEnqueueMany", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pQueueEnqueueMany->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("QueueEnqueueMany(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("QueueSize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("QueueSize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
	}

	if (pScope && phandle)
	{
		pQueueSize = new QueueSize(*pScope, *phandle);
		ObjectInfo* pObj = AddObjectMap(pQueueSize, id, SYMBOL_QUEUESIZE, "QueueSize", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pQueueSize->size, OUTPUT_TYPE_OUTPUT, "size");
	}
	else
	{
		std::string msg = string_format("QueueSize(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pQueueSize;
}

void* Create_RandomShuffleQueue(std::string id, Json::Value pInputItem) {
	RandomShuffleQueue* pRandomShuffleQueue = nullptr;
	Scope* pScope = nullptr;
	std::vector<tensorflow::DataType> vDT;
	RandomShuffleQueue::Attrs attrs;
	std::vector<PartialTensorShape> v_PTS;
	std::string container_ = "";
	std::string sharedname_ = "";
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("RandomShuffleQueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "component_types")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("RandomShuffleQueue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "RandomShuffleQueue::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("shapes_") != "")
				{
					if (attrParser.GetValue_arraySliceTensorshape("shapes_", v_PTS))
					{
						attrs = attrs.Shapes(v_PTS);
					}
				}
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("min_after_dequeue_") != "") attrs = attrs.MinAfterDequeue(attrParser.ConvStrToInt64(attrParser.GetAttribute("min_after_dequeue_")));
				if (attrParser.GetAttribute("seed_") != "") attrs = attrs.Seed(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed_")));
				if (attrParser.GetAttribute("seed2_") != "") attrs = attrs.Seed2(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed2_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					sharedname_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = sharedname_;
				}
			}
		}
		else
		{
			std::string msg = string_format("RandomShuffleQueue pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && vDT.size() > 0)
	{
		DataTypeSlice component_types(vDT);
		pRandomShuffleQueue = new RandomShuffleQueue(*pScope, component_types, attrs);
		vDT.clear();
		ObjectInfo* pObj = AddObjectMap(pRandomShuffleQueue, id, SYMBOL_RANDOMSHUFFLEQUEUE, "RandomShuffleQueue", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pRandomShuffleQueue->handle, OUTPUT_TYPE_OUTPUT, "handle");
	}
	else
	{
		std::string msg = string_format("RandomShuffleQueue(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("RecordInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("RecordInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "RecordInput::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("file_random_seed_") != "") attrs = attrs.FileRandomSeed(attrParser.ConvStrToInt64(attrParser.GetAttribute("file_random_seed_")));
				if (attrParser.GetAttribute("file_shuffle_shift_ratio_") != "") attrs = attrs.FileShuffleShiftRatio(attrParser.ConvStrToFloat(attrParser.GetAttribute("file_shuffle_shift_ratio_")));
				if (attrParser.GetAttribute("file_buffer_size_") != "") attrs = attrs.FileBufferSize(attrParser.ConvStrToInt64(attrParser.GetAttribute("file_buffer_size_")));
				if (attrParser.GetAttribute("file_parallelism_") != "") attrs = attrs.FileParallelism(attrParser.ConvStrToInt64(attrParser.GetAttribute("file_parallelism_")));
				if (attrParser.GetAttribute("batch_size_") != "") attrs = attrs.BatchSize(attrParser.ConvStrToInt64(attrParser.GetAttribute("batch_size_")));
			}
		}
		else
		{
			std::string msg = string_format("RecordInput pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
		std::string msg = string_format("RecordInput(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("SparseAccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("SparseAccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						plocal_step = (Output*)Create_StrToOutput(*m_pScope, "DT_INT64", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("SparseAccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						plocal_step = (Output*)Create_StrToOutput(*m_pScope, "DT_INT64", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("SparseAccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						plocal_step = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("SparseAccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "gradient_shape")
		{
			if (strPinInterface == "Input")
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
				else
				{
					if (!strPinInitial.empty())
						pgradient_shape = (Output*)Create_StrToOutput(*m_pScope, "DT_INT64", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("SparseAccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("SparseAccumulatorApplyGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("SparseAccumulatorApplyGradient pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && plocal_step && pgradient_indices && pgradient_values && pgradient_shape)
	{
		pSparseAccumulatorApplyGradient = new SparseAccumulatorApplyGradient(*pScope, *phandle, *plocal_step, *pgradient_indices, *pgradient_values, *pgradient_shape, has_known_shape);
		ObjectInfo* pObj = AddObjectMap(pSparseAccumulatorApplyGradient, id, SYMBOL_SPARSEACCUMULATORAPPLYGRADIENT, "SparseAccumulatorApplyGradient", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSparseAccumulatorApplyGradient->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("SparseAccumulatorApplyGradient(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("SparseAccumulatorTakeGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("SparseAccumulatorTakeGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pnum_required = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("SparseAccumulatorTakeGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				if (!(dtype = GetDatatypeFromInitial(strPinInitial)))
				{
					std::string msg = string_format("SparseAccumulatorTakeGradient - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
				}
			}
			else
			{
				std::string msg = string_format("SparseAccumulatorTakeGradient - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("SparseAccumulatorTakeGradient pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && pnum_required)
	{
		pSparseAccumulatorTakeGradient = new SparseAccumulatorTakeGradient(*pScope, *phandle, *pnum_required, dtype);
		ObjectInfo* pObj = AddObjectMap(pSparseAccumulatorTakeGradient, id, SYMBOL_SPARSEACCUMULATORTAKEGRADIENT, "SparseAccumulatorTakeGradient", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseAccumulatorTakeGradient->indices, OUTPUT_TYPE_OUTPUT, "indices");
			AddOutputInfo(pObj, &pSparseAccumulatorTakeGradient->values, OUTPUT_TYPE_OUTPUT, "values");
			AddOutputInfo(pObj, &pSparseAccumulatorTakeGradient->shape, OUTPUT_TYPE_OUTPUT, "shape");
		}
	}
	else
	{
		std::string msg = string_format("SparseAccumulatorTakeGradient(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pSparseAccumulatorTakeGradient;
}

void* Create_SparseConditionalAccumulator(std::string id, Json::Value pInputItem) {
	SparseConditionalAccumulator* pSparseConditionalAccumulator = nullptr;
	Scope* pScope = nullptr;
	PartialTensorShape shape;
	DataType dtype;
	SparseConditionalAccumulator::Attrs attrs;
	std::string container_ = "";
	std::string sharedname_ = "";
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("SparseConditionalAccumulator - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "shape")
		{
			if (strPinInterface == "PartialTensorShape")
			{
				shape = GetPartialShapeFromInitial(strPinInitial);
			}
			else
			{
				std::string msg = string_format("SparseConditionalAccumulator - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				if (!(dtype = GetDatatypeFromInitial(strPinInitial)))
				{
					std::string msg = string_format("SparseConditionalAccumulator - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
				}
			}
			else
			{
				std::string msg = string_format("SparseConditionalAccumulator - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "SparseConditionalAccumulator::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					sharedname_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = sharedname_;
				}
			}
		}
		else
		{
			std::string msg = string_format("SparseConditionalAccumulator pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
		std::string msg = string_format("pSparseConditionalAccumulator(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pSparseConditionalAccumulator;
}

void* Create_Stage(std::string id, Json::Value pInputItem) {
	Stage* pStage = nullptr;
	Scope* pScope = nullptr;
	OutputList* pvalues = nullptr;
	Stage::Attrs attrs;
	int iSize = (int)pInputItem.size();
	std::string container_ = "";
	std::string shared_name_ = "";
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("Stage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "values")
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
							pvalues = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("Stage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Stage::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("memory_limit_") != "") attrs = attrs.MemoryLimit(attrParser.ConvStrToInt64(attrParser.GetAttribute("memory_limit_")));
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = shared_name_;
				}
			}
		}
		else
		{
			std::string msg = string_format("Stage pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && pvalues)
	{
		pStage = new Stage(*pScope, *pvalues, attrs);
		ObjectInfo* pObj = AddObjectMap(pStage, id, SYMBOL_STAGE, "Stage", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pStage->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("Stage(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pStage;
}
void* Create_StageClear(std::string id, Json::Value pInputItem) {
	StageClear* pStageClear = nullptr;
	Scope* pScope = nullptr;
	std::vector<DataType> vDT;
	int iSize = (int)pInputItem.size();
	StageClear::Attrs attrs;
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("StageClear - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("StageClear - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "StageClear::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("memory_limit_") != "") attrs = attrs.MemoryLimit(attrParser.ConvStrToInt64(attrParser.GetAttribute("memory_limit_")));
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = shared_name_;
				}
			}
		}
		else
		{
			std::string msg = string_format("StageClear pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && vDT.size() > 0)
	{
		DataTypeSlice dtypes(vDT);
		pStageClear = new StageClear(*pScope, dtypes, attrs);
		ObjectInfo* pObj = AddObjectMap(pStageClear, id, SYMBOL_STAGECLEAR, "StageClear", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pStageClear->operation, OUTPUT_TYPE_OPERATION, "operation");
		}
	}
	else
	{
		std::string msg = string_format("MapClear(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pStageClear;
}

void* Create_StagePeek(std::string id, Json::Value pInputItem) {
	StagePeek* pStagePeek = nullptr;
	Scope* pScope = nullptr;
	Output* index = nullptr;
	std::vector<DataType> vDT;
	int iSize = (int)pInputItem.size();
	StagePeek::Attrs attrs;
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("StagePeek - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
							index = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						index = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("StagePeek - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("StagePeek - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "StagePeek::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("memory_limit_") != "") attrs = attrs.MemoryLimit(attrParser.ConvStrToInt64(attrParser.GetAttribute("memory_limit_")));
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = shared_name_;
				}
			}
		}
		else
		{
			std::string msg = string_format("StagePeek pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && index && vDT.size() > 0)
	{
		DataTypeSlice dtypes(vDT);
		pStagePeek = new StagePeek(*pScope, *index, dtypes, attrs);
		ObjectInfo* pObj = AddObjectMap(pStagePeek, id, SYMBOL_STAGEPEEK, "StagePeek", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pStagePeek->values, OUTPUT_TYPE_OUTPUTLIST, "values");
		}
	}
	else
	{
		std::string msg = string_format("MapPeek(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pStagePeek;
}
void* Create_StageSize(std::string id, Json::Value pInputItem) {
	StageSize* pStageSize = nullptr;
	Scope* pScope = nullptr;
	std::vector<DataType> vDT;
	int iSize = (int)pInputItem.size();
	StageSize::Attrs attrs;
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("StageSize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("StageSize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "MapClear::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("memory_limit_") != "") attrs = attrs.MemoryLimit(attrParser.ConvStrToInt64(attrParser.GetAttribute("memory_limit_")));
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = shared_name_;
				}
			}
		}
		else
		{
			std::string msg = string_format("StageSize pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && vDT.size() > 0)
	{
		DataTypeSlice dtypes(vDT);
		pStageSize = new StageSize(*pScope, dtypes, attrs);
		ObjectInfo* pObj = AddObjectMap(pStageSize, id, SYMBOL_STAGESIZE, "StageSize", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pStageSize->size, OUTPUT_TYPE_OUTPUT, "size");
		}
	}
	else
	{
		std::string msg = string_format("MapSize(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pStageSize;
}
void* Create_TensorArray(std::string id, Json::Value pInputItem) {
	TensorArray* pTensorArray = nullptr;
	Scope* pScope = nullptr;
	Output* psize = nullptr;
	DataType dtype;
	TensorArray::Attrs attrs;
	int iSize = (int)pInputItem.size();
	std::string tensor_array_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("TensorArray - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
						psize = (Output*)Create_StrToOutput(*m_pScope, "DT_INT32", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("TensorArray - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				if (!(dtype = GetDatatypeFromInitial(strPinInitial)))
				{
					std::string msg = string_format("TensorArray - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
				}
			}
			else
			{
				std::string msg = string_format("TensorArray - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "TensorArray::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("element_shape_") != "") attrs = attrs.ElementShape(attrParser.GetValue_PartialTensorShape("element_shape_"));
				if (attrParser.GetAttribute("dynamic_size_") != "") attrs = attrs.DynamicSize(attrParser.ConvStrToBool(attrParser.GetAttribute("dynamic_size_")));
				if (attrParser.GetAttribute("clear_after_read_") != "") attrs = attrs.ClearAfterRead(attrParser.ConvStrToBool(attrParser.GetAttribute("clear_after_read_")));
				if (attrParser.GetAttribute("tensor_array_name_") != "")
				{
					tensor_array_name_ = attrParser.GetAttribute("tensor_array_name_");
					attrs = attrs.TensorArrayName(tensor_array_name_);
				}
			}
		}
		else
		{
			std::string msg = string_format("TensorArray pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && psize)
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
		std::string msg = string_format("TensorArray(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("TensorArrayClose - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("TensorArrayClose - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("TensorArrayClose pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle)
	{
		pTensorArrayClose = new TensorArrayClose(*pScope, *phandle);
		ObjectInfo* pObj = AddObjectMap(pTensorArrayClose, id, SYMBOL_TENSORARRAYCLOSE, "TensorArrayClose", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pTensorArrayClose->operation, OUTPUT_TYPE_OPERATION, "operation");
	}
	else
	{
		std::string msg = string_format("TensorArrayClose(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("TensorArrayConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("TensorArrayConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pflow_in = (Output*)Create_StrToOutput(*m_pScope, "DT_FLOAT", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("TensorArrayConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				if (!(dtype = GetDatatypeFromInitial(strPinInitial)))
				{
					std::string msg = string_format("TensorArrayConcat - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
				}
			}
			else
			{
				std::string msg = string_format("TensorArrayConcat - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "TensorArrayConcat::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("element_shape_except0_") != "") attrs = attrs.ElementShapeExcept0(attrParser.GetValue_PartialTensorShape("element_shape_except0_"));
			}
		}
		else
		{
			std::string msg = string_format("TensorArrayConcat pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && pflow_in)
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
		std::string msg = string_format("TensorArrayConcat(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("TensorArrayGather - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("TensorArrayGather - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
						pindices = (Output*)Create_StrToOutput(*m_pScope, "DT_INT32", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("TensorArrayGather - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pflow_in = (Output*)Create_StrToOutput(*m_pScope, "DT_FLOAT", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("TensorArrayGather - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				if (!(dtype = GetDatatypeFromInitial(strPinInitial)))
				{
					std::string msg = string_format("TensorArrayGather - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
				}
			}
			else
			{
				std::string msg = string_format("TensorArrayGather - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "TensorArrayGather::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("element_shape_") != "") attrs = attrs.ElementShape(attrParser.GetValue_PartialTensorShape("element_shape_"));
			}
		}
		else
		{
			std::string msg = string_format("TensorArrayGather pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && pindices && pflow_in)
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
		std::string msg = string_format("TensorArrayGather(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("TensorArrayGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("TensorArrayGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pflow_in = (Output*)Create_StrToOutput(*m_pScope, "DT_FLOAT", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("TensorArrayGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("TensorArrayGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("TensorArrayGrad pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && pflow_in)
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
		std::string msg = string_format("TensorArrayGrad(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("TensorArrayRead - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("TensorArrayRead - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pindex = (Output*)Create_StrToOutput(*m_pScope, "DT_INT32", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("TensorArrayRead - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pflow_in = (Output*)Create_StrToOutput(*m_pScope, "DT_FLOAT", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("TensorArrayRead - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				if (!(dtype = GetDatatypeFromInitial(strPinInitial)))
				{
					std::string msg = string_format("TensorArrayRead - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
				}
			}
			else
			{
				std::string msg = string_format("TensorArrayRead - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("TensorArrayRead pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && pindex && pflow_in)
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
		std::string msg = string_format("TensorArrayRead(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("TensorArrayScatter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("TensorArrayScatter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
						pindices = (Output*)Create_StrToOutput(*m_pScope, "DT_INT32", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("TensorArrayScatter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("TensorArrayScatter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pflow_in = (Output*)Create_StrToOutput(*m_pScope, "DT_FLOAT", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("TensorArrayScatter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("TensorArrayScatter pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && pindices && pvalue)
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
		std::string msg = string_format("TensorArrayScatter(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("TensorArraySize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("TensorArraySize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pflow_in = (Output*)Create_StrToOutput(*m_pScope, "DT_FLOAT", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("TensorArraySize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("TensorArraySize pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && pflow_in)
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
		std::string msg = string_format("TensorArraySize(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("TensorArraySplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("TensorArraySplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						plengths = (Output*)Create_StrToOutput(*m_pScope, "DT_INT64", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("TensorArraySplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pflow_in = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("TensorArraySplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pflow_in = (Output*)Create_StrToOutput(*m_pScope, "DT_FLOAT", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("TensorArraySplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("TensorArraySplit pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && pvalue && plengths && pflow_in)
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
		std::string msg = string_format("TensorArraySplit(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
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
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("TensorArraySplit - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				std::string msg = string_format("TensorArrayWrite - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pindex = (Output*)Create_StrToOutput(*m_pScope, "DT_INT32", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("TensorArrayWrite - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pvalue = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("TensorArrayWrite - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
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
				else
				{
					if (!strPinInitial.empty())
						pflow_in = (Output*)Create_StrToOutput(*m_pScope, "DT_FLOAT", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("TensorArrayWrite - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else
		{
			std::string msg = string_format("TensorArrayWrite pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && phandle && pindex && pvalue && pflow_in)
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
		std::string msg = string_format("TensorArrayWrite(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pTensorArrayWrite;
}

void* Create_Unstage(std::string id, Json::Value pInputItem) {
	Unstage* pUnstage = nullptr;
	Scope* pScope = nullptr;
	std::vector<tensorflow::DataType> vDT;
	Unstage::Attrs attrs;
	int iSize = (int)pInputItem.size();
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
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
				std::string msg = string_format("Unstage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("Unstage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Unstage::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("capacity_") != "") attrs = attrs.Capacity(attrParser.ConvStrToInt64(attrParser.GetAttribute("capacity_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs.container_ = container_;
				}
				if (attrParser.GetAttribute("memory_limit_") != "") attrs = attrs.MemoryLimit(attrParser.ConvStrToInt64(attrParser.GetAttribute("memory_limit_")));
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs.shared_name_ = shared_name_;
				}
			}
		}
		else
		{
			std::string msg = string_format("Unstage pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			SetLastError(DEF_WARNING, "", 0, msg, false, id.c_str());
		}
	}

	if (pScope && vDT.size() > 0)
	{
		DataTypeSlice dtypes(vDT);
		pUnstage = new Unstage(*pScope, dtypes, attrs);
		vDT.clear();
		ObjectInfo* pObj = AddObjectMap(pUnstage, id, SYMBOL_UNSTAGE, "Unstage", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pUnstage->values, OUTPUT_TYPE_OUTPUTLIST, "values");
	}
	else
	{
		std::string msg = string_format("Unstage(%s) Object create failed.", id.c_str());
		SetLastError(DEF_ERROR, "", 0, msg, false, id.c_str());
	}
	return pUnstage;
}
