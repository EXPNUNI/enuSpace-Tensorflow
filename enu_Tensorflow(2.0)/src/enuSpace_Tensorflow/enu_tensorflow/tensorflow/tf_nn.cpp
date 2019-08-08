#include "stdafx.h"
#include "tf_nn.h"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "jsoncpp/json.h"

#include "GlobalHeader.h"
#include "tensorflow.h"

#include "enuSpaceToTensorflow.h"
#include "AttributeParser.h"
// #include "../../../core/lib/gtl/array_slice.h"
#include "utility_functions.h"
//#include "utilTest.h"

void* Create_AvgPool(std::string id, Json::Value pInputItem) {
	AvgPool* pAvgPool = nullptr;
	Scope* pScope = nullptr;
	Output* value = nullptr;
	std::vector<int> v_ksize;
	std::vector<int> v_strides;
	std::string padding;
	AvgPool::Attrs attrs;


	std::string strTemp;
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
				std::string msg = string_format("warning : AvgPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : AvgPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "ksize")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				v_ksize.clear();
				 GetIntVectorFromInitial(strPinInitial, v_ksize);

			}
			else
			{
				std::string msg = string_format("warning : AvgPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);

			}
			else
			{
				std::string msg = string_format("warning : AvgPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : AvgPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "AvgPool::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}
					
			}
		}
		else
		{
			std::string msg = string_format("warning : AvgPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && value)
	{
		gtl::ArraySlice<int> ksize(v_ksize);
		gtl::ArraySlice<int> strides(v_strides);
		pAvgPool = new AvgPool(*pScope, *value,ksize, strides, padding, attrs);
		ObjectInfo* pObj = AddObjectMap(pAvgPool, id, SYMBOL_AVGPOOL, "AvgPool", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAvgPool->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//ksize.clear();
		//strides.clear();
	}
	else
	{
		std::string msg = string_format("error : AvgPool(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_ksize.clear();
	v_strides.clear();
	return pAvgPool;
}

void* Create_AvgPool3D(std::string id, Json::Value pInputItem) {
	AvgPool3D* pAvgPool3D = nullptr;
	Scope* pScope = nullptr;
	Output* value = nullptr;
	std::vector<int> v_ksize;
	std::vector<int> v_strides;
	std::string padding;
	AvgPool3D::Attrs attrs;
	std::string strTemp ="";
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
				std::string msg = string_format("warning : AvgPool3D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							value = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : AvgPool3D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "ksize")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_ksize);

			}
			else
			{
				std::string msg = string_format("warning : AvgPool3D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);

			}
			else
			{
				std::string msg = string_format("warning : AvgPool3D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : AvgPool3D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "AvgPool3D::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : AvgPool3D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && value)
	{
	
		gtl::ArraySlice<int> strides(v_strides);
		gtl::ArraySlice<int> ksize(v_ksize);
		pAvgPool3D = new AvgPool3D(*pScope, *value, ksize, strides, padding, attrs);
		ObjectInfo* pObj = AddObjectMap(pAvgPool3D, id, SYMBOL_AVGPOOL3D, "AvgPool3D", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAvgPool3D->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//ksize.clear();
		//strides.clear();
	}
	else
	{
		std::string msg = string_format("error : AvgPool3D(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_ksize.clear();
	v_strides.clear();

	return pAvgPool3D;
}

void* Create_AvgPool3DGrad(std::string id, Json::Value pInputItem) {
	AvgPool3DGrad* pAvgPool3DGrad = nullptr;
	Scope* pScope = nullptr;
	Output* orig_input_shape = nullptr;
	Output* grad = nullptr;
	std::vector<int> v_ksize;
	std::vector<int> v_strides;
	std::string padding;
	std::string strTemp = "";
	AvgPool3DGrad::Attrs attrs;

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
				std::string msg = string_format("warning : AvgPool3DGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "orig_input_shape")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							orig_input_shape = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : AvgPool3DGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : AvgPool3DGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "ksize")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_ksize);

			}
			else
			{
				std::string msg = string_format("warning : AvgPool3DGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);

			}
			else
			{
				std::string msg = string_format("warning : AvgPool3DGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : AvgPool3DGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "AvgPool3DGrad::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : AvgPool3DGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && orig_input_shape && grad)
	{
		gtl::ArraySlice<int> ksize(v_ksize);
		gtl::ArraySlice<int> strides(v_strides);
		pAvgPool3DGrad = new AvgPool3DGrad(*pScope, *orig_input_shape,*grad, ksize, strides, padding, attrs);
		ObjectInfo* pObj = AddObjectMap(pAvgPool3DGrad, id, SYMBOL_AVGPOOL3DGRAD, "AvgPool3DGrad", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAvgPool3DGrad->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//ksize.clear();
		//strides.clear();
	}
	else
	{
		std::string msg = string_format("error : AvgPool3DGrad(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_ksize.clear();
	v_strides.clear();

	return pAvgPool3DGrad;
}

void* Create_BiasAdd(std::string id, Json::Value pInputItem) {
	BiasAdd* pBiasAdd = nullptr;
	Scope* pScope = nullptr;
	Output* value = nullptr;
	Output* bias = nullptr;
	BiasAdd::Attrs attrs;
	std::string strTemp ="";
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
				std::string msg = string_format("warning : BiasAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : BiasAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "bias")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							bias = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : BiasAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "BiasAdd::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}

			}
		}
		else
		{
			std::string msg = string_format("warning : BiasAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && value && bias)
	{
		pBiasAdd = new BiasAdd(*pScope, *value, *bias, attrs);
		ObjectInfo* pObj = AddObjectMap(pBiasAdd, id, SYMBOL_BIASADD, "BiasAdd", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pBiasAdd->output, OUTPUT_TYPE_OUTPUT, "output");
		}

	}
	else
	{
		std::string msg = string_format("error : BiasAdd(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}

	return pBiasAdd;
}

void* Create_BiasAddGrad(std::string id, Json::Value pInputItem) {
	BiasAddGrad* pBiasAddGrad = nullptr;
	Scope* pScope = nullptr;
	Output* out_backprop = nullptr;
	BiasAddGrad::Attrs attrs;
	std::string strTemp = "";
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
				std::string msg = string_format("warning : BiasAddGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "out_backprop")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							out_backprop = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : BiasAddGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "BiasAddGrad::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : BiasAddGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && out_backprop)
	{
		pBiasAddGrad = new BiasAddGrad(*pScope, *out_backprop,attrs);
		ObjectInfo* pObj = AddObjectMap(pBiasAddGrad, id, SYMBOL_BIASADDGRAD, "BiasAddGrad", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pBiasAddGrad->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : BiasAddGrad(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}


	return pBiasAddGrad;
}

void* Create_Conv2D(std::string id, Json::Value pInputItem) {
	Conv2D* pConv2D = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* filter = nullptr;
	std::vector<int> v_strides;
	std::string padding;
	Conv2D::Attrs attrs;
	std::string strTemp = "";
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
				std::string msg = string_format("warning : Conv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pInput = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : Conv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "filter")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							filter = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Conv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : Conv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : Conv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Conv2D::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_cudnn_on_gpu_") != "")
					attrs =attrs.UseCudnnOnGpu(attrParser.GetValue_bool("use_cudnn_on_gpu_"));
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : Conv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pInput && filter)
	{
		gtl::ArraySlice<int> strides(v_strides);
		pConv2D = new Conv2D(*pScope, *pInput, *filter, strides, padding,attrs);
		ObjectInfo* pObj = AddObjectMap(pConv2D, id, SYMBOL_CONV2D, "Conv2D", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pConv2D->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//strides.clear();
	}
	else
	{
		std::string msg = string_format("error : Conv2D(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_strides.clear();
	return pConv2D;
}

void* Create_Conv2DBackpropFilter(std::string id, Json::Value pInputItem) {
	Conv2DBackpropFilter* pConv2DBackpropFilter = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* filter_sizes = nullptr;
	Output* out_backprop = nullptr;
	std::vector<int> v_strides;
	std::string padding;
	std::string strTemp = "";
	Conv2DBackpropFilter::Attrs attrs;

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
				std::string msg = string_format("warning : Conv2DBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pInput = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : Conv2DBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "filter_sizes")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							filter_sizes = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Conv2DBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "out_backprop")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							out_backprop = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Conv2DBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : Conv2DBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : Conv2DBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Conv2DBackpropFilter::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_cudnn_on_gpu_") != "")
					attrs =attrs.UseCudnnOnGpu(attrParser.GetValue_bool("use_cudnn_on_gpu_"));
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : Conv2DBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pInput && filter_sizes && out_backprop)
	{
		gtl::ArraySlice<int> strides(v_strides);
		pConv2DBackpropFilter = new Conv2DBackpropFilter(*pScope, *pInput, *filter_sizes,*out_backprop, strides, padding, attrs);
		ObjectInfo* pObj = AddObjectMap(pConv2DBackpropFilter, id, SYMBOL_CONV2DBACKPROPFILTER, "Conv2DBackpropFilter", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pConv2DBackpropFilter->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//strides.clear();

	}
	else
	{
		std::string msg = string_format("error : Conv2DBackpropFilter(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_strides.clear();
	return pConv2DBackpropFilter;
}

void* Create_Conv2DBackpropInput(std::string id, Json::Value pInputItem) {
	Conv2DBackpropInput* pConv2DBackpropInput = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* filter = nullptr;
	Output* out_backprop = nullptr;
	std::vector<int> v_strides;
	std::string padding;
	Conv2DBackpropInput::Attrs attrs;
	std::string strTemp = "";

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
				std::string msg = string_format("warning : Conv2DBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input_sizes")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pInput = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : Conv2DBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "filter")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							filter = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Conv2DBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "out_backprop")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							out_backprop = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Conv2DBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : Conv2DBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : Conv2DBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Conv2DBackpropInput::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("use_cudnn_on_gpu_") != "")
					attrs =attrs.UseCudnnOnGpu(attrParser.GetValue_bool("use_cudnn_on_gpu_"));
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : Conv2DBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pInput && filter && out_backprop)
	{
		gtl::ArraySlice<int> strides(v_strides);
		pConv2DBackpropInput = new Conv2DBackpropInput(*pScope, *pInput, *filter, *out_backprop, strides, padding, attrs);
		ObjectInfo* pObj = AddObjectMap(pConv2DBackpropInput, id, SYMBOL_CONV2DBACKPROPINPUT, "Conv2DBackpropInput", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pConv2DBackpropInput->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//strides.clear();

	}
	else
	{
		std::string msg = string_format("error : Conv2DBackpropInput(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_strides.clear();
	return pConv2DBackpropInput;
}

void* Create_Conv3D(std::string id, Json::Value pInputItem) {
	Conv3D* pConv3D = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* filter = nullptr;
	std::vector<int> v_strides;
	std::string padding;
	Conv3D::Attrs attrs;
	std::string strTemp = "";
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
				std::string msg = string_format("warning : Conv3D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pInput = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : Conv3D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "filter")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							filter = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Conv3D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : Conv3D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : Conv3D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Conv3D::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : Conv3D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pInput && filter )
	{
		gtl::ArraySlice<int> strides(v_strides);
		pConv3D = new Conv3D(*pScope, *pInput, *filter, strides, padding, attrs);
		ObjectInfo* pObj = AddObjectMap(pConv3D, id, SYMBOL_CONV3D, "Conv3D", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pConv3D->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//strides.clear();

	}
	else
	{
		std::string msg = string_format("error : Conv3D(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_strides.clear();
	return pConv3D;
}

void* Create_Conv3DBackpropFilterV2(std::string id, Json::Value pInputItem) {
	Conv3DBackpropFilterV2* pConv3DBackpropFilterV2 = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* filter_sizes = nullptr;
	Output* out_backprop = nullptr;
	std::vector<int> v_strides;
	std::string padding;
	Conv3DBackpropFilterV2::Attrs attrs;
	std::string strTemp = "";
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
				std::string msg = string_format("warning : Conv3DBackpropFilterV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pInput = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : Conv3DBackpropFilterV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "filter_sizes")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							filter_sizes = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Conv3DBackpropFilterV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "out_backprop")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							out_backprop = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Conv3DBackpropFilterV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : Conv3DBackpropFilterV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : Conv3DBackpropFilterV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Conv3DBackpropFilterV2::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : Conv3DBackpropFilterV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pInput && filter_sizes && out_backprop)
	{
		gtl::ArraySlice<int> strides(v_strides);
		pConv3DBackpropFilterV2 = new Conv3DBackpropFilterV2(*pScope, *pInput, *filter_sizes,*out_backprop, strides, padding, attrs);
		ObjectInfo* pObj = AddObjectMap(pConv3DBackpropFilterV2, id, SYMBOL_CONV3DBACKPROPFILTERV2, "Conv3DBackpropFilterV2", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pConv3DBackpropFilterV2->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//strides.clear();

	}
	else
	{
		std::string msg = string_format("error : Conv3DBackpropFilterV2(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_strides.clear();
	return pConv3DBackpropFilterV2;
}

void* Create_Conv3DBackpropInputV2(std::string id, Json::Value pInputItem) {
	Conv3DBackpropInputV2* pConv3DBackpropInputV2 = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* filter = nullptr;
	Output* out_backprop = nullptr;
	std::vector<int> v_strides;
	std::string padding;
	Conv3DBackpropInputV2::Attrs attrs;
	std::string strTemp = "";
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
				std::string msg = string_format("warning : Conv3DBackpropInputV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input_sizes")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pInput = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : Conv3DBackpropInputV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "filter")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							filter = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Conv3DBackpropInputV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "out_backprop")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							out_backprop = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Conv3DBackpropInputV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : Conv3DBackpropInputV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : Conv3DBackpropInputV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Conv3DBackpropInputV2::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : Conv3DBackpropInputV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pInput && filter && out_backprop)
	{
		gtl::ArraySlice<int> strides(v_strides);
		pConv3DBackpropInputV2 = new Conv3DBackpropInputV2(*pScope, *pInput, *filter, *out_backprop, strides, padding, attrs);
		ObjectInfo* pObj = AddObjectMap(pConv3DBackpropInputV2, id, SYMBOL_CONV3DBACKPROPINPUTV2, "Conv3DBackpropInputV2", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pConv3DBackpropInputV2->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//strides.clear();

	}
	else
	{
		std::string msg = string_format("error : Conv3DBackpropInputV2(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_strides.clear();
	return pConv3DBackpropInputV2;
}

void* Create_DepthwiseConv2dNative(std::string id, Json::Value pInputItem) {
	DepthwiseConv2dNative* pDepthwiseConv2dNative = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* filter = nullptr;
	std::vector<int> v_strides;
	std::string padding;
	DepthwiseConv2dNative::Attrs attrs;
	std::string strTemp = "";

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
				std::string msg = string_format("warning : DepthwiseConv2dNative - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pInput = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : DepthwiseConv2dNative - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "filter")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							filter = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DepthwiseConv2dNative - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : DepthwiseConv2dNative - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : DepthwiseConv2dNative - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "DepthwiseConv2dNative::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : DepthwiseConv2dNative - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pInput && filter)
	{
		gtl::ArraySlice<int> strides(v_strides);
		pDepthwiseConv2dNative = new DepthwiseConv2dNative(*pScope, *pInput, *filter, strides, padding, attrs);
		ObjectInfo* pObj = AddObjectMap(pDepthwiseConv2dNative, id, SYMBOL_DEPTHWISECONV2DNATIVE, "DepthwiseConv2dNative", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pDepthwiseConv2dNative->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//strides.clear();

	}
	else
	{
		std::string msg = string_format("error : DepthwiseConv2dNative(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_strides.clear();
	return pDepthwiseConv2dNative;
}

void* Create_DepthwiseConv2dNativeBackpropFilter(std::string id, Json::Value pInputItem) {
	DepthwiseConv2dNativeBackpropFilter* pDepthwiseConv2dNativeBackpropFilter = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* filter_sizes = nullptr;
	Output* out_backprop = nullptr;
	std::vector<int> v_strides;
	std::string padding;
	DepthwiseConv2dNativeBackpropFilter::Attrs attrs;
	std::string strTemp = "";
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
				std::string msg = string_format("warning : DepthwiseConv2dNativeBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pInput = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : DepthwiseConv2dNativeBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "filter_sizes")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							filter_sizes = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DepthwiseConv2dNativeBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "out_backprop")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							out_backprop = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DepthwiseConv2dNativeBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : DepthwiseConv2dNativeBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : DepthwiseConv2dNativeBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "DepthwiseConv2dNativeBackpropFilter::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : DepthwiseConv2dNativeBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pInput && filter_sizes && out_backprop)
	{
		gtl::ArraySlice<int> strides(v_strides);
		pDepthwiseConv2dNativeBackpropFilter = new DepthwiseConv2dNativeBackpropFilter(*pScope, *pInput, *filter_sizes,*out_backprop, strides, padding, attrs);
		ObjectInfo* pObj = AddObjectMap(pDepthwiseConv2dNativeBackpropFilter, id, SYMBOL_DEPTHWISECONV2DNATIVEBACKPROPFILTER, 
										"DepthwiseConv2dNativeBackpropFilter", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pDepthwiseConv2dNativeBackpropFilter->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//strides.clear();

	}
	else
	{
		std::string msg = string_format("error : DepthwiseConv2dNativeBackpropFilter(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_strides.clear();
	return pDepthwiseConv2dNativeBackpropFilter;
}

void* Create_DepthwiseConv2dNativeBackpropInput(std::string id, Json::Value pInputItem) {
	DepthwiseConv2dNativeBackpropInput* pDepthwiseConv2dNativeBackpropInput = nullptr;
	Scope* pScope = nullptr;
	Output* input_sizes = nullptr;
	Output* filter = nullptr;
	Output* out_backprop = nullptr;
	std::vector<int> v_strides;
	std::string padding;
	DepthwiseConv2dNativeBackpropInput::Attrs attrs;
	std::string strTemp = "";
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
				std::string msg = string_format("warning : DepthwiseConv2dNativeBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "input_sizes")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							input_sizes = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : DepthwiseConv2dNativeBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "filter")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							filter = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DepthwiseConv2dNativeBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "out_backprop")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							out_backprop = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DepthwiseConv2dNativeBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : DepthwiseConv2dNativeBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : DepthwiseConv2dNativeBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "DepthwiseConv2dNativeBackpropInput::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : DepthwiseConv2dNativeBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && input_sizes && filter && out_backprop)
	{
		gtl::ArraySlice<int> strides(v_strides);
		pDepthwiseConv2dNativeBackpropInput = new DepthwiseConv2dNativeBackpropInput(*pScope, *input_sizes, *filter, *out_backprop, strides, padding, attrs);
		ObjectInfo* pObj = AddObjectMap(pDepthwiseConv2dNativeBackpropInput, id, SYMBOL_DEPTHWISECONV2DNATIVEBACKPROPINPUT,
			"DepthwiseConv2dNativeBackpropInput", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pDepthwiseConv2dNativeBackpropInput->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//strides.clear();

	}
	else
	{
		std::string msg = string_format("error : DepthwiseConv2dNativeBackpropInput(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_strides.clear();
	return pDepthwiseConv2dNativeBackpropInput;
}

void* Create_Dilation2D(std::string id, Json::Value pInputItem) {
	Dilation2D* pDilation2D = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* filter = nullptr;
	std::vector<int> v_strides;
	std::vector<int> v_rates;
	std::string padding;

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
				std::string msg = string_format("warning : Dilation2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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

			}
			else
			{
				std::string msg = string_format("warning : Dilation2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "filter")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							filter = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Dilation2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : Dilation2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "rates")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_rates);
			}
			else
			{
				std::string msg = string_format("warning : Dilation2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : Dilation2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Dilation2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput && filter)
	{
		gtl::ArraySlice<int> strides(v_strides);
		gtl::ArraySlice<int> rates(v_rates);

		pDilation2D = new Dilation2D(*pScope, *pinput, *filter, strides, rates, padding);
		ObjectInfo* pObj = AddObjectMap(pDilation2D, id, SYMBOL_DILATION2D,"Dilation2D", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pDilation2D->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//strides.clear();
		//rates.clear();
	}
	else
	{
		std::string msg = string_format("error : Dilation2D(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_strides.clear();
	v_rates.clear();
	return pDilation2D;
}

void* Create_Dilation2DBackpropFilter(std::string id, Json::Value pInputItem) {
	Dilation2DBackpropFilter* pDilation2DBackpropFilter = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* filter = nullptr;
	Output* out_backprop = nullptr;
	std::vector<int> v_strides;
	std::vector<int> v_rates;
	std::string padding;

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
				std::string msg = string_format("warning : Dilation2DBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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

			}
			else
			{
				std::string msg = string_format("warning : Dilation2DBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "filter")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							filter = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Dilation2DBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "out_backprop")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							out_backprop = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Dilation2DBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : Dilation2DBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "rates")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_rates);
			}
			else
			{
				std::string msg = string_format("warning : Dilation2DBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : Dilation2DBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Dilation2DBackpropFilter - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput && filter && out_backprop)
	{
		gtl::ArraySlice<int> strides(v_strides);
		gtl::ArraySlice<int> rates(v_rates);

		pDilation2DBackpropFilter = new Dilation2DBackpropFilter(*pScope, *pinput, *filter,*out_backprop ,strides, rates, padding);
		ObjectInfo* pObj = AddObjectMap(pDilation2DBackpropFilter, id, SYMBOL_DILATION2DBACKPROPFILTER, "Dilation2DBackpropFilter", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pDilation2DBackpropFilter->filter_backprop, OUTPUT_TYPE_OUTPUT, "filter_backprop");
		}
		//strides.clear();
		//rates.clear();
	}
	else
	{
		std::string msg = string_format("error : Dilation2DBackpropFilter(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_strides.clear();
	v_rates.clear();
	return pDilation2DBackpropFilter;
}

void* Create_Dilation2DBackpropInput(std::string id, Json::Value pInputItem) {
	Dilation2DBackpropInput* pDilation2DBackpropInput = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* filter = nullptr;
	Output* out_backprop = nullptr;
	std::vector<int> v_strides;
	std::vector<int> v_rates;
	std::string padding;

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
				std::string msg = string_format("warning : Dilation2DBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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

			}
			else
			{
				std::string msg = string_format("warning : Dilation2DBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "filter")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							filter = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Dilation2DBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "out_backprop")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							out_backprop = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Dilation2DBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : Dilation2DBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "rates")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_rates);
			}
			else
			{
				std::string msg = string_format("warning : Dilation2DBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : Dilation2DBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Dilation2DBackpropInput - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput && filter && out_backprop)
	{
		gtl::ArraySlice<int> strides(v_strides);
		gtl::ArraySlice<int> rates(v_rates);

		pDilation2DBackpropInput = new Dilation2DBackpropInput(*pScope, *pinput, *filter, *out_backprop, strides, rates, padding);
		ObjectInfo* pObj = AddObjectMap(pDilation2DBackpropInput, id, SYMBOL_DILATION2DBACKPROPINPUT, "Dilation2DBackpropInput", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pDilation2DBackpropInput->in_backprop, OUTPUT_TYPE_OUTPUT, "in_backprop");
		}
		//strides.clear();
		//rates.clear();
	}
	else
	{
		std::string msg = string_format("error : Dilation2DBackpropInput(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_strides.clear();
	v_rates.clear();
	return pDilation2DBackpropInput;
}

void* Create_Elu(std::string id, Json::Value pInputItem) {
	Elu* pElu = nullptr;
	Scope* pScope = nullptr;
	Output* features = nullptr;

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
				std::string msg = string_format("warning : Elu - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "features")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							features = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : Elu - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Elu - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	
	}
	if (pScope && features)
	{
		pElu = new Elu(*pScope, *features);
		ObjectInfo* pObj = AddObjectMap(pElu, id, SYMBOL_ELU, "Elu", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pElu->activations, OUTPUT_TYPE_OUTPUT, "activations");
		}
	}
	else
	{
		std::string msg = string_format("error : Elu(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pElu;
}

void* Create_FractionalAvgPool(std::string id, Json::Value pInputItem) {
	FractionalAvgPool* pFractionalAvgPool = nullptr;
	Scope* pScope = nullptr;
	Output* value = nullptr;
	std::vector<float> v_pooling_ratio;
	FractionalAvgPool::Attrs attrs;

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
				std::string msg = string_format("warning : FractionalAvgPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : FractionalAvgPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "pooling_ratio")
		{
			if (strPinInterface == "gtl::ArraySlice<float>")
			{
				GetFloatVectorFromInitial(strPinInitial, v_pooling_ratio);
			}
			else
			{
				std::string msg = string_format("warning : FractionalAvgPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "FractionalAvgPool::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("pseudo_random_") != "")
					attrs = attrs.PseudoRandom(attrParser.GetValue_bool("pseudo_random_"));
				if (attrParser.GetAttribute("overlapping_") != "")
					attrs =attrs.Overlapping(attrParser.GetValue_bool("overlapping_"));
				if (attrParser.GetAttribute("deterministic_") != "")
					attrs = attrs.Deterministic(attrParser.GetValue_bool("deterministic_"));
				if (attrParser.GetAttribute("seed_") != "")
					attrs =attrs.Seed(attrParser.GetValue_int64("seed_"));
				if (attrParser.GetAttribute("seed2_") != "")
					attrs =attrs.Seed2(attrParser.GetValue_bool("seed2_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : FractionalAvgPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && value)
	{
		gtl::ArraySlice<float> pooling_ratio(v_pooling_ratio);

		pFractionalAvgPool = new FractionalAvgPool(*pScope, *value, pooling_ratio,attrs);
		ObjectInfo* pObj = AddObjectMap(pFractionalAvgPool, id, SYMBOL_FRACTIONALAVGPOOL, "FractionalAvgPool", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pFractionalAvgPool->output, OUTPUT_TYPE_OUTPUT, "output");
			AddOutputInfo(pObj, &pFractionalAvgPool->row_pooling_sequence, OUTPUT_TYPE_OUTPUT, "row_pooling_sequence");
			AddOutputInfo(pObj, &pFractionalAvgPool->col_pooling_sequence, OUTPUT_TYPE_OUTPUT, "col_pooling_sequence");
		}
		//pooling_ratio.clear();
	}
	else
	{
		std::string msg = string_format("error : Dilation2DBackpropInput(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_pooling_ratio.clear();
	return pFractionalAvgPool;
}

void* Create_FractionalMaxPool(std::string id, Json::Value pInputItem) {
	FractionalMaxPool* pFractionalMaxPool = nullptr;
	Scope* pScope = nullptr;
	Output* value = nullptr;
	std::vector<float> v_pooling_ratio;
	FractionalMaxPool::Attrs attrs;

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
				std::string msg = string_format("warning : FractionalMaxPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : FractionalMaxPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "pooling_ratio")
		{
			if (strPinInterface == "gtl::ArraySlice<float>")
			{
				GetFloatVectorFromInitial(strPinInitial, v_pooling_ratio);
			}
			else
			{
				std::string msg = string_format("warning : FractionalMaxPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "FractionalAvgPool::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("pseudo_random_") != "")
					attrs =attrs.PseudoRandom(attrParser.GetValue_bool("pseudo_random_"));
				if (attrParser.GetAttribute("overlapping_") != "")
					attrs =attrs.Overlapping(attrParser.GetValue_bool("overlapping_"));
				if (attrParser.GetAttribute("deterministic_") != "")
					attrs =attrs.Deterministic(attrParser.GetValue_bool("deterministic_"));
				if (attrParser.GetAttribute("seed_") != "")
					attrs =attrs.Seed(attrParser.GetValue_int64("seed_"));
				if (attrParser.GetAttribute("seed2_") != "")
					attrs = attrs.Seed2(attrParser.GetValue_bool("seed2_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : FractionalMaxPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && value)
	{
		gtl::ArraySlice<float> pooling_ratio(v_pooling_ratio);
		pFractionalMaxPool = new FractionalMaxPool(*pScope, *value, pooling_ratio, attrs);
		ObjectInfo* pObj = AddObjectMap(pFractionalMaxPool, id, SYMBOL_FRACTIONALMAXPOOL, "FractionalMaxPool", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pFractionalMaxPool->output, OUTPUT_TYPE_OUTPUT, "output");
			AddOutputInfo(pObj, &pFractionalMaxPool->row_pooling_sequence, OUTPUT_TYPE_OUTPUT, "row_pooling_sequence");
			AddOutputInfo(pObj, &pFractionalMaxPool->col_pooling_sequence, OUTPUT_TYPE_OUTPUT, "col_pooling_sequence");
		}
		//pooling_ratio.clear();
	}
	else
	{
		std::string msg = string_format("error : FractionalMaxPool(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_pooling_ratio.clear();
	return pFractionalMaxPool;
}

void* Create_FusedBatchNorm(std::string id, Json::Value pInputItem) {
	FusedBatchNorm* pFusedBatchNorm = nullptr;
	Scope* pScope = nullptr;
	Output* x = nullptr;
	Output* scale = nullptr;
	Output* offset = nullptr;
	Output* mean = nullptr;
	Output* variance = nullptr;
	FusedBatchNorm::Attrs attrs;
	std::string strTemp = "";
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
				std::string msg = string_format("warning : FusedBatchNorm - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							x = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : FusedBatchNorm - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "scale")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							scale = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : FusedBatchNorm - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "offset")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							offset = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : FusedBatchNorm - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "mean")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							mean = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : FusedBatchNorm - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "variance")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							variance = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : FusedBatchNorm - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "FusedBatchNorm::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("epsilon_") != "")
					attrs =attrs.Epsilon(attrParser.GetValue_float("epsilon_"));
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}
				if (attrParser.GetAttribute("is_training_") != "")
					attrs =attrs.IsTraining(attrParser.GetValue_bool("is_training_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : FusedBatchNorm - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && x &&scale &&offset &&mean &&variance)
	{
		
		pFusedBatchNorm = new FusedBatchNorm(*pScope, *x, *scale,*offset,*mean,*variance, attrs);
		ObjectInfo* pObj = AddObjectMap(pFusedBatchNorm, id, SYMBOL_FUSEDBATCHNORM, "FusedBatchNorm", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pFusedBatchNorm->y, OUTPUT_TYPE_OUTPUT, "y");
			AddOutputInfo(pObj, &pFusedBatchNorm->batch_mean, OUTPUT_TYPE_OUTPUT, "batch_mean");
			AddOutputInfo(pObj, &pFusedBatchNorm->batch_variance, OUTPUT_TYPE_OUTPUT, "batch_variance");
			AddOutputInfo(pObj, &pFusedBatchNorm->reserve_space_1, OUTPUT_TYPE_OUTPUT, "reserve_space_1");
			AddOutputInfo(pObj, &pFusedBatchNorm->reserve_space_2, OUTPUT_TYPE_OUTPUT, "reserve_space_2");
		}
	}
	else
	{
		std::string msg = string_format("error : FusedBatchNorm(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pFusedBatchNorm;
}

void* Create_FusedBatchNormGrad(std::string id, Json::Value pInputItem) {
	FusedBatchNormGrad* pFusedBatchNormGrad = nullptr;
	Scope* pScope = nullptr;
	Output* y_backprop = nullptr;
	Output* x = nullptr;
	Output* scale = nullptr;
	Output* reserve_space_1 = nullptr;
	Output* reserve_space_2 = nullptr;
	FusedBatchNormGrad::Attrs attrs;
	std::string strTemp = "";
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
				std::string msg = string_format("warning : FusedBatchNormGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "y_backprop")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							y_backprop = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : FusedBatchNormGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							x = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : FusedBatchNormGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "scale")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							scale = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : FusedBatchNormGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "reserve_space_1")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							reserve_space_1 = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : FusedBatchNormGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "reserve_space_2")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							reserve_space_2 = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : FusedBatchNormGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "FusedBatchNormGrad::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("epsilon_") != "")
					attrs=attrs.Epsilon(attrParser.GetValue_float("epsilon_"));
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}
				if (attrParser.GetAttribute("is_training_") != "")
					attrs =attrs.IsTraining(attrParser.GetValue_bool("is_training_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : FusedBatchNormGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&y_backprop&& x &&scale &&reserve_space_1 &&reserve_space_2)
	{
		
		pFusedBatchNormGrad = new FusedBatchNormGrad(*pScope,*y_backprop ,*x, *scale,*reserve_space_1,*reserve_space_2, attrs);
		ObjectInfo* pObj = AddObjectMap(pFusedBatchNormGrad, id, SYMBOL_FUSEDBATCHNORMGRAD, "FusedBatchNormGrad", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pFusedBatchNormGrad->x_backprop, OUTPUT_TYPE_OUTPUT, "x_backprop");
			AddOutputInfo(pObj, &pFusedBatchNormGrad->scale_backprop, OUTPUT_TYPE_OUTPUT, "scale_backprop");
			AddOutputInfo(pObj, &pFusedBatchNormGrad->offset_backprop, OUTPUT_TYPE_OUTPUT, "offset_backprop");
			AddOutputInfo(pObj, &pFusedBatchNormGrad->reserve_space_3, OUTPUT_TYPE_OUTPUT, "reserve_space_3");
			AddOutputInfo(pObj, &pFusedBatchNormGrad->reserve_space_4, OUTPUT_TYPE_OUTPUT, "reserve_space_4");
		}
	}
	else
	{
		std::string msg = string_format("error : FusedBatchNormGrad(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pFusedBatchNormGrad;
}

void* Create_FusedPadConv2D(std::string id, Json::Value pInputItem) {
	FusedPadConv2D* pFusedPadConv2D = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* paddings = nullptr;
	Output* filter = nullptr;
	std::string  mode;
	std::vector<int> v_strides;
	std::string padding;

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
				std::string msg = string_format("warning : FusedPadConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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

			}
			else
			{
				std::string msg = string_format("warning : FusedPadConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "paddings")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							paddings = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : FusedPadConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "filter")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							filter = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : FusedPadConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "mode")
		{
			if (strPinInterface == "StringPiece")
			{
				mode = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : FusedPadConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : FusedPadConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : FusedPadConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : FusedPadConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&pinput&& paddings &&filter)
	{
		gtl::ArraySlice<int> strides(v_strides);
		pFusedPadConv2D = new FusedPadConv2D(*pScope, *pinput, *paddings, *filter, mode, strides, padding);
		ObjectInfo* pObj = AddObjectMap(pFusedPadConv2D, id, SYMBOL_FUSEDPADCONV2D, "FusedPadConv2D", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pFusedPadConv2D->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//strides.clear();
	}
	else
	{
		std::string msg = string_format("error : FusedPadConv2D(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_strides.clear();
	return pFusedPadConv2D;
}

void* Create_FusedResizeAndPadConv2D(std::string id, Json::Value pInputItem) {
	FusedResizeAndPadConv2D* pFusedResizeAndPadConv2D = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* size = nullptr;
	Output* paddings = nullptr;
	Output* filter = nullptr;
	std::string mode;
	std::vector<int> v_strides;
	std::string padding;
	FusedResizeAndPadConv2D::Attrs attrs;

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
				std::string msg = string_format("warning : FusedResizeAndPadConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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

			}
			else
			{
				std::string msg = string_format("warning : FusedResizeAndPadConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							size = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : FusedResizeAndPadConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "paddings")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							paddings = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : FusedResizeAndPadConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "filter")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							filter = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : FusedResizeAndPadConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "mode")
		{
			if (strPinInterface == "StringPiece")
			{
				mode = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : FusedResizeAndPadConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : FusedResizeAndPadConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : FusedResizeAndPadConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "FusedResizeAndPadConv2D::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("resize_align_corners_") != "")
					attrs =attrs.ResizeAlignCorners(attrParser.GetValue_bool("resize_align_corners_"));
			}
			else
			{
				std::string msg = string_format("warning : FusedResizeAndPadConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : FusedResizeAndPadConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&pinput&&size &&paddings &&filter)
	{
		gtl::ArraySlice<int> strides(v_strides);
		pFusedResizeAndPadConv2D = new FusedResizeAndPadConv2D(*pScope, *pinput,*size, *paddings, *filter, mode, strides, padding,attrs);
		ObjectInfo* pObj = AddObjectMap(pFusedResizeAndPadConv2D, id, SYMBOL_FUSEDRESIZEANDPADCONV2D, "FusedResizeAndPadConv2D", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pFusedResizeAndPadConv2D->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//strides.clear();
	}
	else
	{
		std::string msg = string_format("error : FusedResizeAndPadConv2D(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_strides.clear();
	return pFusedResizeAndPadConv2D;
}

void* Create_InTopK(std::string id, Json::Value pInputItem) {
	InTopK* pInTopK = nullptr;
	Scope* pScope = nullptr;
	Output* predictions = nullptr;
	Output* targets = nullptr;
	int64 k = 0;

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
				std::string msg = string_format("warning : InTopK - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "predictions")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							predictions = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : InTopK - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "targets")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							targets = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : InTopK - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "k")
		{
			if (strPinInterface == "int64")
			{
				k = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : InTopK - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : InTopK - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&predictions&&targets)
	{
		pInTopK = new InTopK(*pScope, *predictions, *targets, k);
		ObjectInfo* pObj = AddObjectMap(pInTopK, id, SYMBOL_INTOPK, "InTopK", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pInTopK->precision, OUTPUT_TYPE_OUTPUT, "precision");
		}
	}
	else
	{
		std::string msg = string_format("error : InTopK(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pInTopK;
}

void* Create_L2Loss(std::string id, Json::Value pInputItem) {
	L2Loss* pL2Loss = nullptr;
	Scope* pScope = nullptr;
	Output* t = nullptr;

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
				std::string msg = string_format("warning : L2Loss - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "t")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							t = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : L2Loss - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : L2Loss - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&t)
	{
		pL2Loss = new L2Loss(*pScope, *t);
		ObjectInfo* pObj = AddObjectMap(pL2Loss, id, SYMBOL_L2LOSS, "L2Loss", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pL2Loss->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : L2Loss(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pL2Loss;
}

void* Create_LRN(std::string id, Json::Value pInputItem) {
	LRN* pLRN = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	LRN::Attrs attrs;

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
				std::string msg = string_format("warning : LRN - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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

			}
			else
			{
				std::string msg = string_format("warning : LRN - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "LRN::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("depth_radius_") != "")
					attrs =attrs.DepthRadius(attrParser.GetValue_int64("depth_radius_"));
				if (attrParser.GetAttribute("bias_") != "")
					attrs =attrs.Bias(attrParser.GetValue_float("bias_"));
				if (attrParser.GetAttribute("alpha_") != "")
					attrs=attrs.Alpha(attrParser.GetValue_float("alpha_"));
				if (attrParser.GetAttribute("beta_") != "")
					attrs =attrs.Beta(attrParser.GetValue_float("beta_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : LRN - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&pinput)
	{
		pLRN = new LRN(*pScope, *pinput,attrs);
		ObjectInfo* pObj = AddObjectMap(pLRN, id, SYMBOL_LRN, "LRN", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pLRN->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : LRN(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pLRN;
}

void* Create_LogSoftmax(std::string id, Json::Value pInputItem) {
	LogSoftmax* pLogSoftmax = nullptr;
	Scope* pScope = nullptr;
	Output* logits = nullptr;

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
				std::string msg = string_format("warning : LogSoftmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "logits")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							logits = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : LogSoftmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : LogSoftmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&logits)
	{
		pLogSoftmax = new LogSoftmax(*pScope, *logits);
		ObjectInfo* pObj = AddObjectMap(pLogSoftmax, id, SYMBOL_LOGSOFTMAX, "LogSoftmax", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pLogSoftmax->logsoftmax, OUTPUT_TYPE_OUTPUT, "logsoftmax");
		}
	}
	else
	{
		std::string msg = string_format("error : LogSoftmax(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pLogSoftmax;
}

void* Create_MaxPool(std::string id, Json::Value pInputItem) {
	MaxPool* pMaxPool = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	std::vector<int> v_ksize;
	std::vector<int> v_strides;
	std::string padding;
	MaxPool::Attrs attrs;
	std::string strTemp = "";
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
				std::string msg = string_format("warning : MaxPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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

			}
			else
			{
				std::string msg = string_format("warning : MaxPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "ksize")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_ksize);
			}
			else
			{
				std::string msg = string_format("warning : MaxPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : MaxPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : MaxPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "MaxPool::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}
			}
			else
			{
				std::string msg = string_format("warning : MaxPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : MaxPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&pinput)
	{
		gtl::ArraySlice<int> strides(v_strides);
		gtl::ArraySlice<int> ksize(v_ksize);
		pMaxPool = new MaxPool(*pScope, *pinput, ksize, strides, padding, attrs);
		ObjectInfo* pObj = AddObjectMap(pMaxPool, id, SYMBOL_MAXPOOL, "MaxPool", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pMaxPool->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//ksize.clear();
		//strides.clear();
	}
	else
	{
		std::string msg = string_format("error : MaxPool(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_ksize.clear();
	v_strides.clear();
	return pMaxPool;
}

void* Create_MaxPool3D(std::string id, Json::Value pInputItem) {
	MaxPool3D* pMaxPool3D = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	std::vector<int> v_ksize;
	std::vector<int> v_strides;
	std::string padding;
	MaxPool3D::Attrs attrs;
	std::string strTemp = "";
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
				std::string msg = string_format("warning : MaxPool3D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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

			}
			else
			{
				std::string msg = string_format("warning : MaxPool3D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "ksize")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_ksize);
			}
			else
			{
				std::string msg = string_format("warning : MaxPool3D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : MaxPool3D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : MaxPool3D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "MaxPool3D::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs =attrs.DataFormat(strTemp);
				}
			}
			else
			{
				std::string msg = string_format("warning : MaxPool3D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : MaxPool3D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&pinput)
	{
		gtl::ArraySlice<int> strides(v_strides);
		gtl::ArraySlice<int> ksize(v_ksize);
		pMaxPool3D = new MaxPool3D(*pScope, *pinput, ksize, strides, padding, attrs);
		ObjectInfo* pObj = AddObjectMap(pMaxPool3D, id, SYMBOL_MAXPOOL3D, "MaxPool3D", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pMaxPool3D->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//ksize.clear();
		//strides.clear();
	}
	else
	{
		std::string msg = string_format("error : MaxPool(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_ksize.clear();
	v_strides.clear();
	return pMaxPool3D;
}

void* Create_MaxPool3DGrad(std::string id, Json::Value pInputItem) {
	MaxPool3DGrad* pMaxPool3DGrad = nullptr;
	Scope* pScope = nullptr;
	Output* orig_input = nullptr;
	Output* orig_output = nullptr;
	Output* grad = nullptr;
	std::vector<int> v_ksize;
	std::vector<int> v_strides;
	std::string padding;
	std::string strTemp = "";
	MaxPool3DGrad::Attrs attrs;

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
				std::string msg = string_format("warning : MaxPool3DGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "orig_input")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							orig_input = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : MaxPool3DGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "orig_output")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							orig_output = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : MaxPool3DGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : MaxPool3DGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "ksize")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_ksize);
			}
			else
			{
				std::string msg = string_format("warning : MaxPool3DGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : MaxPool3DGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : MaxPool3DGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "MaxPool3DGrad::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}
			}
			else
			{
				std::string msg = string_format("warning : MaxPool3DGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : MaxPool3DGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&orig_input&&orig_output&&grad)
	{
		gtl::ArraySlice<int> strides(v_strides);
		gtl::ArraySlice<int> ksize(v_ksize);
		pMaxPool3DGrad = new MaxPool3DGrad(*pScope, *orig_input,*orig_output,*grad, ksize, strides, padding, attrs);
		ObjectInfo* pObj = AddObjectMap(pMaxPool3DGrad, id, SYMBOL_MAXPOOL3DGRAD, "MaxPool3DGrad", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pMaxPool3DGrad->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//ksize.clear();
		//strides.clear();
	}
	else
	{
		std::string msg = string_format("error : MaxPool3DGrad(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_ksize.clear();
	v_strides.clear();
	return pMaxPool3DGrad;
}

void* Create_MaxPool3DGradGrad(std::string id, Json::Value pInputItem) {
	MaxPool3DGradGrad* pMaxPool3DGradGrad = nullptr;
	Scope* pScope = nullptr;
	Output* orig_input = nullptr;
	Output* orig_output = nullptr;
	Output* grad = nullptr;
	std::vector<int> v_ksize;
	std::vector<int> v_strides;
	std::string padding;
	MaxPool3DGradGrad::Attrs attrs;
	std::string strTemp = "";

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
				std::string msg = string_format("warning : MaxPool3DGradGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "orig_input")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							orig_input = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : MaxPool3DGradGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "orig_output")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							orig_output = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : MaxPool3DGradGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : MaxPool3DGradGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "ksize")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				
				GetIntVectorFromInitial(strPinInitial, v_ksize);
			}
			else
			{
				std::string msg = string_format("warning : MaxPool3DGradGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : MaxPool3DGradGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : MaxPool3DGradGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "MaxPool3DGradGrad::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}
			}
			else
			{
				std::string msg = string_format("warning : MaxPool3DGradGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : MaxPool3DGradGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&orig_input&&orig_output&&grad)
	{
		gtl::ArraySlice<int> strides(v_strides);
		gtl::ArraySlice<int> ksize(v_ksize);
		pMaxPool3DGradGrad = new MaxPool3DGradGrad(*pScope, *orig_input, *orig_output, *grad, ksize, strides, padding, attrs);
		ObjectInfo* pObj = AddObjectMap(pMaxPool3DGradGrad, id, SYMBOL_MAXPOOL3DGRADGRAD, "MaxPool3DGradGrad", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pMaxPool3DGradGrad->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//ksize.clear();
		//strides.clear();
	}
	else
	{
		std::string msg = string_format("error : MaxPool3DGradGrad(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_ksize.clear();
	v_strides.clear();
	return pMaxPool3DGradGrad;
}

void* Create_MaxPoolGradGrad(std::string id, Json::Value pInputItem) {
	MaxPoolGradGrad* pMaxPoolGradGrad = nullptr;
	Scope* pScope = nullptr;
	Output* orig_input = nullptr;
	Output* orig_output = nullptr;
	Output* grad = nullptr;
	std::vector<int> v_ksize;
	std::vector<int> v_strides;
	std::string padding;
	MaxPoolGradGrad::Attrs attrs;
	std::string strTemp = "";
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
				std::string msg = string_format("warning : MaxPoolGradGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "orig_input")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							orig_input = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : MaxPoolGradGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "orig_output")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							orig_output = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : MaxPoolGradGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : MaxPoolGradGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "ksize")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_ksize);
			}
			else
			{
				std::string msg = string_format("warning : MaxPoolGradGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : MaxPoolGradGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : MaxPoolGradGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "MaxPoolGradGrad::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("data_format_") != "")
				{
					strTemp = attrParser.GetAttribute("data_format_");
					attrs = attrs.DataFormat(strTemp);
				}
			}
			else
			{
				std::string msg = string_format("warning : MaxPoolGradGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : MaxPoolGradGrad - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&orig_input&&orig_output&&grad)
	{
		gtl::ArraySlice<int> strides(v_strides);
		gtl::ArraySlice<int> ksize(v_ksize);
		pMaxPoolGradGrad = new MaxPoolGradGrad(*pScope, *orig_input, *orig_output, *grad, ksize, strides, padding, attrs);
		ObjectInfo* pObj = AddObjectMap(pMaxPoolGradGrad, id, SYMBOL_MAXPOOLGRADGRAD, "MaxPoolGradGrad", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pMaxPoolGradGrad->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//ksize.clear();
		//strides.clear();
	}
	else
	{
		std::string msg = string_format("error : MaxPoolGradGrad(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_ksize.clear();
	v_strides.clear();
	return pMaxPoolGradGrad;
}

void* Create_MaxPoolGradGradWithArgmax(std::string id, Json::Value pInputItem) {
	MaxPoolGradGradWithArgmax* pMaxPoolGradGradWithArgmax = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* grad = nullptr;
	Output* argmax = nullptr;
	std::vector<int> v_ksize;
	std::vector<int> v_strides;
	std::string padding;

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
				std::string msg = string_format("warning : MaxPoolGradGradWithArgmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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

			}
			else
			{
				std::string msg = string_format("warning : MaxPoolGradGradWithArgmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							grad = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : MaxPoolGradGradWithArgmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "argmax")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							argmax = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : MaxPoolGradGradWithArgmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "ksize")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_ksize);
			}
			else
			{
				std::string msg = string_format("warning : MaxPoolGradGradWithArgmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : MaxPoolGradGradWithArgmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : MaxPoolGradGradWithArgmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : MaxPoolGradGradWithArgmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&pinput&&argmax&&grad)
	{
		gtl::ArraySlice<int> strides(v_strides);
		gtl::ArraySlice<int> ksize(v_ksize);
		pMaxPoolGradGradWithArgmax = new MaxPoolGradGradWithArgmax(*pScope, *pinput, *grad, *argmax, ksize, strides, padding);
		ObjectInfo* pObj = AddObjectMap(pMaxPoolGradGradWithArgmax, id, SYMBOL_MAXPOOLGRADGRADWITHARGMAX, "MaxPoolGradGradWithArgmax", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pMaxPoolGradGradWithArgmax->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		//ksize.clear();
		//strides.clear();
	}
	else
	{
		std::string msg = string_format("error : MaxPoolGradGradWithArgmax(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_ksize.clear();
	v_strides.clear();
	return pMaxPoolGradGradWithArgmax;
}

void* Create_MaxPoolWithArgmax(std::string id, Json::Value pInputItem) {
	MaxPoolWithArgmax* pMaxPoolWithArgmax = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	std::vector<int> v_ksize;
	std::vector<int> v_strides;
	std::string padding;
	MaxPoolWithArgmax::Attrs attrs;

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
				std::string msg = string_format("warning : MaxPoolWithArgmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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

			}
			else
			{
				std::string msg = string_format("warning : MaxPoolWithArgmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "ksize")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_ksize);
			}
			else
			{
				std::string msg = string_format("warning : MaxPoolWithArgmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : MaxPoolWithArgmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : MaxPoolWithArgmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "MaxPoolWithArgmax::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("Targmax_") != "")
					attrs =attrs.Targmax(attrParser.GetValue_DataType("Targmax_"));

			}
		}
		else
		{
			std::string msg = string_format("warning : MaxPoolWithArgmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&pinput)
	{
		gtl::ArraySlice<int> strides(v_strides);
		gtl::ArraySlice<int> ksize(v_ksize);
		pMaxPoolWithArgmax = new MaxPoolWithArgmax(*pScope, *pinput, ksize, strides, padding,attrs);
		ObjectInfo* pObj = AddObjectMap(pMaxPoolWithArgmax, id, SYMBOL_MAXPOOLWITHARGMAX, "pMaxPoolWithArgmax", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pMaxPoolWithArgmax->output, OUTPUT_TYPE_OUTPUT, "output");
			AddOutputInfo(pObj, &pMaxPoolWithArgmax->argmax, OUTPUT_TYPE_OUTPUT, "argmax");
		}
		//ksize.clear();
		//strides.clear();
	}
	else
	{
		std::string msg = string_format("error : pMaxPoolWithArgmax(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_ksize.clear();
	v_strides.clear();

	return pMaxPoolWithArgmax;
}

void* Create_QuantizedAvgPool(std::string id, Json::Value pInputItem) {
	QuantizedAvgPool* pQuantizedAvgPool = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* min_input = nullptr;
	Output* max_input = nullptr;
	std::vector<int> v_ksize;
	std::vector<int> v_strides;
	std::string padding;


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
				std::string msg = string_format("warning : QuantizedAvgPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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

			}
			else
			{
				std::string msg = string_format("warning : QuantizedAvgPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "min_input")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							min_input = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedAvgPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "max_input")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							max_input = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedAvgPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "ksize")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_ksize);
			}
			else
			{
				std::string msg = string_format("warning : QuantizedAvgPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : QuantizedAvgPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : QuantizedAvgPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : QuantizedAvgPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&pinput&&max_input&&min_input)
	{
		gtl::ArraySlice<int> strides(v_strides);
		gtl::ArraySlice<int> ksize(v_ksize);
		pQuantizedAvgPool = new QuantizedAvgPool(*pScope, *pinput,*min_input,*max_input, ksize, strides, padding);
		ObjectInfo* pObj = AddObjectMap(pQuantizedAvgPool, id, SYMBOL_QUANTIZEDAVGPOOL, "QuantizedAvgPool", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pQuantizedAvgPool->output, OUTPUT_TYPE_OUTPUT, "output");
			AddOutputInfo(pObj, &pQuantizedAvgPool->min_output, OUTPUT_TYPE_OUTPUT, "min_output");
			AddOutputInfo(pObj, &pQuantizedAvgPool->max_output, OUTPUT_TYPE_OUTPUT, "max_output");
		}
		//ksize.clear();
		//strides.clear();
	}
	else
	{
		std::string msg = string_format("error : QuantizedAvgPool(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_ksize.clear();
	v_strides.clear();
	return pQuantizedAvgPool;
}

void* Create_QuantizedBatchNormWithGlobalNormalization(std::string id, Json::Value pInputItem) {
	QuantizedBatchNormWithGlobalNormalization* pQuantizedBatchNormWithGlobalNormalization = nullptr;
	Scope* pScope = nullptr;
	Output* t = nullptr;
	Output* t_min = nullptr;
	Output* t_max = nullptr;
	Output* m = nullptr;
	Output* m_min = nullptr;
	Output* m_max = nullptr;
	Output* v = nullptr;
	Output* v_min = nullptr;
	Output* v_max = nullptr;
	Output* beta = nullptr;
	Output* beta_min = nullptr;
	Output* beta_max = nullptr;
	Output* gamma = nullptr;
	Output* gamma_min = nullptr;
	Output* gamma_max = nullptr;
	DataType out_type = DT_DOUBLE;
	float variance_epsilon = 0;
	bool scale_after_normalization = false;


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
				std::string msg = string_format("warning : QuantizedBatchNormWithGlobalNormalization - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "t")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							t = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedBatchNormWithGlobalNormalization - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "t_min")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							t_min = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedBatchNormWithGlobalNormalization - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "t_max")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							t_max = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedBatchNormWithGlobalNormalization - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							t = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedBatchNormWithGlobalNormalization - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "m_min")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							t_min = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedBatchNormWithGlobalNormalization - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "m_max")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							t_max = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedBatchNormWithGlobalNormalization - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
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
							t = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedBatchNormWithGlobalNormalization - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "v_min")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							t_min = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedBatchNormWithGlobalNormalization - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "v_max")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							t_max = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedBatchNormWithGlobalNormalization - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "beta")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							t = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedBatchNormWithGlobalNormalization - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "beta_min")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							t_min = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedBatchNormWithGlobalNormalization - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "beta_max")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							t_max = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedBatchNormWithGlobalNormalization - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "gamma")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							t = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedBatchNormWithGlobalNormalization - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "gamma_min")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							t_min = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedBatchNormWithGlobalNormalization - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "gamma_max")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							t_max = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedBatchNormWithGlobalNormalization - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "out_type")
		{
			if (strPinInterface == "DataType")
			{
				
				out_type = GetDatatypeFromInitial(strPinInitial);
			}
		}
		else if (strPinName == "variance_epsilon")
		{
			if (strPinInterface == "DataType")
			{

				variance_epsilon = stof(strPinInitial);
			}
		}
		else if (strPinName == "scale_after_normalization")
		{
			if (strPinInterface == "DataType")
			{

				if (strPinInitial =="true" || strPinInitial =="TRUE")
				{
					scale_after_normalization = true;
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : QuantizedBatchNormWithGlobalNormalization - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&t&&t_min&&t_max&&m&&m_min&&m_max&&v&&v_min&&v_max&&beta&&beta_min&&beta_max&&gamma&&gamma_min&&gamma_max)
	{

		pQuantizedBatchNormWithGlobalNormalization = new QuantizedBatchNormWithGlobalNormalization(
			*pScope, *t, *t_min, *t_max,
			*m, *m_min, *m_max,
			*v, *v_min, *v_max,
			*beta, *beta_min, *beta_max,
			*gamma, *gamma_min, *gamma_max,
			out_type, variance_epsilon, scale_after_normalization);
		ObjectInfo* pObj = AddObjectMap(pQuantizedBatchNormWithGlobalNormalization, id, SYMBOL_QUANTIZEDBATCHNORMWITHGLOBALNORMALIZATION,
			"QuantizedBatchNormWithGlobalNormalization", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pQuantizedBatchNormWithGlobalNormalization->result, OUTPUT_TYPE_OUTPUT, "result");
			AddOutputInfo(pObj, &pQuantizedBatchNormWithGlobalNormalization->result_min, OUTPUT_TYPE_OUTPUT, "result_min");
			AddOutputInfo(pObj, &pQuantizedBatchNormWithGlobalNormalization->result_max, OUTPUT_TYPE_OUTPUT, "result_max");
		}
	}
	else
	{
		std::string msg = string_format("error : QuantizedBatchNormWithGlobalNormalization(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}

	return pQuantizedBatchNormWithGlobalNormalization;
}

void* Create_QuantizedBiasAdd(std::string id, Json::Value pInputItem) {
	QuantizedBiasAdd* pQuantizedBiasAdd = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* bias = nullptr;
	Output* min_input = nullptr;
	Output* max_input = nullptr;
	Output* min_bias = nullptr;
	Output* max_bias = nullptr;
	DataType out_type = DT_DOUBLE;

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
				std::string msg = string_format("warning : QuantizedBiasAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pInput = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedBiasAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "bias")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							bias = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizedBiasAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "min_input")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							min_input = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizedBiasAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "max_input")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							max_input = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizedBiasAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "min_bias")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							min_bias = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizedBiasAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "max_bias")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							max_bias = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : QuantizedBiasAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "out_type")
		{
			if (strPinInterface == "DataType")
			{
				out_type = GetDatatypeFromInitial(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : QuantizedBiasAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : QuantizedBiasAdd - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope &&pInput&&bias&&min_input&&max_input&&min_bias&&max_bias)
	{
		pQuantizedBiasAdd = new QuantizedBiasAdd(*pScope, *pInput, *bias,*min_input,*max_input,*min_bias,*max_bias,out_type);
		ObjectInfo* pObj = AddObjectMap(pQuantizedBiasAdd, id, SYMBOL_QUANTIZEDBIASADD, "QuantizedBiasAdd", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pQuantizedBiasAdd->output, OUTPUT_TYPE_OUTPUT, "output");
			AddOutputInfo(pObj, &pQuantizedBiasAdd->min_out, OUTPUT_TYPE_OUTPUT, "min_out");
			AddOutputInfo(pObj, &pQuantizedBiasAdd->max_out, OUTPUT_TYPE_OUTPUT, "max_out");
		}
	}
	else
	{
		std::string msg = string_format("error : QuantizedBiasAdd(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pQuantizedBiasAdd;
}

void* Create_QuantizedConv2D(std::string id, Json::Value pInputItem) {
	QuantizedConv2D* pQuantizedConv2D = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* filter = nullptr;
	Output* min_input = nullptr;
	Output* max_input = nullptr;
	Output* min_filter = nullptr;
	Output* max_filter = nullptr;
	std::vector<int> v_strides;
	StringPiece padding;
	QuantizedConv2D::Attrs attrs;

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
				std::string msg = string_format("warning : QuantizedConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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

			}
			else
			{
				std::string msg = string_format("warning : QuantizedConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "filter")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							filter = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "min_input")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							min_input = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "max_input")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							max_input = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "min_filter")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							min_input = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "max_filter")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							max_input = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : QuantizedConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : QuantizedConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QuantizedConv2D::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("out_type_") != "")
					attrs = attrs.OutType(attrParser.GetValue_DataType("out_type_"));
			}
			else
			{
				std::string msg = string_format("warning : QuantizedConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : QuantizedConv2D - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&pinput&&max_input&&min_input&&filter&&min_filter&&max_filter)
	{
		gtl::ArraySlice<int> strides(v_strides);
		pQuantizedConv2D = new QuantizedConv2D(*pScope, *pinput,*filter, *min_input, *max_input,*min_filter,*max_filter, strides, padding,attrs);
		ObjectInfo* pObj = AddObjectMap(pQuantizedConv2D, id, SYMBOL_QUANTIZEDCONV2D, "QuantizedConv2D", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pQuantizedConv2D->output, OUTPUT_TYPE_OUTPUT, "output");
			AddOutputInfo(pObj, &pQuantizedConv2D->min_output, OUTPUT_TYPE_OUTPUT, "min_output");
			AddOutputInfo(pObj, &pQuantizedConv2D->max_output, OUTPUT_TYPE_OUTPUT, "max_output");
		}
		//strides.clear();
	}
	else
	{
		std::string msg = string_format("error : QuantizedConv2D(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	v_strides.clear();
	return pQuantizedConv2D;
}

void* Create_QuantizedMaxPool(std::string id, Json::Value pInputItem) {
	QuantizedMaxPool* pQuantizedMaxPool = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* min_input = nullptr;
	Output* max_input = nullptr;
	std::vector<int> v_ksize;
	std::vector<int> v_strides;
	StringPiece padding;


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
				std::string msg = string_format("warning : QuantizedMaxPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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

			}
			else
			{
				std::string msg = string_format("warning : QuantizedMaxPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "min_input")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							min_input = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedMaxPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "max_input")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							max_input = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedMaxPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "ksize")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_ksize);
			}
			else
			{
				std::string msg = string_format("warning : QuantizedMaxPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "strides")
		{
			if (strPinInterface == "gtl::ArraySlice<int>")
			{
				GetIntVectorFromInitial(strPinInitial, v_strides);
			}
			else
			{
				std::string msg = string_format("warning : QuantizedMaxPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "padding")
		{
			if (strPinInterface == "StringPiece")
			{
				padding = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : QuantizedMaxPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : QuantizedMaxPool - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&pinput&&max_input&&min_input)
	{
		gtl::ArraySlice<int> ksize(v_ksize);
		gtl::ArraySlice<int> strides(v_strides);
		pQuantizedMaxPool = new QuantizedMaxPool(*pScope, *pinput, *min_input, *max_input, ksize, strides, padding);
		ObjectInfo* pObj = AddObjectMap(pQuantizedMaxPool, id, SYMBOL_QUANTIZEDMAXPOOL, "QuantizedMaxPool", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pQuantizedMaxPool->output, OUTPUT_TYPE_OUTPUT, "output");
			AddOutputInfo(pObj, &pQuantizedMaxPool->min_output, OUTPUT_TYPE_OUTPUT, "min_output");
			AddOutputInfo(pObj, &pQuantizedMaxPool->max_output, OUTPUT_TYPE_OUTPUT, "max_output");
		}
		//ksize.clear();
		//strides.clear();
	}
	else
	{
		std::string msg = string_format("error : QuantizedMaxPool(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}

	v_ksize.clear();
	v_strides.clear();
	return pQuantizedMaxPool;
}

void* Create_QuantizedRelu(std::string id, Json::Value pInputItem) {
	QuantizedRelu* pQuantizedRelu = nullptr;
	Scope* pScope = nullptr;
	Output* features = nullptr;
	Output* min_features = nullptr;
	Output* max_features = nullptr;
	QuantizedRelu::Attrs attrs;

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
				std::string msg = string_format("warning : QuantizedRelu - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "features")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							features = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedRelu - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "min_features")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							min_features = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedRelu - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "max_features")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							max_features = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedRelu - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QuantizedRelu::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("out_type_") != "")
					attrs =attrs.OutType(attrParser.GetValue_DataType("out_type_"));
			}
			else
			{
				std::string msg = string_format("warning : QuantizedRelu - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : QuantizedRelu - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&features&&min_features&&max_features)
	{
	
		pQuantizedRelu = new QuantizedRelu(*pScope, *features, *min_features, *max_features, attrs);
		ObjectInfo* pObj = AddObjectMap(pQuantizedRelu, id, SYMBOL_QUANTIZEDRELU, "QuantizedRelu", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pQuantizedRelu->activations, OUTPUT_TYPE_OUTPUT, "activations");
			AddOutputInfo(pObj, &pQuantizedRelu->min_activations, OUTPUT_TYPE_OUTPUT, "min_activations");
			AddOutputInfo(pObj, &pQuantizedRelu->max_activations, OUTPUT_TYPE_OUTPUT, "max_activations");
		}
	}
	else
	{
		std::string msg = string_format("error : QuantizedRelu(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pQuantizedRelu;
}

void* Create_QuantizedRelu6(std::string id, Json::Value pInputItem) {
	QuantizedRelu6* pQuantizedRelu6 = nullptr;
	Scope* pScope = nullptr;
	Output* features = nullptr;
	Output* min_features = nullptr;
	Output* max_features = nullptr;
	QuantizedRelu6::Attrs attrs;

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
				std::string msg = string_format("warning : QuantizedRelu6 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "features")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							features = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedRelu6 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "min_features")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							min_features = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedRelu6 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "max_features")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							max_features = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedRelu6 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QuantizedRelu6::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("out_type_") != "")
					attrs =attrs.OutType(attrParser.GetValue_DataType("out_type_"));
			}
			else
			{
				std::string msg = string_format("warning : QuantizedRelu6 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : QuantizedRelu6 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&features&&min_features&&max_features)
	{

		pQuantizedRelu6 = new QuantizedRelu6(*pScope, *features, *min_features, *max_features, attrs);
		ObjectInfo* pObj = AddObjectMap(pQuantizedRelu6, id, SYMBOL_QUANTIZEDRELU6, "QuantizedRelu6", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pQuantizedRelu6->activations, OUTPUT_TYPE_OUTPUT, "activations");
			AddOutputInfo(pObj, &pQuantizedRelu6->min_activations, OUTPUT_TYPE_OUTPUT, "min_activations");
			AddOutputInfo(pObj, &pQuantizedRelu6->max_activations, OUTPUT_TYPE_OUTPUT, "max_activations");
		}
	}
	else
	{
		std::string msg = string_format("error : QuantizedRelu6(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pQuantizedRelu6;
}

void* Create_QuantizedReluX(std::string id, Json::Value pInputItem) {
	QuantizedReluX* pQuantizedReluX = nullptr;
	Scope* pScope = nullptr;
	Output* features = nullptr;
	Output* max_value = nullptr;
	Output* min_features = nullptr;
	Output* max_features = nullptr;
	QuantizedReluX::Attrs attrs;

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
				std::string msg = string_format("warning : QuantizedReluX - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "features")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							features = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedReluX - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "max_value")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							max_value = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedReluX - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "min_features")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							min_features = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedReluX - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "max_features")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							max_features = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : QuantizedReluX - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "QuantizedReluX::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("out_type_") != "")
					attrs = attrs.OutType(attrParser.GetValue_DataType("out_type_"));
			}
			else
			{
				std::string msg = string_format("warning : QuantizedReluX - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : QuantizedReluX - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&features&&max_value&&min_features&&max_features)
	{
		pQuantizedReluX = new QuantizedReluX(*pScope, *features, *max_value,*min_features, *max_features, attrs);
		ObjectInfo* pObj = AddObjectMap(pQuantizedReluX, id, SYMBOL_QUANTIZEDRELUX, "QuantizedReluX", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pQuantizedReluX->activations, OUTPUT_TYPE_OUTPUT, "activations");
			AddOutputInfo(pObj, &pQuantizedReluX->min_activations, OUTPUT_TYPE_OUTPUT, "min_activations");
			AddOutputInfo(pObj, &pQuantizedReluX->max_activations, OUTPUT_TYPE_OUTPUT, "max_activations");
		}
	}
	else
	{
		std::string msg = string_format("error : QuantizedReluX(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pQuantizedReluX;
}

void* Create_Relu(std::string id, Json::Value pInputItem) {
	Relu* pRelu = nullptr;
	Scope* pScope = nullptr;
	Output* features = nullptr;

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
				std::string msg = string_format("warning : Relu - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "features")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							features = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : Relu - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Relu - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&features)
	{
		pRelu = new Relu(*pScope, *features);
		ObjectInfo* pObj = AddObjectMap(pRelu, id, SYMBOL_RELU, "Relu", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pRelu->activations, OUTPUT_TYPE_OUTPUT, "activations");
		}
	}
	else
	{
		std::string msg = string_format("error : Relu(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRelu;
}

void* Create_Relu6(std::string id, Json::Value pInputItem) {
	Relu6* pRelu6 = nullptr;
	Scope* pScope = nullptr;
	Output* features = nullptr;

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
				std::string msg = string_format("warning : Relu6 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "features")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							features = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : Relu6 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Relu6 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope &&features)
	{
		pRelu6 = new Relu6(*pScope, *features);
		ObjectInfo* pObj = AddObjectMap(pRelu6, id, SYMBOL_RELU6, "Relu6", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pRelu6->activations, OUTPUT_TYPE_OUTPUT, "activations");
		}
	}
	else
	{
		std::string msg = string_format("error : Relu6(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRelu6;
}

void* Create_Softmax(std::string id, Json::Value pInputItem) {
	Softmax* pSoftmax = nullptr;
	Scope* pScope = nullptr;
	Output* plogits = nullptr;

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
				std::string msg = string_format("warning : Softmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "logits")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							plogits = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Softmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Softmax - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && plogits)
	{
		pSoftmax = new Softmax(*pScope, *plogits);
		ObjectInfo* pObj = AddObjectMap(pSoftmax, id, SYMBOL_SOFTMAX, "softmax", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSoftmax->softmax, OUTPUT_TYPE_OUTPUT, "softmax");
	}
	else
	{
		std::string msg = string_format("error : softmax(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSoftmax;
}

void* Create_SoftmaxCrossEntropyWithLogits(std::string id, Json::Value pInputItem) {
	SoftmaxCrossEntropyWithLogits* pSoftmaxCrossEntropyWithLogits = nullptr;
	Scope* pScope = nullptr;
	Output* features = nullptr;
	Output* labels = nullptr;

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
				std::string msg = string_format("warning : SoftmaxCrossEntropyWithLogits - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "features")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							features = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SoftmaxCrossEntropyWithLogits - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "labels")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							labels = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SoftmaxCrossEntropyWithLogits - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SoftmaxCrossEntropyWithLogits - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && features && labels)
	{
		pSoftmaxCrossEntropyWithLogits = new SoftmaxCrossEntropyWithLogits(*pScope, *features, *labels);
		ObjectInfo* pObj = AddObjectMap(pSoftmaxCrossEntropyWithLogits, id, SYMBOL_SOFTMAXCROSSENTROPYWITHLOGITS, "SoftmaxCrossEntropyWithLogits", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSoftmaxCrossEntropyWithLogits->backprop, OUTPUT_TYPE_OUTPUT, "backprop");
			AddOutputInfo(pObj, &pSoftmaxCrossEntropyWithLogits->loss, OUTPUT_TYPE_OUTPUT, "loss");
		}	
	}
	else
	{
		std::string msg = string_format("error : SoftmaxCrossEntropyWithLogits(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSoftmaxCrossEntropyWithLogits;
}

void* Create_Softplus(std::string id, Json::Value pInputItem) {
	Softplus* pSoftplus = nullptr;
	Scope* pScope = nullptr;
	Output* pfeatures = nullptr;

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
				std::string msg = string_format("warning : Softplus - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "features")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pfeatures = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : Softplus - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Softplus - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pfeatures)
	{
		pSoftplus = new Softplus(*pScope, *pfeatures);
		ObjectInfo* pObj = AddObjectMap(pSoftplus, id, SYMBOL_SOFTMAX, "Softplus", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSoftplus->activations, OUTPUT_TYPE_OUTPUT, "activations");
	}
	else
	{
		std::string msg = string_format("error : Softplus(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSoftplus;
}

void* Create_Softsign(std::string id, Json::Value pInputItem) {
	
	Softsign* pSoftsign = nullptr;
	Scope* pScope = nullptr;
	Output* pfeatures = nullptr;

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
				std::string msg = string_format("warning : Softsign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "features")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pfeatures = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : Softsign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Softsign - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pfeatures)
	{
		pSoftsign = new Softsign(*pScope, *pfeatures);
		ObjectInfo* pObj = AddObjectMap(pSoftsign, id, SYMBOL_SOFTSIGN, "Softsign", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSoftsign->activations, OUTPUT_TYPE_OUTPUT, "activations");
	}
	else
	{
		std::string msg = string_format("error : Softsign(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}

	return pSoftsign;
}

void* Create_SparseSoftmaxCrossEntropyWithLogits(std::string id, Json::Value pInputItem) {
	SparseSoftmaxCrossEntropyWithLogits* pSparseSoftmaxCrossEntropyWithLogits = nullptr;
	Scope* pScope = nullptr;
	Output* pfeatures = nullptr;
	Output* plabels = nullptr;

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
				std::string msg = string_format("warning : SparseSoftmaxCrossEntropyWithLogits - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "features")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pfeatures = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : SparseSoftmaxCrossEntropyWithLogits - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "labels")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							plabels = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SparseSoftmaxCrossEntropyWithLogits - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SparseSoftmaxCrossEntropyWithLogits - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	
	if (pScope && pfeatures && plabels)
	{
		pSparseSoftmaxCrossEntropyWithLogits = new SparseSoftmaxCrossEntropyWithLogits(*pScope, *pfeatures,*plabels);
		ObjectInfo* pObj = AddObjectMap(pSparseSoftmaxCrossEntropyWithLogits, id, SYMBOL_SOFTMAXCROSSENTROPYWITHLOGITS, "SparseSoftmaxCrossEntropyWithLogits", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSparseSoftmaxCrossEntropyWithLogits->loss, OUTPUT_TYPE_OUTPUT, "loss");
			AddOutputInfo(pObj, &pSparseSoftmaxCrossEntropyWithLogits->backprop, OUTPUT_TYPE_OUTPUT, "backprop");
		}
			
	}
	else
	{
		std::string msg = string_format("error : SparseSoftmaxCrossEntropyWithLogits(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSparseSoftmaxCrossEntropyWithLogits;
}

void* Create_TopK(std::string id, Json::Value pInputItem) {
	TopK* pTopK = nullptr;
	Scope* pScope = nullptr;
	Output* pinput = nullptr;
	Output* pk = nullptr;
	TopK::Attrs attrs;
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
				std::string msg = string_format("warning : TopK - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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

			}
			else
			{
				std::string msg = string_format("warning : TopK - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "k")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pk = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TopK - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "TopK::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("Sorted_") !="")
				{
					attrs = attrs.Sorted(attrParser.GetValue_bool("Sorted_"));
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : TopK - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput && pk)
	{
		pTopK = new TopK(*pScope, *pinput, *pk);
		ObjectInfo* pObj = AddObjectMap(pTopK, id, SYMBOL_TOPK, "TopK", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pTopK->values, OUTPUT_TYPE_OUTPUT, "values");
			AddOutputInfo(pObj, &pTopK->indices, OUTPUT_TYPE_OUTPUT, "indices");
		}

	}
	else
	{
		std::string msg = string_format("error : TopK(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pTopK;
}

