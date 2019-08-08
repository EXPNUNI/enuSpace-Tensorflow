#include "stdafx.h"
#include "tf_random.h"

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

void* Create_Multinomial(std::string id, Json::Value pInputItem) {
	Multinomial* pMultinomial = nullptr;
	Scope* pScope = nullptr;
	Output* logit = nullptr;
	Output* num_samples = nullptr;
	Multinomial::Attrs attrs;

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
				std::string msg = string_format("warning : Multinomial - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							logit = (Output*)pOutputObj->pOutput;
							if (logit)
							{
								Input ptest(*logit);
								DataType dtype1 = ptest.data_type();
								if (dtype1 == DT_INT32)
								{
									std::string msg = string_format("warning : Multinomial - Logit Datatype misssMatch. double,float change.", id.c_str(), strPinName.c_str());
									PrintMessage(msg);
									logit = nullptr;
								}
							}
							
							

						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Multinomial - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_samples")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							num_samples = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : num_samples - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Multinomial::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("seed_") != "")
					attrs = attrs.Seed(attrParser.GetValue_int64("seed_"));
				if (attrParser.GetAttribute("seed2_") != "")
					attrs = attrs.Seed2(attrParser.GetValue_int64("seed2_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : Multinomial - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && logit && num_samples)
	{
		pMultinomial = new Multinomial(*pScope, *logit, *num_samples, attrs);
		ObjectInfo* pObj = AddObjectMap(pMultinomial, id, SYMBOL_MULTINOMIAL, "Multinomial", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pMultinomial->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : Multinomial(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pMultinomial;
}

void* Create_ParameterizedTruncatedNormal(std::string id, Json::Value pInputItem) {
	ParameterizedTruncatedNormal* pParameterizedTruncatedNormal = nullptr;
	Scope* pScope = nullptr;
	Output* shape = nullptr;
	Output* means = nullptr;
	Output* stdevs = nullptr;
	Output* minvals = nullptr;
	Output* maxvals = nullptr;
	ParameterizedTruncatedNormal::Attrs attrs;

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
				std::string msg = string_format("warning : ParameterizedTruncatedNormal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ParameterizedTruncatedNormal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "means")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							means = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ParameterizedTruncatedNormal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "stdevs")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							stdevs = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ParameterizedTruncatedNormal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "minvals")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							minvals = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ParameterizedTruncatedNormal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "maxvals")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							maxvals = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ParameterizedTruncatedNormal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ParameterizedTruncatedNormal::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("seed_") != "")
					attrs = attrs.Seed(attrParser.GetValue_int64("seed_"));
				if (attrParser.GetAttribute("seed2_") != "")
					attrs = attrs.Seed2(attrParser.GetValue_int64("seed2_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : ParameterizedTruncatedNormal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && shape && means&& stdevs && minvals && maxvals)
	{
		/*
		Input tempInput(*shape);
		Tensor orgtensor = tempInput.tensor();
		int idim = orgtensor.dims();
		TensorShape shape = TensorShape();
		shape.AddDim(idim);
		Tensor tensor(DT_INT32, shape);
		for (int i = 0; i < idim; i++)
		{
			int64 idim = orgtensor.dim_size(i);
			tensor.flat<int>()(i) = idim;
		}
		Input input(tensor);
		*/
		pParameterizedTruncatedNormal = new ParameterizedTruncatedNormal(*pScope, *shape, *means, *stdevs, *minvals, *maxvals, attrs);
		ObjectInfo* pObj = AddObjectMap(pParameterizedTruncatedNormal, id, SYMBOL_PARAMETERIZEDTRUNCATEDNORMAL, "ParameterizedTruncatedNormal", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pParameterizedTruncatedNormal->output, OUTPUT_TYPE_OUTPUT, "output");
		}
		
	}
	else
	{
		std::string msg = string_format("error : ParameterizedTruncatedNormal(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}


	return pParameterizedTruncatedNormal;
}

void* Create_RandomGamma(std::string id, Json::Value pInputItem) {
	RandomGamma* pRandomGamma = nullptr;
	Scope* pScope = nullptr;
	Output* shape = nullptr;
	Output* alpha = nullptr;
	RandomGamma::Attrs attrs;
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
				std::string msg = string_format("warning : RandomGamma - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : RandomGamma - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : RandomGamma - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "RandomGamma::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("seed_") != "")
					attrs = attrs.Seed(attrParser.GetValue_int64("seed_"));
				if (attrParser.GetAttribute("seed2_") != "")
					attrs = attrs.Seed2(attrParser.GetValue_int64("seed2_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : RandomGamma - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && shape && alpha)
	{
		/*
		Input tempInput(*shape);
		Tensor orgtensor = tempInput.tensor();
		int idim = orgtensor.dims();
		TensorShape shape = TensorShape();
		shape.AddDim(idim);
		Tensor tensor(DT_INT32, shape);
		for (int i = 0; i < idim; i++)
		{
			int64 idim = orgtensor.dim_size(i);
			tensor.flat<int>()(i) = idim;
		}
		Input input(tensor);
		*/
		pRandomGamma = new RandomGamma(*pScope, *shape, *alpha, attrs);
		ObjectInfo* pObj = AddObjectMap(pRandomGamma, id, SYMBOL_RANDOMGAMMA, "RandomGamma", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pRandomGamma->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : RandomGamma(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRandomGamma;
}

void* Create_RandomNormal(std::string id, Json::Value pInputItem) {
	RandomNormal* pRandomNormal = nullptr;
	Scope* pScope = nullptr;
	Output* pShape = nullptr;
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : Scope, 연결 핀 : Scope
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
					pShape = (Output*)pObj->pObject;		// SYMBOL_INPUT은 자체가 Input 임.
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
				dtype = GetDatatypeFromInitial(strPinInitial);
				if (dtype == DT_INVALID)
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
			if (strPinInterface == "RandomNormal::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("seed_") != "")
					attrs = attrs.Seed(attrParser.GetValue_int64("seed_"));
				if (attrParser.GetAttribute("seed2_") != "")
					attrs = attrs.Seed2(attrParser.GetValue_int64("seed2_"));
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
		/*
		Tensor orgtensor = pShape->tensor();
		int idim = orgtensor.dims();
		TensorShape shape = TensorShape();
		shape.AddDim(idim);
		Tensor tensor(DT_INT32, shape);
		for (int i = 0; i < idim; i++)
		{
			int64 idim = orgtensor.dim_size(i);
			tensor.flat<int>()(i) = idim;
		}
		Input input(tensor);
		*/
		pRandomNormal = new RandomNormal(*pScope, *pShape, dtype, attrs);
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

//void* Create_RandomPoisson(std::string id, Json::Value pInputItem) {
//	RandomPoisson* pRandomPoisson = nullptr;
//	Scope* pScope = nullptr;
//	Output* shape = nullptr;
//	Output* rate = nullptr;
//	RandomPoisson::Attrs attrs;
//	int iSize = (int)pInputItem.size();
//	for (int subindex = 0; subindex < iSize; ++subindex)
//	{
//		Json::Value ItemValue = pInputItem[subindex];
//
//		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
//		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
//		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
//		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
//		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
//		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
//		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
//		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
//		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]
//
//		if (strPinName == "scope")
//		{
//			// 입력심볼 : #Scope, 입력심볼의 핀 : Scope, 연결 핀 : Scope
//			if (strPinInterface == "Scope")
//			{
//				pScope = m_pScope;
//			}
//			else
//			{
//				std::string msg = string_format("warning : RandomPoisson - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
//				PrintMessage(msg);
//			}
//		}
//		else if (strPinName == "shape")
//		{
//			if (strPinInterface == "Input")
//			{
//				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
//				if (pObj)
//				{
//					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
//					if (pOutputObj)
//					{
//						if (pOutputObj->pOutput)
//						{
//							shape = (Output*)pOutputObj->pOutput;
//						}
//					}
//				}
//			}
//			else
//			{
//				std::string msg = string_format("warning : RandomPoisson - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
//				PrintMessage(msg);
//			}
//		}
//		else if (strPinName == "rate")
//		{
//			if (strPinInterface == "Input")
//			{
//				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
//				if (pObj)
//				{
//					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
//					if (pOutputObj)
//					{
//						if (pOutputObj->pOutput)
//						{
//							rate = (Output*)pOutputObj->pOutput;
//						}
//					}
//				}
//			}
//			else
//			{
//				std::string msg = string_format("warning : RandomPoisson - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
//				PrintMessage(msg);
//			}
//		}
//		else if (strPinName == "attrs")
//		{
//			if (strPinInterface == "RandomPoisson::Attrs")
//			{
//				CAttributeParser attrParser(strPinInterface, strPinInitial);
//				if (attrParser.GetAttribute("seed_") != "")
//					attrs = attrs.Seed(attrParser.GetValue_int64("seed_"));
//				if (attrParser.GetAttribute("seed2_") != "")
//					attrs = attrs.Seed2(attrParser.GetValue_int64("seed2_"));
//			}
//		}
//		else
//		{
//			std::string msg = string_format("warning : RandomPoisson - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
//			PrintMessage(msg);
//		}
//	}
//	if (pScope && shape && rate)
//	{
//		/*
//		Input tempInput(*shape);
//		Tensor orgtensor = tempInput.tensor();
//		int idim = orgtensor.dims();
//		TensorShape shape = TensorShape();
//		shape.AddDim(idim);
//		Tensor tensor(DT_INT32, shape);
//		for (int i = 0; i < idim; i++)
//		{
//			int64 idim = orgtensor.dim_size(i);
//			tensor.flat<int>()(i) = idim;
//		}
//		Input input(tensor);
//		*/
//		pRandomPoisson = new RandomPoisson(*pScope, *shape, *rate, attrs);
//		ObjectInfo* pObj = AddObjectMap(pRandomPoisson, id, SYMBOL_RANDOMPOISSON, "RandomPoisson", pInputItem);
//		if (pObj)
//		{
//			AddOutputInfo(pObj, &pRandomPoisson->output, OUTPUT_TYPE_OUTPUT, "output");
//		}
//	}
//	else
//	{
//		std::string msg = string_format("error : RandomPoisson(%s) Object create failed.", id.c_str());
//		PrintMessage(msg);
//	}
//	return pRandomPoisson;
//}

void* Create_RandomShuffle(std::string id, Json::Value pInputItem) {
	RandomShuffle* pRandomShuffle = nullptr;
	Scope* pScope = nullptr;
	Output* value = nullptr;
	RandomShuffle::Attrs attrs;
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
				std::string msg = string_format("warning : RandomShuffle - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : RandomShuffle - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "RandomShuffle::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("seed_") != "")
					attrs = attrs.Seed(attrParser.GetValue_int64("seed_"));
				if (attrParser.GetAttribute("seed2_") != "")
					attrs = attrs.Seed2(attrParser.GetValue_int64("seed2_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : RandomShuffle - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && value)
	{

		*pRandomShuffle = RandomShuffle(*pScope, *value, attrs);
		ObjectInfo* pObj = AddObjectMap(pRandomShuffle, id, SYMBOL_RANDOMSHUFFLE, "RandomShuffle", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pRandomShuffle->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : RandomShuffle(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRandomShuffle;
}

void* Create_RandomUniform(std::string id, Json::Value pInputItem) {
	RandomUniform* pRandomUniform = nullptr;
	Scope* pScope = nullptr;
	Output* shape = nullptr;
	DataType dtype;
	RandomUniform::Attrs attrs;
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
				std::string msg = string_format("warning : RandomUniform - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : RandomUniform - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				dtype = GetDatatypeFromInitial(strPinInitial);
				if(dtype == DT_INVALID)
				{
					std::string msg = string_format("warning : RandomUniform - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : RandomUniform - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "RandomUniform::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("seed_") != "")
					attrs = attrs.Seed(attrParser.GetValue_int64("seed_"));
				if (attrParser.GetAttribute("seed2_") != "")
					attrs = attrs.Seed2(attrParser.GetValue_int64("seed2_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : RandomUniform - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && shape)
	{
		pRandomUniform = new RandomUniform(*pScope,*shape,dtype,attrs);
		ObjectInfo* pObj = AddObjectMap(pRandomUniform, id, SYMBOL_RANDOMUNIFORM, "RandomUniform", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pRandomUniform->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : RandomUniform(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRandomUniform;
}

void* Create_RandomUniformInt(std::string id, Json::Value pInputItem) {
	RandomUniformInt* pRandomUniformInt = nullptr;
	Scope* pScope = nullptr;
	Output* shape = nullptr;
	Output* minval = nullptr;
	Output* maxval = nullptr;
	RandomUniformInt::Attrs attrs;

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
				std::string msg = string_format("warning : RandomUniformInt - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : RandomUniformInt - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "minval")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							minval = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : RandomUniformInt - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "maxval")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							maxval = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : RandomUniformInt - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "RandomUniformInt::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("seed_") != "")
					attrs = attrs.Seed(attrParser.GetValue_int64("seed_"));
				if (attrParser.GetAttribute("seed2_") != "")
					attrs = attrs.Seed2(attrParser.GetValue_int64("seed2_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : RandomUniformInt - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && shape && minval && maxval)
	{
		
		pRandomUniformInt = new RandomUniformInt(*pScope, *shape, *minval, *maxval, attrs);
		ObjectInfo* pObj = AddObjectMap(pRandomUniformInt, id, SYMBOL_RANDOMUNIFORMINT, "RandomUniformInt", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pRandomUniformInt->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : RandomUniformInt(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRandomUniformInt;
}

void* Create_TruncatedNormal(std::string id, Json::Value pInputItem) {
	TruncatedNormal* pTruncatedNormal = nullptr;
	Scope* pScope = nullptr;
	Output* shape = nullptr;
	DataType dtype;
	TruncatedNormal::Attrs attrs;

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
				std::string msg = string_format("warning : TruncatedNormal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							shape = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : TruncatedNormal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				dtype = GetDatatypeFromInitial(strPinInitial);
				if(dtype == DT_INVALID)
				{
					std::string msg = string_format("warning : TruncatedNormal - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : TruncatedNormal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "TruncatedNormal::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("seed_") != "")
					attrs = attrs.Seed(attrParser.GetValue_int64("seed_"));
				if (attrParser.GetAttribute("seed2_") != "")
					attrs = attrs.Seed2(attrParser.GetValue_int64("seed2_"));
			}
		}
		else
		{
			std::string msg = string_format("warning : TruncatedNormal - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && shape )
	{

		pTruncatedNormal = new TruncatedNormal(*pScope, *shape, dtype, attrs);
		ObjectInfo* pObj = AddObjectMap(pTruncatedNormal, id, SYMBOL_TRUNCATEDNORMAL, "TruncatedNormal", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pTruncatedNormal->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : TruncatedNormal(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
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
			// 입력심볼 : #Scope, 입력심볼의 핀 : Scope, 연결 핀 : Scope
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
				dtype = GetDatatypeFromInitial(strPinInitial);
				if (dtype == DT_INVALID)
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
				if (attrParser.GetAttribute("seed_") != "")
					attrs = attrs.Seed(attrParser.GetValue_int64("seed_"));
				if (attrParser.GetAttribute("seed2_") != "")
					attrs = attrs.Seed2(attrParser.GetValue_int64("seed2_"));
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