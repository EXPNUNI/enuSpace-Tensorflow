#include "stdafx.h"
#include "tf_image_ops.h"

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

void* Create_AdjustContrast(std::string id, Json::Value pInputItem) {
	AdjustContrast* pAdjustContrast = nullptr;
	Scope* pScope = nullptr;
	Output *pimages = nullptr;
	Output *pcontrast_factor = nullptr;
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
				std::string msg = string_format("warning : AdjustContrast - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "images")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pimages = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AdjustContrast - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "contrast_factor")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pcontrast_factor = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AdjustContrast - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : AdjustContrast pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pimages && pcontrast_factor)
	{
		pAdjustContrast = new AdjustContrast(*pScope, *pimages, *pcontrast_factor);
		ObjectInfo* pObj = AddObjectMap(pAdjustContrast, id, SYMBOL_ADJUSTCONTRAST, "AdjustContrast", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAdjustContrast->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : AdjustContrast(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAdjustContrast;
}

void* Create_AdjustHue(std::string id, Json::Value pInputItem) {
	AdjustHue* pAdjustHue = nullptr;
	Scope* pScope = nullptr;
	Output *pimages = nullptr;
	Output *pdelta = nullptr;
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
				std::string msg = string_format("warning : AdjustHue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "images")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pimages = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AdjustHue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pdelta = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AdjustHue - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : AdjustHue pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pimages && pdelta)
	{
		pAdjustHue = new AdjustHue(*pScope, *pimages, *pdelta);
		ObjectInfo* pObj = AddObjectMap(pAdjustHue, id, SYMBOL_ADJUSTHUE, "AdjustHue", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAdjustHue->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : AdjustHue(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAdjustHue;
}

void* Create_AdjustSaturation(std::string id, Json::Value pInputItem) {
	AdjustSaturation* pAdjustSaturation = nullptr;
	Scope* pScope = nullptr;
	Output *pimages = nullptr;
	Output *pscale = nullptr;
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
				std::string msg = string_format("warning : AdjustSaturation - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "images")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pimages = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AdjustSaturation - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
							pscale = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AdjustSaturation - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : AdjustSaturation pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pimages && pscale)
	{
		pAdjustSaturation = new AdjustSaturation(*pScope, *pimages, *pscale);
		ObjectInfo* pObj = AddObjectMap(pAdjustSaturation, id, SYMBOL_ADJUSTSATURATION, "AdjustSaturation", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAdjustSaturation->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : AdjustSaturation(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAdjustSaturation;
}

void* Create_CropAndResize(std::string id, Json::Value pInputItem) {
	CropAndResize* pCropAndResize = nullptr;
	Scope* pScope = nullptr;
	Output *pimages = nullptr;
	Output *pboxes = nullptr;
	Output *pbox_ind = nullptr;
	Output *pcrop_size = nullptr;
	CropAndResize::Attrs attrs;
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
				std::string msg = string_format("warning : CropAndResize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "images")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pimages = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : CropAndResize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "boxes")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pboxes = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : CropAndResize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "box_ind")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pbox_ind = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : CropAndResize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "crop_size")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pcrop_size = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : CropAndResize - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "CropAndResize::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("method_") != "") attrs.Method(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("method_")));
				if (attrParser.GetAttribute("extrapolation_value_") != "") attrs.ExtrapolationValue(attrParser.ConvStrToFloat(attrParser.GetAttribute("extrapolation_value_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : CropAndResize pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pimages && pboxes && pbox_ind && pcrop_size)
	{
		pCropAndResize = new CropAndResize(*pScope, *pimages, *pboxes, *pbox_ind, *pcrop_size,attrs);
		ObjectInfo* pObj = AddObjectMap(pCropAndResize, id, SYMBOL_CROPANDRESIZE, "CropAndResize", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pCropAndResize->crops, OUTPUT_TYPE_OUTPUT, "crops");
		}
	}
	else
	{
		std::string msg = string_format("error : CropAndResize(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pCropAndResize;
}

void* Create_CropAndResizeGradBoxes(std::string id, Json::Value pInputItem) {
	CropAndResizeGradBoxes* pCropAndResizeGradBoxes = nullptr;
	Scope* pScope = nullptr;
	Output *pgrads = nullptr;
	Output *pimages = nullptr;
	Output *pboxes = nullptr;
	Output *pbox_ind = nullptr;
	CropAndResizeGradBoxes::Attrs attrs;
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
				std::string msg = string_format("warning : CropAndResizeGradBoxes - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "grads")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pgrads = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : CropAndResizeGradBoxes - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "images")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pimages = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : CropAndResizeGradBoxes - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "boxes")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pboxes = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : CropAndResizeGradBoxes - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "box_ind")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pbox_ind = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : CropAndResizeGradBoxes - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "CropAndResizeGradBoxes::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("method_") != "") attrs.Method(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("method_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : CropAndResizeGradBoxes pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pimages && pboxes && pgrads && pbox_ind)
	{
		pCropAndResizeGradBoxes = new CropAndResizeGradBoxes(*pScope, *pgrads, *pimages, *pboxes, *pbox_ind, attrs);
		ObjectInfo* pObj = AddObjectMap(pCropAndResizeGradBoxes, id, SYMBOL_CROPANDRESIZEGRADBOXES, "CropAndResizeGradBoxes", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pCropAndResizeGradBoxes->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : CropAndResizeGradBoxes(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pCropAndResizeGradBoxes;
}

void* Create_CropAndResizeGradImage(std::string id, Json::Value pInputItem) {
	CropAndResizeGradImage* pCropAndResizeGradImage = nullptr;
	Scope* pScope = nullptr;
	Output *pgrads = nullptr;
	Output *pboxes = nullptr;
	Output *pbox_ind = nullptr;
	Output *pimage_size = nullptr;
	DataType T;
	CropAndResizeGradImage::Attrs attrs;
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
				std::string msg = string_format("warning : CropAndResizeGradImage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "grads")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pgrads = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : CropAndResizeGradImage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "boxes")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pboxes = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : CropAndResizeGradImage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "box_ind")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pbox_ind = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : CropAndResizeGradImage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "image_size")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pimage_size = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : CropAndResizeGradImage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "T")
		{
			if (strPinInterface == "DataType")
			{
				if (!(T = GetDatatypeFromInitial(strPinInitial)))
				{
					std::string msg = string_format("warning : CropAndResizeGradImage - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : CropAndResizeGradImage - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "CropAndResizeGradImage::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("method_") != "") attrs.Method(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("method_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : CropAndResizeGradImage pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pgrads && pboxes && pbox_ind && pimage_size)
	{
		pCropAndResizeGradImage = new CropAndResizeGradImage(*pScope, *pgrads, *pboxes, *pbox_ind, *pimage_size, T, attrs);
		ObjectInfo* pObj = AddObjectMap(pCropAndResizeGradImage, id, SYMBOL_CROPANDRESIZEGRADIMAGE, "CropAndResizeGradImage", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pCropAndResizeGradImage->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : CropAndResizeGradImage(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pCropAndResizeGradImage;
}

void* Create_DecodeGif(std::string id, Json::Value pInputItem) {
	DecodeGif* pDecodeGif = nullptr;
	Scope* pScope = nullptr;
	Output *pcontents = nullptr;
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
				std::string msg = string_format("warning : DecodeGif - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "contents")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pcontents = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DecodeGif - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : DecodeGif pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pcontents)
	{
		pDecodeGif = new DecodeGif(*pScope, *pcontents);
		ObjectInfo* pObj = AddObjectMap(pDecodeGif, id, SYMBOL_DECODEGIF, "DecodeGif", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pDecodeGif->image, OUTPUT_TYPE_OUTPUT, "image");
		}
	}
	else
	{
		std::string msg = string_format("error : DecodeGif(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pDecodeGif;
}

void* Create_DecodeJpeg(std::string id, Json::Value pInputItem) {
	DecodeJpeg* pDecodeJpeg = nullptr;
	Scope* pScope = nullptr;
	Output *pcontents = nullptr;
	DecodeJpeg::Attrs attrs;
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
				std::string msg = string_format("warning : DecodeJpeg - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "contents")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pcontents = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DecodeJpeg - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "DecodeJpeg::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("channels_") != "") attrs.Channels(attrParser.ConvStrToInt64(attrParser.GetAttribute("channels_")));
				if (attrParser.GetAttribute("ratio_") != "") attrs.Channels(attrParser.ConvStrToInt64(attrParser.GetAttribute("ratio_")));
				if (attrParser.GetAttribute("fancy_upscaling_") != "") attrs.FancyUpscaling(attrParser.ConvStrToBool(attrParser.GetAttribute("fancy_upscaling_")));
				if (attrParser.GetAttribute("try_recover_truncated_") != "") attrs.TryRecoverTruncated(attrParser.ConvStrToBool(attrParser.GetAttribute("try_recover_truncated_")));
				if (attrParser.GetAttribute("acceptable_fraction_") != "") attrs.AcceptableFraction(attrParser.ConvStrToFloat(attrParser.GetAttribute("acceptable_fraction_")));
				if (attrParser.GetAttribute("dct_method_") != "") attrs.DctMethod(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("dct_method_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : DecodeJpeg pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pcontents)
	{
		pDecodeJpeg = new DecodeJpeg(*pScope, *pcontents, attrs);
		ObjectInfo* pObj = AddObjectMap(pDecodeJpeg, id, SYMBOL_DECODEJPEG, "DecodeJpeg", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pDecodeJpeg->image, OUTPUT_TYPE_OUTPUT, "image");
		}
	}
	else
	{
		std::string msg = string_format("error : DecodeJpeg(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pDecodeJpeg;
}

void* Create_DecodePng(std::string id, Json::Value pInputItem) {
	DecodePng* pDecodePng = nullptr;
	Scope* pScope = nullptr;
	Output *pcontents = nullptr;
	DecodePng::Attrs attrs;
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
				std::string msg = string_format("warning : DecodePng - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "contents")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pcontents = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DecodePng - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "DecodePng::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("channels_") != "") attrs.Channels(attrParser.ConvStrToInt64(attrParser.GetAttribute("channels_")));
				if (attrParser.GetAttribute("dtype_") != "") attrs.Channels(attrParser.ConvStrToDataType(attrParser.GetAttribute("dtype_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : DecodePng pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pcontents)
	{
		pDecodePng = new DecodePng(*pScope, *pcontents, attrs);
		ObjectInfo* pObj = AddObjectMap(pDecodePng, id, SYMBOL_DECODEPNG, "DecodePng", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pDecodePng->image, OUTPUT_TYPE_OUTPUT, "image");
		}
	}
	else
	{
		std::string msg = string_format("error : DecodeJpeg(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pDecodePng;
}

void* Create_DrawBoundingBoxes(std::string id, Json::Value pInputItem) {
	DrawBoundingBoxes* pDrawBoundingBoxes = nullptr;
	Scope* pScope = nullptr;
	Output *pimages = nullptr;
	Output *pboxes = nullptr;
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
				std::string msg = string_format("warning : DrawBoundingBoxes - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "images")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pimages = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DrawBoundingBoxes - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "boxes")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pboxes = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : DrawBoundingBoxes - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : DrawBoundingBoxes pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pimages && pboxes)
	{
		pDrawBoundingBoxes = new DrawBoundingBoxes(*pScope, *pimages, *pboxes);
		ObjectInfo* pObj = AddObjectMap(pDrawBoundingBoxes, id, SYMBOL_DRAWBOUNDINGBOXES, "DrawBoundingBoxes", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pDrawBoundingBoxes->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : DrawBoundingBoxes(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pDrawBoundingBoxes;
}

void* Create_EncodeJpeg(std::string id, Json::Value pInputItem) {
	EncodeJpeg* pEncodeJpeg = nullptr;
	Scope* pScope = nullptr;
	Output *pimage = nullptr;
	EncodeJpeg::Attrs attrs;
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
				std::string msg = string_format("warning : EncodeJpeg - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "image")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pimage = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : EncodeJpeg - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "EncodeJpeg::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("format_") != "") attrs.Format(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("format_")));
				if (attrParser.GetAttribute("quality_") != "") attrs.Quality(attrParser.ConvStrToInt64(attrParser.GetAttribute("quality_")));
				if (attrParser.GetAttribute("progressive_") != "") attrs.Progressive(attrParser.ConvStrToBool(attrParser.GetAttribute("progressive_")));
				if (attrParser.GetAttribute("optimize_size_") != "") attrs.OptimizeSize(attrParser.ConvStrToBool(attrParser.GetAttribute("optimize_size_")));
				if (attrParser.GetAttribute("chroma_downsampling_") != "") attrs.ChromaDownsampling(attrParser.ConvStrToBool(attrParser.GetAttribute("chroma_downsampling_")));
				if (attrParser.GetAttribute("density_unit_") != "") attrs.DensityUnit(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("density_unit_")));
				if (attrParser.GetAttribute("x_density_") != "") attrs.XDensity(attrParser.ConvStrToInt64(attrParser.GetAttribute("x_density_")));
				if (attrParser.GetAttribute("y_density_") != "") attrs.YDensity(attrParser.ConvStrToInt64(attrParser.GetAttribute("y_density_")));
				if (attrParser.GetAttribute("xmp_metadata_") != "") attrs.XmpMetadata(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("xmp_metadata_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : EncodeJpeg pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pimage)
	{
		pEncodeJpeg = new EncodeJpeg(*pScope, *pimage, attrs);
		ObjectInfo* pObj = AddObjectMap(pEncodeJpeg, id, SYMBOL_ENCODEJPEG, "EncodeJpeg", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pEncodeJpeg->contents, OUTPUT_TYPE_OUTPUT, "contents");
		}
	}
	else
	{
		std::string msg = string_format("error : EncodeJpeg(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pEncodeJpeg;
}

void* Create_EncodePng(std::string id, Json::Value pInputItem) {
	EncodePng* pEncodePng = nullptr;
	Scope* pScope = nullptr;
	Output *pimage = nullptr;
	EncodePng::Attrs attrs;
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
				std::string msg = string_format("warning : EncodePng - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "image")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pimage = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : EncodePng - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "EncodePng::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("compression_") != "") attrs.Compression(attrParser.ConvStrToInt64(attrParser.GetAttribute("compression_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : EncodePng pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pimage)
	{
		pEncodePng = new EncodePng(*pScope, *pimage, attrs);
		ObjectInfo* pObj = AddObjectMap(pEncodePng, id, SYMBOL_ENCODEPNG, "EncodePng", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pEncodePng->contents, OUTPUT_TYPE_OUTPUT, "contents");
		}
	}
	else
	{
		std::string msg = string_format("error : EncodePng(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pEncodePng;
}

void* Create_ExtractGlimpse(std::string id, Json::Value pInputItem) {
	ExtractGlimpse* pExtractGlimpse = nullptr;
	Scope* pScope = nullptr;
	Output *pinput = nullptr;
	Output *psize = nullptr;
	Output *poffsets = nullptr;
	ExtractGlimpse::Attrs attrs;
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
				std::string msg = string_format("warning : ExtractGlimpse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : ExtractGlimpse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : ExtractGlimpse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "offsets")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							poffsets = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ExtractGlimpse - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ExtractGlimpse::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("centered_") != "") attrs.Centered(attrParser.ConvStrToBool(attrParser.GetAttribute("centered_")));
				if (attrParser.GetAttribute("normalized_") != "") attrs.Normalized(attrParser.ConvStrToBool(attrParser.GetAttribute("normalized_")));
				if (attrParser.GetAttribute("uniform_noise_") != "") attrs.UniformNoise(attrParser.ConvStrToBool(attrParser.GetAttribute("uniform_noise_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ExtractGlimpse pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pinput && psize && poffsets)
	{
		pExtractGlimpse = new ExtractGlimpse(*pScope, *pinput, *psize, *poffsets, attrs);
		ObjectInfo* pObj = AddObjectMap(pExtractGlimpse, id, SYMBOL_EXTRACTGLIMPSE, "ExtractGlimpse", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pExtractGlimpse->glimpse, OUTPUT_TYPE_OUTPUT, "glimpse");
		}
	}
	else
	{
		std::string msg = string_format("error : ExtractGlimpse(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pExtractGlimpse;
}

void* Create_HSVToRGB(std::string id, Json::Value pInputItem) {
	HSVToRGB* pHSVToRGB = nullptr;
	Scope* pScope = nullptr;
	Output *pimages = nullptr;
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
				std::string msg = string_format("warning : HSVToRGB - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "images")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pimages = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : HSVToRGB - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : HSVToRGB pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pimages)
	{
		pHSVToRGB = new HSVToRGB(*pScope, *pimages);
		ObjectInfo* pObj = AddObjectMap(pHSVToRGB, id, SYMBOL_HSVTORGB, "HSVToRGB", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pHSVToRGB->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : HSVToRGB(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pHSVToRGB;
}

void* Create_NonMaxSuppression(std::string id, Json::Value pInputItem) {
	NonMaxSuppression* pNonMaxSuppression = nullptr;
	Scope* pScope = nullptr;
	Output *pboxes = nullptr;
	Output *pscores = nullptr;
	Output *pmax_output_size = nullptr;
	NonMaxSuppression::Attrs attrs;
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
				std::string msg = string_format("warning : NonMaxSuppression - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "boxes")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pboxes = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : NonMaxSuppression - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "scores")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pscores = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : NonMaxSuppression - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "max_output_size")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pmax_output_size = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : NonMaxSuppression - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "NonMaxSuppression::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("timeout_ms_") != "") attrs.IouThreshold(attrParser.ConvStrToFloat(attrParser.GetAttribute("iou_threshold_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : NonMaxSuppression pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pboxes && pscores && pmax_output_size)
	{
		pNonMaxSuppression = new NonMaxSuppression(*pScope, *pboxes, *pscores, *pmax_output_size, attrs);
		ObjectInfo* pObj = AddObjectMap(pNonMaxSuppression, id, SYMBOL_NONMAXSUPPRESSION, "NonMaxSuppression", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pNonMaxSuppression->selected_indices, OUTPUT_TYPE_OUTPUT, "selected_indices");
		}
	}
	else
	{
		std::string msg = string_format("error : NonMaxSuppression(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pNonMaxSuppression;
}

void* Create_RGBToHSV(std::string id, Json::Value pInputItem) {
	RGBToHSV* pRGBToHSV = nullptr;
	Scope* pScope = nullptr;
	Output *pimages = nullptr;
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
				std::string msg = string_format("warning : RGBToHSV - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "images")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pimages = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : RGBToHSV - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : RGBToHSV pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pimages)
	{
		pRGBToHSV = new RGBToHSV(*pScope, *pimages);
		ObjectInfo* pObj = AddObjectMap(pRGBToHSV, id, SYMBOL_RGBTOHSV, "RGBToHSV", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pRGBToHSV->output, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : RGBToHSV(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRGBToHSV;
}

void* Create_ResizeArea(std::string id, Json::Value pInputItem) {
	ResizeArea* pResizeArea = nullptr;
	Scope* pScope = nullptr;
	Output *pimages = nullptr;
	Output *psize = nullptr;
	ResizeArea::Attrs attrs;
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
				std::string msg = string_format("warning : ResizeArea - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "images")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pimages = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResizeArea - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : ResizeArea - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResizeArea::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("align_corners_") != "") attrs.AlignCorners(attrParser.ConvStrToBool(attrParser.GetAttribute("align_corners_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResizeArea pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pimages && psize)
	{
		pResizeArea = new ResizeArea(*pScope, *pimages, *psize, attrs);
		ObjectInfo* pObj = AddObjectMap(pResizeArea, id, SYMBOL_RESIZEAREA, "ResizeArea", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pResizeArea->resized_images, OUTPUT_TYPE_OUTPUT, "resized_images");
		}
	}
	else
	{
		std::string msg = string_format("error : ResizeArea(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pResizeArea;
}

void* Create_ResizeBicubic(std::string id, Json::Value pInputItem) {
	ResizeBicubic* pResizeBicubic = nullptr;
	Scope* pScope = nullptr;
	Output *pimages = nullptr;
	Output *psize = nullptr;
	ResizeBicubic::Attrs attrs;
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
				std::string msg = string_format("warning : ResizeBicubic - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "images")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pimages = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResizeBicubic - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : ResizeBicubic - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResizeBicubic::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("align_corners_") != "") attrs.AlignCorners(attrParser.ConvStrToBool(attrParser.GetAttribute("align_corners_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResizeBicubic pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pimages && psize)
	{
		pResizeBicubic = new ResizeBicubic(*pScope, *pimages, *psize, attrs);
		ObjectInfo* pObj = AddObjectMap(pResizeBicubic, id, SYMBOL_RESIZEBICUBIC, "ResizeBicubic", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pResizeBicubic->resized_images, OUTPUT_TYPE_OUTPUT, "resized_images");
		}
	}
	else
	{
		std::string msg = string_format("error : ResizeBicubic(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pResizeBicubic;
}

void* Create_ResizeBilinear(std::string id, Json::Value pInputItem) {
	ResizeBilinear* pResizeBilinear = nullptr;
	Scope* pScope = nullptr;
	Output *pimages = nullptr;
	Output *psize = nullptr;
	ResizeBilinear::Attrs attrs;
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
				std::string msg = string_format("warning : ResizeBilinear - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "images")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pimages = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResizeBilinear - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : ResizeBilinear - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResizeBilinear::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("align_corners_") != "") attrs.AlignCorners(attrParser.ConvStrToBool(attrParser.GetAttribute("align_corners_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResizeBilinear pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pimages && psize)
	{
		pResizeBilinear = new ResizeBilinear(*pScope, *pimages, *psize, attrs);
		ObjectInfo* pObj = AddObjectMap(pResizeBilinear, id, SYMBOL_RESIZEBILINEAR, "ResizeBilinear", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pResizeBilinear->resized_images, OUTPUT_TYPE_OUTPUT, "resized_images");
		}
	}
	else
	{
		std::string msg = string_format("error : ResizeBilinear(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pResizeBilinear;
}

void* Create_ResizeNearestNeighbor(std::string id, Json::Value pInputItem) {
	ResizeNearestNeighbor* pResizeNearestNeighbor = nullptr;
	Scope* pScope = nullptr;
	Output *pimages = nullptr;
	Output *psize = nullptr;
	ResizeNearestNeighbor::Attrs attrs;
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
				std::string msg = string_format("warning : ResizeNearestNeighbor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "images")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pimages = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ResizeNearestNeighbor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
				std::string msg = string_format("warning : ResizeNearestNeighbor - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ResizeNearestNeighbor::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("align_corners_") != "") attrs.AlignCorners(attrParser.ConvStrToBool(attrParser.GetAttribute("align_corners_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : ResizeNearestNeighbor pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pimages && psize)
	{
		pResizeNearestNeighbor = new ResizeNearestNeighbor(*pScope, *pimages, *psize, attrs);
		ObjectInfo* pObj = AddObjectMap(pResizeNearestNeighbor, id, SYMBOL_RESIZENEARESTNEIGHBOR, "ResizeNearestNeighbor", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pResizeNearestNeighbor->resized_images, OUTPUT_TYPE_OUTPUT, "resized_images");
		}
	}
	else
	{
		std::string msg = string_format("error : ResizeNearestNeighbor(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pResizeNearestNeighbor;
}

void* Create_SampleDistortedBoundingBox(std::string id, Json::Value pInputItem) {
	SampleDistortedBoundingBox* pSampleDistortedBoundingBox = nullptr;
	Scope* pScope = nullptr;
	Output *pimage_size = nullptr;
	Output *pbounding_boxes = nullptr;
	SampleDistortedBoundingBox::Attrs attrs;
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
				std::string msg = string_format("warning : SampleDistortedBoundingBox - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "image_size")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pimage_size = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SampleDistortedBoundingBox - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "bounding_boxes")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pbounding_boxes = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SampleDistortedBoundingBox - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "SampleDistortedBoundingBox::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("seed_") != "") attrs.Seed(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed_")));
				if (attrParser.GetAttribute("seed2_") != "") attrs.Seed2(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed2_")));
				if (attrParser.GetAttribute("min_object_covered_") != "") attrs.MinObjectCovered(attrParser.ConvStrToFloat(attrParser.GetAttribute("min_object_covered_")));
				if (attrParser.GetAttribute("aspect_ratio_range_") != "") attrs.AspectRatioRange(attrParser.ConvStrToArraySlicefloat(attrParser.GetAttribute("aspect_ratio_range_")));
				if (attrParser.GetAttribute("area_range_") != "") attrs.AreaRange(attrParser.ConvStrToArraySlicefloat(attrParser.GetAttribute("area_range_")));
				if (attrParser.GetAttribute("max_attempts_") != "") attrs.MaxAttempts(attrParser.ConvStrToInt64(attrParser.GetAttribute("max_attempts_")));
				if (attrParser.GetAttribute("use_image_if_no_bounding_boxes_") != "") attrs.UseImageIfNoBoundingBoxes(attrParser.ConvStrToBool(attrParser.GetAttribute("use_image_if_no_bounding_boxes_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : SampleDistortedBoundingBox pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pimage_size && pbounding_boxes)
	{
		pSampleDistortedBoundingBox = new SampleDistortedBoundingBox(*pScope, *pimage_size, *pbounding_boxes, attrs);
		ObjectInfo* pObj = AddObjectMap(pSampleDistortedBoundingBox, id, SYMBOL_SAMPLEDISTORTEDBOUNDINGBOX, "SampleDistortedBoundingBox", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSampleDistortedBoundingBox->begin, OUTPUT_TYPE_OUTPUT, "begin");
			AddOutputInfo(pObj, &pSampleDistortedBoundingBox->size, OUTPUT_TYPE_OUTPUT, "size");
			AddOutputInfo(pObj, &pSampleDistortedBoundingBox->bboxes, OUTPUT_TYPE_OUTPUT, "bboxes");
		}
	}
	else
	{
		std::string msg = string_format("error : SampleDistortedBoundingBox(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSampleDistortedBoundingBox;
}
