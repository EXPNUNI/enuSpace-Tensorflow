#include "stdafx.h"
#include "tf_math.h"


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

void* Create_Abs(std::string id, Json::Value pInputItem) {
	Abs* pAbs = nullptr;
	Scope* pScope = nullptr;
	return pAbs;
}

void* Create_Acos(std::string id, Json::Value pInputItem) {
	Acos* pAcos = nullptr;
	Scope* pScope = nullptr;
	return pAcos;
}

void* Create_Add(std::string id, Json::Value pInputItem) {
	Scope* pScope = nullptr;
	Output* pX = nullptr;
	Output* pY = nullptr;
	Add* pAdd = nullptr;

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
				std::string msg = string_format("warning : Add - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, "output");
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pX = (Output*)pOutputObj->pOutput;
						}
					}
					// pX = pObj->pOutput;
				}
			}
			else
			{
				std::string msg = string_format("warning : Add - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "y")
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
							pY = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Add - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Add pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pX && pY)
	{
		pAdd = new Add(*pScope, *pX, *pY);
		ObjectInfo* pObj = AddObjectMap(pAdd, id, SYMBOL_ADD, "Add", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAdd->z, OUTPUT_TYPE_OUTPUT, "output");
		}
	}
	else
	{
		std::string msg = string_format("error : Add(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAdd;
}

void* Create_AddN(std::string id, Json::Value pInputItem) {
	AddN* pAddN = nullptr;
	Scope* pScope = nullptr;
	return pAddN;
}

void* Create_All(std::string id, Json::Value pInputItem) {
	All* pAll = nullptr;
	Scope* pScope = nullptr;
	return pAll;
}

void* Create_Any(std::string id, Json::Value pInputItem) {
	Any* pAny = nullptr;
	Scope* pScope = nullptr;
	return pAny;
}

void* Create_ApproximateEqual(std::string id, Json::Value pInputItem) {
	ApproximateEqual* pApproximateEqual = nullptr;
	Scope* pScope = nullptr;
	return pApproximateEqual;
}

void* Create_ArgMax(std::string id, Json::Value pInputItem) {
	ArgMax* pArgMax = nullptr;
	Scope* pScope = nullptr;
	return pArgMax;
}

void* Create_ArgMin(std::string id, Json::Value pInputItem) {
	ArgMin* pArgMin = nullptr;
	Scope* pScope = nullptr;
	return pArgMin;
}

void* Create_Asin(std::string id, Json::Value pInputItem) {
	Asin* pAsin = nullptr;
	Scope* pScope = nullptr;
	return pAsin;
}

void* Create_Atan(std::string id, Json::Value pInputItem) {
	Atan* pAtan = nullptr;
	Scope* pScope = nullptr;
	return pAtan;
}

void* Create_Atan2(std::string id, Json::Value pInputItem) {
	//Atan2* pAtan2 = nullptr;
	Scope* pScope = nullptr;
	return NULL;
}

void* Create_BatchMatMul(std::string id, Json::Value pInputItem) {
	BatchMatMul* pBatchMatMul = nullptr;
	Scope* pScope = nullptr;
	return pBatchMatMul;
}

void* Create_Betainc(std::string id, Json::Value pInputItem) {
	Betainc* pBetainc = nullptr;
	Scope* pScope = nullptr;
	return pBetainc;
}

void* Create_Bincount(std::string id, Json::Value pInputItem) {
	Bincount* pBincount = nullptr;
	Scope* pScope = nullptr;
	return pBincount;
}

void* Create_Bucketize(std::string id, Json::Value pInputItem) {
	//Bucketize* pBucketize = nullptr;
	Scope* pScope = nullptr;
	return NULL;
}

void* Create_Cast(std::string id, Json::Value pInputItem) {
	Cast* pCast = nullptr;
	Scope* pScope = nullptr;
	return pCast;
}

void* Create_Ceil(std::string id, Json::Value pInputItem) {
	Ceil* pCeil = nullptr;
	Scope* pScope = nullptr;
	return pCeil;
}

void* Create_Complex(std::string id, Json::Value pInputItem) {
	Complex* pComplex = nullptr;
	Scope* pScope = nullptr;
	return pComplex;
}

void* Create_ComplexAbs(std::string id, Json::Value pInputItem) {
	ComplexAbs* pComplexAbs = nullptr;
	Scope* pScope = nullptr;
	return pComplexAbs;
}

void* Create_Conj(std::string id, Json::Value pInputItem) {
	Conj* pConj = nullptr;
	Scope* pScope = nullptr;
	return pConj;
}

void* Create_Cos(std::string id, Json::Value pInputItem) {
	Cos* pCos = nullptr;
	Scope* pScope = nullptr;
	return pCos;
}

void* Create_Cross(std::string id, Json::Value pInputItem) {
	Cross* pCross = nullptr;
	Scope* pScope = nullptr;
	return pCross;
}

void* Create_Cumprod(std::string id, Json::Value pInputItem) {
	Cumprod* pCumprod = nullptr;
	Scope* pScope = nullptr;
	return pCumprod;
}

void* Create_Cumsum(std::string id, Json::Value pInputItem) {
	Cumsum* pCumsum = nullptr;
	Scope* pScope = nullptr;
	return pCumsum;
}

void* Create_Digamma(std::string id, Json::Value pInputItem) {
	Digamma* pDigamma = nullptr;
	Scope* pScope = nullptr;
	return pDigamma;
}

void* Create_Div(std::string id, Json::Value pInputItem) {
	Scope* pScope = nullptr;
	Output* pX = nullptr;
	Output* pY = nullptr;
	Div* pDiv = nullptr;

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
				std::string msg = string_format("warning : Div - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "x")
		{
			if (strPinInterface == "Input")
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
								pX = (Output*)pOutputObj->pOutput;
							}
						}
						// pX = pObj->pOutput;
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Div - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "y")
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
							pY = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Div - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Div pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pX && pY)
	{
		pDiv = new Div(*pScope, *pX, *pY);
		ObjectInfo* pObj = AddObjectMap(pDiv, id, SYMBOL_DIV, "Div", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pDiv->z, OUTPUT_TYPE_OUTPUT, "output");
		// pObj->pOutput = &pDiv->z;
	}
	else
	{
		std::string msg = string_format("error : Div(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pDiv;
}

void* Create_Equal(std::string id, Json::Value pInputItem) {
	Equal* pEqual = nullptr;
	Scope* pScope = nullptr;
	return pEqual;
}

void* Create_Erf(std::string id, Json::Value pInputItem) {
	Erf* pErf = nullptr;
	Scope* pScope = nullptr;
	return pErf;
}

void* Create_Erfc(std::string id, Json::Value pInputItem) {
	Erfc* pErfc = nullptr;
	Scope* pScope = nullptr;
	return pErfc;
}

void* Create_Exp(std::string id, Json::Value pInputItem) {
	Exp* pExp = nullptr;
	Scope* pScope = nullptr;
	return pExp;
}

void* Create_Expm1(std::string id, Json::Value pInputItem) {
	Expm1* pExpm1 = nullptr;
	Scope* pScope = nullptr;
	return pExpm1;
}

void* Create_Floor(std::string id, Json::Value pInputItem) {
	Floor* pFloor = nullptr;
	Scope* pScope = nullptr;
	return pFloor;
}

void* Create_FloorDiv(std::string id, Json::Value pInputItem) {
	FloorDiv* pFloorDiv = nullptr;
	Scope* pScope = nullptr;
	return pFloorDiv;
}

void* Create_FloorMod(std::string id, Json::Value pInputItem) {
	FloorMod* pFloorMod = nullptr;
	Scope* pScope = nullptr;
	return pFloorMod;
}

void* Create_Greater(std::string id, Json::Value pInputItem) {
	Greater* pGreater = nullptr;
	Scope* pScope = nullptr;
	return pGreater;
}

void* Create_GreaterEqual(std::string id, Json::Value pInputItem) {
	GreaterEqual* pGreaterEqual = nullptr;
	Scope* pScope = nullptr;
	return pGreaterEqual;
}

void* Create_Igamma(std::string id, Json::Value pInputItem) {
	Igamma* pIgamma = nullptr;
	Scope* pScope = nullptr;
	return pIgamma;
}

void* Create_Igammac(std::string id, Json::Value pInputItem) {
	Igammac* pIgammac = nullptr;
	Scope* pScope = nullptr;
	return pIgammac;
}

void* Create_Imag(std::string id, Json::Value pInputItem) {
	Imag* pImag = nullptr;
	Scope* pScope = nullptr;
	return pImag;
}

void* Create_IsInf(std::string id, Json::Value pInputItem) {
	IsInf* pIsInf = nullptr;
	Scope* pScope = nullptr;
	return pIsInf;
}

void* Create_IsNan(std::string id, Json::Value pInputItem) {
	IsNan* pIsNan = nullptr;
	Scope* pScope = nullptr;
	return pIsNan;
}

void* Create_Less(std::string id, Json::Value pInputItem) {
	Less* pLess = nullptr;
	Scope* pScope = nullptr;
	return pLess;
}

void* Create_LessEqual(std::string id, Json::Value pInputItem) {
	LessEqual* pLessEqual = nullptr;
	Scope* pScope = nullptr;
	return pLessEqual;
}

void* Create_Lgamma(std::string id, Json::Value pInputItem) {
	Lgamma* pLgamma = nullptr;
	Scope* pScope = nullptr;
	return pLgamma;
}

void* Create_LinSpace(std::string id, Json::Value pInputItem) {
	LinSpace* pLinSpace = nullptr;
	Scope* pScope = nullptr;
	return pLinSpace;
}

void* Create_Log(std::string id, Json::Value pInputItem) {
	Log* pLog = nullptr;
	Scope* pScope = nullptr;
	return pLog;
}

void* Create_Log1p(std::string id, Json::Value pInputItem) {
	Log1p* pLog1p = nullptr;
	Scope* pScope = nullptr;
	return pLog1p;
}

void* Create_LogicalAnd(std::string id, Json::Value pInputItem) {
	LogicalAnd* pLogicalAnd = nullptr;
	Scope* pScope = nullptr;
	return pLogicalAnd;
}

void* Create_LogicalNot(std::string id, Json::Value pInputItem) {
	LogicalNot* pLogicalNot = nullptr;
	Scope* pScope = nullptr;
	return pLogicalNot;
}

void* Create_LogicalOr(std::string id, Json::Value pInputItem) {
	LogicalOr* pLogicalOr = nullptr;
	Scope* pScope = nullptr;
	return pLogicalOr;
}

void* Create_MatMul(std::string id, Json::Value pInputItem) {
	Scope* pScope = nullptr;
	Output* pA = nullptr;
	Output* pB = nullptr;
	MatMul::Attrs attrs;
	MatMul* pMatMul = nullptr;

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
				std::string msg = string_format("warning : MatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "a")
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
							pA = (Output*)pOutputObj->pOutput;
						}
					}
				}

			}
			else
			{
				std::string msg = string_format("warning : MatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "b")
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
							pB = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : MatMul - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "tensorflow::ops::MatMul::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				attrs.TransposeA(attrParser.ConvStrToBool(attrParser.GetAttribute("TransposeA")));
				attrs.TransposeB(attrParser.ConvStrToBool(attrParser.GetAttribute("TransposeB")));
			}
		}
		else
		{
			std::string msg = string_format("warning : MatMul pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pA && pB)
	{
		pMatMul = new MatMul(*pScope, *pA, *pB, attrs);
		ObjectInfo* pObj = AddObjectMap(pMatMul, id, SYMBOL_MATMUL, "MatMul", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pMatMul->product, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("error : MatMul(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pMatMul;
}

void* Create_Max(std::string id, Json::Value pInputItem) {
	Max* pMax = nullptr;
	Scope* pScope = nullptr;
	return pMax;
}

void* Create_Maximum(std::string id, Json::Value pInputItem) {
	Maximum* pMaximum = nullptr;
	Scope* pScope = nullptr;
	return pMaximum;
}

void* Create_Mean(std::string id, Json::Value pInputItem) {
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	Output* pAxis = nullptr;
	Mean* pMean = nullptr;

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
				std::string msg = string_format("warning : Mean - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, "output");
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
				std::string msg = string_format("warning : Mean - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "axis")
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
							pAxis = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Mean - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Mean pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pInput && pAxis)
	{
		pMean = new Mean(*pScope, *pInput, *pAxis);

		ObjectInfo* pObj = AddObjectMap(pMean, id, SYMBOL_MEAN, "Mean", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pMean->output, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("error : Mean(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pMean;
}

void* Create_Min(std::string id, Json::Value pInputItem) {
	Min* pMin = nullptr;
	Scope* pScope = nullptr;
	return pMin;
}

void* Create_Minimum(std::string id, Json::Value pInputItem) {
	Minimum* pMinimum = nullptr;
	Scope* pScope = nullptr;
	return pMinimum;
}

void* Create_Mod(std::string id, Json::Value pInputItem) {
	Mod* pMod = nullptr;
	Scope* pScope = nullptr;
	return pMod;
}

void* Create_Multiply(std::string id, Json::Value pInputItem) {
	Scope* pScope = nullptr;
	Output* pX = nullptr;
	Output* pY = nullptr;
	Multiply* pMultiply = nullptr;

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
				std::string msg = string_format("warning : Multiply - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, "output");
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pX = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Multiply - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "y")
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
							pY = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Multiply - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Multiply pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pX && pY)
	{
		pMultiply = new Multiply(*pScope, *pX, *pY);
		ObjectInfo* pObj = AddObjectMap(pMultiply, id, SYMBOL_MULTIPLY, "Multiply", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pMultiply->z, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("error : Multiply(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pMultiply;;
}

void* Create_Negate(std::string id, Json::Value pInputItem) {
	Negate* pNegate = nullptr;
	Scope* pScope = nullptr;
	return pNegate;
}

void* Create_NotEqual(std::string id, Json::Value pInputItem) {
	NotEqual* pNotEqual = nullptr;
	Scope* pScope = nullptr;
	return pNotEqual;
}

void* Create_Polygamma(std::string id, Json::Value pInputItem) {
	Polygamma* pPolygamma = nullptr;
	Scope* pScope = nullptr;
	return pPolygamma;
}

void* Create_Pow(std::string id, Json::Value pInputItem) {
	Pow* pPow = nullptr;
	Scope* pScope = nullptr;
	return pPow;
}

void* Create_Prod(std::string id, Json::Value pInputItem) {
	Prod* pProd = nullptr;
	Scope* pScope = nullptr;
	return pProd;
}

void* Create_QuantizeDownAndShrinkRange(std::string id, Json::Value pInputItem) {
	QuantizeDownAndShrinkRange* pQuantizeDownAndShrinkRange = nullptr;
	Scope* pScope = nullptr;
	return pQuantizeDownAndShrinkRange;
}

void* Create_QuantizedMatMul(std::string id, Json::Value pInputItem) {
	QuantizedMatMul* pQuantizedMatMul = nullptr;
	Scope* pScope = nullptr;
	return pQuantizedMatMul;
}

void* Create_QuantizedMul(std::string id, Json::Value pInputItem) {
	QuantizedMul* pQuantizedMul = nullptr;
	Scope* pScope = nullptr;
	return pQuantizedMul;
}

void* Create_Range(std::string id, Json::Value pInputItem) {
	Range* pRange = nullptr;
	Scope* pScope = nullptr;
	return pRange;
}

void* Create_Real(std::string id, Json::Value pInputItem) {
	Real* pReal = nullptr;
	Scope* pScope = nullptr;
	return pReal;
}

void* Create_RealDiv(std::string id, Json::Value pInputItem) {
	RealDiv* pRealDiv = nullptr;
	Scope* pScope = nullptr;
	return pRealDiv;
}

void* Create_Reciprocal(std::string id, Json::Value pInputItem) {
	Reciprocal* pReciprocal = nullptr;
	Scope* pScope = nullptr;
	return pReciprocal;
}

void* Create_RequantizationRange(std::string id, Json::Value pInputItem) {
	RequantizationRange* pRequantizationRange = nullptr;
	Scope* pScope = nullptr;
	return pRequantizationRange;
}

void* Create_Requantize(std::string id, Json::Value pInputItem) {
	Requantize* pRequantize = nullptr;
	Scope* pScope = nullptr;
	return pRequantize;
}

void* Create_Rint(std::string id, Json::Value pInputItem) {
	Rint* pRint = nullptr;
	Scope* pScope = nullptr;
	return pRint;
}

void* Create_Round(std::string id, Json::Value pInputItem) {
	Round* pRound = nullptr;
	Scope* pScope = nullptr;
	return pRound;
}

void* Create_Rsqrt(std::string id, Json::Value pInputItem) {
	Rsqrt* pRsqrt = nullptr;
	Scope* pScope = nullptr;
	return pRsqrt;
}

void* Create_SegmentMax(std::string id, Json::Value pInputItem) {
	SegmentMax* pSegmentMax = nullptr;
	Scope* pScope = nullptr;
	return pSegmentMax;
}

void* Create_SegmentMean(std::string id, Json::Value pInputItem) {
	SegmentMean* pSegmentMean = nullptr;
	Scope* pScope = nullptr;
	return pSegmentMean;
}

void* Create_SegmentMin(std::string id, Json::Value pInputItem) {
	SegmentMin* pSegmentMin = nullptr;
	Scope* pScope = nullptr;
	return pSegmentMin;
}

void* Create_SegmentProd(std::string id, Json::Value pInputItem) {
	SegmentProd* pSegmentProd = nullptr;
	Scope* pScope = nullptr;
	return pSegmentProd;
}

void* Create_SegmentSum(std::string id, Json::Value pInputItem) {
	SegmentSum* pSegmentSum = nullptr;
	Scope* pScope = nullptr;
	return pSegmentSum;
}

void* Create_Sigmoid(std::string id, Json::Value pInputItem) {
	Sigmoid* pSigmoid = nullptr;
	Scope* pScope = nullptr;
	return pSigmoid;
}

void* Create_Sign(std::string id, Json::Value pInputItem) {
	Sign* pSign = nullptr;
	Scope* pScope = nullptr;
	return pSign;
}

void* Create_Sin(std::string id, Json::Value pInputItem) {
	Sin* pSin = nullptr;
	Scope* pScope = nullptr;
	return pSin;
}

void* Create_SparseMatMul(std::string id, Json::Value pInputItem) {
	SparseMatMul* pSparseMatMul = nullptr;
	Scope* pScope = nullptr;
	return pSparseMatMul;
}

void* Create_SparseSegmentMean(std::string id, Json::Value pInputItem) {
	SparseSegmentMean* pSparseSegmentMean = nullptr;
	Scope* pScope = nullptr;
	return pSparseSegmentMean;
}

void* Create_SparseSegmentMeanGrad(std::string id, Json::Value pInputItem) {
	SparseSegmentMeanGrad* pSparseSegmentMeanGrad = nullptr;
	Scope* pScope = nullptr;
	return pSparseSegmentMeanGrad;
}

void* Create_SparseSegmentSqrtN(std::string id, Json::Value pInputItem) {
	SparseSegmentSqrtN* pSparseSegmentSqrtN = nullptr;
	Scope* pScope = nullptr;
	return pSparseSegmentSqrtN;
}

void* Create_SparseSegmentSqrtNGrad(std::string id, Json::Value pInputItem) {
	SparseSegmentSqrtNGrad* pSparseSegmentSqrtNGrad = nullptr;
	Scope* pScope = nullptr;
	return pSparseSegmentSqrtNGrad;
}

void* Create_SparseSegmentSum(std::string id, Json::Value pInputItem) {
	SparseSegmentSum* pSparseSegmentSum = nullptr;
	Scope* pScope = nullptr;
	return pSparseSegmentSum;
}

void* Create_Sqrt(std::string id, Json::Value pInputItem) {
	Sqrt* pSqrt = nullptr;
	Scope* pScope = nullptr;
	return pSqrt;
}

void* Create_Square(std::string id, Json::Value pInputItem) {
	Scope* pScope = nullptr;
	Output* pX = nullptr;
	Square* pSquare = nullptr;

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
				std::string msg = string_format("warning : Square - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, "output");
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pX = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Square - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Square pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pX)
	{
		pSquare = new Square(*pScope, *pX);
		ObjectInfo* pObj = AddObjectMap(pSquare, id, SYMBOL_SQUARE, "Square", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSquare->y, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("error : Square(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSquare;
}

void* Create_SquaredDifference(std::string id, Json::Value pInputItem) {
	SquaredDifference* pSquaredDifference = nullptr;
	Scope* pScope = nullptr;
	return pSquaredDifference;
}

void* Create_Subtract(std::string id, Json::Value pInputItem) {
	Scope* pScope = nullptr;
	Output* pX = nullptr;
	Output* pY = nullptr;
	Subtract* pSubtract = nullptr;

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
				std::string msg = string_format("warning : Subtract - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
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
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, "output");
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pX = (Output*)pOutputObj->pOutput;
						}
					}

					// pX = pObj->pOutput;
				}
			}
			else
			{
				std::string msg = string_format("warning : Subtract - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "y")
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
							pY = (Output*)pOutputObj->pOutput;
						}
					}

					// pY = pObj->pOutput;
				}
			}
			else
			{
				std::string msg = string_format("warning : Subtract - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Subtract pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pX && pY)
	{
		pSubtract = new Subtract(*pScope, *pX, *pY);
		ObjectInfo* pObj = AddObjectMap(pSubtract, id, SYMBOL_SUBTRACT, "Subtract", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, &pSubtract->z, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("error : Subtract(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSubtract;
}

void* Create_Sum(std::string id, Json::Value pInputItem) {
	Sum* pSum = nullptr;
	Scope* pScope = nullptr;
	return pSum;
}

void* Create_Tan(std::string id, Json::Value pInputItem) {
	Tan* pTan = nullptr;
	Scope* pScope = nullptr;
	return pTan;
}

void* Create_Tanh(std::string id, Json::Value pInputItem) {
	Tanh* pTanh = nullptr;
	Scope* pScope = nullptr;
	return pTanh;
}

void* Create_TruncateDiv(std::string id, Json::Value pInputItem) {
	TruncateDiv* pTruncateDiv = nullptr;
	Scope* pScope = nullptr;
	return pTruncateDiv;
}

void* Create_TruncateMod(std::string id, Json::Value pInputItem) {
	TruncateMod* pTruncateMod = nullptr;
	Scope* pScope = nullptr;
	return pTruncateMod;
}

void* Create_UnsortedSegmentMax(std::string id, Json::Value pInputItem) {
	UnsortedSegmentMax* pUnsortedSegmentMax = nullptr;
	Scope* pScope = nullptr;
	return pUnsortedSegmentMax;
}

void* Create_UnsortedSegmentSum(std::string id, Json::Value pInputItem) {
	UnsortedSegmentSum* pUnsortedSegmentSum = nullptr;
	Scope* pScope = nullptr;
	return pUnsortedSegmentSum;
}

void* Create_Where3(std::string id, Json::Value pInputItem) {
	Where3* pWhere3 = nullptr;
	Scope* pScope = nullptr;
	return pWhere3;
}

void* Create_Zeta(std::string id, Json::Value pInputItem) {
	Zeta* pZeta = nullptr;
	Scope* pScope = nullptr;
	return pZeta;
}