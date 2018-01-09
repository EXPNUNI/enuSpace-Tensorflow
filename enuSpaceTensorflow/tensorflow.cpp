// tensorflow.cpp : Defines the initialization routines for the DLL.
//

#include "stdafx.h"
#include "tensorflow.h"
#include "utility_functions.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

//
//TODO: If this DLL is dynamically linked against the MFC DLLs,
//		any functions exported from this DLL which call into
//		MFC must have the AFX_MANAGE_STATE macro added at the
//		very beginning of the function.
//
//		For example:
//
//		extern "C" BOOL PASCAL EXPORT ExportedFunction()
//		{
//			AFX_MANAGE_STATE(AfxGetStaticModuleState());
//			// normal function body here
//		}
//
//		It is very important that this macro appear in each
//		function, prior to any calls into MFC.  This means that
//		it must appear as the first statement within the 
//		function, even before any object variable declarations
//		as their constructors may generate calls into the MFC
//		DLL.
//
//		Please see MFC Technical Notes 33 and 58 for additional
//		details.
//

// CtensorflowApp

BEGIN_MESSAGE_MAP(CtensorflowApp, CWinApp)
END_MESSAGE_MAP()

 
// CtensorflowApp construction

CtensorflowApp::CtensorflowApp()
{
	// TODO: add construction code here,
	// Place all significant initialization in InitInstance
}


// The one and only CtensorflowApp object

CtensorflowApp theApp;


// CtensorflowApp initialization
CString g_strDllPath;

BOOL CtensorflowApp::InitInstance()
{
	CWinApp::InitInstance();

	HINSTANCE hInstance = AfxGetInstanceHandle();
	wchar_t szPath[MAX_PATH];
	GetModuleFileName(hInstance, szPath, MAX_PATH);

	wchar_t drive[MAX_PATH];               // ����̺� ��
	wchar_t dir[MAX_PATH];                 // ���丮 ���
	wchar_t fname[MAX_PATH];			   // ���ϸ�
	wchar_t ext[MAX_PATH];                 // Ȯ���� ��

	_wsplitpath_s(szPath, drive, dir, fname, ext);
	g_strDllPath.Format(L"%s%s", drive, dir);

	if (AllocConsole())
	{
		freopen("CONIN$", "rb", stdin);
		freopen("CONOUT$", "wb", stdout);
		freopen("CONOUT$", "wb", stderr);
	}

	return TRUE;
}

/////////////////////////////////////////////////////////////////////////

#include "enuSpaceToTensorflow.h"

void(*g_fcbSetValue)(wchar_t*, double) = NULL;
VariableStruct (*g_fcbGetValue)(wchar_t*) = NULL;
void(*g_fcbSetArrayValue)(wchar_t*, void*, int, int) = NULL;
void(*g_fcbSetReShapeArrayValue)(wchar_t*, void*, int, int) = NULL;
VariableStruct (*g_fcbGetArrayValue)(wchar_t*) = NULL;
void(*g_fcbPrintMessage)(wchar_t*, wchar_t*) = NULL;

CMap<CString, LPCWSTR, VariableStruct*, VariableStruct*> g_DBMapList;

CString StringToCString(std::string str)
{
	CString result;
	result = CString::CStringT(CA2CT(str.c_str()));
	return result;
}

std::string CStringToString(CString reqStr)
{
	std::string result;
	result = std::string(CT2CA(reqStr.operator LPCWSTR()));
	return result;
}

int GetArrayIndexFromDimension(CString strOrgDim, CString strDimension)
{
	CString Token;
	CString Seperator = _T("[]");
	int Position = 0;
	CString strBuffer = strOrgDim;

	bool berror = false;
	int idimcount = 0;
	int idim[20];

	Token = strBuffer.Tokenize(Seperator, Position);
	while (Token != L"")
	{
		Token.Trim();
		int iValue = _wtoi(Token);
		if (iValue > 0)
		{
			idim[idimcount] = iValue;
			idimcount++;
		}
		else
		{
			berror = true;
			break;
		}
		Token = strBuffer.Tokenize(Seperator, Position);
	}

	if (berror)
		return 0;

	/////////////////////////////////////////////////////////
	int iIndex = 0;
	strBuffer = strDimension;
	Position = 0;

	int isdimcount = 0;
	int isdim[20];

	Token = strBuffer.Tokenize(Seperator, Position);
	while (Token != L"")
	{
		Token.Trim();
		int iValue = _wtoi(Token);
		if (iValue >= 0)
		{
			isdim[isdimcount] = iValue;
			isdimcount++;
		}
		else
		{
			berror = true;
			break;
		}
		Token = strBuffer.Tokenize(Seperator, Position);
	}

	if (berror || isdimcount != idimcount)
		return 0;

	for (int i = 0; i < idimcount; i++)
	{
		int imux = 1;
		for (int j = i + 1; j < idimcount; j++)
		{
			imux = imux * idim[j];
		}
		iIndex = iIndex + isdim[i] * imux;
	}

	return iIndex;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// enuSpace interface functions.
extern "C" __declspec(dllexport) void SetCallBack_SetValue(void fcbSetValue(wchar_t*, double));
extern "C" __declspec(dllexport) void SetCallBack_GetValue(VariableStruct fcbGetValue(wchar_t*));
extern "C" __declspec(dllexport) void SetCallBack_SetArrayValue(void fcbSetArrayValue(wchar_t*, void*, int, int));
extern "C" __declspec(dllexport) void SetCallBack_GetArrayValue(VariableStruct fcbGetArrayValue(wchar_t*));
extern "C" __declspec(dllexport) void SetCallBack_SetReShapeArrayValue(void fcbSetReShapeArrayValue(wchar_t*, void*, int, int));
extern "C" __declspec(dllexport) void SetCallBack_PrintMessage(void fcbPrintMessage(wchar_t*, wchar_t*));

extern "C" __declspec(dllexport) int GetTaskType();
extern "C" __declspec(dllexport) bool IsEnableTransfer(wchar_t* pFromType, wchar_t* pToType);
extern "C" __declspec(dllexport) bool IsTaskStopWhenModify();

extern "C" __declspec(dllexport) bool OnInit();
extern "C" __declspec(dllexport) bool OnLoad();
extern "C" __declspec(dllexport) bool OnUnload();
extern "C" __declspec(dllexport) bool OnTask();
extern "C" __declspec(dllexport) void OnModeChange(int iMode);
extern "C" __declspec(dllexport) void ExecuteFunction(wchar_t* pStrFunction);


extern "C" __declspec(dllexport) void OnEditComponent(wchar_t* pStrSymbolName, wchar_t* pStrID);
extern "C" __declspec(dllexport) void OnShowComponent(wchar_t* pStrSymbolName, wchar_t* pStrID);
extern "C" __declspec(dllexport) bool OnShowHelp(wchar_t* pStrSymbolName);
//////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////
extern "C" __declspec(dllexport) void SetCallBack_SetValue(void fcbSetValue(wchar_t*, double))
{
	g_fcbSetValue = fcbSetValue;
}

extern "C" __declspec(dllexport) void SetCallBack_GetValue(VariableStruct fcbGetValue(wchar_t*))
{
	g_fcbGetValue = fcbGetValue;
}

extern "C" __declspec(dllexport) void SetCallBack_SetArrayValue(void fcbSetArrayValue(wchar_t*, void*, int, int))
{
	g_fcbSetArrayValue = fcbSetArrayValue;
}

extern "C" __declspec(dllexport) void SetCallBack_GetArrayValue(VariableStruct fcbGetArrayValue(wchar_t*))
{
	g_fcbGetArrayValue = fcbGetArrayValue;
}

extern "C" __declspec(dllexport) void SetCallBack_SetReShapeArrayValue(void fcbSetReShapeArrayValue(wchar_t*, void*, int, int))
{
	g_fcbSetReShapeArrayValue = fcbSetReShapeArrayValue;
}
extern "C" __declspec(dllexport) void SetCallBack_PrintMessage(void fcbPrintMessage(wchar_t*, wchar_t*))
{
	g_fcbPrintMessage = fcbPrintMessage;
}

// �������̽� �� ������ ����ü Ŭ���� ����.
void InterfaceDataMapClear()
{
	POSITION mappos = g_DBMapList.GetStartPosition();
	while (mappos)
	{
		VariableStruct *pObject = NULL;
		CString srtVar = L"";
		g_DBMapList.GetNextAssoc(mappos, srtVar, pObject);
		if (srtVar != L"")
		{
			if (pObject)
			{
				delete pObject;
			}
		}
	}
	g_DBMapList.RemoveAll();
}

// enuSpace�� �޸� ����� �ٸ����, enuSpace�� �޸𸮸� ���Ҵ��� �� �����ϴ� �Լ�
// strVariable=ID_OBJECT.input[12][12] (�迭�� SHAPE ���� ����) pSrc=���� �������� ������, iType=���� �������� Ÿ��, iSize=���� �������� ������
// �Էµ� �迭�� ������ enuSpace�� �迭 ������ ��
void SetReShapeArrayValue(std::string strVariable, void* pSrc, int iType, int iSize)
{
	VariableStruct* pData = NULL;
	std::wstring widestr = std::wstring(strVariable.begin(), strVariable.end());

	CString strVal = (wchar_t*)widestr.c_str();

	CString strDim;
	int iPos = strVal.Find(L"[");
	if (iPos > 1)
	{
		strDim = strVal.Right(strVal.GetLength() - iPos);
		strVal = strVal.Left(iPos);
	}

	g_DBMapList.Lookup(strVal, pData);
	if (pData)
	{
		// ��û�� ������ ������ Ÿ�� �� �迭�� ���̰� �����ϴٸ�, ���� ������Ʈ ����.
		if (pData->array.size == iSize && pData->type == iType)
		{
			// ArraySize�� �����ϳ� dimension���� ����Ȱ�� �Ÿ𸮸��� ������ ������Ʈ ����.
			if (wcscmp(pData->array.dimension, strDim.GetBuffer(0)) != 0)
			{
				// enuSpace Reshape ������ ����, enuSpace�� �޸� ������ ������ dimension ������ ������Ʈ ����.
				if (g_fcbSetReShapeArrayValue)
				{
					g_fcbSetReShapeArrayValue((wchar_t*)widestr.c_str(), pSrc, iType, iSize);
					wcscpy_s(pData->array.dimension, strDim);
				}
				return;
			}

			int itemSize = 0;
			void* pTarget = NULL;
			switch (iType)
			{
				case DEF_BOOL:
				{
					itemSize = iSize * sizeof(bool);
					pTarget = ((bool*)pData->pValue);
					memcpy(pTarget, pSrc, itemSize);
					break;
				}
				case DEF_INT:
				{
					itemSize = iSize * sizeof(int);
					pTarget = ((int*)pData->pValue);
					memcpy(pTarget, pSrc, itemSize);
					break;
				}
				case DEF_FLOAT:
				{
					itemSize = iSize * sizeof(float);
					pTarget = ((float*)pData->pValue);
					memcpy(pTarget, pSrc, itemSize);
					break;
				}
				case DEF_DOUBLE:
				{
					itemSize = iSize * sizeof(double);
					pTarget = ((double*)pData->pValue);
					memcpy(pTarget, pSrc, itemSize);
					break;
				}
				case DEF_STRING:
				{
					for (int i = 0; i < iSize; i++)
					{
						std::string strValue = *((std::string*)pSrc + i);
						pTarget = ((std::string*)pData->pValue);
						((CString*)pTarget+i)->SetString(StringToCString(strValue));
					}
					break;
				}
			}
			return;
		}
		// �޸��� ������ Ÿ�� �� �迭�� ����� �ٸ��ٸ�, �� ����Ʈ���� ���� ���� �� �޸� RESHAPE �� �� ������Ʈ �Լ� ȣ��
		else
		{
			delete pData;
			pData = NULL;
			g_DBMapList.RemoveKey(strVal);

			if (g_fcbSetReShapeArrayValue)
			{
				g_fcbSetReShapeArrayValue((wchar_t*)widestr.c_str(), pSrc, iType, iSize);
			}
			// ���������� Ÿ�� ����Ʈ ���� �߰� ����.
		}
	}

	// �� ����Ʈ���� �˻��� �������� ���Ͽ��ٸ�, ���� ������Ʈ ��û��. �� ������Ʈ ��û�Լ��� �������� Ÿ�� �� ARRAY ������ �ٸ� ���
	// �޸��� ReShape ������ ���� ������Ʈ ����
	// ����� �޸��� ������ �ʸ���Ʈ�� �߰��Ͽ�, ���� ���� ����� ���� ó���� �� �ֵ��� �߰���.
	if (pData == NULL)
	{
		if (g_fcbGetArrayValue)
		{
			std::wstring widestr = std::wstring(strVariable.begin(), strVariable.end());
			VariableStruct Data = g_fcbGetArrayValue(strVal.GetBuffer(0));
			if (Data.pValue)
			{
				// ������ �������� ������ �� Ÿ���� ��� �� ������Ʈ ����.
				if (Data.type == iType && Data.array.size == iSize)
				{
					int itemSize = 0;
					void* pTarget = NULL;
					switch (iType)
					{
						case DEF_BOOL:
						{
							itemSize = iSize * sizeof(bool);
							pTarget = ((bool*)Data.pValue);
							memcpy(pTarget, pSrc, itemSize);
							break;
						}
						case DEF_INT:
						{
							itemSize = iSize * sizeof(int);
							pTarget = ((int*)Data.pValue);
							memcpy(pTarget, pSrc, itemSize);
							break;
						}
						case DEF_FLOAT:
						{
							itemSize = iSize * sizeof(float);
							pTarget = ((float*)Data.pValue);
							memcpy(pTarget, pSrc, itemSize);
							break;
						}
						case DEF_DOUBLE:
						{
							itemSize = iSize * sizeof(double);
							pTarget = ((double*)Data.pValue);
							memcpy(pTarget, pSrc, itemSize);
							break;
						}
						case DEF_STRING:
						{
							for (int i = 0; i < iSize; i++)
							{
								std::string strValue = *((std::string*)pSrc + i);
								pTarget = ((std::string*)Data.pValue);
								((CString*)pTarget + i)->SetString(StringToCString(strValue));
							}
							break;
						}
					}
					// ����Ʈ �߰� 
					VariableStruct* pNewData = new VariableStruct;
					*pNewData = Data;
					g_DBMapList.SetAt(strVal, pNewData);
				}
				// �Էµ� ������, Ÿ���� enuSpace�� ������, Ÿ���� �ٸ��ٸ� RESHAPE �� �� ������Ʈ ����
				else
				{
					if (g_fcbSetReShapeArrayValue)
					{
						if (iType == DEF_STRING)
						{
							CString *pString = new CString[iSize];
							for (int i = 0; i < iSize; i++)
							{
								std::string strValue = *((std::string*)pSrc + i);
								((CString*)pString + i)->SetString(StringToCString(strValue));
							}
							g_fcbSetReShapeArrayValue((wchar_t*)widestr.c_str(), pString, iType, iSize);
							delete[] pString;
						}
						else
							g_fcbSetReShapeArrayValue((wchar_t*)widestr.c_str(), pSrc, iType, iSize);
					}
				}
			}
			else
			{
				PrintMessage(strings::Printf("error : SetArrayValue (Unknown Variable id(%s)", strVariable.c_str()));
				return;			// enuSpace�� ���� ������ ��û��.
			}
		}
	}
}

// �迭�� ���� �Լ� 
// strVarialbe=ID_OBJECT.input[0], pSrc=�����͸� ������ �迭�� �ּ�, iType=�������� Ÿ��, iSize=�������� ������
// ID_OBJECT.input[10]�� ������ 10���� Array�� ������
// ID_OBJECT.input[5], iSize�� ���� 3�� �Է��Ͽ��� ���, 5��° ���� 7��°���� ���縦 ������.
// ���� �������� ���� ���� �������� ���� �Ѿ�� ��� ���� �������� ����.
void SetArrayValue(std::string strVariable, void* pSrc, int iType, int iSize)
{
	VariableStruct* pData = NULL;
	std::wstring widestr = std::wstring(strVariable.begin(), strVariable.end());

	CString strVal = (wchar_t*)widestr.c_str();
	CString strDim;

	// ���� ������ �迭�� �̸��� ���ԵǾ� ���� ����. ����ü ������ ���Ե�.
	int iPos = strVal.Find(L"[");
	if (iPos > 1)
	{
		strDim = strVal.Right(strVal.GetLength() - iPos);
		strVal = strVal.Left(iPos);
	}
	g_DBMapList.Lookup(strVal, pData);

	// ���� ���Ͽ� �������� ����ü ������ ���ٸ�, enuSpace�� �ش� ������ ������ �ּҸ� ��û��.
	if (g_fcbSetArrayValue && pData == NULL)
	{
		VariableStruct Data = g_fcbGetArrayValue(strVal.GetBuffer(0));
		if (Data.pValue)
		{
			VariableStruct* pNewData = new VariableStruct;
			*pNewData = Data;
			g_DBMapList.SetAt((wchar_t*)widestr.c_str(), pNewData);

			pData = pNewData;
		}
	}

	// ����� �޸��� ������ �˻��ϰ� ������ �������, ���� ����.
	if (pData && pData->pValue)
	{
		int itemSize = 0;
		void* pTarget = NULL;
		int iIndex = GetArrayIndexFromDimension(pData->array.dimension, strDim);
		if (iIndex + iSize <= pData->array.size)
		{
			switch (iType)
			{
				case DEF_BOOL:
				{
					itemSize = iSize * sizeof(bool);
					pTarget = ((bool*)pData->pValue + iIndex);
					memcpy(pTarget, pSrc, itemSize);
					break;
				}
				case DEF_INT:
				{
					itemSize = iSize * sizeof(int);
					pTarget = ((int*)pData->pValue + iIndex);
					memcpy(pTarget, pSrc, itemSize);
					break;
				}
				case DEF_FLOAT:
				{
					itemSize = iSize * sizeof(float);
					pTarget = ((float*)pData->pValue + iIndex);
					memcpy(pTarget, pSrc, itemSize);
					break;
				}
				case DEF_DOUBLE:
				{
					itemSize = iSize * sizeof(double);
					pTarget = ((double*)pData->pValue + iIndex);
					memcpy(pTarget, pSrc, itemSize);
					break;
				}
				case DEF_STRING:
				{
					pTarget = pData->pValue;
					for (int i = 0; i < iSize; i++)
					{
						std::string strValue = *((std::string*)pData->pValue + i);
						((CString*)pTarget + i)->SetString(StringToCString(strValue));
					}
					break;
				}
			}
		}
	}
}

// �迭������ ���� ��û�ϴ� �Լ� ��) OD_OBJECT.input[10] input[10]������ ���� ��ȯ��.
double GetArrayValue(std::string strVariable)
{
	VariableStruct* pData = NULL;
	std::wstring widestr = std::wstring(strVariable.begin(), strVariable.end());

	CString strVal = (wchar_t*)widestr.c_str();
	CString strDim;

	// ���� ������ �迭�� �̸��� ���ԵǾ� ���� ����. ����ü ������ ���Ե�.
	int iPos = strVal.Find(L"[");
	if (iPos > 1)
	{
		strDim = strVal.Right(strVal.GetLength() - iPos);
		strVal = strVal.Left(iPos);
	}
	g_DBMapList.Lookup(strVal, pData);

	if (g_fcbGetArrayValue && pData == NULL)
	{
		std::wstring widestr = std::wstring(strVariable.begin(), strVariable.end());
		VariableStruct Data = g_fcbGetArrayValue(strVal.GetBuffer(0));
		if (Data.pValue)
		{
			VariableStruct* pNewData = new VariableStruct;
			*pNewData = Data;
			g_DBMapList.SetAt(strVal, pNewData);

			pData = pNewData;
		}
	}

	double fReturn = 0;
	if (pData && pData->pValue)
	{
		// ���� �迭������ �迭 ������ �Էµ� ��û�� �迭�� ������ �̿��Ͽ� Array�� ��ġ�� ȹ����.
		int iIndex = GetArrayIndexFromDimension(pData->array.dimension, strDim);
		if (iIndex <= pData->array.size)
		{
			switch (pData->type)
			{
				case DEF_INT:
					fReturn = *(int*)pData->pValue;
					break;
				case DEF_FLOAT:
					fReturn = *(float*)pData->pValue;
					break;
				case DEF_DOUBLE:
					fReturn = *(double*)pData->pValue;
					break;
				case DEF_BOOL:
					if (*(bool*)pData->pValue == TRUE)
						fReturn = 1;
					else
						fReturn = 0;
					break;
				case DEF_STRING:
					fReturn = 0;				// NOT SUPPORT
					break;
				default:
					break;
			}
		}
	}
	return fReturn;
}

void SetValue(std::string strVariable, double fValue)
{
	double fReturn = 0;
	VariableStruct* pData = NULL;
	std::wstring widestr = std::wstring(strVariable.begin(), strVariable.end());
	g_DBMapList.Lookup((wchar_t*)widestr.c_str(), pData);

	if (g_fcbSetValue && pData == NULL)
	{
		VariableStruct Data = g_fcbGetValue((wchar_t*)widestr.c_str());
		if (Data.pValue)
		{
			VariableStruct* pNewData = new VariableStruct;
			*pNewData = Data;
			g_DBMapList.SetAt((wchar_t*)widestr.c_str(), pNewData);

			pData = pNewData;
		}
	}

	if (pData && pData->pValue)
	{
		switch (pData->type)
		{
		case DEF_INT:
			*(int*)pData->pValue = (int)fValue;
			break;
		case DEF_FLOAT:
			*(float*)pData->pValue = (float)fValue;
			break;
		case DEF_DOUBLE:
			*(double*)pData->pValue = fValue;
			break;
		case DEF_BOOL:
			if (fValue == 1)
				*(bool*)pData->pValue = true;
			else
				*(bool*)pData->pValue = false;
			break;
		case DEF_STRING:

			break;
		default:
			break;
		}
	}
}

double GetValue(std::string strVariable)
{
	double fReturn = 0;
	VariableStruct* pData = NULL;
	std::wstring widestr = std::wstring(strVariable.begin(), strVariable.end());
	g_DBMapList.Lookup((wchar_t*)widestr.c_str(), pData);

	if (g_fcbGetValue && pData == NULL)
	{
		VariableStruct Data = g_fcbGetValue((wchar_t*)widestr.c_str());
		if (Data.type)
		{
			VariableStruct* pNewData = new VariableStruct;
			*pNewData = Data;
			g_DBMapList.SetAt((wchar_t*)widestr.c_str(), pNewData);

			pData = pNewData;
		}
	}

	if (pData && pData->pValue)
	{
		switch (pData->type)
		{
		case DEF_INT:
			fReturn = *(int*)pData->pValue;
			break;
		case DEF_FLOAT:
			fReturn = *(float*)pData->pValue;
			break;
		case DEF_DOUBLE:
			fReturn = *(double*)pData->pValue;
			break;
		case DEF_BOOL:
			if (*(bool*)pData->pValue == TRUE)
				fReturn = 1;
			else
				fReturn = 0;
			break;
		case DEF_STRING:
			fReturn = 0;				// NOT SUPPORT
			break;
		default:
			break;
		}
	}
	return fReturn;
}

void PrintMessage(std::string strMessage, std::string strID)
{
	if (g_fcbPrintMessage && m_bShowDebugMessage)
	{
		std::string strTenMessage = string_format("tensorflow -> %s", strMessage.c_str());
		std::string strId = string_format("%s", strID.c_str());
		g_fcbPrintMessage(StringToCString(strTenMessage).GetBuffer(0), StringToCString(strId).GetBuffer(0));
	}
}

extern "C" __declspec(dllexport) int GetTaskType()
{
	return TASK_TYPE_FLOW_COMPONENT_PAGE;
}

extern "C" __declspec(dllexport) bool IsEnableTransfer(wchar_t* pFromType, wchar_t* pToType)
{
	CString strFromType = pFromType;
	CString strToType = pToType;
	if (strFromType == L"Output" && strToType == L"Input")
		return true;
	else if (strFromType == L"Input" && strToType == L"Input")
		return true;
	else if (strFromType == L"Output" && strToType == L"Output")
		return true;
	else if (strFromType == L"Output" && strToType == L"std::vector<tensorflow::Output>")
		return true;
	else if (strFromType == L"OutputList" && strToType == L"InputList")
		return true;
	else if (strFromType == L"OutputList" && strToType == L"std::vector<tensorflow::Output>")
		return true;
	else if (strFromType == L"string" && strToType == L"string")
		return true;
	else if (strFromType == L"Tensor" && strToType == L"Input")
		return true;
	else if (strFromType == L"Input::Initializer" && strToType == L"Input")
		return true;
	else if (strFromType == L"Input::Initializer" && strToType == L"Input::Initializer")
		return true;
	else if (strFromType == L"Output" && strToType == L"OutputList")
		return true;
	else if (strFromType == L"OutputList" && strToType == L"OutputList")
		return true;
	else if (strFromType == L"Operation" && strToType == L"std::vector<tensorflow::Operation>")
		return true;
	else if (strFromType == L"Operation" && strToType == L"std::vector<tensorflow::Output>")
		return true;
	else if (strFromType == L"Output" && strToType == L"Operation")
		return true;
	else if (strFromType == L"Operation" && strToType == L"Output")
		return true;
	else if (strFromType == L"Tensor" && strToType == L"Operation")
		return true;
	else if (strFromType == L"InputList" && strToType == L"std::vector<tensorflow::Output>")
		return true;

	else if (strFromType == L"ops::Variable" && (strToType == L"Input" || strToType == L"std::vector<tensorflow::Output>"))
		return true;
	else if (strFromType == L"ops::Multiply" && (strToType == L"Input" || strToType == L"std::vector<tensorflow::Output>"))
		return true;
	else if (strFromType == L"ops::Add" && (strToType == L"Input" || strToType == L"std::vector<tensorflow::Output>"))
		return true;
	else if (strFromType == L"ops::Subtract" && (strToType == L"Input" || strToType == L"std::vector<tensorflow::Output>"))
		return true;
	else if (strFromType == L"ops::Div" && (strToType == L"Input" || strToType == L"std::vector<tensorflow::Output>"))
		return true;
	else if (strFromType == L"ops::Square" && (strToType == L"Input" || strToType == L"std::vector<tensorflow::Output>"))
		return true;
	else if (strFromType == L"ops::Mean" && (strToType == L"Input" || strToType == L"std::vector<tensorflow::Output>"))
		return true;
	else if (strFromType == L"ops::RandomNormal" && (strToType == L"Input" || strToType == L"std::vector<tensorflow::Output>"))
		return true;
	else if (strFromType == L"ops::Placeholder" && (strToType == L"Input" || strToType == L"std::vector<tensorflow::Output>"))
		return true;

	else if (strFromType == L"ops::ApplyGradientDescent" && (strToType == L"Input" || strToType == L"std::vector<tensorflow::Output>"))
		return true;

	

	else
		return false;
}

extern "C" __declspec(dllexport) bool IsTaskStopWhenModify()
{
	return true;
}

extern "C" __declspec(dllexport) bool OnLoad()
{
	try
	{
		Load_Tensorflow();
		return true;
	}
	catch (...)
	{

	}
	return false;
}

bool bProcessing = false;
int iLoopCycle = 0;
extern "C" __declspec(dllexport) bool OnInit()
{
	try
	{
		InterfaceDataMapClear();
		ObjectMapClear();

		CString dirName = g_strDllPath;
		CString strExt = L"/*.json";

		CFileFind finder;

		BOOL bWorking = finder.FindFile(dirName + strExt);

		while (bWorking)
		{
			bWorking = finder.FindNextFile();
			if (finder.IsDots())
			{
				continue;
			}

			CString logic_file = finder.GetFilePath();

			std::string filename = CStringToString(logic_file);

			CString str;
			CFile pFile;

			if (pFile.Open(logic_file, CFile::modeRead | CFile::typeBinary))
			{
				pFile.Seek(2, CFile::begin);
				size_t fLength = (size_t)pFile.GetLength() - 2;
				TCHAR *th = (TCHAR*)malloc(fLength + sizeof(TCHAR));
				memset(th, 0, fLength + sizeof(TCHAR));
				pFile.Read(th, fLength);

				str = th;
				free(th);
				pFile.Close();

				std::string covertStr;
				covertStr = CStringToString(str);

				WCHAR drive[_MAX_DRIVE];
				WCHAR dir[_MAX_DIR];
				WCHAR fname[_MAX_FNAME];
				WCHAR ext[_MAX_EXT];
				_wsplitpath_s(logic_file, drive, dir, fname, ext);

				///////////////////////////////////////////////////////////////////
				// binary data file
				CString datafilename;
				datafilename.Format(L"%s%s%s.dat", drive, dir, fname);
				if (_wfopen_s(&m_FileData, datafilename.GetBuffer(0), L"rb") != NULL)
				{
					m_FileData = NULL;
				}

				std::string pagename;
				pagename = CStringToString(fname);

				Init_Tensorflow(covertStr, pagename);

				///////////////////////////////////////////////////////////////////
				// binary data file close
				if (m_FileData)
				{
					fclose(m_FileData);
					m_FileData = NULL;
				}
			}
		}
		finder.Close();

		iLoopCycle = 0;
		return true;
	}
	catch (...)
	{

	}
	return false;
}

extern "C" __declspec(dllexport) bool OnTask()
{
	try
	{
		if (m_bContinusLoop == false)
		{
			Task_Tensorflow();
			iLoopCycle++;

			if (g_fcbPrintMessage)
			{
				CString strMessage;
				strMessage.Format(L"tensorflow -> event cycle ...................%d", iLoopCycle);
				g_fcbPrintMessage(strMessage.GetBuffer(0), L"");
			}
			return true;
		}
		else if (m_bContinusLoop && m_iSimulationMode == DEF_MODE_STEP)
		{
			Task_Tensorflow();
			iLoopCycle++;

			if (g_fcbPrintMessage)
			{
				CString strMessage;
				strMessage.Format(L"tensorflow -> event cycle ...................%d", iLoopCycle);
				g_fcbPrintMessage(strMessage.GetBuffer(0), L"");
			}
			return true;
		}
		else
		{
			if (bProcessing == false)
			{
				while (m_bContinusLoop && m_iSimulationMode == DEF_MODE_RUN)
				{
					bProcessing = true;
					Task_Tensorflow();
					iLoopCycle++;

					if (g_fcbPrintMessage)
					{
						CString strMessage;
						strMessage.Format(L"tensorflow -> contineous cycle ...................%d", iLoopCycle);
						g_fcbPrintMessage(strMessage.GetBuffer(0), L"");
					}
				}
				bProcessing = false;
			}
		}

		return true;
	}
	catch (...)
	{

	}
	return false;
}

extern "C" __declspec(dllexport) bool OnUnload()
{
	try
	{
		FreeConsole();
		Unload_Tensorflow();
		return true;
	}
	catch (...)
	{

	}
	return false;
}

extern "C" __declspec(dllexport) void OnEditComponent(wchar_t* pStrSymbolName, wchar_t* pStrID)
{

}
extern "C" __declspec(dllexport) void OnShowComponent(wchar_t* pStrSymbolName, wchar_t* pStrID)
{

}

extern "C" __declspec(dllexport) void OnModeChange(int iMode)
{
	m_iSimulationMode = iMode;
}

extern "C" __declspec(dllexport) void ExecuteFunction(wchar_t* pStrFunction)
{
	try
	{
		CString strFunction = pStrFunction;
		if (strFunction.Find(L"ShowDebugMessage") == 0)
		{
			CString Value = strFunction.Right(strFunction.GetLength() - 16);
			Value.Trim();
			Value.Trim(L"(");
			Value.Trim(L")");
			Value.Trim();
			Value.MakeLower();
			if (Value == L"true" || Value == L"1")
				m_bShowDebugMessage = true;
			else
				m_bShowDebugMessage = false;
		}
		else if (strFunction.Find(L"SetInfiniteLoop") == 0)
		{
			CString Value = strFunction.Right(strFunction.GetLength() - 15);
			Value.Trim();
			Value.Trim(L"(");
			Value.Trim(L")");
			Value.Trim();
			Value.MakeLower();
			if (Value == L"true" || Value == L"1")
				m_bContinusLoop = true;
			else
				m_bContinusLoop = false;
		}

		if (g_fcbPrintMessage)
		{
			CString strMsg;
			strMsg.Format(L"enuSpace to ExecuteFunction call - %s", strFunction);
			g_fcbPrintMessage(strMsg.GetBuffer(0), L"");
		}
		return;
	}
	catch (...)
	{

	}
	return;
}

// HELP Interface
extern "C" __declspec(dllexport) bool OnShowHelp(wchar_t* pStrSymbolName)
{
	try
	{
		CString strComponent = pStrSymbolName;
		int SymbolType = GetSymbolType(CStringToString(strComponent));
		if (SymbolType == SYMBOL_NONE)
		{
			return false;
		}

		strComponent.MakeLower();
		strComponent.Remove(L'#');
		CString strAddress;

		CString strCategory = StringToCString(GetCategoryName(SymbolType));

		strAddress.Format(L"https://expnuni.gitbooks.io/enuspacetensorflow/content/%s/%s.html", strCategory, strComponent);
		ShellExecute(NULL, L"open", L"chrome.exe", strAddress, NULL, SW_SHOW);
		return true;
	}
	catch (...)
	{

	}
	return false;
}
//////////////////////////////////////////////////////////////////////////////////////////////

