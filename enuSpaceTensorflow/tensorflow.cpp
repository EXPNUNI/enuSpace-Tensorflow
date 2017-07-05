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

	wchar_t drive[MAX_PATH];               // 드라이브 명
	wchar_t dir[MAX_PATH];                 // 디렉토리 경로
	wchar_t fname[MAX_PATH];			   // 파일명
	wchar_t ext[MAX_PATH];                 // 확장자 명

	_wsplitpath_s(szPath, drive, dir, fname, ext);
	g_strDllPath.Format(L"%s%s", drive, dir);

	return TRUE;
}


#include "enuSpaceToTensorflow.h"

void(*g_fcbSetValue)(wchar_t*, double) = NULL;
DBDataStruct* (*g_fcbGetValue)(wchar_t*) = NULL;
void(*g_fcbSetArrayValue)(wchar_t*, void*, int, int) = NULL;
DBDataStruct* (*g_fcbGetArrayValue)(wchar_t*) = NULL;
void(*g_fcbPrintMessage)(wchar_t*) = NULL;

CMap<CString, LPCWSTR, DBDataStruct*, DBDataStruct*> g_DBMapList;

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

//////////////////////////////////////////////////////////////////////////////////////////////
// enuSpace interface functions.
extern "C" __declspec(dllexport) void SetCallBack_SetValue(void fcbSetValue(wchar_t*, double));
extern "C" __declspec(dllexport) void SetCallBack_GetValue(DBDataStruct* fcbGetValue(wchar_t*));
extern "C" __declspec(dllexport) void SetCallBack_SetArrayValue(void fcbSetArrayValue(wchar_t*, void*, int, int));
extern "C" __declspec(dllexport) void SetCallBack_GetArrayValue(DBDataStruct* fcbGetArrayValue(wchar_t*));
extern "C" __declspec(dllexport) void SetCallBack_PrintMessage(void fcbPrintMessage(wchar_t*));

extern "C" __declspec(dllexport) int GetTaskType();
extern "C" __declspec(dllexport) bool IsEnableTransfer(wchar_t* pFromType, wchar_t* pToType);

extern "C" __declspec(dllexport) bool OnInit();
extern "C" __declspec(dllexport) bool OnLoad();
extern "C" __declspec(dllexport) bool OnUnload();
extern "C" __declspec(dllexport) bool OnTask();

extern "C" __declspec(dllexport) void OnEditComponent(wchar_t* pStrSymbolName, wchar_t* pStrID);
extern "C" __declspec(dllexport) void OnShowComponent(wchar_t* pStrSymbolName, wchar_t* pStrID);
//////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////
extern "C" __declspec(dllexport) void SetCallBack_SetValue(void fcbSetValue(wchar_t*, double))
{
	g_fcbSetValue = fcbSetValue;
}

extern "C" __declspec(dllexport) void SetCallBack_GetValue(DBDataStruct* fcbGetValue(wchar_t*))
{
	g_fcbGetValue = fcbGetValue;
}

extern "C" __declspec(dllexport) void SetCallBack_SetArrayValue(void fcbSetArrayValue(wchar_t*, void*, int, int))
{
	g_fcbSetArrayValue = fcbSetArrayValue;
}

extern "C" __declspec(dllexport) void SetCallBack_GetArrayValue(DBDataStruct* fcbGetArrayValue(wchar_t*))
{
	g_fcbGetArrayValue = fcbGetArrayValue;
}

extern "C" __declspec(dllexport) void SetCallBack_PrintMessage(void fcbPrintMessage(wchar_t*))
{
	g_fcbPrintMessage = fcbPrintMessage;
}

void SetArrayValue(std::string strVariable, void* pSrc, int iType, int iSize)
{
	if (g_fcbSetArrayValue)
	{
		std::wstring widestr = std::wstring(strVariable.begin(), strVariable.end());
		g_fcbSetArrayValue((wchar_t*)widestr.c_str(), pSrc, iType, iSize);
	}
}

DBDataStruct* GetArrayValue(std::string strVariable)
{
	if (g_fcbGetArrayValue)
	{
		std::wstring widestr = std::wstring(strVariable.begin(), strVariable.end());
		return g_fcbGetArrayValue((wchar_t*)widestr.c_str());
	}
	return NULL;
}

void SetValue(std::string strVariable, double fValue)
{
	double fReturn = 0;
	DBDataStruct* pData = NULL;
	std::wstring widestr = std::wstring(strVariable.begin(), strVariable.end());
	g_DBMapList.Lookup((wchar_t*)widestr.c_str(), pData);

	if (g_fcbSetValue && pData == NULL)
	{
		pData = g_fcbGetValue((wchar_t*)widestr.c_str());
		g_DBMapList.SetAt((wchar_t*)widestr.c_str(), pData);
	}

	if (pData && pData->pValue)
	{
		switch (pData->itype)
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
	DBDataStruct* pData = NULL;
	std::wstring widestr = std::wstring(strVariable.begin(), strVariable.end());
	g_DBMapList.Lookup((wchar_t*)widestr.c_str(), pData);

	if (g_fcbGetValue && pData == NULL)
	{
		pData = g_fcbGetValue((wchar_t*)widestr.c_str());
		if (pData)
			g_DBMapList.SetAt((wchar_t*)widestr.c_str(), pData);
	}

	if (pData && pData->pValue)
	{
		switch (pData->itype)
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

void PrintMessage(std::string strMessage)
{
	std::string strTenMessage = string_format("tensorflow -> %s", strMessage.c_str());
	if (g_fcbPrintMessage)
	{
		g_fcbPrintMessage(StringToCString(strTenMessage).GetBuffer(0));
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
	else if (strFromType == L"Output" && strToType == L"std::vector(tensorflow::Output)")
		return true;

	else if (strFromType == L"ops::Variable" && (strToType == L"Input" || strToType == L"std::vector(tensorflow::Output)"))
		return true;
	else if (strFromType == L"ops::Multiply" && (strToType == L"Input" || strToType == L"std::vector(tensorflow::Output)"))
		return true;
	else if (strFromType == L"ops::Add" && (strToType == L"Input" || strToType == L"std::vector(tensorflow::Output)"))
		return true;
	else if (strFromType == L"ops::Subtract" && (strToType == L"Input" || strToType == L"std::vector(tensorflow::Output)"))
		return true;
	else if (strFromType == L"ops::Div" && (strToType == L"Input" || strToType == L"std::vector(tensorflow::Output)"))
		return true;
	else if (strFromType == L"ops::Square" && (strToType == L"Input" || strToType == L"std::vector(tensorflow::Output)"))
		return true;
	else if (strFromType == L"ops::Mean" && (strToType == L"Input" || strToType == L"std::vector(tensorflow::Output)"))
		return true;
	else if (strFromType == L"ops::RandomNormal" && (strToType == L"Input" || strToType == L"std::vector(tensorflow::Output)"))
		return true;
	else if (strFromType == L"ops::Placeholder" && (strToType == L"Input" || strToType == L"std::vector(tensorflow::Output)"))
		return true;

	else if (strFromType == L"ops::ApplyGradientDescent" && (strToType == L"Input" || strToType == L"std::vector(tensorflow::Output)"))
		return true;

	else
		return false;
}

extern "C" __declspec(dllexport) bool OnLoad()
{
	return Load_Tensorflow();
}

extern "C" __declspec(dllexport) bool OnInit()
{
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
		PrintMessage(filename);

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

			std::string pagename;
			pagename = CStringToString(fname);

			Init_Tensorflow(covertStr, pagename);
		}
	}
	finder.Close();

	return true;
}

extern "C" __declspec(dllexport) bool OnTask()
{
	return Task_Tensorflow();
}

extern "C" __declspec(dllexport) bool OnUnload()
{
	return Unload_Tensorflow();
}

extern "C" __declspec(dllexport) void OnEditComponent(wchar_t* pStrSymbolName, wchar_t* pStrID)
{

}
extern "C" __declspec(dllexport) void OnShowComponent(wchar_t* pStrSymbolName, wchar_t* pStrID)
{

}
//////////////////////////////////////////////////////////////////////////////////////////////
