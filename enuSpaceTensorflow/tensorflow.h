// tensorflow.h : main header file for the tensorflow DLL
//

#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols


// CtensorflowApp
// See tensorflow.cpp for the implementation of this class
//

class CtensorflowApp : public CWinApp
{
public:
	CtensorflowApp();

// Overrides
public:
	virtual BOOL InitInstance();

	DECLARE_MESSAGE_MAP()
};

////////////////////////////////////////////////////////////////////////////////////////////
// Description : enuSpace - plugin (tensorflow)
//               This plugin library is made in Expansion & Universal Cooperation.  
//               This core library used tensorflow project(https://www.tensorflow.org/). 
// Start date  : 2017.05
// homepage    : http://www.enu-tech.co.kr
// Technical Support e-mail : master@enu-tech.co.kr
// Copyright (C) ENU Corporation, 이엔유주식회사, ENU Co., Ltd
// All rights reserved.
////////////////////////////////////////////////////////////////////////////////////////////

#include <string>

/////////////////////////////////////////////////////////////////////////////////
#define DEF_UNKNOWN								-1
#define DEF_INT									0
#define DEF_FLOAT								1
#define DEF_DOUBLE								2
#define DEF_BOOL								3
#define DEF_STRING								4
#define DEF_STRUCT								5
#define DEF_VARIABLE							6
#define DEF_OBJECT								7

#define TASK_TYPE_UNKNOWN						0			// [v3.5] mod : Task의 타입 정의 (알수없는 타입)
#define TASK_TYPE_PROCESS						1			// [v3.5] mod : Task의 타입 정의	(연산처리용 타입)
#define TASK_TYPE_FLOW_COMPONENT_TOTAL			2			// [v3.5] mod : Task의 타입 정의	(FLOW 컴포넌트 전체 단위 타입)
#define TASK_TYPE_FLOW_COMPONENT_PAGE			3			// [v3.5] add : Task의 타입 정의	(FLOW컴포넌트 페이지 단위 타입)
#define TASK_TYPE_FUNCTION_COMPONENT			4			// [v3.5] add : Task의 타입 정의 (함수형 컴포넌트 타입)



#define DEF_NAME_LEN							64
#define DEF_MAXTEXT_LEN							2048
#define DEF_LABELTEXT_LEN						1024

struct arrayInfo
{
	int size;
	int position;								// enuSpace 버젼 (added v3.15)
	wchar_t dimension[DEF_LABELTEXT_LEN];

public:arrayInfo()
{
	size = 0;
	position = 0;
	wcscpy_s(dimension, L"");
}
};

struct VariableStruct
{
	wchar_t name[DEF_NAME_LEN];
	int     type;
	void*   pValue;
	wchar_t strValue[DEF_MAXTEXT_LEN];
	arrayInfo array;

public:VariableStruct()
{
	wcscpy_s(name, L"");			// [ver.56] add : 초기화 추가.
	type = DEF_UNKNOWN;
	pValue = NULL;
	wcscpy_s(strValue, L"N/A");
}
};

void InterfaceDataMapClear();
void SetArrayValue(std::string strVariable, void* pSrc, int iType, int iSize);
void SetReShapeArrayValue(std::string strVariable, void* pSrc, int iType, int iSize);		// enuSpace의 메모리 사이즈가 다른경우, enuSpace의 메모리를 재할당후 값 복사하는 함수
double GetArrayValue(std::string strVariable);
void SetValue(std::string strVariable, double fValue);
double GetValue(std::string strVariable);
void PrintMessage(std::string strMessage, std::string strID = "");
int GetArrayIndexFromDimension(CString strOrgDim, CString strDimension);