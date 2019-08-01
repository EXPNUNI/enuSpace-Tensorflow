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

#include <vector>

struct VariableItem
{
	int iType;							// 변수의 타입 (int, float, bool, double, string, struct)
	CString strType;					// 변수의 타입 문자열 이름
	int iTotalSize;						// 변수의 사이즈(배열 포함)
	int iUnitSize;						// 변수한개의 사이즈.
	CString strName;					// 변수의 이름
	CString strInitial;					// 변수의 초기값
	void* pStructItem;					// 타입정보가 구조체인경우, 해당 구조체 정보를 가리키는 인자.
	CString strDims;					// 문자열의 DIMENSION 정보.
	std::vector<int> Dims;				// 배열의 DIMENSION 정보.
	CString strDescription;				// (UI)변수의 설명
	CString strUnit;					// (UI)변수의 단위
	bool bReadOnly;						// (UI)읽기모드
	bool bVisibility;					// (UI)디스플레이 모드
	byte color[3];						// (UI)색상
	CString min;						// (UI)최소값
	CString max;						// (UI)최대값
public:VariableItem()
{
	iType = DEF_UNKNOWN;
	iTotalSize = 0;
	iUnitSize = 0;
	pStructItem = NULL;
	bReadOnly = false;
	bVisibility = true;
	color[0] = color[1] = color[2] = 0;
}
};
// 구조체의 정보를 담고 있는 구조체.
struct StructItem
{
	CString strName;					// 구조체의 이름.
	CString strFile;					// 선언된 파일이름.
	CString strDescription;				// 구조체의 설명.
	int iSize;							// 구조체의 사이즈.
	std::vector<VariableItem> varList;	// 변수 아이템의 리스트.
	bool bEnuObject;					// 본 구조체가 상위 부모 클래스가 EnuObject인지 확인		
	bool bEnuLink;						// 본 구조체가 상뷔 부모 클래스가 EnuLink인지 확인.
public:StructItem()
{
	iSize = 0;
	bEnuObject = false;
	bEnuLink = false;
}
};

struct arrayInfo
{
	int size;									// 변수 배열의 개수		ex) 100 
	int position;								// 배열 변수의 위치 정보	ex) 10
	wchar_t dimension[DEF_LABELTEXT_LEN];		// 문자열 배열 정보		ex) [10][10]

public:arrayInfo()
{
	size = 0;
	position = 0;
	wcscpy_s(dimension, L"");
}
};

struct VariableStruct
{
	wchar_t name[DEF_NAME_LEN];				// 변수 이름.
	int     type;							// 변수의 타입 (단일변수 또는 구조체)
	void*   pValue;							// 변수의 주소
	wchar_t strValue[DEF_MAXTEXT_LEN];		// 단일 변수에 대한 문자열 값
	arrayInfo array;						// 배열 정보
	StructItem* pStructItem;				// 구조체 선언정보
	VariableItem* pVariableItem;			// 변수 구조체 정보.
	int iSize;								// 변수의 메모리 사이즈
public:VariableStruct()
{
	wcscpy_s(name, L"");
	type = DEF_UNKNOWN;
	pValue = NULL;
	wcscpy_s(strValue, L"N/A");
	pStructItem = NULL;
	pVariableItem = NULL;
	iSize = 0;
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