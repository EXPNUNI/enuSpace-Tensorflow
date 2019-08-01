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
// Copyright (C) ENU Corporation, �̿����ֽ�ȸ��, ENU Co., Ltd
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

#define TASK_TYPE_UNKNOWN						0			// [v3.5] mod : Task�� Ÿ�� ���� (�˼����� Ÿ��)
#define TASK_TYPE_PROCESS						1			// [v3.5] mod : Task�� Ÿ�� ����	(����ó���� Ÿ��)
#define TASK_TYPE_FLOW_COMPONENT_TOTAL			2			// [v3.5] mod : Task�� Ÿ�� ����	(FLOW ������Ʈ ��ü ���� Ÿ��)
#define TASK_TYPE_FLOW_COMPONENT_PAGE			3			// [v3.5] add : Task�� Ÿ�� ����	(FLOW������Ʈ ������ ���� Ÿ��)
#define TASK_TYPE_FUNCTION_COMPONENT			4			// [v3.5] add : Task�� Ÿ�� ���� (�Լ��� ������Ʈ Ÿ��)



#define DEF_NAME_LEN							64
#define DEF_MAXTEXT_LEN							2048
#define DEF_LABELTEXT_LEN						1024

#include <vector>

struct VariableItem
{
	int iType;							// ������ Ÿ�� (int, float, bool, double, string, struct)
	CString strType;					// ������ Ÿ�� ���ڿ� �̸�
	int iTotalSize;						// ������ ������(�迭 ����)
	int iUnitSize;						// �����Ѱ��� ������.
	CString strName;					// ������ �̸�
	CString strInitial;					// ������ �ʱⰪ
	void* pStructItem;					// Ÿ�������� ����ü�ΰ��, �ش� ����ü ������ ����Ű�� ����.
	CString strDims;					// ���ڿ��� DIMENSION ����.
	std::vector<int> Dims;				// �迭�� DIMENSION ����.
	CString strDescription;				// (UI)������ ����
	CString strUnit;					// (UI)������ ����
	bool bReadOnly;						// (UI)�б���
	bool bVisibility;					// (UI)���÷��� ���
	byte color[3];						// (UI)����
	CString min;						// (UI)�ּҰ�
	CString max;						// (UI)�ִ밪
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
// ����ü�� ������ ��� �ִ� ����ü.
struct StructItem
{
	CString strName;					// ����ü�� �̸�.
	CString strFile;					// ����� �����̸�.
	CString strDescription;				// ����ü�� ����.
	int iSize;							// ����ü�� ������.
	std::vector<VariableItem> varList;	// ���� �������� ����Ʈ.
	bool bEnuObject;					// �� ����ü�� ���� �θ� Ŭ������ EnuObject���� Ȯ��		
	bool bEnuLink;						// �� ����ü�� ��� �θ� Ŭ������ EnuLink���� Ȯ��.
public:StructItem()
{
	iSize = 0;
	bEnuObject = false;
	bEnuLink = false;
}
};

struct arrayInfo
{
	int size;									// ���� �迭�� ����		ex) 100 
	int position;								// �迭 ������ ��ġ ����	ex) 10
	wchar_t dimension[DEF_LABELTEXT_LEN];		// ���ڿ� �迭 ����		ex) [10][10]

public:arrayInfo()
{
	size = 0;
	position = 0;
	wcscpy_s(dimension, L"");
}
};

struct VariableStruct
{
	wchar_t name[DEF_NAME_LEN];				// ���� �̸�.
	int     type;							// ������ Ÿ�� (���Ϻ��� �Ǵ� ����ü)
	void*   pValue;							// ������ �ּ�
	wchar_t strValue[DEF_MAXTEXT_LEN];		// ���� ������ ���� ���ڿ� ��
	arrayInfo array;						// �迭 ����
	StructItem* pStructItem;				// ����ü ��������
	VariableItem* pVariableItem;			// ���� ����ü ����.
	int iSize;								// ������ �޸� ������
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
void SetReShapeArrayValue(std::string strVariable, void* pSrc, int iType, int iSize);		// enuSpace�� �޸� ����� �ٸ����, enuSpace�� �޸𸮸� ���Ҵ��� �� �����ϴ� �Լ�
double GetArrayValue(std::string strVariable);
void SetValue(std::string strVariable, double fValue);
double GetValue(std::string strVariable);
void PrintMessage(std::string strMessage, std::string strID = "");
int GetArrayIndexFromDimension(CString strOrgDim, CString strDimension);