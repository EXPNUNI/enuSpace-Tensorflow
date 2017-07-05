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

struct DBDataStruct
{
	bool bLocal;
	void* pNode;
	CString tagid;
	CString sysvariable;
	CString variable;
	CString value;
	CString sysid;
	CString pagename;
	CString history;
	CString description;
	CString tablename;

	int itype;				// variable type
	void* pValue;
	void* pOldValue;

	int arraysize;
	CString dimension;

public:DBDataStruct()
{
	bLocal = true;
	pNode = nullptr;
	itype = DEF_UNKNOWN;
	pValue = nullptr;
	pOldValue = nullptr;
	arraysize = 0;
}
};

void SetArrayValue(std::string, void* pSrc, int iType, int iSize);
DBDataStruct* GetArrayValue(std::string strVariable);
void SetValue(std::string strVariable, double fValue);
double GetValue(std::string strVariable);
void PrintMessage(std::string strMessage);