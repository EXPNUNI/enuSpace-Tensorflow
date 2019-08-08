#include "stdafx.h"
#include "EnuObj.h"
#include "GlobalHeader.h"



CString g_strDllPath;
HANDLE g_hConsole = NULL;
CPtrArray *g_enuObject = NULL;
double g_DT;


std::map<EnuObject*, Object_Info* > m_Object_MapList;
std::map<EnuObject*, Link_Info* > m_Link_MapList;
//////////////////////////////////////////////////////////////



std::map<std::string, ObjectInfo* > m_ObjectMapList;
std::map<std::string, FetchInfo* > m_RunMapList;
Scope* m_pScope = nullptr;								// set the current scope node pointer.

std::map<std::string, int>	m_SymbolList;
bool m_bShowDebugMessage = true;
bool m_bContinusLoop = false;
int m_iSimulationMode = DEF_MODE_EDIT;

FILE* m_FileData = NULL;

///////////////////////////////////////////////////////////////////////////////////////
