#pragma once


#ifndef _TF_STATE_HEADER_
#define _TF_STATE_HEADER_

#include <string>
#include "include/json/json.h"


void* Create_Assign(std::string id, Json::Value pInputItem);
void* Create_AssignAdd(std::string id, Json::Value pInputItem);
void* Create_AssignSub(std::string id, Json::Value pInputItem);
void* Create_CountUpTo(std::string id, Json::Value pInputItem);
void* Create_DestroyTemporaryVariable(std::string id, Json::Value pInputItem);
void* Create_IsVariableInitialized(std::string id, Json::Value pInputItem);
void* Create_ScatterAdd(std::string id, Json::Value pInputItem);
void* Create_ScatterDiv(std::string id, Json::Value pInputItem);
void* Create_ScatterMul(std::string id, Json::Value pInputItem);
void* Create_ScatterNdAdd(std::string id, Json::Value pInputItem);
void* Create_ScatterNdSub(std::string id, Json::Value pInputItem);
void* Create_ScatterNdUpdate(std::string id, Json::Value pInputItem);
void* Create_ScatterSub(std::string id, Json::Value pInputItem);
void* Create_ScatterUpdate(std::string id, Json::Value pInputItem);
void* Create_TemporaryVariable(std::string id, Json::Value pInputItem);
void* Create_Variable(std::string id, Json::Value pInputItem);
#endif