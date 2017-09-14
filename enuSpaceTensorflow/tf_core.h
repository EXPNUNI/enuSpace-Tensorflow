#pragma once



#ifndef _TF_CORE_HEADER_
#define _TF_CORE_HEADER_

#include <string>
#include "include/json/json.h"


void* Create_ClientSession(std::string id, Json::Value pInputItem);
void* Create_Input(std::string id, Json::Value pInputItem);
void* Create_Input_Initializer(std::string id, Json::Value pInputItem);
void* Create_InputList(std::string id, Json::Value pInputItem);
void* Create_Operation(std::string id, Json::Value pInputItem);
void* Create_Output(std::string id, Json::Value pInputItem);
void* Create_Scope(std::string id, Json::Value pInputItem);
void* Create_Status(std::string id, Json::Value pInputItem);
void* Create_Tensor(std::string id, Json::Value pInputItem);
void* Create_FeedType(std::string id, Json::Value pInputItem);
void* Create_Const(std::string id, Json::Value pInputItem);
void* Create_Const_ex(std::string id, Json::Value pInputItem);

#endif