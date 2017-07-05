#pragma once


#ifndef _TF_CONTROL_FLOW_OPS_HEADER_
#define _TF_CONTROL_FLOW_OPS_HEADER_

#include <string>
#include "include/json/json.h"


void* Create_Abort(std::string id, Json::Value pInputItem);
void* Create_ControlTrigger(std::string id, Json::Value pInputItem);
void* Create_LoopCond(std::string id, Json::Value pInputItem);
void* Create_Merge(std::string id, Json::Value pInputItem);
void* Create_NextIteration(std::string id, Json::Value pInputItem);
void* Create_RefNextIteration(std::string id, Json::Value pInputItem);
void* Create_RefSelect(std::string id, Json::Value pInputItem);
void* Create_RefSwitch(std::string id, Json::Value pInputItem);
void* Create_Switch(std::string id, Json::Value pInputItem);

#endif