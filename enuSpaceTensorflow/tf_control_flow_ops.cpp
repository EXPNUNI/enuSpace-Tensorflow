#include "stdafx.h"
#include "tf_control_flow_ops.h"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "include/json/json.h"

#include "GlobalHeader.h"
#include "tensorflow.h"
#include "utility_functions.h"
#include "enuSpaceToTensorflow.h"

void* Create_Abort(std::string id, Json::Value pInputItem) {
	Abort* pAbort = nullptr;
	Scope* pScope = nullptr;
	return pAbort;
}

void* Create_ControlTrigger(std::string id, Json::Value pInputItem) {
	ControlTrigger* pControlTrigger = nullptr;
	Scope* pScope = nullptr;
	return pControlTrigger;
}

void* Create_LoopCond(std::string id, Json::Value pInputItem) {
	LoopCond* pLoopCond = nullptr;
	Scope* pScope = nullptr;
	return pLoopCond;
}

void* Create_Merge(std::string id, Json::Value pInputItem) {
	Merge* pMerge = nullptr;
	Scope* pScope = nullptr;
	return pMerge;
}

void* Create_NextIteration(std::string id, Json::Value pInputItem) {
	NextIteration* pNextIteration = nullptr;
	Scope* pScope = nullptr;
	return pNextIteration;
}

void* Create_RefNextIteration(std::string id, Json::Value pInputItem) {
	RefNextIteration* pRefNextIteration = nullptr;
	Scope* pScope = nullptr;
	return pRefNextIteration;
}

void* Create_RefSelect(std::string id, Json::Value pInputItem) {
	RefSelect* pRefSelect = nullptr;
	Scope* pScope = nullptr;
	return pRefSelect;
}

void* Create_RefSwitch(std::string id, Json::Value pInputItem) {
	RefSwitch* pRefSwitch = nullptr;
	Scope* pScope = nullptr;
	return pRefSwitch;
}

void* Create_Switch(std::string id, Json::Value pInputItem) {
	Switch* pSwitch = nullptr;
	Scope* pScope = nullptr;
	return pSwitch;
}