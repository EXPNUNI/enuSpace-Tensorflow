#include "stdafx.h"
#include "tf_no.h"


#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "include/json/json.h"

#include "GlobalHeader.h"
#include "tensorflow.h"
#include "utility_functions.h"
#include "enuSpaceToTensorflow.h"

void* Create_NoOp(std::string id, Json::Value pInputItem) {
	NoOp* pNoOp = nullptr;
	Scope* pScope = nullptr;
	return pNoOp;
}