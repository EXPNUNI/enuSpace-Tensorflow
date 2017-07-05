#pragma once

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/random_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/client/client_session.h"

#include "include/json/json.h"
#include "GlobalHeader.h"

bool Load_Tensorflow();
bool Init_Tensorflow(std::string logic_file, std::string page_name);
bool Task_Tensorflow();
bool Unload_Tensorflow();

ObjectInfo* LookupFromObjectMap(std::string strid);
ObjectInfo* AddObjectMap(void* pCreate, std::string id, int iSymbol, std::string type_name, Json::Value pInputItem);
FetchInfo* AddRunObjectMap(ObjectInfo* pRunObj);
void ObjectMapClear();

bool AddOutputInfo(ObjectInfo* pObjectInfo, void* pOutput, int iType, std::string strname);
OutputInfo* LookupFromOutputMap(ObjectInfo* pObjectInfo, std::string strname);

void AddSymbolList();
int GetSymbolType(std::string strSymbolName);
void* Create_Symbol(int iSymbol, std::string id, Json::Value pInputItem);

