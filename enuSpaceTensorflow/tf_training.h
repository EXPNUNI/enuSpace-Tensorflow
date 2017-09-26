#pragma once

#ifndef _TF_TRAINING_HEADER_
#define _TF_TRAINING_HEADER_

#include <string>
#include "include/json/json.h"


void* Create_ApplyAdadelta(std::string id, Json::Value pInputItem);
void* Create_ApplyAdagrad(std::string id, Json::Value pInputItem);
void* Create_ApplyAdagradDA(std::string id, Json::Value pInputItem);
void* Create_ApplyAdam(std::string id, Json::Value pInputItem);
void* Create_ApplyCenteredRMSProp(std::string id, Json::Value pInputItem);
void* Create_ApplyFtrl(std::string id, Json::Value pInputItem);
void* Create_ApplyGradientDescent(std::string id, Json::Value pInputItem);
void* Create_ApplyMomentum(std::string id, Json::Value pInputItem);
void* Create_ApplyProximalAdagrad(std::string id, Json::Value pInputItem);
void* Create_ApplyProximalGradientDescent(std::string id, Json::Value pInputItem);
void* Create_ApplyRMSProp(std::string id, Json::Value pInputItem);
void* Create_ResourceApplyAdadelta(std::string id, Json::Value pInputItem);
void* Create_ResourceApplyAdagrad(std::string id, Json::Value pInputItem);
void* Create_ResourceApplyAdagradDA(std::string id, Json::Value pInputItem);
void* Create_ResourceApplyAdam(std::string id, Json::Value pInputItem);
void* Create_ResourceApplyCenteredRMSProp(std::string id, Json::Value pInputItem);
void* Create_ResourceApplyFtrl(std::string id, Json::Value pInputItem);
void* Create_ResourceApplyGradientDescent(std::string id, Json::Value pInputItem);
void* Create_ResourceApplyMomentum(std::string id, Json::Value pInputItem);
void* Create_ResourceApplyProximalAdagrad(std::string id, Json::Value pInputItem);
void* Create_ResourceApplyProximalGradientDescent(std::string id, Json::Value pInputItem);
void* Create_ResourceApplyRMSProp(std::string id, Json::Value pInputItem);
void* Create_ResourceSparseApplyAdadelta(std::string id, Json::Value pInputItem);
void* Create_ResourceSparseApplyAdagrad(std::string id, Json::Value pInputItem);
void* Create_ResourceSparseApplyAdagradDA(std::string id, Json::Value pInputItem);
void* Create_ResourceSparseApplyCenteredRMSProp(std::string id, Json::Value pInputItem);
void* Create_ResourceSparseApplyFtrl(std::string id, Json::Value pInputItem);
void* Create_ResourceSparseApplyMomentum(std::string id, Json::Value pInputItem);
void* Create_ResourceSparseApplyProximalAdagrad(std::string id, Json::Value pInputItem);
void* Create_ResourceSparseApplyProximalGradientDescent(std::string id, Json::Value pInputItem);
void* Create_ResourceSparseApplyRMSProp(std::string id, Json::Value pInputItem);
void* Create_SparseApplyAdadelta(std::string id, Json::Value pInputItem);
void* Create_SparseApplyAdagrad(std::string id, Json::Value pInputItem);
void* Create_SparseApplyAdagradDA(std::string id, Json::Value pInputItem);
void* Create_SparseApplyCenteredRMSProp(std::string id, Json::Value pInputItem);
void* Create_SparseApplyFtrl(std::string id, Json::Value pInputItem);
void* Create_SparseApplyMomentum(std::string id, Json::Value pInputItem);
void* Create_SparseApplyProximalAdagrad(std::string id, Json::Value pInputItem);
void* Create_SparseApplyProximalGradientDescent(std::string id, Json::Value pInputItem);
void* Create_SparseApplyRMSProp(std::string id, Json::Value pInputItem);
void* Create_GradientDescentOptimizer(std::string id, Json::Value pInputItem);

#endif