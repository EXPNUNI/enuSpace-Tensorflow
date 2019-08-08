#pragma once

#include <string>
#include <stdarg.h>  // For va_start, etc.
#include <memory>    // For std::unique_ptr
#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"
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

using namespace tensorflow;
using namespace std;

inline string trim_left(const string& str)
{
	int n = str.find_first_not_of(" \t\v\n");
	return n == string::npos ? str : str.substr(n, str.length());
}

inline string trim_right(const string& str)
{
	int n = str.find_last_not_of(" \t\v\n");
	return n == string::npos ? str : str.substr(0, n + 1);
}

inline string trim(const string& str)
{
	return trim_right(trim_left(str));
}



std::string string_format(const std::string fmt_str, ...);
bool GetArraryFromInitial(std::string strinitial, std::vector<double>& arrayvals, std::vector<int64>& arraydims);
bool GetArrayDimsFromShape(std::string strShape, std::vector<int64>& arraydims, std::vector<int64>& arrayslice);
bool GetDoubleVectorFromInitial(std::string strinitial, std::vector<double>& arrayvals);
bool GetFloatVectorFromInitial(std::string strinitial, std::vector<float>& arrayvals);
bool GetIntVectorFromInitial(std::string strinitial, std::vector<int>& arrayvals);
bool GetBoolVectorFromInitial(std::string strinitial, std::vector<bool>& arrayvals);
bool GetStringVectorFromInitial(std::string strinitial, std::vector<std::string>& arrayvals);

///////////////////////////////////////////////////////////////////////////////////////////////
DataType GetDatatypeFromInitial(std::string strinitial);
bool GetDatatypeSliceFromInitial(std::string strinitial, std::vector<DataType>& arrayvals);
bool GetArrayShapeFromInitial(std::string strinitial, std::vector<PartialTensorShape>& vec_PTS);
TensorShape GetShapeFromInitial(std::string strinitial);
PartialTensorShape GetPartialShapeFromInitial(std::string strinitial);

bool GetInt64VectorFromInitial(std::string strinitial, std::vector<int64>& arrayvals);

///////////////////////////////////////////////////////////////////////////////////////////
void* Create_StrToOutput(Scope& pScope, std::string strPinType, std::string strPinShape, std::string strPinInitial);
void* Create_StrToOutputList(Scope& pScope, std::string strPinType, std::string strPinShape, std::string strPinInitial);
Tensor* Create_StrToTensor(std::string strPinType, std::string strPinShape, std::string strPinInitial);
bool GetArrayDimsFromStrVal(std::string strVal, std::vector<int64>& arraydims, std::vector<int64>& arrayslice, std::vector<double>& arrayVal);
bool GetArrayDimsFromStrVal(std::string strVal, std::vector<int64>& arraydims, std::vector<int64>& arrayslice, std::vector<std::string>& arrayStrVal);
bool GetArrayDimsFromStrVal(std::string strVal, std::vector<int64>& arraydims, std::vector<int64>& arrayslice, std::vector<std::complex<double>>& arrayComplexVal);
bool isString(std::string strVal);

void* Create_BinaryToOutput(Scope& pScope, std::string strPinType, std::string strPinShape, FILE* pFile, int iPos);
Tensor* Create_BinaryToTensor(std::string strPinType, std::string strPinShape, FILE* pFile, int iPos);

void* Create_ArrayStrToOutput(Scope& pScope, std::string strPinType, std::string strPinShape, std::string strInitvalue);
Tensor* Create_ArrayStrToTensor(std::string strPinType, std::string strPinShape, std::string strInitvalue);
