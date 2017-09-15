#include "stdafx.h"
#include "utility_functions.h"


#include <string>
#include <cstdarg>
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow.h"
#include "enuSpaceToTensorflow.h"


std::string string_format(const std::string fmt_str, ...) 
{
	int final_n, n = ((int)fmt_str.size()) * 2; /* Reserve two times as much as the length of the fmt_str */
	std::string str;
	std::unique_ptr<char[]> formatted;
	va_list ap;
	while (1) {
		formatted.reset(new char[n]); /* Wrap the plain char array into the unique_ptr */
		strcpy(&formatted[0], fmt_str.c_str());
		va_start(ap, fmt_str);
		final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
		va_end(ap);
		if (final_n < 0 || final_n >= n)
			n += abs(final_n - n + 1);
		else
			break;
	}
	return std::string(formatted.get());
}

// strPinInitial = "{3.f, 2.f}"; 2by1 => 1 => 2
// strPinInitial = "{{3.f, 2.f}, {-1.f, 0.f}}";	2by2 => 
// strPinInitial = "{{3.f, 2.f}, {-1.f, 0.f}, {3.f, 2.f}}";   3by2 => 1,3
// strPinInitial = "{{3.f, 2.f, 1.f}, {-1.f, 0.f, 0.f}, {3.f, 2.f, 1.f}}";   3by3 => 1,3
// strPinInitial = "{{{3.f, 2.f}, {-1.f, 0.f}, {3.f, 2.f}}, {{3.f, 2.f}, {-1.f, 0.f}, {3.f, 2.f}}}";   2by3by2 => 1,2,3
bool GetArraryFromInitial(std::string strinitial, std::vector<double>& arrayvals, std::vector<int64>& arraydims)
{
	std::string val;
	bool bdims[10] = { false,false,false,false,false,false,false,false,false,false};
	int64 idims[10] = {0,0,0,0,0,0,0,0,0,0};
	int ibrace = 0;

	for (std::string::size_type i = 0; i < strinitial.size(); ++i)
	{
		if (strinitial[i] != ' ')
		{
			if (strinitial[i] == '{')
			{
				if (bdims[ibrace] == true)
					return false;

				bdims[ibrace] = true;
				ibrace++;
			}
			else if (strinitial[i] == '}')
			{
				ibrace--;
				idims[ibrace] = idims[ibrace] + 1;
				if (bdims[ibrace] == false)
					return false;

				bdims[ibrace] = false;
				if (val != "")
				{
					arrayvals.push_back(std::stod(val));
					val = "";
				}
			}
			else if (strinitial[i] == ',')
			{
				idims[ibrace-1] = idims[ibrace-1] + 1;
				if (val != "")
				{
					arrayvals.push_back(std::stod(val));
					val = "";
				}
			}
			else
			{
				val = val + strinitial[i];
			}
		}
	}

	arraydims.push_back(idims[0]);

	for (int j = 0; j < 9; j++)
	{
		if (idims[j + 1] != 0)
		{
			int64 ival = idims[j + 1] / idims[j];
			if (ival * idims[j] != idims[j + 1])
				return false;
			arraydims.push_back(ival);
		}
	}
	
	return true;
}

bool GetArrayDimsFromShape(std::string strShape, std::vector<int64>& arraydims, std::vector<int64>& arrayslice)
{
	strShape = trim(strShape);
	if (strShape.length() > 0)
	{
		int64 islice = 0;
		std::string val;
		for (std::string::size_type i = 0; i < strShape.size(); i++)
		{
			if (strShape[0] == '[')
			{
				if (strShape[i] == '[')
				{
					val = "";
				}
				else if (strShape[i] == ']')
				{
					if (val != "")
					{
						arraydims.push_back(std::stoi(val));
						islice++;
					}
				}
				else
				{
					val = val + strShape[i];
				}
			}
			else
			{
				if (strShape[i] == '{')
				{
					val = "";
				}
				else if (strShape[i] == '}' || strShape[i] == ',')
				{
					if (val !="")
					{
						arraydims.push_back(std::stoi(val));
						islice++;
					}
					
				}
				else
				{
					val = val + strShape[i];
				}
			}
		}

		arrayslice.push_back(islice);
	}

	return true;
}

bool GetDoubleVectorFromInitial(std::string strinitial, std::vector<double>& arrayvals)
{
	std::string val;
	for (std::string::size_type i = 0; i < strinitial.size(); i++)
	{
		if (strinitial[i] == ';' || strinitial[i] == ',')
		{
			arrayvals.push_back(std::stod(val));
			val = "";
		}
		else
		{
			if (strinitial[i] != '{' &&  strinitial[i] != '}')
				val = val + strinitial[i];
		}
	}

	if (val.length() > 0)
		arrayvals.push_back(std::stod(val));
	return true;
}

bool GetFloatVectorFromInitial(std::string strinitial, std::vector<float>& arrayvals)
{
	std::string val;
	for (std::string::size_type i = 0; i < strinitial.size(); i++)
	{
		if (strinitial[i] == ';' || strinitial[i] == ',')
		{
			arrayvals.push_back(std::stof(val));
			val = "";
		}
		else
		{
			if (strinitial[i] != '{' &&  strinitial[i] != '}')
				val = val + strinitial[i];
		}
	}

	if (val.length() > 0)
		arrayvals.push_back(std::stof(val));
	return true;
}

bool GetIntVectorFromInitial(std::string strinitial, std::vector<int>& arrayvals)
{
	std::string val;
	for (std::string::size_type i = 0; i < strinitial.size(); i++)
	{
		if (strinitial[i] == ';'|| strinitial[i] == ',')
		{
			arrayvals.push_back(std::stoi(val));
			val = "";
		}
		else
		{
			if (strinitial[i] != '{' &&  strinitial[i] !='}')
				val = val + strinitial[i];
		
		}
	}

	if (val.length() > 0)
		arrayvals.push_back(std::stoi(val));
	return true;
}

bool GetInt64VectorFromInitial(std::string strinitial, std::vector<int64>& arrayvals)
{
	std::string val;
	for (std::string::size_type i = 0; i < strinitial.size(); i++)
	{
		if (strinitial[i] == ';' || strinitial[i] == ',')
		{
			arrayvals.push_back(std::stoi(val));
			val = "";
		}
		else
		{
			val = val + strinitial[i];
		}
	}

	if (val.length() > 0)
		arrayvals.push_back(std::stoi(val));
	return true;
}

bool GetBoolVectorFromInitial(std::string strinitial, std::vector<bool>& arrayvals)
{
	std::string val;
	for (std::string::size_type i = 0; i < strinitial.size(); i++)
	{
		if (strinitial[i] == ';' || strinitial[i] == ',')
		{
			if (val == "1" || val == "true" || val == "TRUE" || val == "True")
				arrayvals.push_back(true);
			else
				arrayvals.push_back(false);
			val = "";
		}
		else
		{
			val = val + strinitial[i];
		}
	}

	if (val.length() > 0)
	{
		if (val == "1" || val == "true" || val == "TRUE" || val == "True")
			arrayvals.push_back(true);
		else
			arrayvals.push_back(false);
	}
	return true;
}

bool GetStringVectorFromInitial(std::string strinitial, std::vector<std::string>& arrayvals)
{
	std::string val;
	for (std::string::size_type i = 0; i < strinitial.size(); i++)
	{
		if (strinitial[i] == ';' )
		{
			trim(val);
			arrayvals.push_back(val);
			val = "";
		}
		else
		{
			val = val + strinitial[i];
		}
	}

	if (val.length() > 0)
	{
		trim(val);
		arrayvals.push_back(val);
	}
return true;
}
DataType GetDatatypeFromInitial(std::string strinitial)
{

	strinitial = trim(strinitial);
	DataType dt;
	if (strinitial == "DT_INVALID")
		dt = DT_INVALID;
	else if (strinitial == "DT_FLOAT")
		dt = DT_FLOAT;
	else if (strinitial == "DT_DOUBLE")
		dt = DT_DOUBLE;
	else if (strinitial == "DT_INT32")
		dt = DT_INT32;
	else if (strinitial == "DT_UINT8")
		dt = DT_UINT8;
	else if (strinitial == "DT_INT16")
		dt = DT_INT16;
	else if (strinitial == "DT_INT8")
		dt = DT_INT8;
	else if (strinitial == "DT_STRING")
		dt = DT_STRING;
	else if (strinitial == "DT_COMPLEX64")
		dt = DT_COMPLEX64;
	else if (strinitial == "DT_INT64")
		dt = DT_INT64;
	else if (strinitial == "DT_BOOL")
		dt = DT_BOOL;
	else if (strinitial == "DT_QINT8")
		dt = DT_QINT8;
	else if (strinitial == "DT_QUINT8")
		dt = DT_QUINT8;
	else if (strinitial == "DT_QINT32")
		dt = DT_QINT32;
	else if (strinitial == "DT_BFLOAT16")
		dt = DT_BFLOAT16;
	else if (strinitial == "DT_QINT16")
		dt = DT_QINT16;
	else if (strinitial == "DT_QUINT16")
		dt = DT_QUINT16;
	else if (strinitial == "DT_UINT16")
		dt = DT_UINT16;
	else if (strinitial == "DT_COMPLEX128")
		dt = DT_COMPLEX128;
	else if (strinitial == "DT_HALF")
		dt = DT_HALF;
	else if (strinitial == "DT_RESOURCE")
		dt = DT_RESOURCE;
	else if (strinitial == "DT_FLOAT_REF")
		dt = DT_FLOAT_REF;
	else if (strinitial == "DT_DOUBLE_REF")
		dt = DT_DOUBLE_REF;
	else if (strinitial == "DT_INT32_REF")
		dt = DT_INT32_REF;
	else if (strinitial == "DT_UINT8_REF")
		dt = DT_UINT8_REF;
	else if (strinitial == "DT_INT16_REF")
		dt = DT_INT16_REF;
	else if (strinitial == "DT_INT8_REF")
		dt = DT_INT8_REF;
	else if (strinitial == "DT_STRING_REF")
		dt = DT_STRING_REF;
	else if (strinitial == "DT_COMPLEX64_REF")
		dt = DT_COMPLEX64_REF;
	else if (strinitial == "DT_INT64_REF")
		dt = DT_INT64_REF;
	else if (strinitial == "DT_BOOL_REF")
		dt = DT_BOOL_REF;
	else if (strinitial == "DT_QINT8_REF")
		dt = DT_QINT8_REF;
	else if (strinitial == "DT_QUINT8_REF")
		dt = DT_QUINT8_REF;
	else if (strinitial == "DT_QINT32_REF")
		dt = DT_QINT32_REF;
	else if (strinitial == "DT_BFLOAT16_REF")
		dt = DT_BFLOAT16_REF;
	else if (strinitial == "DT_QINT16_REF")
		dt = DT_QINT16_REF;
	else if (strinitial == "DT_QUINT16_REF")
		dt = DT_QUINT16_REF;
	else if (strinitial == "DT_UINT16_REF")
		dt = DT_UINT16_REF;
	else if (strinitial == "DT_COMPLEX128_REF")
		dt = DT_COMPLEX128_REF;
	else if (strinitial == "DT_HALF_REF")
		dt = DT_HALF_REF;
	else if (strinitial == "DT_RESOURCE_REF")
		dt = DT_RESOURCE_REF;
	else if (strinitial == "DataType_INT_MIN_SENTINEL_DO_NOT_USE_")
		dt = DataType_INT_MIN_SENTINEL_DO_NOT_USE_;
	else if (strinitial == "DataType_INT_MAX_SENTINEL_DO_NOT_USE_")
		dt = DataType_INT_MAX_SENTINEL_DO_NOT_USE_;
	else
		dt = DT_INVALID;
	return dt;
}

bool GetDatatypeSliceFromInitial(std::string strinitial, std::vector<DataType>& arrayvals)
{
	strinitial = trim(strinitial);
	std::string val;
	for (std::string::size_type i = 0; i < strinitial.size(); i++)
	{
		if (strinitial[i] == ';' || strinitial[i] == ',')
		{
			DataType dtype;
			dtype = GetDatatypeFromInitial(val);
			if (dtype !=DT_INVALID)
				arrayvals.push_back(dtype);
			val = "";
		}
		else
		{
			if (strinitial[i] != '{' && strinitial[i] != '}')
			{
				val = val + strinitial[i];
			}
			
		}
	}

	if (val.length() > 0)
	{
		DataType dtype;
		dtype = GetDatatypeFromInitial(val);
		if (dtype != DT_INVALID)
			arrayvals.push_back(dtype);
	}
	return true;
}

bool GetArrayShapeFromInitial(std::string strinitial, std::vector<PartialTensorShape>& vec_PTS)
{
	//3;2;
	//3;2;3;2;
	//2,2,3;2,2,3;
	std::vector<int64> arraydims;
	std::string val;
	strinitial= trim(strinitial);
	int64 iDimSize = 1;
	bool bOpen = false;
	int iOpen = 0; //{갯수 판단하여 다음 배열인지 체크한다.
	
	for (std::string::size_type i = 0; i < strinitial.size(); i++)
	{
		//[2][2];[2][2];
		//{2,2},{2,2};
		if (strinitial[0] == '[')
		{
			if (strinitial[i] == '[')
			{
				val = "";
			}
			else if (strinitial[i] == ']')
			{
				arraydims.push_back(std::stoi(val));
			}
			else if (strinitial[i] == ',' || strinitial[i] == ';')
			{
				PartialTensorShape partts(arraydims);
				vec_PTS.push_back(partts);
				arraydims.clear();
				val = "";
			}
			else
			{
				val = val + strinitial[i];
			}
		}
		else
		{
			if (strinitial[i] == ',')
			{
				if (bOpen)
				{
					arraydims.push_back(stoll(val));
					val = "";
				}
				else
				{
					val = "";
				}

			}
			else if (strinitial[i] == '{')
			{
				bOpen = true;
				val = "";
				
			}
			else if (strinitial[i] == '}')
			{
				if (val != "")
				{
					arraydims.push_back(stoll(val));
					val = "";
					std::string msg = string_format("test %d", arraydims.size());
					PrintMessage(msg);
					PartialTensorShape partts(arraydims);
					vec_PTS.push_back(partts);
					arraydims.clear();
				}
				
				bOpen = false;
				
				

			}
			else
			{
				val = val + strinitial[i];
			}
		}
	}

	if (bOpen == true)
	{
		vec_PTS.clear();
	}
	/*
	if (val.length() > 0)  //사용자가 마지막 구분자를 닫지않았다.
	{
		if (arraydims.size() != 0)
		{
			if ()
			{
			}
			arraydims.push_back(stoll(val));
			PartialTensorShape partts(arraydims);
			vec_PTS.push_back(partts);
			arraydims.clear();
		}
	}
	*/
	int isize = vec_PTS.size();

	std::string msg = string_format("vec_pts %d", isize);
	PrintMessage(msg);
	arraydims.clear();

	return 	true;
}

TensorShape GetShapeFromInitial(std::string strinitial)
{
	TensorShape tempTS;
	std::string val;
	int64 iDimSize = 1;
	//{2, 2, 3}
	for (std::string::size_type i = 0; i < strinitial.size(); i++)
	{
		if (strinitial[0] == '[')
		{
			if (strinitial[i] == '[')
			{
				val = "";
			}
			else if (strinitial[i] == ']')
			{
				if (val != "")
				{
					iDimSize = iDimSize * stoll(val);
					val = "";
				}
			}
			else if (strinitial[i] == ';')
			{
				if (val != "")
				{
					iDimSize = iDimSize * stoll(val);
					val = "";
				}
			}
			else
			{
				val = val + strinitial[i];
			}
		}
		else
		{
			if (strinitial[i] == ';' || strinitial[i] == ',')
			{
				if (val != "")
				{
					iDimSize = iDimSize * stoll(val);
					val = "";
				}
				
			}
			else if (strinitial[i] == '}')
			{
				if (val != "")
				{
					iDimSize = iDimSize * stoll(val);
					val = "";
				}
			}
			else if (strinitial[i] == '{')
			{
				val = "";
			}
			else
			{
				val = val + strinitial[i];
			}
		}
	}

	tempTS.AddDim(iDimSize);
	return tempTS;
}
PartialTensorShape GetPartialShapeFromInitial(std::string strinitial)
{
	std::string val;
	std::vector<int64> arraydims;
	for (std::string::size_type i = 0; i < strinitial.size(); i++)
	{
		if (strinitial[0] == '[')
		{
			if (strinitial[i] == '[')
			{
				val = "";
			}
			else if (strinitial[i] == ']')
			{
				if (val!="")
				{
					arraydims.push_back(stoll(val));
					val = "";
				}
			}
			else if (strinitial[i] == ';')
			{
				if (val !="")
				{
					arraydims.push_back(stoll(val));
					val = "";
				}
			}
			else
			{
				val = val + strinitial[i];
			}
		}
		else
		{
			if (strinitial[i] == ';' || strinitial[i] == ',')
			{
				if (val != "")
				{
					arraydims.push_back(stoll(val));
					val = "";
				}				
			}
			else if (strinitial[i] == '}')
			{
				if (val != "")
				{
					arraydims.push_back(stoll(val));
					val = "";
				}
			}
			else if (strinitial[i] == '{')
			{
				val = "";
			}
			else
			{
				val = val + strinitial[i];
			}
		}
	}

	if (val.length() > 0)
	{
		arraydims.push_back(stoll(val));
		val = "";
	}
	PartialTensorShape tempTS(arraydims);
	arraydims.clear();
	
	return tempTS;
}

void* Create_StrToOutputList(Scope& pScope, std::string strPinType, std::string strPinShape, std::string strPinInitial)
{
	OutputList* pOutputList = new OutputList();
	Output* pOutput = nullptr;
	Tensor* pTensor = nullptr;
	std::vector<std::string> arrayStrVal;
	std::string val = "";
	std::string strPinLocalType = strPinType;
	int iDim = 0;
	DataType itype = DT_INVALID;
	if (strPinType != "" &&  strPinType != "auto")
	{
		itype = GetDatatypeFromInitial(strPinType);
		if (itype == DT_INVALID)
		{
			if (strPinType != "double") itype = DT_DOUBLE;
			else if (strPinType != "float") itype = DT_FLOAT;
			else if (strPinType != "int") itype = DT_INT32;
			else if (strPinType != "bool") itype = DT_BOOL;
			else if (strPinType != "string") itype = DT_STRING;
			else
			{
				std::string msg = string_format("error : DataType (%s).", strPinType.c_str());
				return pOutputList;
			}
		}
	}
	else
	{
		if (!strPinInitial.empty())
		{
			if (strPinInitial.find('\"') != -1)
				strPinLocalType = "DT_STRING";
			else if (strPinInitial.find('j') != -1 || strPinInitial.find('J') != -1)
				strPinLocalType = "DT_COMPLEX128";
			else if (strPinInitial.find('f') != -1)
				strPinLocalType = "DT_FLOAT";
			else if (strPinInitial.find('.') != -1 || strPinInitial.find('E') != -1 || strPinInitial.find('e') != -1)
				strPinLocalType = "DT_DOUBLE";
			else
				strPinLocalType = "DT_INT32";
		}
	}

	for (std::string::size_type i = 0; i < strPinInitial.size(); i++)
	{
		if (strPinInitial[i] == '{')
		{
			iDim++;
			val = val + strPinInitial[i];
		}
		else if (strPinInitial[i] == '}')
		{
			iDim--;
			val = val + strPinInitial[i];
		}
		else if (strPinInitial[i] == ',' || strPinInitial[i] == ';')
		{
			if (iDim == 0)
			{
				if (val != "")
				{
					arrayStrVal.push_back(val.c_str());
					val = "";
				}
			}
			else
			{
				val = val + strPinInitial[i];
			}
		}
		else
		{
			if (strPinInitial[i] != '\n' && strPinInitial[i] != '\r')
				val = val + strPinInitial[i];
		}
	}
	if (iDim == 0)
	{
		if (val != "")
		{
			arrayStrVal.push_back(val.c_str());
			val = "";
		}
	}
	for (std::string::size_type i = 0; i < arrayStrVal.size(); i++)
	{
		pOutput = ((Output*)Create_StrToOutput(pScope, strPinLocalType, strPinShape, arrayStrVal[i]));
		pOutputList->push_back(*pOutput);
		delete pOutput;
	}

	return pOutputList;
}
Tensor* Create_StrToTensor(std::string strPinType, std::string strPinShape, std::string strPinInitial)
{
	Tensor* pTensor = nullptr;
	DataType itype = DT_INVALID;

	std::vector<int64> array_slice;
	std::vector<int64> arraydims;
	std::vector<double> arrayVal;
	std::vector<std::string> arrayStrVal;
	std::vector<std::complex<double>> arrayComplexVal;
	if (strPinType != "" &&  strPinType != "auto")
	{
		itype = GetDatatypeFromInitial(strPinType);
		if (itype == DT_INVALID)
		{
			if (strPinType != "double") itype = DT_DOUBLE;
			else if (strPinType != "float") itype = DT_FLOAT;
			else if (strPinType != "int") itype = DT_INT32;
			else if (strPinType != "bool") itype = DT_BOOL;
			else if (strPinType != "string") itype = DT_STRING;
			else
			{
				std::string msg = string_format("error : DataType (%s).", strPinType.c_str());
				return pTensor;
			}
		}
	}
	else
	{
		if (!strPinInitial.empty())
		{
			if (strPinInitial.find('\"') != -1)
				itype = DT_STRING;
			else if (strPinInitial.find('j') != -1 || strPinInitial.find('J') != -1)
				itype = DT_COMPLEX128;
			else if (strPinInitial.find('f') != -1)
				itype = DT_FLOAT;
			else if (strPinInitial.find('.') != -1 || strPinInitial.find('E') != -1 || strPinInitial.find('e') != -1)
				itype = DT_DOUBLE;
			else
				itype = DT_INT32;
		}
	}
	if (itype == DT_STRING)
	{
		GetArrayDimsFromStrVal(strPinInitial, arraydims, array_slice, arrayStrVal);
	}
	else if (itype == DT_COMPLEX64 || itype == DT_COMPLEX128)
	{
		GetArrayDimsFromStrVal(strPinInitial, arraydims, array_slice, arrayComplexVal);
	}
	else
	{
		GetArrayDimsFromStrVal(strPinInitial, arraydims, array_slice, arrayVal);
	}
	switch (itype)
	{
	case DT_DOUBLE:
	{
		std::vector<double> arrayvals;
		arrayvals = arrayVal;

		gtl::ArraySlice< int64 > arraySlice(arraydims);
		pTensor = new Tensor(DT_DOUBLE, TensorShape(arraySlice));

		int i = 0;
		for (std::vector<double>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
		{
			pTensor->flat<double>()(i) = *it;
			i++;
		}
		arraySlice.clear();
		arrayvals.clear();
	}
	break;
	case DT_COMPLEX64:
	{
		std::vector<complex64> arrayvals;
		std::complex<float> cfVal;
		for (unsigned int i = 0; i < arrayComplexVal.size(); i++)
		{
			cfVal.real((float)(arrayComplexVal[i].real()));
			cfVal.imag((float)(arrayComplexVal[i].imag()));
			arrayvals.push_back(cfVal);
		}

		gtl::ArraySlice< int64 > arraySlice(arraydims);
		pTensor = new Tensor(DT_COMPLEX64, TensorShape(arraySlice));

		int i = 0;
		for (std::vector<complex64>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
		{
			pTensor->flat<complex64>()(i) = *it;
			i++;
		}
		arraySlice.clear();
		arrayvals.clear();
	}
	break;
	case DT_COMPLEX128:
	{
		std::vector<complex128> arrayvals;
		arrayvals = arrayComplexVal;

		gtl::ArraySlice< int64 > arraySlice(arraydims);
		pTensor = new Tensor(DT_COMPLEX128, TensorShape(arraySlice));

		int i = 0;
		for (std::vector<complex128>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
		{
			pTensor->flat<complex128>()(i) = *it;
			i++;
		}
		arraySlice.clear();
		arrayvals.clear();
	}
	break;
	case DT_FLOAT:
	{
		std::vector<float> arrayvals;
		for (unsigned int i = 0; i < arrayVal.size(); i++)
		{
			arrayvals.push_back((float)arrayVal[i]);
		}

		gtl::ArraySlice< int64 > arraySlice(arraydims);
		pTensor = new Tensor(DT_FLOAT, TensorShape(arraySlice));

		int i = 0;
		for (std::vector<float>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
		{
			pTensor->flat<float>()(i) = *it;
			i++;
		}
		arraySlice.clear();
		arrayvals.clear();
	}
	break;
	case DT_BFLOAT16:
	{
		std::vector<bfloat16> arrayvals;
		for (unsigned int i = 0; i < arrayVal.size(); i++)
		{
			arrayvals.push_back((bfloat16)arrayVal[i]);
		}

		gtl::ArraySlice< int64 > arraySlice(arraydims);
		pTensor = new Tensor(DT_BFLOAT16, TensorShape(arraySlice));

		int i = 0;
		for (std::vector<bfloat16>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
		{
			pTensor->flat<bfloat16>()(i) = *it;
			i++;
		}
		arraySlice.clear();
		arrayvals.clear();
	}
	break;
	case DT_INT8:
	{
		std::vector<int8> arrayvals;

		for (unsigned int i = 0; i < arrayVal.size(); i++)
		{
			arrayvals.push_back((int8)arrayVal[i]);
		}

		gtl::ArraySlice< int64 > arraySlice(arraydims);
		pTensor = new Tensor(DT_INT8, TensorShape(arraySlice));

		int i = 0;
		for (std::vector<int8>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
		{
			pTensor->flat<int8>()(i) = *it;
			i++;
		}
		arraySlice.clear();
		arrayvals.clear();
	}
	break;
	case DT_INT16:
	{
		std::vector<int16> arrayvals;

		for (unsigned int i = 0; i < arrayVal.size(); i++)
		{
			arrayvals.push_back((int16)arrayVal[i]);
		}

		gtl::ArraySlice< int64 > arraySlice(arraydims);
		pTensor = new Tensor(DT_INT16, TensorShape(arraySlice));

		int i = 0;
		for (std::vector<int16>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
		{
			pTensor->flat<int16>()(i) = *it;
			i++;
		}
		arraySlice.clear();
		arrayvals.clear();
	}
	break;
	case DT_UINT8:
	{
		std::vector<uint8> arrayvals;

		for (unsigned int i = 0; i < arrayVal.size(); i++)
		{
			arrayvals.push_back((uint8)arrayVal[i]);
		}

		gtl::ArraySlice< int64 > arraySlice(arraydims);
		pTensor = new Tensor(DT_INT8, TensorShape(arraySlice));

		int i = 0;
		for (std::vector<uint8>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
		{
			pTensor->flat<uint8>()(i) = *it;
			i++;
		}
		arraySlice.clear();
		arrayvals.clear();
	}
	break;
	case DT_UINT16:
	{
		std::vector<uint16> arrayvals;

		for (unsigned int i = 0; i < arrayVal.size(); i++)
		{
			arrayvals.push_back((uint16)arrayVal[i]);
		}

		gtl::ArraySlice< int64 > arraySlice(arraydims);
		pTensor = new Tensor(DT_INT16, TensorShape(arraySlice));

		int i = 0;
		for (std::vector<uint16>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
		{
			pTensor->flat<uint16>()(i) = *it;
			i++;
		}
		arraySlice.clear();
		arrayvals.clear();
	}
	break;
	case DT_INT32:
	{
		std::vector<int32_t> arrayvals;

		for (unsigned int i = 0; i < arrayVal.size(); i++)
		{
			arrayvals.push_back((int32_t)arrayVal[i]);
		}

		gtl::ArraySlice< int64 > arraySlice(arraydims);
		pTensor = new Tensor(DT_INT32, TensorShape(arraySlice));

		int i = 0;
		for (std::vector<int32_t>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
		{
			pTensor->flat<int32_t>()(i) = *it;
			i++;
		}
		arraySlice.clear();
		arrayvals.clear();
	}
	break;
	case DT_INT64:
	{
		std::vector<int64_t> arrayvals;

		for (unsigned int i = 0; i < arrayVal.size(); i++)
		{
			arrayvals.push_back((int64_t)arrayVal[i]);
		}

		gtl::ArraySlice< int64 > arraySlice(arraydims);
		pTensor = new Tensor(DT_INT64, TensorShape(arraySlice));

		int i = 0;
		for (std::vector<int64_t>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
		{
			pTensor->flat<int64_t>()(i) = *it;
			i++;
		}
		arraySlice.clear();
		arrayvals.clear();
	}
	break;
	case DT_QINT8:
	{
		std::vector<qint8> arrayvals;

		for (unsigned int i = 0; i < arrayVal.size(); i++)
		{
			arrayvals.push_back((int8_t)arrayVal[i]);
		}

		gtl::ArraySlice< int64 > arraySlice(arraydims);
		pTensor = new Tensor(DT_QINT8, TensorShape(arraySlice));

		int i = 0;
		for (std::vector<qint8>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
		{
			pTensor->flat<qint8>()(i) = *it;
			i++;
		}
		arraySlice.clear();
		arrayvals.clear();
	}
	break;
	case DT_QINT16:
	{
		std::vector<qint16> arrayvals;

		for (unsigned int i = 0; i < arrayVal.size(); i++)
		{
			arrayvals.push_back((qint16)arrayVal[i]);
		}

		gtl::ArraySlice< int64 > arraySlice(arraydims);
		pTensor = new Tensor(DT_QINT16, TensorShape(arraySlice));

		int i = 0;
		for (std::vector<qint16>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
		{
			pTensor->flat<qint16>()(i) = *it;
			i++;
		}
		arraySlice.clear();
		arrayvals.clear();
	}
	break;
	case DT_QINT32:
	{
		std::vector<qint32> arrayvals;

		for (unsigned int i = 0; i < arrayVal.size(); i++)
		{
			arrayvals.push_back((qint32)((float)arrayVal[i]));
		}

		gtl::ArraySlice< int64 > arraySlice(arraydims);
		pTensor = new Tensor(DT_QINT32, TensorShape(arraySlice));

		int i = 0;
		for (std::vector<qint32>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
		{
			pTensor->flat<qint32>()(i) = *it;
			i++;
		}
		arraySlice.clear();
		arrayvals.clear();
	}
	break;
	case DT_QUINT8:
	{
		std::vector<quint8> arrayvals;

		for (unsigned int i = 0; i < arrayVal.size(); i++)
		{
			arrayvals.push_back((quint8)arrayVal[i]);
		}

		gtl::ArraySlice< int64 > arraySlice(arraydims);
		pTensor = new Tensor(DT_QUINT8, TensorShape(arraySlice));

		int i = 0;
		for (std::vector<quint8>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
		{
			pTensor->flat<quint8>()(i) = *it;
			i++;
		}
		arraySlice.clear();
		arrayvals.clear();
	}
	break;
	case DT_QUINT16:
	{
		std::vector<quint16> arrayvals;
		for (unsigned int i = 0; i < arrayVal.size(); i++)
		{
			arrayvals.push_back((quint16)arrayVal[i]);
		}

		gtl::ArraySlice< int64 > arraySlice(arraydims);
		pTensor = new Tensor(DT_QUINT16, TensorShape(arraySlice));

		int i = 0;
		for (std::vector<quint16>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
		{
			pTensor->flat<quint16>()(i) = *it;
			i++;
		}
		arraySlice.clear();
		arrayvals.clear();
	}
	break;
	case DT_BOOL:
	{
		std::vector<bool> arrayvals;
		for (unsigned int i = 0; i < arrayVal.size(); i++)
		{
			arrayvals.push_back((bool)arrayVal[i]);
		}

		gtl::ArraySlice< int64 > arraySlice(arraydims);
		pTensor = new Tensor(DT_BOOL, TensorShape(arraySlice));

		int i = 0;
		for (std::vector<bool>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
		{
			pTensor->flat<bool>()(i) = *it;
			i++;
		}
		arraySlice.clear();
		arrayvals.clear();
	}
	break;
	case DT_STRING:
	{
		std::vector<std::string> arrayvals;
		arrayvals = arrayStrVal;
		gtl::ArraySlice< int64 > arraySlice(arraydims);
		pTensor = new Tensor(DT_STRING, TensorShape(arraySlice));

		int i = 0;
		for (std::vector<std::string>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
		{
			pTensor->flat<std::string>()(i) = *it;
			i++;
		}
		arraySlice.clear();
	}
	break;
	default:
	{
		std::string msg = string_format("error : DataType (%s) is not spported in tensor data.", strPinType.c_str());
		return pTensor;
	}
	}


	array_slice.clear();
	arraydims.clear();
	arrayVal.clear();
	arrayStrVal.clear();
	arrayComplexVal.clear();

	return pTensor;
}

void* Create_StrToOutput(Scope& pScope, std::string strPinType, std::string strPinShape, std::string strPinInitial)
{
	Output* pOutput = new Output();

	Tensor* pTensor = Create_StrToTensor(strPinType, strPinShape, strPinInitial);

	if (pTensor)
	{
		*pOutput = Const(pScope, *pTensor);
		delete pTensor;
	}

	return pOutput;
}
bool GetArrayDimsFromStrVal(std::string strVal, std::vector<int64>& arraydims, std::vector<int64>& arrayslice, std::vector<double>& arrayVal)
{
	if (strVal.length() > 0)
	{
		int islice = 0;
		std::string val = "";
		int iDim = 0;
		int iMaxDim = 0;
		std::vector<int> idims;
		for (std::string::size_type i = 0; i < strVal.size(); i++)
		{
			if (strVal[i] == '{')
			{
				iDim++;
				if (iMaxDim < iDim)
					idims.push_back(0);
				iMaxDim = max(iMaxDim, iDim);
			}
			else if (strVal[i] == '}')
			{
				idims[iDim - 1] = idims[iDim - 1] + 1;
				if (val != "")
				{
					arrayVal.push_back(std::stod(val.c_str()));
				}
				iDim--;
				val = "";
			}
			else if (strVal[i] == ',')
			{
				idims[iDim - 1] = idims[iDim - 1] + 1;
				if (val != "")
				{
					arrayVal.push_back(std::stod(val.c_str()));
					val = "";
				}
			}
			else
			{
				if ((strVal[i] >= 0x30 && strVal[i] <= 0x39) || strVal[i] == '.' || strVal[i] == '-' || strVal[i] == '+' || strVal[i] == 'e' || strVal[i] == 'E')
					val = val + strVal[i];
			}
		}
		val = "";
		if (iMaxDim == 0)
		{
			for (std::string::size_type i = 0; i < strVal.size(); i++)
			{
				if (strVal[i] == ',' || strVal[i] == ';')
				{
					if (val != "")
					{
						arrayVal.push_back(std::stod(val.c_str()));
						val = "";
					}
				}
				else
				{
					if ((strVal[i] >= 0x30 && strVal[i] <= 0x39) || strVal[i] == '.' || strVal[i] == '-' || strVal[i] == '+' || strVal[i] == 'e' || strVal[i] == 'E')
						val = val + strVal[i];
				}
			}
			if (val != "")
			{
				arrayVal.push_back(std::stod(val.c_str()));
				val = "";
			}
		}

		for (int k = 1; k < idims.size(); k++)
		{
			int iDatdNum = 1;
			for (int j = 0; j < k; j++)
			{
				iDatdNum = iDatdNum*idims[j];
			}
			idims[k] = idims[k] / iDatdNum;
		}
		for (unsigned int l = 0; l < idims.size(); l++)
		{
			arraydims.push_back(idims[l]);
		}

//		if (iMaxDim == 0) arraydims.push_back(arrayVal.size());
		arrayslice.push_back(iMaxDim);
	}

	return true;
}
bool GetArrayDimsFromStrVal(std::string strVal, std::vector<int64>& arraydims, std::vector<int64>& arrayslice, std::vector<std::string>& arrayStrVal)
{
	if (strVal.length() > 0)
	{
		int islice = 0;
		std::string val = "";
		int iDim = 0;
		int iMaxDim = 0;
		std::vector<int> idims;
		for (std::string::size_type i = 0; i < strVal.size(); i++)
		{
			if (strVal[i] == '{')
			{
				iDim++;
				if (iMaxDim < iDim)
					idims.push_back(0);
				iMaxDim = max(iMaxDim, iDim);
			}
			else if (strVal[i] == '}')
			{
				idims[iDim - 1] = idims[iDim - 1] + 1;
				if (val != "")
				{
					arrayStrVal.push_back(val.c_str());
				}
				iDim--;
				val = "";
			}
			else if (strVal[i] == ',')
			{
				idims[iDim - 1] = idims[iDim - 1] + 1;
				if (val != "")
				{
					arrayStrVal.push_back(val.c_str());
					val = "";
				}
			}
			else
			{
				if (strVal[i] != '\n' && strVal[i] != '\r')
					val = val + strVal[i];
			}
		}
		val = "";
		if (iMaxDim == 0)
		{
			for (std::string::size_type i = 0; i < strVal.size(); i++)
			{
				if (strVal[i] == ',' || strVal[i] == ';')
				{
					if (val != "")
					{
						arrayStrVal.push_back(val.c_str());
						val = "";
					}
				}
				else
				{
					if (strVal[i] != '\n' && strVal[i] != '\r')
						val = val + strVal[i];
				}
			}
			if (val != "")
			{
				arrayStrVal.push_back(val.c_str());
				val = "";
			}
		}

		for (int k = 1; k < idims.size(); k++)
		{
			int iDatdNum = 1;
			for (int j = 0; j < k; j++)
			{
				iDatdNum = iDatdNum*idims[j];
			}
			idims[k] = idims[k] / iDatdNum;
		}
		for (unsigned int l = 0; l < idims.size(); l++)
		{
			arraydims.push_back(idims[l]);
		}
//		if (iMaxDim == 0)
//			arraydims.push_back(arrayStrVal.size());

		arrayslice.push_back(iMaxDim);
	}

	return true;
}
bool GetArrayDimsFromStrVal(std::string strVal, std::vector<int64>& arraydims, std::vector<int64>& arrayslice, std::vector<std::complex<double>>& arrayComplexVal)
{
	if (strVal.length() > 0)
	{
		int islice = 0;
		std::string val = "";
		std::string valReal = "";
		std::string valImag = "";
		int iDim = 0;
		int iMaxDim = 0;
		std::vector<int> idims;
		std::complex<double> cVal;
		for (std::string::size_type i = 0; i < strVal.size(); i++)
		{
			if (strVal[i] == '{')
			{
				iDim++;
				if (iMaxDim < iDim)
					idims.push_back(0);
				iMaxDim = max(iMaxDim, iDim);
			}
			else if (strVal[i] == '}')
			{
				idims[iDim - 1] = idims[iDim - 1] + 1;
				if (val != "")
				{
					valReal = val;
					val = "";
				}
				if (!valReal.empty()) cVal.real(std::stod(valReal.c_str()));
				if (!valImag.empty()) cVal.imag(std::stod(valImag.c_str()));
				if (!valReal.empty() || !valImag.empty()) arrayComplexVal.push_back(cVal);
				cVal.real(0.0);
				cVal.imag(0.0);
				iDim--;
				valReal = "";
				valImag = "";
			}
			else if (strVal[i] == ',')
			{
				idims[iDim - 1] = idims[iDim - 1] + 1;
				if (val != "")
				{
					valReal = val;
					val = "";
				}
				if (!valReal.empty()) cVal.real(std::stod(valReal.c_str()));
				if (!valImag.empty()) cVal.imag(std::stod(valImag.c_str()));
				if (!valReal.empty() || !valImag.empty()) arrayComplexVal.push_back(cVal);
				cVal.real(0.0);
				cVal.imag(0.0);
				valReal = "";
				valImag = "";
			}
			else
			{
				if ((strVal[i] >= 0x30 && strVal[i] <= 0x39) || strVal[i] == '.' || strVal[i] == '-' || strVal[i] == '+' || strVal[i] == 'e' || strVal[i] == 'E' || strVal[i] == 'j' || strVal[i] == 'J')
				{
					if (!val.empty() && (strVal[i] == '+' || strVal[i] == '-'))
					{
						valReal = val;
						val = "";
						//						val = val + strVal[i];
					}
					if (!val.empty() && (strVal[i] == 'j' || strVal[i] == 'J'))
					{
						valImag = val;
						val = "";
					}
					else
						val = val + strVal[i];

				}
			}
		}
		val = "";
		if (iMaxDim == 0)
		{
			for (std::string::size_type i = 0; i < strVal.size(); i++)
			{
				if (strVal[i] == ',' || strVal[i] == ';')
				{
					if (val != "")
					{
						val = "";
					}
					if (!valReal.empty()) cVal.real(std::stod(valReal.c_str()));
					if (!valImag.empty()) cVal.imag(std::stod(valImag.c_str()));
					if (!valReal.empty() || !valImag.empty())
					{
						arrayComplexVal.push_back(cVal);
					}
					cVal.real(0.0);
					cVal.imag(0.0);
					valReal = "";
					valImag = "";
				}
				else
				{
					if ((strVal[i] >= 0x30 && strVal[i] <= 0x39) || strVal[i] == '.' || strVal[i] == '-' || strVal[i] == '+' || strVal[i] == 'e' || strVal[i] == 'E' || strVal[i] == 'j' || strVal[i] == 'J')
					{
						if (!val.empty() && (strVal[i] == '+' || strVal[i] == '-'))
						{
							valReal = val;
							val = "";
							//							val = val + strVal[i];
						}
						if (!val.empty() && (strVal[i] == 'j' || strVal[i] == 'J'))
						{
							valImag = val;
							val = "";
						}
						else
							val = val + strVal[i];
					}
				}
			}

			if (val != "")
			{
				valReal = val;
				val = "";
			}
			if (!valReal.empty()) cVal.real(std::stod(valReal.c_str()));
			if (!valImag.empty()) cVal.imag(std::stod(valImag.c_str()));
			if (!valReal.empty() || !valImag.empty())
			{
				arrayComplexVal.push_back(cVal);
			}
			cVal.real(0.0);
			cVal.imag(0.0);
			valReal = "";
			valImag = "";
		}

		for (int k = 1; k < idims.size(); k++)
		{
			int iDatdNum = 1;
			for (int j = 0; j < k; j++)
			{
				iDatdNum = iDatdNum*idims[j];
			}
			idims[k] = idims[k] / iDatdNum;
		}
		for (unsigned int l = 0; l < idims.size(); l++)
		{
			arraydims.push_back(idims[l]);
		}

//		if (iMaxDim == 0) arraydims.push_back(arrayComplexVal.size());
		arrayslice.push_back(iMaxDim);
	}

	return true;
}
bool isString(std::string strVal)
{
	bool bIsString = true;
	for (size_t i = 0; i < strVal.size(); i++)
	{
		if ((strVal[i] < 0x20 && strVal[i] != 0x0A && strVal[i] != 0x0D && strVal[i] != 0x09)  || strVal[i] == 0xff) return false;
	}
	return bIsString;
}

