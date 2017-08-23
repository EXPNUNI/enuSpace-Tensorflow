#include "stdafx.h"
#include "utility_functions.h"


#include <string>
#include <cstdarg>
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow.h"

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
	if (strShape.length() > 0)
	{
		int64 islice = 0;
		std::string val;
		for (std::string::size_type i = 0; i < strShape.size(); i++)
		{
			if (strShape[i] == '[')
			{
				val = "";
			}
			else if (strShape[i] == ']')
			{
				arraydims.push_back(std::stoi(val));
				islice++;
			}
			else
			{
				val = val + strShape[i];
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
		if (strinitial[i] == ';' || strinitial[i] == ',')
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


DataTypeSlice GetDatatypeSliceFromInitial(std::string strinitial)
{
	std::string val;
	std::vector<DataType> arrayvals;
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
			val = val + strinitial[i];
		}
	}

	if (val.length() > 0)
	{
		DataType dtype;
		dtype = GetDatatypeFromInitial(val);
		if (dtype != DT_INVALID)
			arrayvals.push_back(dtype);
	}
	DataTypeSlice DT(arrayvals);
	arrayvals.clear();
	return DT;
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
			if (strinitial[i] == '{')
			{
				bOpen = true;
			}
			else if (strinitial[i] == '}' || strinitial[i] == ';')
			{
				if (arraydims.size() != 0)
				{
					bOpen = false;
					PartialTensorShape partts(arraydims);
					vec_PTS.push_back(partts);
					arraydims.clear();
				}

			}
			else
			{
				val = val + strinitial[i];
			}
		}
	}
	if (val.length() > 0)  //사용자가 마지막 구분자를 닫지않았다.
	{
		if (arraydims.size() != 0)
		{
			PartialTensorShape partts(arraydims);
			vec_PTS.push_back(partts);
			arraydims.clear();
		}
	}
	int isize = vec_PTS.size();

	//std::string msg = string_format("test %d", isize);
	//PrintMessage(msg);
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
		if (strinitial[i] == ',')
		{
			iDimSize = iDimSize * stoll (val);
			val = "";
		}
		if (strinitial[i] == '{' || strinitial[i] == '}')
		{

		}
		else
		{
			val = val + strinitial[i];
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
		if (strinitial[i] == ';' || strinitial[i] == ',')
		{
			arraydims.push_back(stoll(val));
			val = "";
			continue;
		}
		if (strinitial[i] == '{' || strinitial[i] == '}')
		{

		}
		else
		{
			val = val + strinitial[i];
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