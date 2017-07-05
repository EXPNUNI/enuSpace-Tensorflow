#include "stdafx.h"
#include "utility_functions.h"


#include <string>
#include <cstdarg>

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
	return true;
}

bool GetDoubleVectorFormInitial(std::string strinitial, std::vector<double>& arrayvals)
{
	std::string val;
	for (std::string::size_type i = 0; i < strinitial.size(); i++)
	{
		if (strinitial[i] == ';')
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

bool GetFloatVectorFormInitial(std::string strinitial, std::vector<float>& arrayvals)
{
	std::string val;
	for (std::string::size_type i = 0; i < strinitial.size(); i++)
	{
		if (strinitial[i] == ';')
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

bool GetIntVectorFormInitial(std::string strinitial, std::vector<int>& arrayvals)
{
	std::string val;
	for (std::string::size_type i = 0; i < strinitial.size(); i++)
	{
		if (strinitial[i] == ';')
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

bool GetBoolVectorFormInitial(std::string strinitial, std::vector<bool>& arrayvals)
{
	std::string val;
	for (std::string::size_type i = 0; i < strinitial.size(); i++)
	{
		if (strinitial[i] == ';')
		{
			if (val == "1" || val == "true" )
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
		if (val == "1" || val == "true")
			arrayvals.push_back(true);
		else
			arrayvals.push_back(false);
	}
	return true;
}

bool GetStringVectorFormInitial(std::string strinitial, std::vector<std::string>& arrayvals)
{
	std::string val;
	for (std::string::size_type i = 0; i < strinitial.size(); i++)
	{
		if (strinitial[i] == ';')
		{
			arrayvals.push_back(val);
			val = "";
		}
		else
		{
			val = val + strinitial[i];
		}
	}

	if (val.length() > 0)
		arrayvals.push_back(val);
	return true;
}
