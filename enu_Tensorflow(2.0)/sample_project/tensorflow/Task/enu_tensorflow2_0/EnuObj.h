#pragma once
#pragma pack(1)
#include <string>

#pragma pack(push, 1) 

#define _readonly(value);
#define _visibility(value);
#define _color(value);
#define _min(value);
#define _max(value);
#define _unit(value);
#define _desc(value);


struct EnuObject
{
	wchar_t id[32]; _visibility(false)
	wchar_t type[32]; _visibility(false)
	EnuObject *from; _visibility(false)
	EnuObject *to; _visibility(false)
};

struct EnuParam
{
	int type;
	void* param;
};
struct TRANSFER : EnuObject
{
	int _TypeIndex; _visibility(false)
	EnuParam param_from;
	EnuParam param_to;
};
struct EnuMulfunc : EnuObject
{
	int _TypeIndex; _visibility(false)
	int Action; _unit(bmnls) _color(#808080) _visibility(false) // Instructor Action
	double Delay; _unit(sec) _color(#808080) _visibility(false) // Malfunction Delay Time
	double Ramp; _unit(sec) _color(#808080) _visibility(false) // Malfunction Ramp Time
	double Severity; _unit(dmnls) _color(#808080) _visibility(false) // Malfunction Severity
	int FailureStatus; _unit(dmnls) _color(#808080) _visibility(false) // Malfunction Status
	double FailureValue; _unit(dmnls) _color(#808080) _visibility(false) // Malfunction Value
	double DT; _unit(1 / sec) _color(#808080) _visibility(false) // Malfunction Change Rate
	double Timer; _unit(sec) _color(#808080) _visibility(false) // Malfunction Delay Timer
	EnuMulfunc()
	{}
};

#pragma pack(pop) 
