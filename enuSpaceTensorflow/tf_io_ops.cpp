#include "stdafx.h"
#include "tf_io_ops.h"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "include/json/json.h"

#include "GlobalHeader.h"
#include "tensorflow.h"
#include "utility_functions.h"
#include "enuSpaceToTensorflow.h"
#include "AttributeParser.h"

void* Create_FixedLengthRecordReader(std::string id, Json::Value pInputItem) {
	FixedLengthRecordReader* pFixedLengthRecordReader = nullptr;
	Scope* pScope = nullptr;
	int64 record_bytes = 0;
	FixedLengthRecordReader::Attrs attrs;
	int iSize = (int)pInputItem.size();
	std::string container_ = "";
	std::string shared_name_ = "";
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : FixedLengthRecordReader - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "record_bytes")
		{
			if (strPinInterface == "Input")
			{
				if(strPinInitial !="")
					record_bytes = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : FixedLengthRecordReader - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "FixedLengthRecordReader::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("header_bytes_") != "") attrs = attrs.HeaderBytes(attrParser.ConvStrToInt64(attrParser.GetAttribute("header_bytes_")));
				if (attrParser.GetAttribute("footer_bytes_") != "") attrs = attrs.FooterBytes(attrParser.ConvStrToInt64(attrParser.GetAttribute("footer_bytes_")));
				if (attrParser.GetAttribute("hop_bytes_") != "") attrs = attrs.HopBytes(attrParser.ConvStrToInt64(attrParser.GetAttribute("hop_bytes_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs = attrs.Container(container_);
				}
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs = attrs.SharedName(shared_name_);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : FixedLengthRecordReader pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pFixedLengthRecordReader = new FixedLengthRecordReader(*pScope, record_bytes, attrs);
		ObjectInfo* pObj = AddObjectMap(pFixedLengthRecordReader, id, SYMBOL_FIXEDLENGTHRECORDREADER, "FixedLengthRecordReader", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pFixedLengthRecordReader->reader_handle, OUTPUT_TYPE_OUTPUT, "reader_handle");
		}
	}
	else
	{
		std::string msg = string_format("error : FixedLengthRecordReader(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pFixedLengthRecordReader;
}

void* Create_IdentityReader(std::string id, Json::Value pInputItem) {
	IdentityReader* pIdentityReader = nullptr;
	Scope* pScope = nullptr;
	IdentityReader::Attrs attrs;
	int iSize = (int)pInputItem.size();
	std::string container_ = "";
	std::string shared_name_ = "";
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : IdentityReader - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "IdentityReader::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs = attrs.Container(container_);
				}
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs = attrs.SharedName(shared_name_);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : IdentityReader pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pIdentityReader = new IdentityReader(*pScope, attrs);
		ObjectInfo* pObj = AddObjectMap(pIdentityReader, id, SYMBOL_IDENTITYREADER, "IdentityReader", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pIdentityReader->reader_handle, OUTPUT_TYPE_OUTPUT, "reader_handle");
		}
	}
	else
	{
		std::string msg = string_format("error : IdentityReader(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pIdentityReader;
}
void* Create_LMDBReader(std::string id, Json::Value pInputItem) {
	LMDBReader* pLMDBReaderReader = nullptr;
	Scope* pScope = nullptr;
	LMDBReader::Attrs attrs;
	int iSize = (int)pInputItem.size();
	std::string container_ = "";
	std::string shared_name_ = "";
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : LMDBReader - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "LMDBReader::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs = attrs.Container(container_);
				}
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs = attrs.SharedName(shared_name_);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : LMDBReader pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pLMDBReaderReader = new LMDBReader(*pScope, attrs);
		ObjectInfo* pObj = AddObjectMap(pLMDBReaderReader, id, SYMBOL_LMDBREADER, "LMDBReader", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pLMDBReaderReader->reader_handle, OUTPUT_TYPE_OUTPUT, "reader_handle");
		}
	}
	else
	{
		std::string msg = string_format("error : LMDBReader(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pLMDBReaderReader;
}

void* Create_MatchingFiles(std::string id, Json::Value pInputItem) {
	MatchingFiles* pMatchingFiles = nullptr;
	Scope* pScope = nullptr;
	Output* ppattern = nullptr;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : MatchingFiles - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "pattern")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							ppattern = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						ppattern = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : MatchingFiles - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : MatchingFiles pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && ppattern)
	{
		pMatchingFiles = new MatchingFiles(*pScope, *ppattern);
		ObjectInfo* pObj = AddObjectMap(pMatchingFiles, id, SYMBOL_MATCHINGFILES, "MatchingFiles", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pMatchingFiles->filenames, OUTPUT_TYPE_OUTPUT, "filenames");
		}
	}
	else
	{
		std::string msg = string_format("error : MatchingFiles(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pMatchingFiles;
}

void* Create_MergeV2Checkpoints(std::string id, Json::Value pInputItem) {
	MergeV2Checkpoints* pMergeV2Checkpoints = nullptr;
	Scope* pScope = nullptr;
	Output *pcheckpoint_prefixes = nullptr;
	Output *pdestination_prefix = nullptr;
	MergeV2Checkpoints::Attrs attrs;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : MergeV2Checkpoints - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "checkpoint_prefixes")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pcheckpoint_prefixes = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : MergeV2Checkpoints - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "destination_prefix")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pdestination_prefix = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : MergeV2Checkpoints - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "MergeV2Checkpoints::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("delete_old_dirs_") != "") attrs = attrs.DeleteOldDirs(attrParser.ConvStrToBool(attrParser.GetAttribute("delete_old_dirs_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : MergeV2Checkpoints pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pcheckpoint_prefixes && pdestination_prefix)
	{
		pMergeV2Checkpoints = new MergeV2Checkpoints(*pScope,*pcheckpoint_prefixes,*pdestination_prefix, attrs);
		ObjectInfo* pObj = AddObjectMap(pMergeV2Checkpoints, id, SYMBOL_MERGEV2CHECKPOINTS, "MergeV2Checkpoints", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pMergeV2Checkpoints->operation, OUTPUT_TYPE_OPERATION, "operation");
		}
	}
	else
	{
		std::string msg = string_format("error : MergeV2Checkpoints(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pMergeV2Checkpoints;
}

void* Create_ReadFile(std::string id, Json::Value pInputItem) {
	tensorflow::ops::ReadFile* pReadFile = nullptr;
	Scope* pScope = nullptr;
	Output *pfilename = nullptr;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ReadFile - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "filename")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pfilename = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if(!strPinInitial.empty())
						pfilename = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : ReadFile - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : ReadFile pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pfilename)
	{
		pReadFile = new tensorflow::ops::ReadFile(*pScope, *pfilename);
		ObjectInfo* pObj = AddObjectMap(pReadFile, id, SYMBOL_READFILE, "ReadFile", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pReadFile->contents, OUTPUT_TYPE_OUTPUT, "contents");
		}
	}
	else
	{
		std::string msg = string_format("error : ReadFile(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pReadFile;
}

void* Create_ReaderNumRecordsProduced(std::string id, Json::Value pInputItem) {
	ReaderNumRecordsProduced* pReaderNumRecordsProduced = nullptr;
	Scope* pScope = nullptr;
	Output *preader_handle = nullptr;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ReaderNumRecordsProduced - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "reader_handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							preader_handle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ReaderNumRecordsProduced - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : ReaderNumRecordsProduced pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && preader_handle)
	{
		pReaderNumRecordsProduced = new ReaderNumRecordsProduced(*pScope, *preader_handle);
		ObjectInfo* pObj = AddObjectMap(pReaderNumRecordsProduced, id, SYMBOL_READERNUMRECORDSPRODUCED, "ReaderNumRecordsProduced", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pReaderNumRecordsProduced->records_produced, OUTPUT_TYPE_OUTPUT, "records_produced");
		}
	}
	else
	{
		std::string msg = string_format("error : ReaderNumRecordsProduced(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pReaderNumRecordsProduced;
}

void* Create_ReaderNumWorkUnitsCompleted(std::string id, Json::Value pInputItem) {
	ReaderNumWorkUnitsCompleted* pReaderNumWorkUnitsCompleted = nullptr;
	Scope* pScope = nullptr;
	Output *preader_handle = nullptr;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ReaderNumWorkUnitsCompleted - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "reader_handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							preader_handle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ReaderNumWorkUnitsCompleted - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : ReaderNumWorkUnitsCompleted pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && preader_handle)
	{
		pReaderNumWorkUnitsCompleted = new ReaderNumWorkUnitsCompleted(*pScope, *preader_handle);
		ObjectInfo* pObj = AddObjectMap(pReaderNumWorkUnitsCompleted, id, SYMBOL_READERNUMWORKUNITSCOMPLETED, "ReaderNumWorkUnitsCompleted", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pReaderNumWorkUnitsCompleted->units_completed, OUTPUT_TYPE_OUTPUT, "units_completed");
		}
	}
	else
	{
		std::string msg = string_format("error : ReaderNumWorkUnitsCompleted(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pReaderNumWorkUnitsCompleted;
}

void* Create_ReaderRead(std::string id, Json::Value pInputItem) {
	ReaderRead* pReaderRead = nullptr;
	Scope* pScope = nullptr;
	Output *preader_handle = nullptr;
	Output *pqueue_handle = nullptr;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ReaderRead - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "reader_handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							preader_handle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ReaderRead - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "queue_handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pqueue_handle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ReaderRead - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : ReaderRead pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && preader_handle && pqueue_handle)
	{
		pReaderRead = new ReaderRead(*pScope, *preader_handle, *pqueue_handle);
		ObjectInfo* pObj = AddObjectMap(pReaderRead, id, SYMBOL_READERREAD, "ReaderRead", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pReaderRead->key, OUTPUT_TYPE_OUTPUT, "key");
			AddOutputInfo(pObj, &pReaderRead->value, OUTPUT_TYPE_OUTPUT, "value");
		}
	}
	else
	{
		std::string msg = string_format("error : ReaderRead(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pReaderRead;
}

void* Create_ReaderReadUpTo(std::string id, Json::Value pInputItem) {
	ReaderReadUpTo* pReaderReadUpTo = nullptr;
	Scope* pScope = nullptr;
	Output *preader_handle = nullptr;
	Output *pqueue_handle = nullptr;
	Output *pnum_records = nullptr;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ReaderReadUpTo - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "reader_handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							preader_handle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ReaderReadUpTo - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "queue_handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pqueue_handle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ReaderReadUpTo - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_records")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pnum_records = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pnum_records = (Output*)Create_StrToOutput(*m_pScope, "DT_INT64", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : ReaderReadUpTo - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : ReaderReadUpTo pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && preader_handle && pqueue_handle && pnum_records)
	{
		pReaderReadUpTo = new ReaderReadUpTo(*pScope, *preader_handle, *pqueue_handle, *pnum_records);
		ObjectInfo* pObj = AddObjectMap(pReaderReadUpTo, id, SYMBOL_READERREADUPTO, "ReaderReadUpTo", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pReaderReadUpTo->keys, OUTPUT_TYPE_OUTPUT, "keys");
			AddOutputInfo(pObj, &pReaderReadUpTo->values, OUTPUT_TYPE_OUTPUT, "values");
		}
	}
	else
	{
		std::string msg = string_format("error : ReaderReadUpTo(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pReaderReadUpTo;
}

void* Create_ReaderReset(std::string id, Json::Value pInputItem) {
	ReaderReset* pReaderReset = nullptr;
	Scope* pScope = nullptr;
	Output *preader_handle = nullptr;
	Output *pqueue_handle = nullptr;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ReaderReset - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "reader_handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							preader_handle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ReaderReset - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : ReaderReset pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && preader_handle)
	{
		pReaderReset = new ReaderReset(*pScope, *preader_handle);
		ObjectInfo* pObj = AddObjectMap(pReaderReset, id, SYMBOL_READERRESET, "ReaderReset", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pReaderReset->operation, OUTPUT_TYPE_OPERATION, "operation");
		}
	}
	else
	{
		std::string msg = string_format("error : ReaderReset(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pReaderReset;
}

void* Create_ReaderRestoreState(std::string id, Json::Value pInputItem) {
	ReaderRestoreState* pReaderRestoreState = nullptr;
	Scope* pScope = nullptr;
	Output *preader_handle = nullptr;
	Output *pstate = nullptr;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : pReaderRestoreState - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "reader_handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							preader_handle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : pReaderRestoreState - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "state")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pstate = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : pReaderRestoreState - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : pReaderRestoreState pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && preader_handle && pstate)
	{
		pReaderRestoreState = new ReaderRestoreState(*pScope, *preader_handle, *pstate);
		ObjectInfo* pObj = AddObjectMap(pReaderRestoreState, id, SYMBOL_READERRESTORESTATE, "ReaderRestoreState", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pReaderRestoreState->operation, OUTPUT_TYPE_OPERATION, "operation");
		}
	}
	else
	{
		std::string msg = string_format("error : pReaderRestoreState(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pReaderRestoreState;
}

void* Create_ReaderSerializeState(std::string id, Json::Value pInputItem) {
	ReaderSerializeState* pReaderSerializeState = nullptr;
	Scope* pScope = nullptr;
	Output *preader_handle = nullptr;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ReaderSerializeState - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "reader_handle")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							preader_handle = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ReaderSerializeState - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : ReaderSerializeState pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && preader_handle)
	{
		pReaderSerializeState = new ReaderSerializeState(*pScope, *preader_handle);
		ObjectInfo* pObj = AddObjectMap(pReaderSerializeState, id, SYMBOL_READERSERIALIZESTATE, "ReaderSerializeState", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pReaderSerializeState->state, OUTPUT_TYPE_OUTPUT, "state");
		}
	}
	else
	{
		std::string msg = string_format("error : ReaderSerializeState(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pReaderSerializeState;
}

void* Create_Restore(std::string id, Json::Value pInputItem) {
	Restore* pRestore = nullptr;
	Scope* pScope = nullptr;
	Output *pfile_pattern = nullptr;
	Output *ptensor_name = nullptr;
	DataType dt;
	Restore::Attrs attrs;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Restore - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "file_pattern")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pfile_pattern = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pfile_pattern = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : Restore - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "tensor_name")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							ptensor_name = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						ptensor_name = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : Restore - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "dt")
		{
			if (strPinInterface == "DataType")
			{
				if (!(dt = GetDatatypeFromInitial(strPinInitial)))
				{
					std::string msg = string_format("warning : Restore - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : Restore - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "Restore::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("preferred_shard_") != "") attrs = attrs.PreferredShard(attrParser.ConvStrToInt64(attrParser.GetAttribute("preferred_shard_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : Restore pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pfile_pattern && ptensor_name)
	{
		pRestore = new Restore(*pScope, *pfile_pattern, *ptensor_name, dt, attrs);
		ObjectInfo* pObj = AddObjectMap(pRestore, id, SYMBOL_RESTORE, "Restore", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pRestore->tensor, OUTPUT_TYPE_OUTPUT, "tensor");
		}
	}
	else
	{
		std::string msg = string_format("error : Restore(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRestore;
}

void* Create_RestoreSlice(std::string id, Json::Value pInputItem) {
	RestoreSlice* pRestoreSlice = nullptr;
	Scope* pScope = nullptr;
	Output *pfile_pattern = nullptr;
	Output *ptensor_name = nullptr;
	Output *pshape_and_slice = nullptr;
	DataType dt;
	RestoreSlice::Attrs attrs;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : RestoreSlice - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "file_pattern")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pfile_pattern = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pfile_pattern = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : RestoreSlice - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "tensor_name")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							ptensor_name = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						ptensor_name = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : RestoreSlice - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "shape_and_slice")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pshape_and_slice = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pshape_and_slice = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : RestoreSlice - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "dt")
		{
			if (strPinInterface == "DataType")
			{
				if (!(dt = GetDatatypeFromInitial(strPinInitial)))
				{
					std::string msg = string_format("warning : RestoreSlice - %s(%s) unknown type(%s).", id.c_str(), strPinName.c_str(), strPinInitial.c_str());
					PrintMessage(msg);
				}
			}
			else
			{
				std::string msg = string_format("warning : RestoreSlice - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "RestoreSlice::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("preferred_shard_") != "") attrs = attrs.PreferredShard(attrParser.ConvStrToInt64(attrParser.GetAttribute("preferred_shard_")));
			}
		}
		else
		{
			std::string msg = string_format("warning : RestoreSlice pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pfile_pattern && ptensor_name && pshape_and_slice)
	{
		pRestoreSlice = new RestoreSlice(*pScope, *pfile_pattern, *ptensor_name,*pshape_and_slice, dt, attrs);
		ObjectInfo* pObj = AddObjectMap(pRestoreSlice, id, SYMBOL_RESTORESLICE, "RestoreSlice", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pRestoreSlice->tensor, OUTPUT_TYPE_OUTPUT, "tensor");
		}
	}
	else
	{
		std::string msg = string_format("error : RestoreSlice(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRestoreSlice;
}

void* Create_RestoreV2(std::string id, Json::Value pInputItem) {
	RestoreV2* pRestoreV2 = nullptr;
	Scope* pScope = nullptr;
	Output *pprefix = nullptr;
	Output *ptensor_names = nullptr;
	Output *pshape_and_slices = nullptr;
	std::vector<tensorflow::DataType> vDT;
	RestoreSlice::Attrs attrs;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : RestoreV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "prefix")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pprefix = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pprefix = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : RestoreV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "tensor_names")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							ptensor_names = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						ptensor_names = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : RestoreV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "shape_and_slices")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pshape_and_slices = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pshape_and_slices = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : RestoreV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "dtypes")
		{
			if (strPinInterface == "DataTypeSlice")
			{
				if (!strPinInitial.empty())
					GetDatatypeSliceFromInitial(strPinInitial, vDT);
			}
			else
			{
				std::string msg = string_format("warning : RestoreV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : RestoreV2 pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pprefix && ptensor_names && pshape_and_slices && vDT.size() > 0)
	{
		DataTypeSlice dtypes(vDT);
		pRestoreV2 = new RestoreV2(*pScope, *pprefix, *ptensor_names, *pshape_and_slices, dtypes);
		vDT.clear();
		ObjectInfo* pObj = AddObjectMap(pRestoreV2, id, SYMBOL_RESTOREV2, "RestoreV2", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pRestoreV2->tensors, OUTPUT_TYPE_OUTPUTLIST, "tensors");
		}
	}
	else
	{
		std::string msg = string_format("error : RestoreV2(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pRestoreV2;
}

void* Create_Save(std::string id, Json::Value pInputItem) {
	Save* pSave = nullptr;
	Scope* pScope = nullptr;
	Output *pfilename = nullptr;
	Output *ptensor_names = nullptr;
	OutputList *pdata = nullptr;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Save - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "filename")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pfilename = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pfilename = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : Save - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "tensor_names")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							ptensor_names = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						ptensor_names = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : Save - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "data")
		{
			if (strPinInterface == "InputList")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pdata = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
					{
						pdata = (OutputList*)Create_StrToOutputList(*m_pScope, strAutoPinType, "", strPinInitial);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Save - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : Save pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pfilename && ptensor_names && pdata)
	{
		pSave = new Save(*pScope, *pfilename, *ptensor_names, *pdata);
		ObjectInfo* pObj = AddObjectMap(pSave, id, SYMBOL_SAVE, "Save", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSave->operation, OUTPUT_TYPE_OPERATION, "operation");
		}
	}
	else
	{
		std::string msg = string_format("error : Save(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSave;
}

void* Create_SaveSlices(std::string id, Json::Value pInputItem) {
	SaveSlices* pSaveSlices = nullptr;
	Scope* pScope = nullptr;
	Output *pfilename = nullptr;
	Output *ptensor_names = nullptr;
	Output *pshape_and_slices = nullptr;
	OutputList *pdata = nullptr;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SaveSlices - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "filename")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pfilename = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pfilename = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : SaveSlices - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "tensor_names")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							ptensor_names = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						ptensor_names = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : SaveSlices - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "shapes_and_slices")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pshape_and_slices = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pshape_and_slices = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : SaveSlices - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "data")
		{
			if (strPinInterface == "InputList")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pdata = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SaveSlices - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SaveSlices pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pfilename && ptensor_names && pshape_and_slices && pdata )
	{
		pSaveSlices = new SaveSlices(*pScope, *pfilename, *ptensor_names, *pshape_and_slices, *pdata);
		ObjectInfo* pObj = AddObjectMap(pSaveSlices, id, SYMBOL_SAVESLICES, "SaveSlices", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSaveSlices->operation, OUTPUT_TYPE_OPERATION, "operation");
		}
	}
	else
	{
		std::string msg = string_format("error : SaveSlices(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSaveSlices;
}

void* Create_SaveV2(std::string id, Json::Value pInputItem) {
	SaveV2* pSaveV2 = nullptr;
	Scope* pScope = nullptr;
	Output *pprefix = nullptr;
	Output *ptensor_names = nullptr;
	Output *pshape_and_slices = nullptr;
	OutputList *ptensors = nullptr;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : SaveV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "prefix")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pprefix = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pprefix = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : SaveV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "tensor_names")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							ptensor_names = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						ptensor_names = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : SaveV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "shape_and_slices")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pshape_and_slices = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pshape_and_slices = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : SaveV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "tensors")
		{
			if (strPinInterface == "InputList")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							ptensors = (OutputList*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : SaveV2 - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : SaveV2 pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pprefix && ptensor_names && pshape_and_slices && ptensors)
	{
		pSaveV2 = new SaveV2(*pScope, *pprefix, *ptensor_names, *pshape_and_slices, *ptensors);
		ObjectInfo* pObj = AddObjectMap(pSaveV2, id, SYMBOL_SAVEV2, "SaveV2", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pSaveV2->operation, OUTPUT_TYPE_OPERATION, "operation");
		}
	}
	else
	{
		std::string msg = string_format("error : SaveV2(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pSaveV2;
}

void* Create_ShardedFilename(std::string id, Json::Value pInputItem) {
	ShardedFilename* pShardedFilename = nullptr;
	Scope* pScope = nullptr;
	Output *pbasename = nullptr;
	Output *pshard = nullptr;
	Output *pnum_shards = nullptr;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ShardedFilename - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "basename")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pbasename = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pbasename = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : ShardedFilename - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "shard")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pshard = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pshard = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : ShardedFilename - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_shards")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pnum_shards = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pnum_shards = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : ShardedFilename - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : ShardedFilename pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && pbasename && pshard && pnum_shards)
	{
		pShardedFilename = new ShardedFilename(*pScope, *pbasename, *pshard, *pnum_shards);
		ObjectInfo* pObj = AddObjectMap(pShardedFilename, id, SYMBOL_SHARDEDFILENAME, "ShardedFilename", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pShardedFilename->filename, OUTPUT_TYPE_OUTPUT, "filename");
		}
	}
	else
	{
		std::string msg = string_format("error : ShardedFilename(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pShardedFilename;
}

void* Create_ShardedFilespec(std::string id, Json::Value pInputItem) {
	ShardedFilespec* pShardedFilespec = nullptr;
	Scope* pScope = nullptr;
	Output *pbasename = nullptr;
	Output *pnum_shards = nullptr;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : ShardedFilespec - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "basename")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pbasename = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pbasename = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : ShardedFilespec - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_shards")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pnum_shards = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pnum_shards = (Output*)Create_StrToOutput(*m_pScope, strAutoPinType, "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : ShardedFilespec - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : ShardedFilespec pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}
	if (pScope && pbasename && pnum_shards)
	{
		pShardedFilespec = new ShardedFilespec(*pScope, *pbasename, *pnum_shards);
		ObjectInfo* pObj = AddObjectMap(pShardedFilespec, id, SYMBOL_SHARDEDFILESPEC, "ShardedFilespec", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pShardedFilespec->filename, OUTPUT_TYPE_OUTPUT, "filename");
		}
	}
	else
	{
		std::string msg = string_format("error : ShardedFilespec(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pShardedFilespec;
}

void* Create_TFRecordReader(std::string id, Json::Value pInputItem) {
	TFRecordReader* pTFRecordReader = nullptr;
	Scope* pScope = nullptr;
	TFRecordReader::Attrs attrs;
	int iSize = (int)pInputItem.size();
	std::string container_;
	std::string shared_name_;
	std::string compression_type_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : TFRecordReader - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "TFRecordReader::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs = attrs.Container(container_);
				}
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs = attrs.SharedName(shared_name_);
				}
				if (attrParser.GetAttribute("compression_type_") != "")
				{
					compression_type_ = attrParser.GetAttribute("compression_type_");
					attrs = attrs.CompressionType(compression_type_);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : TFRecordReader pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pTFRecordReader = new TFRecordReader(*pScope, attrs);
		ObjectInfo* pObj = AddObjectMap(pTFRecordReader, id, SYMBOL_TFRECORDREADER, "TFRecordReader", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pTFRecordReader->reader_handle, OUTPUT_TYPE_OUTPUT, "reader_handle");
		}
	}
	else
	{
		std::string msg = string_format("error : TFRecordReader(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pTFRecordReader;
}

void* Create_TextLineReader(std::string id, Json::Value pInputItem) {
	TextLineReader* pTextLineReader = nullptr;
	Scope* pScope = nullptr;
	TextLineReader::Attrs attrs;
	int iSize = (int)pInputItem.size();
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : TextLineReader - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "TextLineReader::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("skip_header_lines_") != "") attrs = attrs.SkipHeaderLines(attrParser.ConvStrToInt64(attrParser.GetAttribute("skip_header_lines_")));
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs = attrs.Container(container_);
				}
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs = attrs.SharedName(shared_name_);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : TextLineReader pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pTextLineReader = new TextLineReader(*pScope, attrs);
		ObjectInfo* pObj = AddObjectMap(pTextLineReader, id, SYMBOL_TEXTLINEREADER, "TextLineReader", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pTextLineReader->reader_handle, OUTPUT_TYPE_OUTPUT, "reader_handle");
		}
	}
	else
	{
		std::string msg = string_format("error : TextLineReader(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pTextLineReader;
}

void* Create_WholeFileReader(std::string id, Json::Value pInputItem) {
	WholeFileReader* pWholeFileReader = nullptr;
	Scope* pScope = nullptr;
	WholeFileReader::Attrs attrs;
	int iSize = (int)pInputItem.size();
	std::string container_;
	std::string shared_name_;
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : WholeFileReader - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "WholeFileReader::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("container_") != "")
				{
					container_ = attrParser.GetAttribute("container_");
					attrs = attrs.Container(container_);
				}
				if (attrParser.GetAttribute("shared_name_") != "")
				{
					shared_name_ = attrParser.GetAttribute("shared_name_");
					attrs = attrs.SharedName(shared_name_);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : WholeFileReader pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope)
	{
		pWholeFileReader = new WholeFileReader(*pScope, attrs);
		ObjectInfo* pObj = AddObjectMap(pWholeFileReader, id, SYMBOL_WHOLEFILEREADER, "WholeFileReader", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pWholeFileReader->reader_handle, OUTPUT_TYPE_OUTPUT, "reader_handle");
		}
	}
	else
	{
		std::string msg = string_format("error : WholeFileReader(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pWholeFileReader;
}

void* Create_WriteFile(std::string id, Json::Value pInputItem) {
	tensorflow::ops::WriteFile* pWriteFile = nullptr;
	Scope* pScope = nullptr;
	Output *pfilename = nullptr;
	Output *pcontents = nullptr;
	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
		std::string strAutoPinType = ItemValue.get("pin-datatype", "").asString();						//DT_DOUBLE
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : WriteFile - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "filename")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pfilename = (Output*)pOutputObj->pOutput;
						}
					}
				}
				else
				{
					if (!strPinInitial.empty())
						pfilename = (Output*)Create_StrToOutput(*m_pScope, "DT_STRING", "", strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : WriteFile - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "contents")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pcontents = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : WriteFile - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else
		{
			std::string msg = string_format("warning : WriteFile pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && pfilename && pcontents)
	{
		pWriteFile = new tensorflow::ops::WriteFile(*pScope, *pfilename, *pcontents);
		ObjectInfo* pObj = AddObjectMap(pWriteFile, id, SYMBOL_WRITEFILE, "WriteFile", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pWriteFile->operation, OUTPUT_TYPE_OPERATION, "operation");
		}
	}
	else
	{
		std::string msg = string_format("error : WriteFile(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pWriteFile;
}