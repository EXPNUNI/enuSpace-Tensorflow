#pragma once


#ifndef _TF_IO_OPS_HEADER_
#define _TF_IO_OPS_HEADER_

#include <string>
#include "jsoncpp/json.h"

void* Create_FixedLengthRecordReader(std::string id, Json::Value pInputItem);
void* Create_IdentityReader(std::string id, Json::Value pInputItem);
void* Create_LMDBReader(std::string id, Json::Value pInputItem);
void* Create_MatchingFiles(std::string id, Json::Value pInputItem);
void* Create_MergeV2Checkpoints(std::string id, Json::Value pInputItem);
void* Create_ReadFile(std::string id, Json::Value pInputItem);
void* Create_ReaderNumRecordsProduced(std::string id, Json::Value pInputItem);
void* Create_ReaderNumWorkUnitsCompleted(std::string id, Json::Value pInputItem);
void* Create_ReaderRead(std::string id, Json::Value pInputItem);
void* Create_ReaderReadUpTo(std::string id, Json::Value pInputItem);
void* Create_ReaderReset(std::string id, Json::Value pInputItem);
void* Create_ReaderRestoreState(std::string id, Json::Value pInputItem);
void* Create_ReaderSerializeState(std::string id, Json::Value pInputItem);
void* Create_Restore(std::string id, Json::Value pInputItem);
void* Create_RestoreSlice(std::string id, Json::Value pInputItem);
void* Create_RestoreV2(std::string id, Json::Value pInputItem);
void* Create_Save(std::string id, Json::Value pInputItem);
void* Create_SaveSlices(std::string id, Json::Value pInputItem);
void* Create_SaveV2(std::string id, Json::Value pInputItem);
void* Create_ShardedFilename(std::string id, Json::Value pInputItem);
void* Create_ShardedFilespec(std::string id, Json::Value pInputItem);
void* Create_TFRecordReader(std::string id, Json::Value pInputItem);
void* Create_TextLineReader(std::string id, Json::Value pInputItem);
void* Create_WholeFileReader(std::string id, Json::Value pInputItem);
void* Create_WriteFile(std::string id, Json::Value pInputItem);

#endif