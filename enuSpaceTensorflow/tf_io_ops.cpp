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

void* Create_FixedLengthRecordReader(std::string id, Json::Value pInputItem) {
	FixedLengthRecordReader* pFixedLengthRecordReader = nullptr;
	Scope* pScope = nullptr;
	return pFixedLengthRecordReader;
}

void* Create_IdentityReader(std::string id, Json::Value pInputItem) {
	IdentityReader* pIdentityReader = nullptr;
	Scope* pScope = nullptr;
	return pIdentityReader;
}

void* Create_MatchingFiles(std::string id, Json::Value pInputItem) {
	MatchingFiles* pMatchingFiles = nullptr;
	Scope* pScope = nullptr;
	return pMatchingFiles;
}

void* Create_MergeV2Checkpoints(std::string id, Json::Value pInputItem) {
	MergeV2Checkpoints* pMergeV2Checkpoints = nullptr;
	Scope* pScope = nullptr;
	return pMergeV2Checkpoints;
}

void* Create_ReadFile(std::string id, Json::Value pInputItem) {
	//ReadFile* pReadFile = nullptr;
	Scope* pScope = nullptr;
	return NULL;
}

void* Create_ReaderNumRecordsProduced(std::string id, Json::Value pInputItem) {
	ReaderNumRecordsProduced* pReaderNumRecordsProduced = nullptr;
	Scope* pScope = nullptr;
	return pReaderNumRecordsProduced;
}

void* Create_ReaderNumWorkUnitsCompleted(std::string id, Json::Value pInputItem) {
	ReaderNumWorkUnitsCompleted* pReaderNumWorkUnitsCompleted = nullptr;
	Scope* pScope = nullptr;
	return pReaderNumWorkUnitsCompleted;
}

void* Create_ReaderRead(std::string id, Json::Value pInputItem) {
	ReaderRead* pReaderRead = nullptr;
	Scope* pScope = nullptr;
	return pReaderRead;
}

void* Create_ReaderReadUpTo(std::string id, Json::Value pInputItem) {
	ReaderReadUpTo* pReaderReadUpTo = nullptr;
	Scope* pScope = nullptr;
	return pReaderReadUpTo;
}

void* Create_ReaderReset(std::string id, Json::Value pInputItem) {
	ReaderReset* pReaderReset = nullptr;
	Scope* pScope = nullptr;
	return pReaderReset;
}

void* Create_ReaderRestoreState(std::string id, Json::Value pInputItem) {
	ReaderRestoreState* pReaderRestoreState = nullptr;
	Scope* pScope = nullptr;
	return pReaderRestoreState;
}

void* Create_ReaderSerializeState(std::string id, Json::Value pInputItem) {
	ReaderSerializeState* pReaderSerializeState = nullptr;
	Scope* pScope = nullptr;
	return pReaderSerializeState;
}

void* Create_Restore(std::string id, Json::Value pInputItem) {
	Restore* pRestore = nullptr;
	Scope* pScope = nullptr;
	return pRestore;
}

void* Create_RestoreSlice(std::string id, Json::Value pInputItem) {
	RestoreSlice* pRestoreSlice = nullptr;
	Scope* pScope = nullptr;
	return pRestoreSlice;
}

void* Create_RestoreV2(std::string id, Json::Value pInputItem) {
	RestoreV2* pRestoreV2 = nullptr;
	Scope* pScope = nullptr;
	return pRestoreV2;
}

void* Create_Save(std::string id, Json::Value pInputItem) {
	Save* pSave = nullptr;
	Scope* pScope = nullptr;
	return pSave;
}

void* Create_SaveSlices(std::string id, Json::Value pInputItem) {
	SaveSlices* pSaveSlices = nullptr;
	Scope* pScope = nullptr;
	return pSaveSlices;
}

void* Create_SaveV2(std::string id, Json::Value pInputItem) {
	SaveV2* pSaveV2 = nullptr;
	Scope* pScope = nullptr;
	return pSaveV2;
}

void* Create_ShardedFilename(std::string id, Json::Value pInputItem) {
	ShardedFilename* pShardedFilename = nullptr;
	Scope* pScope = nullptr;
	return pShardedFilename;
}

void* Create_ShardedFilespec(std::string id, Json::Value pInputItem) {
	ShardedFilespec* pShardedFilespec = nullptr;
	Scope* pScope = nullptr;
	return pShardedFilespec;
}

void* Create_TFRecordReader(std::string id, Json::Value pInputItem) {
	TFRecordReader* pTFRecordReader = nullptr;
	Scope* pScope = nullptr;
	return pTFRecordReader;
}

void* Create_TextLineReader(std::string id, Json::Value pInputItem) {
	TextLineReader* pTextLineReader = nullptr;
	Scope* pScope = nullptr;
	return pTextLineReader;
}

void* Create_WholeFileReader(std::string id, Json::Value pInputItem) {
	WholeFileReader* pWholeFileReader = nullptr;
	Scope* pScope = nullptr;
	return pWholeFileReader;
}

void* Create_WriteFile(std::string id, Json::Value pInputItem) {
	//WriteFile* pWriteFile = nullptr;
	Scope* pScope = nullptr;
	return NULL;
}