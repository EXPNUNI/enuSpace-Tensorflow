#include "stdafx.h"
#include "tf_candidate_sampling_ops.h"

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

void* Create_AllCandidateSampler(std::string id, Json::Value pInputItem) {
	AllCandidateSampler* pAllCandidateSampler = nullptr;
	Scope* pScope = nullptr;
	Output* true_classes = nullptr;
	int64 num_true;
	int64 num_sampled;
	bool unique;
	AllCandidateSampler::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
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
				std::string msg = string_format("warning : AllCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "true_classes")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, "output");
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							true_classes = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : AllCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_true")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				num_true = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : AllCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_sampled")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				num_sampled = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : AllCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "unique")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "bool")
			{
				if (strPinInitial == "true" || strPinInitial == "TRUE" || strPinInitial == "1")
					unique = true;
				else
					unique = false;
			}
			else
			{
				std::string msg = string_format("warning : AllCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "AllCandidateSampler::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("seed_") != "")
				{
					attrs.Seed(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed_")));
				}
				if (attrParser.GetAttribute("seed2_") != "")
				{
					attrs.Seed2(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed2_")));
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : AllCandidateSampler pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && true_classes)
	{
		pAllCandidateSampler = new AllCandidateSampler(*pScope, *true_classes, num_true, num_sampled, unique, attrs);
		ObjectInfo* pObj = AddObjectMap(pAllCandidateSampler, id, SYMBOL_ALLCANDIDATESAMPLER, "AllCandidateSampler", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pAllCandidateSampler->sampled_candidates, OUTPUT_TYPE_OUTPUT, "sampled_candidates");
			AddOutputInfo(pObj, &pAllCandidateSampler->true_expected_count, OUTPUT_TYPE_OUTPUT, "true_expected_count");
			AddOutputInfo(pObj, &pAllCandidateSampler->sampled_expected_count, OUTPUT_TYPE_OUTPUT, "sampled_expected_count");
		}
	}
	else
	{
		std::string msg = string_format("error : AllCandidateSampler(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pAllCandidateSampler;
}

void* Create_ComputeAccidentalHits(std::string id, Json::Value pInputItem) {
	ComputeAccidentalHits* pComputeAccidentalHits = nullptr;
	Scope* pScope = nullptr;
	Output* true_classes = nullptr;
	Output* sampled_candidates = nullptr;
	int64 num_true;
	ComputeAccidentalHits::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
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
				std::string msg = string_format("warning : ComputeAccidentalHits - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "true_classes")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, "output");
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							true_classes = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ComputeAccidentalHits - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "sampled_candidates")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, "output");
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							sampled_candidates = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ComputeAccidentalHits - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_true")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				num_true = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : ComputeAccidentalHits - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "ComputeAccidentalHits::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("seed_") != "")
				{
					attrs.Seed(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed_")));
				}
				if (attrParser.GetAttribute("seed2_") != "")
				{
					attrs.Seed2(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed2_")));
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : ComputeAccidentalHits pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && true_classes && sampled_candidates)
	{
		pComputeAccidentalHits = new ComputeAccidentalHits(*pScope, *true_classes, *sampled_candidates, num_true, attrs);
		ObjectInfo* pObj = AddObjectMap(pComputeAccidentalHits, id, SYMBOL_COMPUTEACCIDENTALHITS, "ComputeAccidentalHits", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pComputeAccidentalHits->indices, OUTPUT_TYPE_OUTPUT, "indices");
			AddOutputInfo(pObj, &pComputeAccidentalHits->ids, OUTPUT_TYPE_OUTPUT, "ids");
			AddOutputInfo(pObj, &pComputeAccidentalHits->weights, OUTPUT_TYPE_OUTPUT, "weights");
		}
	}
	else
	{
		std::string msg = string_format("error : ComputeAccidentalHits(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pComputeAccidentalHits;
}

void* Create_FixedUnigramCandidateSampler(std::string id, Json::Value pInputItem) {
	FixedUnigramCandidateSampler* pFixedUnigramCandidateSampler = nullptr;
	Scope* pScope = nullptr;
	Output* true_classes = nullptr;
	int64 num_true;
	int64 num_sampled;
	bool unique;
	int64 range_max;
	FixedUnigramCandidateSampler::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
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
				std::string msg = string_format("warning : FixedUnigramCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "true_classes")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, "output");
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							true_classes = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : FixedUnigramCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_true")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				num_true = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : FixedUnigramCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_sampled")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				num_sampled = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : FixedUnigramCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "unique")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "bool")
			{
				if (strPinInitial == "true" || strPinInitial == "TRUE" || strPinInitial == "1")
					unique = true;
				else
					unique = false;
			}
			else
			{
				std::string msg = string_format("warning : FixedUnigramCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "range_max")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				range_max = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : FixedUnigramCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "FixedUnigramCandidateSampler::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("vocab_file_") != "")
				{
					attrs.VocabFile(attrParser.ConvStrToStringPiece(attrParser.GetAttribute("vocab_file_")));
				}
				if (attrParser.GetAttribute("distortion_") != "")
				{
					attrs.Distortion(attrParser.ConvStrToFloat(attrParser.GetAttribute("distortion_")));
				}
				if (attrParser.GetAttribute("num_reserved_ids_") != "")
				{
					attrs.NumReservedIds(attrParser.ConvStrToInt64(attrParser.GetAttribute("num_reserved_ids_")));
				}
				if (attrParser.GetAttribute("num_shards_") != "")
				{
					attrs.NumShards(attrParser.ConvStrToInt64(attrParser.GetAttribute("num_shards_")));
				}
				if (attrParser.GetAttribute("shard_") != "")
				{
					attrs.Shard(attrParser.ConvStrToInt64(attrParser.GetAttribute("shard_")));
				}
				if (attrParser.GetAttribute("unigrams_") != "")
				{
					attrs.Unigrams(attrParser.ConvStrToArraySlicefloat(attrParser.GetAttribute("unigrams_")));
				}
				if (attrParser.GetAttribute("seed_") != "")
				{
					attrs.Seed(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed_")));
				}
				if (attrParser.GetAttribute("seed2_") != "")
				{
					attrs.Seed2(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed2_")));
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : FixedUnigramCandidateSampler pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && true_classes)
	{
		pFixedUnigramCandidateSampler = new FixedUnigramCandidateSampler(*pScope, *true_classes, num_true, num_sampled, unique, range_max, attrs);
		ObjectInfo* pObj = AddObjectMap(pFixedUnigramCandidateSampler, id, SYMBOL_FIXEDUNIGRAMCANDIDATESAMPLER, "FixedUnigramCandidateSampler", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pFixedUnigramCandidateSampler->sampled_candidates, OUTPUT_TYPE_OUTPUT, "sampled_candidates");
			AddOutputInfo(pObj, &pFixedUnigramCandidateSampler->true_expected_count, OUTPUT_TYPE_OUTPUT, "true_expected_count");
			AddOutputInfo(pObj, &pFixedUnigramCandidateSampler->sampled_expected_count, OUTPUT_TYPE_OUTPUT, "sampled_expected_count");
		}
	}
	else
	{
		std::string msg = string_format("error : FixedUnigramCandidateSampler(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pFixedUnigramCandidateSampler;
}

void* Create_LearnedUnigramCandidateSampler(std::string id, Json::Value pInputItem) {
	LearnedUnigramCandidateSampler* pLearnedUnigramCandidateSampler = nullptr;
	Scope* pScope = nullptr;
	Output* true_classes = nullptr;
	int64 num_true;
	int64 num_sampled;
	bool unique;
	int64 range_max;
	LearnedUnigramCandidateSampler::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
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
				std::string msg = string_format("warning : LearnedUnigramCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "true_classes")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, "output");
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							true_classes = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : LearnedUnigramCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_true")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				num_true = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : LearnedUnigramCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_sampled")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				num_sampled = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : LearnedUnigramCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "unique")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "bool")
			{
				if (strPinInitial == "true" || strPinInitial == "TRUE" || strPinInitial == "1")
					unique = true;
				else
					unique = false;
			}
			else
			{
				std::string msg = string_format("warning : LearnedUnigramCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "range_max")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				range_max = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : LearnedUnigramCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "LearnedUnigramCandidateSampler::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("seed_") != "")
				{
					attrs.Seed(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed_")));
				}
				if (attrParser.GetAttribute("seed2_") != "")
				{
					attrs.Seed2(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed2_")));
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : LearnedUnigramCandidateSampler pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && true_classes)
	{
		pLearnedUnigramCandidateSampler = new LearnedUnigramCandidateSampler(*pScope, *true_classes, num_true, num_sampled, unique, range_max, attrs);
		ObjectInfo* pObj = AddObjectMap(pLearnedUnigramCandidateSampler, id, SYMBOL_LEARNEDUNIGRAMCANDIDATESAMPLER, "LearnedUnigramCandidateSampler", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pLearnedUnigramCandidateSampler->sampled_candidates, OUTPUT_TYPE_OUTPUT, "sampled_candidates");
			AddOutputInfo(pObj, &pLearnedUnigramCandidateSampler->true_expected_count, OUTPUT_TYPE_OUTPUT, "true_expected_count");
			AddOutputInfo(pObj, &pLearnedUnigramCandidateSampler->sampled_expected_count, OUTPUT_TYPE_OUTPUT, "sampled_expected_count");
		}
	}
	else
	{
		std::string msg = string_format("error : LearnedUnigramCandidateSampler(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pLearnedUnigramCandidateSampler;
}

void* Create_LogUniformCandidateSampler(std::string id, Json::Value pInputItem) {
	LogUniformCandidateSampler* pLogUniformCandidateSampler = nullptr;
	Scope* pScope = nullptr;
	Output* true_classes = nullptr;
	int64 num_true;
	int64 num_sampled;
	bool unique;
	int64 range_max;
	LogUniformCandidateSampler::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
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
				std::string msg = string_format("warning : LogUniformCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "true_classes")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, "output");
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							true_classes = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : LogUniformCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_true")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				num_true = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : LogUniformCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_sampled")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				num_sampled = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : LogUniformCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "unique")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "bool")
			{
				if (strPinInitial == "true" || strPinInitial == "TRUE" || strPinInitial == "1")
					unique = true;
				else
					unique = false;
			}
			else
			{
				std::string msg = string_format("warning : LogUniformCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "range_max")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				range_max = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : LogUniformCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "LogUniformCandidateSampler::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("seed_") != "")
				{
					attrs.Seed(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed_")));
				}
				if (attrParser.GetAttribute("seed2_") != "")
				{
					attrs.Seed2(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed2_")));
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : LogUniformCandidateSampler pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && true_classes)
	{
		pLogUniformCandidateSampler = new LogUniformCandidateSampler(*pScope, *true_classes, num_true, num_sampled, unique, range_max, attrs);
		ObjectInfo* pObj = AddObjectMap(pLogUniformCandidateSampler, id, SYMBOL_LOGUNIFORMCANDIDATESAMPLER, "LogUniformCandidateSampler", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pLogUniformCandidateSampler->sampled_candidates, OUTPUT_TYPE_OUTPUT, "sampled_candidates");
			AddOutputInfo(pObj, &pLogUniformCandidateSampler->true_expected_count, OUTPUT_TYPE_OUTPUT, "true_expected_count");
			AddOutputInfo(pObj, &pLogUniformCandidateSampler->sampled_expected_count, OUTPUT_TYPE_OUTPUT, "sampled_expected_count");
		}
	}
	else
	{
		std::string msg = string_format("error : LogUniformCandidateSampler(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pLogUniformCandidateSampler;
}

void* Create_UniformCandidateSampler(std::string id, Json::Value pInputItem) {
	UniformCandidateSampler* pUniformCandidateSampler = nullptr;
	Scope* pScope = nullptr;
	Output* true_classes = nullptr;
	int64 num_true;
	int64 num_sampled;
	bool unique;
	int64 range_max;
	UniformCandidateSampler::Attrs attrs;

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
		std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
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
				std::string msg = string_format("warning : UniformCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "true_classes")
		{
			if (strPinInterface == "Input")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, "output");
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							true_classes = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : UniformCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_true")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				num_true = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : UniformCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "num_sampled")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				num_sampled = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : UniformCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "unique")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "bool")
			{
				if (strPinInitial == "true" || strPinInitial == "TRUE" || strPinInitial == "1")
					unique = true;
				else
					unique = false;
			}
			else
			{
				std::string msg = string_format("warning : UniformCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "range_max")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "int64")
			{
				range_max = stoll(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : UniformCandidateSampler - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg);
			}
		}
		else if (strPinName == "attrs")
		{
			if (strPinInterface == "UniformCandidateSampler::Attrs")
			{
				CAttributeParser attrParser(strPinInterface, strPinInitial);
				if (attrParser.GetAttribute("seed_") != "")
				{
					attrs.Seed(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed_")));
				}
				if (attrParser.GetAttribute("seed2_") != "")
				{
					attrs.Seed2(attrParser.ConvStrToInt64(attrParser.GetAttribute("seed2_")));
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : UniformCandidateSampler pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg);
		}
	}

	if (pScope && true_classes)
	{
		pUniformCandidateSampler = new UniformCandidateSampler(*pScope, *true_classes, num_true, num_sampled, unique, range_max, attrs);
		ObjectInfo* pObj = AddObjectMap(pUniformCandidateSampler, id, SYMBOL_UNIFORMCANDIDATESAMPLER, "UniformCandidateSampler", pInputItem);
		if (pObj)
		{
			AddOutputInfo(pObj, &pUniformCandidateSampler->sampled_candidates, OUTPUT_TYPE_OUTPUT, "sampled_candidates");
			AddOutputInfo(pObj, &pUniformCandidateSampler->true_expected_count, OUTPUT_TYPE_OUTPUT, "true_expected_count");
			AddOutputInfo(pObj, &pUniformCandidateSampler->sampled_expected_count, OUTPUT_TYPE_OUTPUT, "sampled_expected_count");
		}
	}
	else
	{
		std::string msg = string_format("error : UniformCandidateSampler(%s) Object create failed.", id.c_str());
		PrintMessage(msg);
	}
	return pUniformCandidateSampler;
}
