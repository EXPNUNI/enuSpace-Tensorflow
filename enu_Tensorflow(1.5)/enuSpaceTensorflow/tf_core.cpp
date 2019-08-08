#include "stdafx.h"
#include "tf_core.h"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "include/json/json.h"

#include "GlobalHeader.h"
#include "tensorflow.h"
#include "utility_functions.h"
#include "enuSpaceToTensorflow.h"

// ClientSession에 연결된 연결선 정보를 따라서 Fect 벡터리스트에 추가하여 실행시 각 객체의 값을 업데이트 수행함
void AddPrevFetchObject(ObjectInfo* pfetchObj, FetchInfo* pFetchInfo)
{
	if (pfetchObj)
	{
		int iSize = (int)pfetchObj->param.size();
		for (int subindex = 0; subindex < iSize; ++subindex)
		{
			Json::Value ItemValue = pfetchObj->param[subindex];

			std::string strPinName = ItemValue.get("pin-name", "").asString();								// val
			std::string strPinType = ItemValue.get("pin-type", "").asString();								// double
			std::string strPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
			std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
			std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
			std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
			std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
			std::string strPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
			std::string strPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

			if (strInSymbolId.empty() == false && strInSymbolPinName.empty() == false)
			{
				ObjectInfo* pPrevfetchObj = LookupFromObjectMap(strInSymbolId);
				if (pPrevfetchObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pPrevfetchObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->type == OUTPUT_TYPE_OUTPUT)
						{
							// 이전에 추가한 객체인지 확인후 리스트에 추가 수행.
							const std::map<std::string, ObjectInfo*>::const_iterator aLookup = pFetchInfo->fetch_object_map.find(strInSymbolId);
							const bool bExists = aLookup != pFetchInfo->fetch_object_map.end();
							if (bExists == false)
							{
								pFetchInfo->fetch_object_map.insert(std::pair<std::string, ObjectInfo*>(strInSymbolId, pPrevfetchObj));

								pFetchInfo->output.fetch_object.push_back(pPrevfetchObj);
								pFetchInfo->output.fetch_outputs.push_back(*(Output*)pOutputObj->pOutput);
								pFetchInfo->output.pin_names.push_back(strInSymbolPinName);

								// 이전 심볼에 대하여 OUTPUT 객체이면 추가 수행.
								AddPrevFetchObject(pPrevfetchObj, pFetchInfo);
							}
						}
						else if (pOutputObj->type == OUTPUT_TYPE_OUTPUTLIST)
						{
							// 이전에 추가한 객체인지 확인후 리스트에 추가 수행.
							const std::map<std::string, ObjectInfo*>::const_iterator aLookup = pFetchInfo->fetch_object_map.find(strInSymbolId);

							const bool bExists = aLookup != pFetchInfo->fetch_object_map.end();
							if (bExists == false)
							{
								AddPrevFetchObject(pPrevfetchObj, pFetchInfo);
							}
						}
						// OUTPUT_TYPE_OUTPUT_ETC의 타입으로 AddSymbolicGradient객체가 있음. 본 로직을 만났을 경우, 업데이트 수행은 하지 않지만 뒤로 검색은 진행함.
						else if (pOutputObj->type == OUTPUT_TYPE_OUTPUT_ETC)
						{
							// 이전에 추가한 객체인지 확인후 리스트에 추가 수행.
							const std::map<std::string, ObjectInfo*>::const_iterator aLookup = pFetchInfo->fetch_object_map.find(strInSymbolId);

							const bool bExists = aLookup != pFetchInfo->fetch_object_map.end();
							if (bExists == false)
							{
								AddPrevFetchObject(pPrevfetchObj, pFetchInfo);
							}
						}
					}
				}
			}
		}
	}
}

void* Create_ClientSession(std::string id, Json::Value pInputItem) {
	Scope* pScope = nullptr;
	ClientSession* pSession = nullptr;
	ObjectInfo* pSessionObj = nullptr;
	FetchInfo* pFetchInfo = nullptr;
	std::string msgParam = string_format("#%s", id.c_str());

	std::string  device = "/cpu:0";      // device set interface

	int iSize = (int)pInputItem.size();
	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();                        // val
		std::string strPinType = ItemValue.get("pin-type", "").asString();                        // double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();                  // 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();               // ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();                  // ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();         // ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();   // ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();               // tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();                     // [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Scope")
			{
				if (m_pScope)
				{
					ClientSession* pCreateSession = new ClientSession(*m_pScope);
					pSession = pCreateSession;

					pSessionObj = AddObjectMap(pSession, id, SYMBOL_CLIENTSESSION, "ClientSession", pInputItem);
					pSessionObj->pScope = m_pScope;      // Scope 객체 설정 (실행시 Scope가 ok 상태인지 체크후 실행

					pFetchInfo = AddRunObjectMap(pSessionObj);
				}
				else
				{
					std::string msg = string_format("warning : ClientSession - %s(%s) scope information missed.", id.c_str(), strPinName.c_str());
					PrintMessage(msg, msgParam);
				}
			}
			else
			{
				std::string msg = string_format("warning : ClientSession - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "inputs")
		{
			if (strPinInterface == "FeedType")
			{
				if (pFetchInfo)
				{
					ObjectInfo* pfetchObj = LookupFromObjectMap(strInSymbolId);
					if (pfetchObj)
					{
						if (pfetchObj->type == SYMBOL_FEEDTYPE)
						{
							FeedTypeObject* pFeedType = (FeedTypeObject*)pfetchObj->pObject;
							pFetchInfo->FeedType.insert({ *pFeedType->pOutput, *pFeedType->pInitializer });
						}
						else
						{
							std::string msg = string_format("warning : ClientSession - %s(%s) transfer information missed (lookup output map).", id.c_str(), strPinName.c_str());
							PrintMessage(msg, msgParam);
						}
					}
				}
			}
		}
		else if (strPinName == "fetch_outputs")
		{
			if (strPinInterface == "std::vector<tensorflow::Output>")
			{
				if (pFetchInfo)
				{
					ObjectInfo* pfetchObj = LookupFromObjectMap(strInSymbolId);
					if (pfetchObj)
					{
						OutputInfo* pOutputObj = LookupFromOutputMap(pfetchObj, strInSymbolPinName);
						if (pOutputObj)
						{
							if (pOutputObj->type == OUTPUT_TYPE_OUTPUT)
							{
								pFetchInfo->fetch_object_map.insert(std::pair<std::string, ObjectInfo*>(strInSymbolId, pfetchObj));

								pOutputObj->bConnect = true;
								pFetchInfo->output.fetch_object.push_back(pfetchObj);
								pFetchInfo->output.fetch_outputs.push_back(*(Output*)pOutputObj->pOutput);
								pFetchInfo->output.pin_names.push_back(strInSymbolPinName);

								// 이전 심볼에 대하여 OUTPUT 객체이면 추가 수행.
								AddPrevFetchObject(pfetchObj, pFetchInfo);
							}
							else if (pOutputObj->type == OUTPUT_TYPE_OUTPUTLIST)
							{
								pFetchInfo->fetch_object_map.insert(std::pair<std::string, ObjectInfo*>(strInSymbolId, pfetchObj));

								pOutputObj->bConnect = true;
								pFetchInfo->output_list.fetch_object.push_back(pfetchObj);
								pFetchInfo->output_list.fetch_outputs.push_back(*(OutputList*)pOutputObj->pOutput);
								pFetchInfo->output_list.pin_names.push_back(strInSymbolPinName);

								// ClientSession에 OUTPUTLIST객체를 만났을 경우에만 추가를 수행하고, 뒤에 나오는 객체에 대해서는 추가하지 않음.
								// AddPrevFetchObject(pfetchObj, pFetchInfo);
							}
							// OUTPUT_TYPE_OUTPUT_ETC의 타입으로 AddSymbolicGradient객체가 있음. 본 로직을 만났을 경우, 업데이트 수행은 하지 않지만 뒤로 검색은 진행함.
							else if (pOutputObj->type == OUTPUT_TYPE_OUTPUT_ETC)
							{
								AddPrevFetchObject(pfetchObj, pFetchInfo);
							}
							else
							{
								std::string msg = string_format("warning : ClientSession - %s(%s) transfer information missed (lookup type is not output).", id.c_str(), strPinName.c_str());
								PrintMessage(msg, msgParam);
							}
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ClientSession - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "run_outputs")
		{
			if (strPinInterface == "std::vector<tensorflow::Operation>")
			{
				if (pFetchInfo)
				{
					ObjectInfo* pfetchObj = LookupFromObjectMap(strInSymbolId);
					if (pfetchObj)
					{
						OutputInfo* pOutputObj = LookupFromOutputMap(pfetchObj, strInSymbolPinName);
						if (pOutputObj)
						{
							if (pOutputObj->type == OUTPUT_TYPE_OPERATION)
							{
								pFetchInfo->fetch_object_map.insert(std::pair<std::string, ObjectInfo*>(strInSymbolId, pfetchObj));

								pFetchInfo->output.run_outputs.push_back(*(Operation*)pOutputObj->pOutput);
								pFetchInfo->output.pin_names.push_back(strInSymbolPinName);
							}
							else if (pOutputObj->type == OUTPUT_TYPE_OUTPUT)
							{
								std::string msg = string_format("warning : ClientSession - %s(%s) not support output object.", id.c_str(), strPinName.c_str());
								PrintMessage(msg, msgParam);
							}
							else if (pOutputObj->type == OUTPUT_TYPE_OUTPUTLIST)
							{
								std::string msg = string_format("warning : ClientSession - %s(%s) not support operationlist object.", id.c_str(), strPinName.c_str());
								PrintMessage(msg, msgParam);
							}
							else
							{
								std::string msg = string_format("warning : ClientSession - %s(%s) transfer information missed (lookup type is not output).", id.c_str(), strPinName.c_str());
								PrintMessage(msg, msgParam);
							}
						}
						else
						{
							std::string msg = string_format("warning : ClientSession - %s(%s) transfer information missed (lookup output map).", id.c_str(), strPinName.c_str());
							PrintMessage(msg, msgParam);
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : ClientSession - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		// device set interface
		else if (strPinName == "device")
		{
			if (strPinInterface == "device" && !strPinInitial.empty())
			{
				device = strPinInitial;
			}
		}
	}
	
	if (pSession == nullptr)
	{
		std::string msg = string_format("error : ClientSession(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	else
	{
		if (m_pScope->ok() == false)
			return pSession;

		// device set interface
		GraphDef def;
		m_pScope->ToGraphDef(&def);

		for (int i = 0; i < def.node_size(); ++i)
		{
			auto node = def.mutable_node(i);
			if (node->device().empty()) 
			{
				node->set_device(device);
			}
		}

		////////////////////////////////////////////////////////////////////////////////////
		// init variables
		std::map<std::string, ObjectInfo*>::iterator vit;

		std::string strPinSave;
		std::string strBinPinType;
		std::string strBinPinShape;
		int iPinBinPos = -1;

		for (vit = m_ObjectMapList.begin(); vit != m_ObjectMapList.end(); ++vit)
		{
			ObjectInfo* pTar = vit->second;
			if (pTar)
			{
				if (pTar->type == SYMBOL_VARIABLE)
				{
					tensorflow::DataType dtype = DT_DOUBLE;
					std::string strdatatype;
					std::string initvalues;
					bool bRun = false;

					int iSize = (int)pTar->param.size();
					for (int subindex = 0; subindex < iSize; ++subindex)
					{
						Json::Value ItemValue = pTar->param[subindex];

						std::string strSubPinName = ItemValue.get("pin-name", "").asString();								// val
						std::string strSubPinType = ItemValue.get("pin-type", "").asString();								// double
						std::string strSubPinInitial = ItemValue.get("pin-initial", "").asString();						// 1;2;3;4
						std::string strSubInSymbolName = ItemValue.get("in-symbol-name", "").asString();					// ""
						std::string strSubInSymbolId = ItemValue.get("in-symbol-id", "").asString();						// ""
						std::string strSubInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();			// ""
						std::string strSubInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();	// ""
						std::string strSubPinInterface = ItemValue.get("pin-interface", "").asString();					// tensorflow::Input::Initializer 
						std::string strSubPinShape = ItemValue.get("pin-shape", "").asString();							// [2][2]

						if (strSubPinName == "initvalues")
						{
							if (strSubPinInterface == "Input")
							{
								ObjectInfo* pObj = LookupFromObjectMap(strSubInSymbolId);
								if (pObj)
								{
									if (pObj->type == SYMBOL_CONST)
									{
										Output *pOutput = (Output*)(pObj->pObject);

										std::vector< Output > init_obj;
										std::vector<tensorflow::Tensor> outputs;
										auto assign = Assign(*m_pScope, ((Variable*)pTar->pObject)->ref, *pOutput);
										init_obj.push_back(assign);

										Status st;
										st = pSession->Run(init_obj, &outputs);
										if (st.code() != error::OK)
										{
											std::string msg = string_format("error: %s.", st.error_message().c_str());
											PrintMessage(msg, msgParam);
										}

										bRun = true;
										break;
									}
								}
								else
								{
									initvalues = strSubPinInitial;
									strBinPinType = strSubPinType;
									strBinPinShape = strSubPinShape;

									strPinSave = ItemValue.get("pin-save", "").asString();
									if (strPinSave == "binary")
									{
										std::string strPinBinPos = ItemValue.get("pin-binary-pos", "").asString();
										iPinBinPos = stoi(strPinBinPos);
									}
								}
							}
						}
						else if (strSubPinName == "dtype")
						{
							if (!strSubPinInitial.empty())
							{
								strdatatype = strSubPinInitial;
							}
						}
					}

					if (bRun == false)
					{
						if (strPinSave == "binary") 
						{
							if (iPinBinPos != -1 && m_FileData)
							{
								Output* pOutput = nullptr;
								// float, [10]
								pOutput = (Output*)Create_BinaryToOutput(*m_pScope, strBinPinType, strBinPinShape, m_FileData, iPinBinPos);
								if (pOutput)
								{
									std::vector< Output > init_obj;
									std::vector<tensorflow::Tensor> outputs;
									auto assign = Assign(*m_pScope, ((Variable*)pTar->pObject)->ref, *pOutput);
									init_obj.push_back(assign);

									Status st;
									st = pSession->Run(init_obj, &outputs);
									if (st.code() != error::OK)
									{
										std::string msg = string_format("error: %s.", st.error_message().c_str());
										PrintMessage(msg, msgParam);
									}
								}
								else 
								{
									std::string msg = string_format("error : variable init - %s binary information missed 1.", pTar->id.c_str());
									PrintMessage(msg, msgParam);
								}
							}
							else
							{
								std::string msg = string_format("error : variable init - %s binary information missed 2.", pTar->id.c_str());
								PrintMessage(msg, msgParam);
							}
						}
						else
						{
							// initvalues = {1.0f, 2.0f}, datatype = DT_FLOAT, shape = [10] or null 
							if (!strdatatype.empty())
							{
								Output* pOutput = nullptr;
								pOutput = (Output*)Create_StrToOutput(*m_pScope, strdatatype, strBinPinShape, initvalues);
								if (pOutput)
								{
									std::vector< Output > init_obj;
									std::vector<tensorflow::Tensor> outputs;
									auto assign = Assign(*m_pScope, ((Variable*)pTar->pObject)->ref, *pOutput);
									init_obj.push_back(assign);

									Status st;
									st = pSession->Run(init_obj, &outputs);
									if (st.code() != error::OK)
									{
										std::string msg = string_format("error: %s.", st.error_message().c_str());
										PrintMessage(msg, msgParam);
									}
								}
								else
								{
									std::string msg = string_format("error : variable init - %s Create_StrToOutput information missed.", pTar->id.c_str());
									PrintMessage(msg, msgParam);
								}
							}
							// initvalues = 1.0;2.0 datatype= "", strBinPinType="string, int, float, bool, double", strBinPinShape = [10][10]
							else if (strdatatype.empty())
							{
								Output* pOutput = nullptr;
								pOutput = (Output*)Create_ArrayStrToOutput(*m_pScope, strBinPinType, strBinPinShape, initvalues);
								if (pOutput)
								{
									std::vector< Output > init_obj;
									std::vector<tensorflow::Tensor> outputs;
									auto assign = Assign(*m_pScope, ((Variable*)pTar->pObject)->ref, *pOutput);
									init_obj.push_back(assign);

									Status st;
									st = pSession->Run(init_obj, &outputs);
									if (st.code() != error::OK)
									{
										std::string msg = string_format("error: %s.", st.error_message().c_str());
										PrintMessage(msg, msgParam);
									}
								}
								else
								{
									std::string msg = string_format("error : variable init - %s Create_ArrayStrToOutput information missed.", pTar->id.c_str());
									PrintMessage(msg, msgParam);
								}
							}
							else
							{
								std::string msg = string_format("warning : variable init - %s text information missed.", pTar->id.c_str());
								PrintMessage(msg, msgParam);
							}
						}
					}
				}
			}
		}
		////////////////////////////////////////////////////////////////////////////////////
	}

	return pSession;
}

void* Create_Input_Initializer(std::string id, Json::Value pInputItem) {
	Input::Initializer* pInput_Initializer = nullptr;
	Scope* pScope = nullptr;
	Tensor* ptensor = nullptr;
	std::string msgParam = string_format("#%s", id.c_str());

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

		if (strPinName == "t")
		{
			if (strPinInterface == "Tensor")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					if (pObj->type == SYMBOL_TENSOR)
					{
						ptensor = (Tensor*)pObj->pObject;						
					}
				}
				else
				{
					if (!strPinInitial.empty())
						ptensor = Create_StrToTensor(strAutoPinType, "", strPinInitial);
					else
					{
						std::string strPinSave = ItemValue.get("pin-save", "").asString();
						if (strPinSave == "binary")
						{
							std::string strPinBinPos = ItemValue.get("pin-binary-pos", "").asString();
							int ipos = stoi(strPinBinPos);
							if (ipos != -1 && m_FileData)
							{
								ptensor = Create_BinaryToTensor(strPinType, strPinShape, m_FileData, ipos);
							}
							else
							{
								std::string msg = string_format("warning : %s(%s) binary data missed.", id.c_str(), strPinName.c_str());
								PrintMessage(msg, msgParam);
							}
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Initializer - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else
		{
			std::string msg = string_format("warning : Initializer pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}
	if (ptensor)
	{
		pInput_Initializer = new Input::Initializer(*ptensor);
		AddObjectMap(pInput_Initializer, id, SYMBOL_INPUT_INITIALIZER, "output", pInputItem);
	}

	return pInput_Initializer;
}

void* Create_InputList(std::string id, Json::Value pInputItem) {
	Scope* pScope = nullptr;
	OutputList* pInputList = nullptr;
	OutputList outputlist;
	int iCheck_Pintype = 0;
	std::string msgParam = string_format("#%s", id.c_str());


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

		if (strPinName == "outputlist")
		{
			if (strInSymbolPinInterface == "Input" || strInSymbolPinInterface == "Output" || strInSymbolPinInterface == "OutputList")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->type == OUTPUT_TYPE_OUTPUT)
						{
							outputlist.push_back(*((Output*)pOutputObj->pOutput));
						}
						else if (pOutputObj->type == OUTPUT_TYPE_OUTPUTLIST)
						{
							OutputList* outputcopylist = (OutputList*)pOutputObj->pOutput;
							outputlist.reserve(outputlist.size() + outputcopylist->size());
							outputlist.insert(outputlist.end(), outputcopylist->begin(), outputcopylist->end());
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : InputList - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else
		{
			std::string msg = string_format("warning : InputList pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (!outputlist.empty())
	{
		pInputList = new OutputList(outputlist);
		ObjectInfo* pObj = AddObjectMap(pInputList, id, SYMBOL_INPUTLIST, "output", pInputItem);
 		if (pObj)
 			AddOutputInfo(pObj, pInputList, OUTPUT_TYPE_OUTPUTLIST, "output");
	}
	else
	{
		std::string msg = string_format("error : InputList(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}

	return pInputList;
}

void* Create_Operation(std::string id, Json::Value pInputItem) {
	Operation* pOperation = nullptr;
	Scope* pScope = nullptr;
	Output* pInput = nullptr;
	int n_count = 0;
	std::string msgParam = string_format("#%s", id.c_str());

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

		if (strPinName == "operation")
		{
			if (strPinInterface == "Operation")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->type == OUTPUT_TYPE_OUTPUT)
						{
							pInput = (Output*)pOutputObj->pOutput;
							n_count++;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Operation - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else
		{
			std::string msg = string_format("warning : Operation pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pInput && n_count == 1)
	{
		pOperation = new Operation(pInput->node());
		ObjectInfo* pObj = AddObjectMap(pOperation, id, SYMBOL_OPERATION, "output", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, pOperation, OUTPUT_TYPE_OPERATION, "output");
	}
	else if (n_count > 1)
	{
		std::string msg = string_format("error : Operation(%s) more than one transfer is not supported.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	else
	{
		std::string msg = string_format("error : Operation(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	
	return pOperation;
}

void* Create_Output(std::string id, Json::Value pInputItem) {
	Output* pOutput = nullptr;
	Scope* pScope = nullptr;
	Output* pInOutput = nullptr;
	Operation* pOperation = nullptr;
	int index = 0;
	std::string msgParam = string_format("#%s", id.c_str());

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

		if (strPinName == "output")
		{
			if (strPinInterface == "Output")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->type == OUTPUT_TYPE_OUTPUT)
						{
							pInOutput = (Output*)pOutputObj->pOutput;
						}
						else if (pOutputObj->type == OUTPUT_TYPE_OPERATION)
						{
							pOperation = (Operation*)pOutputObj->pOutput;
						}
						else
						{
							std::string msg = string_format("warning : Output - %s(%s) pin type only support output, operation.", id.c_str(), strPinName.c_str());
							PrintMessage(msg, msgParam);
						}
					}
					else
					{
						std::string msg = string_format("warning : Output - %s(%s) could not find form output map.", id.c_str(), strPinName.c_str());
						PrintMessage(msg, msgParam);
					}
				}
				else
				{
					std::string msg = string_format("warning : Output - %s(%s)  could not find form object map.", id.c_str(), strPinName.c_str());
					PrintMessage(msg, msgParam);
				}
			}
			else
			{
				std::string msg = string_format("warning : Output - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "index")
		{
			if (strPinInterface == "int32")
			{
				if (strPinInitial == "")
				{
					index = 0;
				}
				else
				{
					index = stoll(strPinInitial);
				}
			}
			else
			{
				std::string msg = string_format("warning : Output - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else
		{
			std::string msg = string_format("warning : Output pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pInOutput)
	{
		if (pInOutput->node())
		{
			pOutput = new Output(pInOutput->node());
			ObjectInfo* pObj = AddObjectMap(pOutput, id, SYMBOL_OUTPUT, "Output", pInputItem);
			if (pObj)
				AddOutputInfo(pObj, pOutput, OUTPUT_TYPE_OUTPUT, "output");
		}
		else
		{
			std::string msg = string_format("warning : Output(%s) input pin object node is null.", id.c_str());
			PrintMessage(msg, msgParam);
		}
	}
	else if (pOperation)
	{
		pOutput = new Output(*pOperation, index);
		ObjectInfo* pObj = AddObjectMap(pOutput, id, SYMBOL_OUTPUT, "Output", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, pOutput, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("error : Output(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}

	return pOutput;
}

void* Create_Scope(std::string id, Json::Value pInputItem) {

	Scope* pRoot = new Scope(Scope::NewRootScope());
	m_pScope = pRoot;
	AddObjectMap(pRoot, id, SYMBOL_SCOPE, "Scope", pInputItem);
	return pRoot;
}

void* Create_Status(std::string id, Json::Value pInputItem) {
	Status* pStatus = nullptr;
	Scope* pScope = nullptr;
	std::string msgParam = string_format("#%s", id.c_str());

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

		if (strPinName == "status")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : tensorflow::Scope, 연결 핀 : tensorflow::Scope
			if (strPinInterface == "Status")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
					if (pOutputObj)
					{
						if (pOutputObj->pOutput)
						{
							pStatus = (Status*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : Status - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else
		{
			std::string msg = string_format("warning : Status pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pStatus)
	{
		ObjectInfo* pObj = AddObjectMap(pStatus, id, SYMBOL_STATUS, "Status", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, pStatus, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("error : Status(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}

	return pStatus;
}

void* Create_Tensor(std::string id, Json::Value pInputItem) {
	Tensor* pTensor = nullptr;

	tensorflow::TensorShape shape;
	std::string strinitvalues;
	std::string strdatatype;
	tensorflow::DataType dtype = DT_DOUBLE;

	std::string strPinSave;
	int iPinBinPos = -1;
	std::string strBinPinType;
	std::string strBinPinShape;
	std::string msgParam = string_format("#%s", id.c_str());

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

		if (strPinName == "dtype")
		{
			if (!strPinInitial.empty())
			{
				dtype = GetDatatypeFromInitial(strPinInitial);
				if (dtype == DT_INVALID)
				{
					std::string msg = string_format("warning : Tensor - %s(%s) unknown dtype", id.c_str(), strPinName.c_str());
					PrintMessage(msg, msgParam);
				}
				strdatatype = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : Tensor - %s(%s) dtype is not initialized", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "initvalues")
		{
			if (!strPinInitial.empty())
				strinitvalues = strPinInitial;
			else
			{
				strPinSave = ItemValue.get("pin-save", "").asString();
				std::string strPinBinPos = ItemValue.get("pin-binary-pos", "").asString();
				iPinBinPos = stoi(strPinBinPos);
				strBinPinType = strPinType;
				strBinPinShape = strPinShape;
			}
		}
		else
		{
			std::string msg = string_format("warning : Tensor pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (dtype != DT_INVALID)
	{
		if (strPinSave != "binary")
			pTensor = Create_StrToTensor(strdatatype, "", strinitvalues);
		else
		{
			if (iPinBinPos != -1 && m_FileData)
			{
				pTensor = Create_BinaryToTensor(strBinPinType, strBinPinShape, m_FileData, iPinBinPos);
				AddObjectMap(pTensor, id, SYMBOL_TENSOR, "Tensor", pInputItem);
				return pTensor;
			}
			else
			{
				std::string msg = string_format("warning : %s(%s) binary data missed.", id.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		AddObjectMap(pTensor, id, SYMBOL_TENSOR, "Tensor", pInputItem);
	}

	return pTensor;
}

void* Create_Input(std::string id, Json::Value pInputItem)
{
	Input* pInput = nullptr;
	tensorflow::TensorShape shape;
	tensorflow::DataType dtype;
	std::string strinitvalues;
	std::string strdatatype;
	Tensor* pTensor = nullptr;
	Output* pOutput = nullptr;

	std::string strPinSave;
	int iPinBinPos = -1;
	std::string strBinPinType;
	std::string strBinPinShape;
	std::string msgParam = string_format("#%s", id.c_str());

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

		if (strPinName == "dtype")
		{
			if (!strPinInitial.empty())
			{
				dtype = GetDatatypeFromInitial(strPinInitial);
				if (dtype == DT_INVALID)
				{
					std::string msg = string_format("warning : Input_ex - %s(%s) unknown dtype", id.c_str(), strPinName.c_str());
					PrintMessage(msg, msgParam);
				}
				strdatatype = strPinInitial;
			}
			else
			{
				std::string msg = string_format("warning : Input_ex - %s(%s) dtype is not initialized", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "initvalues")
		{
			if(strInSymbolPinInterface == "Tensor")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					if (pObj->type == SYMBOL_TENSOR)
					{
						pTensor = (Tensor*)pObj->pObject;
					}
				}
			}
			else if (!strPinInitial.empty())
			{
				strinitvalues = strPinInitial;
			}
			else
			{
				strPinSave = ItemValue.get("pin-save", "").asString();
				std::string strPinBinPos = ItemValue.get("pin-binary-pos", "").asString();
				iPinBinPos = stoi(strPinBinPos);
				strBinPinType = strPinType;
				strBinPinShape = strPinShape;

				if (strPinBinPos != "binary")
				{
					std::string msg = string_format("warning : Input - %s(%s) binary information missed.", id.c_str(), strPinName.c_str());
					PrintMessage(msg, msgParam);
				}
			}
		}
		else
		{
			std::string msg = string_format("warning : Input_ex pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pTensor)
	{
		pInput = new Input(*pTensor);
		ObjectInfo* pObj = AddObjectMap(pInput, id, SYMBOL_INPUT, "input", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, pInput, OUTPUT_TYPE_INPUT, "input");
	}
	else if(strinitvalues != "" && dtype)
	{
		pTensor = Create_StrToTensor(strdatatype, "", strinitvalues);
		pInput = new Input(*pTensor);
		ObjectInfo* pObj = AddObjectMap(pInput, id, SYMBOL_INPUT, "input", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, pInput, OUTPUT_TYPE_INPUT, "input");
	}
	else
	{
		if (strPinSave == "binary")
		{
			if (iPinBinPos != -1 && m_FileData)
			{
				pTensor = Create_BinaryToTensor(strBinPinType, strBinPinShape, m_FileData, iPinBinPos);
				pInput = new Input(*pTensor);
				ObjectInfo* pObj = AddObjectMap(pInput, id, SYMBOL_INPUT, "input", pInputItem);
				if (pObj)
					AddOutputInfo(pObj, pInput, OUTPUT_TYPE_INPUT, "input");
			}
			else
			{
				std::string msg = string_format("warning : %s(%s) binary data missed.", id.c_str());
				PrintMessage(msg, msgParam);
			}
		}
	}

	return pInput;
}

void* Create_FeedType(std::string id, Json::Value pInputItem)
{
	ClientSession::FeedType* pFeedType;
	Output* pinput = nullptr;
	Input::Initializer* initializer = nullptr;
	OutputHash* outputhash = nullptr;
	std::string msgParam = string_format("#%s", id.c_str());

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

		if (strPinName == "input")
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
							pinput = (Output*)pOutputObj->pOutput;
						}
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : FeedType - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "initializer")
		{
			if (strPinInterface == "Input::Initializer")
			{
				ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
				if (pObj)
				{
					if (pObj->type == SYMBOL_INPUT_INITIALIZER)
						initializer = (Input::Initializer*)pObj->pObject;
					else
					{
						std::string msg = string_format("warning : FeedType - %s(%s) not support another type(input initializer).", id.c_str(), strPinName.c_str());
						PrintMessage(msg, msgParam);
					}
				}
			}
			else
			{
				std::string msg = string_format("warning : FeedType - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "outputhash")
		{

		}
		else
		{
			std::string msg = string_format("warning : FeedType pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pinput && initializer)
	{
		FeedTypeObject* pFeedType = new FeedTypeObject;
		pFeedType->pOutput = pinput;
		pFeedType->pInitializer = initializer;

		AddObjectMap(pFeedType, id, SYMBOL_FEEDTYPE, "output", pInputItem);
	}
	return pFeedType;
}


void* Create_Const(std::string id, Json::Value pInputItem)
{
	Scope* pScope = nullptr;
	Output* pOutput = new Output();
	Tensor* pTensor = nullptr;
	std::string msgParam = string_format("#%s", id.c_str());

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
			// 입력심볼 : #Scope, 입력심볼의 핀 : Scope, 연결 핀 : Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Const - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "val")
		{
			if (strInSymbolPinName == "" && strPinInterface == "Input::Initializer")
			{
				std::string strPinSave = ItemValue.get("pin-save", "").asString();
				if (strPinSave == "binary")
				{
					std::string strPinBinPos = ItemValue.get("pin-binary-pos", "").asString();
					int iPinBinPos = stoi(strPinBinPos);
					std::string strBinPinType = strPinType;
					std::string strBinPinShape = strPinShape;

					if (iPinBinPos != -1 && m_FileData)
					{
						pTensor = Create_BinaryToTensor(strBinPinType, strBinPinShape, m_FileData, iPinBinPos);
					}
					else
					{
						std::string msg = string_format("warning : Const - %s(%s) binary information missed.", id.c_str(), strPinName.c_str());
						PrintMessage(msg, msgParam);
					}
				}
				else
				{
					std::vector<int64> array_slice;
					std::vector<int64> arraydims;
					GetArrayDimsFromShape(strPinShape, arraydims, array_slice);

					if (strPinType == "double")
					{
						std::vector<double> arrayvals;
						GetDoubleVectorFromInitial(strPinInitial, arrayvals);

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
					else if (strPinType == "float")
					{
						std::vector<float> arrayvals;
						GetFloatVectorFromInitial(strPinInitial, arrayvals);

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
					else if (strPinType == "int")
					{
						std::vector<int> arrayvals;
						GetIntVectorFromInitial(strPinInitial, arrayvals);

						gtl::ArraySlice< int64 > arraySlice(arraydims);
						pTensor = new Tensor(DT_INT32, TensorShape(arraySlice));

						int i = 0;
						for (std::vector<int>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
						{
							pTensor->flat<int>()(i) = *it;
							i++;
						}
						arraySlice.clear();
						arrayvals.clear();
					}
					else if (strPinType == "bool")
					{
						std::vector<bool> arrayvals;
						GetBoolVectorFromInitial(strPinInitial, arrayvals);

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
					else if (strPinType == "string")
					{
						std::vector<std::string> arrayvals;
						GetStringVectorFromInitial(strPinInitial, arrayvals);

						gtl::ArraySlice< int64 > arraySlice(arraydims);
						pTensor = new Tensor(DT_STRING, TensorShape(arraySlice));

						int i = 0;
						for (std::vector<std::string>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
						{
							pTensor->flat<std::string>()(i) = *it;
							i++;
						}
						arraySlice.clear();
						arrayvals.clear();
					}
					else
					{
						std::string msg = string_format("warning : Const - %s(val-initvalue) transfer information missed.", id.c_str());
						PrintMessage(msg, msgParam);
					}

					array_slice.clear();
					arraydims.clear();
				}
			}
		}
	}
	if (pScope == nullptr)
	{
		std::string msg = string_format("warning : Const - %s(scope) transfer information missed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	if (pTensor == nullptr)
	{
		std::string msg = string_format("warning : Const - %s(val) transfer information missed.", id.c_str());
		PrintMessage(msg, msgParam);
	}

	if (pScope && pTensor)
	{
		*pOutput = Const(*pScope, *pTensor);
		ObjectInfo* pObj = AddObjectMap(pOutput, id, SYMBOL_CONST, "Const", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, pOutput, OUTPUT_TYPE_OUTPUT, "output");
	}
	else
	{
		std::string msg = string_format("error : Const(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}

	if (pTensor)
		delete pTensor;

	return pOutput;
}

void* Create_Const_ex(std::string id, Json::Value pInputItem)
{
	Scope* pScope = nullptr;
	Output* pOutput = new Output();
	Tensor* pTensor = nullptr;
	DataType dtype = DT_DOUBLE;
	std::string strVal = "";
	std::string strShape = "";
	int iSize = (int)pInputItem.size();

	int iPinBinPos= -1;
	std::string strBinPinType;
	std::string strBinPinShape;
	std::string strPinSave;
	std::string msgParam = string_format("#%s", id.c_str());

	for (int subindex = 0; subindex < iSize; ++subindex)
	{
		Json::Value ItemValue = pInputItem[subindex];

		std::string strPinName = ItemValue.get("pin-name", "").asString();                        // val
		std::string strPinType = ItemValue.get("pin-type", "").asString();                        // double
		std::string strPinInitial = ItemValue.get("pin-initial", "").asString();                  // 1;2;3;4
		std::string strInSymbolName = ItemValue.get("in-symbol-name", "").asString();               // ""
		std::string strInSymbolId = ItemValue.get("in-symbol-id", "").asString();                  // ""
		std::string strInSymbolPinName = ItemValue.get("in-symbol-pin-name", "").asString();         // ""
		std::string strInSymbolPinInterface = ItemValue.get("in-symbol-pin-interface", "").asString();   // ""
		std::string strPinInterface = ItemValue.get("pin-interface", "").asString();               // tensorflow::Input::Initializer 
		std::string strPinShape = ItemValue.get("pin-shape", "").asString();                     // [2][2]

		if (strPinName == "scope")
		{
			// 입력심볼 : #Scope, 입력심볼의 핀 : Scope, 연결 핀 : Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Const_ex - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "dtype")
		{
			if (strPinInterface == "DataType")
			{
				dtype = GetDatatypeFromInitial(strPinInitial);
			}
			else
			{
				std::string msg = string_format("warning : Const_ex - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "val")
		{
			if (strInSymbolPinName == "" && strPinInterface == "Input::Initializer")
			{
				strVal = strPinInitial;
				strShape = strPinShape;

				strPinSave = ItemValue.get("pin-save", "").asString();
				if (strPinSave == "binary")
				{
					std::string strPinBinPos = ItemValue.get("pin-binary-pos", "").asString();
					iPinBinPos = stoi(strPinBinPos);
					strBinPinType = strPinType;
					strBinPinShape = strPinShape;
				}
			}
			else
			{
				std::string msg = string_format("warning : Const_ex - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
	}

	if (strVal != "")
	{
		std::vector<int64> array_slice;
		std::vector<int64> arraydims;
		GetArrayDimsFromShape(strShape, arraydims, array_slice);

		if (dtype == DT_DOUBLE)
		{
			std::vector<double> arrayvals;
			GetDoubleVectorFromInitial(strVal, arrayvals);

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
		else if (dtype == DT_FLOAT)
		{
			std::vector<float> arrayvals;
			GetFloatVectorFromInitial(strVal, arrayvals);

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
		else if (dtype == DT_INT32)
		{
			std::vector<int> arrayvals;
			GetIntVectorFromInitial(strVal, arrayvals);

			gtl::ArraySlice< int64 > arraySlice(arraydims);
			pTensor = new Tensor(DT_INT32, TensorShape(arraySlice));

			int i = 0;
			for (std::vector<int>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
			{
				pTensor->flat<int>()(i) = *it;
				i++;
			}
			arraySlice.clear();
			arrayvals.clear();
		}
		else if (dtype == DT_INT64)
		{
			std::vector<int64> arrayvals;
			GetInt64VectorFromInitial(strVal, arrayvals);

			gtl::ArraySlice< int64 > arraySlice(arraydims);
			pTensor = new Tensor(DT_INT64, TensorShape(arraySlice));

			int i = 0;
			for (std::vector<int64>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
			{
				pTensor->flat<int64>()(i) = *it;
				i++;
			}
			arraySlice.clear();
			arrayvals.clear();
		}
		else if (dtype == DT_BOOL)
		{
			std::vector<bool> arrayvals;
			GetBoolVectorFromInitial(strVal, arrayvals);

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
		else if (dtype == DT_STRING)
		{
			std::vector<std::string> arrayvals;
			GetStringVectorFromInitial(strVal, arrayvals);

			gtl::ArraySlice< int64 > arraySlice(arraydims);
			pTensor = new Tensor(DT_STRING, TensorShape(arraySlice));

			int i = 0;
			for (std::vector<std::string>::iterator it = arrayvals.begin(); it != arrayvals.end(); it++)
			{
				pTensor->flat<std::string>()(i) = *it;
				i++;
			}
			arraySlice.clear();
			arrayvals.clear();
		}
		else
		{
			std::string msg = string_format("warning : Const_ex - %s(val-initvalue) transfer information missed.", id.c_str());
			PrintMessage(msg, msgParam);
		}

		array_slice.clear();
		arraydims.clear();
	}
	else
	{
		if (strPinSave == "binary")
		{
			if (iPinBinPos != -1 && m_FileData)
			{
				pTensor = Create_BinaryToTensor(strBinPinType, strBinPinShape, m_FileData, iPinBinPos);
			}
			else
			{
				std::string msg = string_format("warning : Const_ex - %s(%s) binary information missed.", id.c_str());
				PrintMessage(msg, msgParam);
			}
		}
	}

	if (pScope == nullptr)
	{
		std::string msg = string_format("warning : Const_ex - %s(scope) transfer information missed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
	if (pTensor == nullptr)
	{
		std::string msg = string_format("warning : Const_ex - %s(val) transfer information missed.", id.c_str());
		PrintMessage(msg, msgParam);
	}

	if (pScope && pTensor)
	{
		*pOutput = Const(*pScope, *pTensor);
		ObjectInfo* pObj = AddObjectMap(pOutput, id, SYMBOL_CONST, "Const", pInputItem);
		if (pObj)
			AddOutputInfo(pObj, pOutput, OUTPUT_TYPE_OUTPUT, "output");
		// pObj->pOutput = pOutput;
	}
	else
	{
		std::string msg = string_format("error : Const_ex(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}

	if (pTensor)
		delete pTensor;

	return pOutput;
}


#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"


#include "tensorflow/cc/framework/gradient_checker.h"

void* Create_AddSymbolicGradients(std::string id, Json::Value pInputItem)
{
	Scope* pScope = nullptr;
	OutputList inputlist;
	OutputList outputlist;
	std::string msgParam = string_format("#%s", id.c_str());

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
			// 입력심볼 : #Scope, 입력심볼의 핀 : Scope, 연결 핀 : Scope
			if (strPinInterface == "Scope")
			{
				pScope = m_pScope;
			}
			else
			{
				std::string msg = string_format("warning : Variable - %s(%s) transfer information missed.", id.c_str(), strPinName.c_str());
				PrintMessage(msg, msgParam);
			}
		}
		else if (strPinName == "outputs")
		{
			ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
			if (pObj)
			{
				OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
				if (pOutputObj)
				{
					if (pOutputObj->pOutput)
					{
						outputlist.push_back(*((Output*)pOutputObj->pOutput));
					}
				}
			}
		}
		else if (strPinName == "inputs")
		{
			ObjectInfo* pObj = LookupFromObjectMap(strInSymbolId);
			if (pObj)
			{
				OutputInfo* pOutputObj = LookupFromOutputMap(pObj, strInSymbolPinName);
				if (pOutputObj)
				{
					if (pOutputObj->pOutput)
					{
						inputlist.push_back(*((Output*)pOutputObj->pOutput));
					}
				}
			}
		}

		else
		{
			std::string msg = string_format("warning : Variable pin name - %s(%s) unknown value.", id.c_str(), strPinName.c_str());
			PrintMessage(msg, msgParam);
		}
	}

	if (pScope && outputlist.size()>0 && inputlist.size()>0)
	{
		std::vector<Output> grad_outputs;

		std::vector<Output> grad_inputs;
		grad_inputs.reserve(outputlist.size());
		for (const Output& output : outputlist)
		{
			grad_inputs.emplace_back(ops::OnesLike(*pScope, output));
		}
		Status st; 
		st = AddSymbolicGradients(*pScope, outputlist, inputlist, grad_inputs, &grad_outputs);
		if (st.code() != error::OK)
		{
			std::string msg = string_format("error: %s.", st.error_message().c_str());
			PrintMessage(msg, msgParam);
		}
		else
		{
			int iSize = grad_outputs.size();
			if (iSize > 0)
			{
				ObjectInfo* pObj = AddObjectMap(&grad_outputs, id, SYMBOL_ADDSYMBOLICGRADIENTS, "AddSymbolicGradients", pInputItem);

				int i = 0;
				for (std::vector<Output>::iterator it = grad_outputs.begin(); it != grad_outputs.end(); it++)
				{
					std::string newid = string_format("grad_outputs[%d]", i);
					Output* pOutput = new Output(it->node());
					AddOutputInfo(pObj, pOutput, OUTPUT_TYPE_OUTPUT_ETC, newid);
					i++;
				}
			}
		}
	}
	else
	{
		std::string msg = string_format("error : AddSymbolicGradients(%s) Object create failed.", id.c_str());
		PrintMessage(msg, msgParam);
	}
}