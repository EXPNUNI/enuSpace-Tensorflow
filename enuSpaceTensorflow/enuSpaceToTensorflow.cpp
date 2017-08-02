
#include "stdafx.h"
#include "enuSpaceToTensorflow.h"
#include "tensorflow.h"
#include "utility_functions.h"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "include/json/json.h"

#include "GlobalHeader.h"

#include "tf_array_ops.h"
#include "tf_candidate_sampling_ops.h"
#include "tf_control_flow_ops.h"
#include "tf_core.h"
#include "tf_data_flow_ops.h"
#include "tf_image_ops.h"
#include "tf_io_ops.h"
#include "tf_logging_ops.h"
#include "tf_math.h"
#include "tf_nn.h"
#include "tf_no.h"
#include "tf_parsing_op.h"
#include "tf_random.h"
#include "tf_sparse_ops.h"
#include "tf_state.h"
#include "tf_string.h"
#include "tf_training.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
// dll extern execute functions 
bool Load_Tensorflow()
{
	AddSymbolList();

	return true;
}

bool Init_Tensorflow(std::string config_doc, std::string page_name)
{

	Json::Value root;
	Json::Reader reader;

	bool parsingSuccessful = reader.parse(config_doc, root);
	if (!parsingSuccessful)
	{
		return false;
	}

	// scope object create.
	Json::Value InputItem;
	Create_Symbol(SYMBOL_SCOPE, page_name, InputItem);

	const Json::Value plugins = root["object"];
	for (int index = 0; index < (int)plugins.size(); ++index)
	{
		Json::Value Item = plugins[index];

		std::string strSymbolName = Item.get("symbol-name", "").asString();
		std::string strId = Item.get("id", "").asString();

		Json::Value InputItem = Item["input"];

		int iType = GetSymbolType(strSymbolName);
		Create_Symbol(iType, strId, InputItem);
	}

	return true;
}

bool Task_Tensorflow()
{
	std::vector<std::string>::iterator pinname;
	std::vector<ObjectInfo*>::iterator vObjIt;
	std::map<std::string, FetchInfo*>::iterator vit;
	for (vit = m_RunMapList.begin(); vit != m_RunMapList.end(); ++vit)
	{
		FetchInfo* pTar = vit->second;
		if (pTar->pSession)
		{
			if (pTar->pSession->pScope)
			{
				if (pTar->pSession->pScope->ok())
				{
					if (pTar->pSession->type == SYMBOL_CLIENTSESSION)
					{
						if (pTar->fetch_object.size() > 0)
						{
							ClientSession* pClientSession = (ClientSession*)pTar->pSession->pObject;
							TF_CHECK_OK(pClientSession->Run(pTar->fetch_outputs, &pTar->outputs));

							vObjIt = pTar->fetch_object.begin();
							pinname = pTar->pin_names.begin();
							

							// result msg
							for (std::vector<tensorflow::Tensor>::iterator it = pTar->outputs.begin(); it != pTar->outputs.end(); it++)
							{
								ObjectInfo* pObjet = *vObjIt;
								std::string strpinname = *pinname;

								int iNum = it->NumElements();
								int iType = it->dtype();

								void* pData = nullptr;
								int iDataType = DEF_UNKNOWN;

								switch (iType)
								{
								case DT_DOUBLE:
									pData = new double[iNum];
									iDataType = DEF_DOUBLE;
									break;
								case DT_FLOAT:
									pData = new float[iNum];
									iDataType = DEF_FLOAT;
									break;
								case DT_INT32:
								case DT_INT64:
									pData = new int[iNum];
									iDataType = DEF_INT;
									break;
								}

								for (int i = 0; i < iNum; i++)
								{
									if (iType == DT_DOUBLE)
									{
										auto flat = it->flat<double>();
										PrintMessage(strings::Printf("[%d] = %8.6f", i, flat(i)));

										*((double*)pData+i) = flat(i);
									}
									else if (iType == DT_FLOAT)
									{
										auto flat = it->flat<float>();
										PrintMessage(strings::Printf("[%d] = %8.6f", i, flat(i)));
										*((float*)pData + i) = flat(i);
									}
									else if (iType == DT_INT32)
									{
										auto flat = it->flat<int>();
										PrintMessage(strings::Printf("[%d] = %d", i, flat(i)));
										*((int*)pData + i) = flat(i);
									}
									else if (iType == DT_INT64)
									{
										auto flat = it->flat<int64>();
										PrintMessage(strings::Printf("[%d] = %d", i, flat(i)));
										*((int*)pData + i) = flat(i);
									}
								}

								if (pData)
								{
									std::string strdim;
									TensorShape shape = it->shape();
									int idim = shape.dims();
									for (int i = 0; i < idim; i++)
									{
										int64 idim = shape.dim_size(i);
										strdim += strings::Printf("[%d]", idim);
									}

									pTar->outputs;
									LookupFromOutputMap(pObjet, "");

									std::string strVariable;
									strVariable = pObjet->id + ".result_"+ strpinname + strdim;
									SetReShapeArrayValue(strVariable, pData, iDataType, iNum);

									delete[] pData;
								}

								vObjIt++;
								pinname++;

								std::string strvalue = it->DebugString();
								PrintMessage(strvalue);
							}
						}
					}
				}
			}
			else
			{
				PrintMessage("error : scope fail status");
			}
		}
	}

	return true;
}

bool Unload_Tensorflow()
{
	ObjectMapClear();
	m_SymbolList.clear();

	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////
void AddSymbolList()
{
	m_SymbolList.insert(std::pair<std::string, int>("#BatchToSpace", SYMBOL_BATCHTOSPACE));
	m_SymbolList.insert(std::pair<std::string, int>("#BatchToSpaceND", SYMBOL_BATCHTOSPACEND));
	m_SymbolList.insert(std::pair<std::string, int>("#Bitcast", SYMBOL_BITCAST));
	m_SymbolList.insert(std::pair<std::string, int>("#BroadcastDynamicShape", SYMBOL_BROADCASTDYNAMICSHAPE));
	m_SymbolList.insert(std::pair<std::string, int>("#CheckNumerics", SYMBOL_CHECKNUMERICS));
	m_SymbolList.insert(std::pair<std::string, int>("#Concat", SYMBOL_CONCAT));
	m_SymbolList.insert(std::pair<std::string, int>("#DepthToSpace", SYMBOL_DEPTHTOSPACE));
	m_SymbolList.insert(std::pair<std::string, int>("#Dequantize", SYMBOL_DEQUANTIZE));
	m_SymbolList.insert(std::pair<std::string, int>("#Diag", SYMBOL_DIAG));
	m_SymbolList.insert(std::pair<std::string, int>("#DiagPart", SYMBOL_DIAGPART));
	m_SymbolList.insert(std::pair<std::string, int>("#EditDistance", SYMBOL_EDITDISTANCE));
	m_SymbolList.insert(std::pair<std::string, int>("#ExpandDims", SYMBOL_EXPANDDIMS));
	m_SymbolList.insert(std::pair<std::string, int>("#ExtractImagePatches", SYMBOL_EXTRACTIMAGEPATCHES));
	m_SymbolList.insert(std::pair<std::string, int>("#FakeQuantWithMinMaxArgs", SYMBOL_FAKEQUANTWITHMINMAXARGS));
	m_SymbolList.insert(std::pair<std::string, int>("#FakeQuantWithMinMaxArgsGradient", SYMBOL_FAKEQUANTWITHMINMAXARGSGRADIENT));
	m_SymbolList.insert(std::pair<std::string, int>("#FakeQuantWithMinMaxVars", SYMBOL_FAKEQUANTWITHMINMAXVARS));
	m_SymbolList.insert(std::pair<std::string, int>("#FakeQuantWithMinMaxVarsGradient", SYMBOL_FAKEQUANTWITHMINMAXVARSGRADIENT));
	m_SymbolList.insert(std::pair<std::string, int>("#FakeQuantWithMinMaxVarsPerChannel", SYMBOL_FAKEQUANTWITHMINMAXVARSPERCHANNEL));
	m_SymbolList.insert(std::pair<std::string, int>("#FakeQuantWithMinMaxVarsPerChannelGradient", SYMBOL_FAKEQUANTWITHMINMAXVARSPERCHANNELGRADIENT));
	m_SymbolList.insert(std::pair<std::string, int>("#Fill", SYMBOL_FILL));
	m_SymbolList.insert(std::pair<std::string, int>("#Gather", SYMBOL_GATHER));
	m_SymbolList.insert(std::pair<std::string, int>("#GatherNd", SYMBOL_GATHERND));
	m_SymbolList.insert(std::pair<std::string, int>("#Identity", SYMBOL_IDENTITY));
	m_SymbolList.insert(std::pair<std::string, int>("#ImmutableConst", SYMBOL_IMMUTABLECONST));
	m_SymbolList.insert(std::pair<std::string, int>("#InvertPermutation", SYMBOL_INVERTPERMUTATION));
	m_SymbolList.insert(std::pair<std::string, int>("#MatrixBandPart", SYMBOL_MATRIXBANDPART));
	m_SymbolList.insert(std::pair<std::string, int>("#MatrixDiag", SYMBOL_MATRIXDIAG));
	m_SymbolList.insert(std::pair<std::string, int>("#MatrixDiagPart", SYMBOL_MATRIXDIAGPART));
	m_SymbolList.insert(std::pair<std::string, int>("#MatrixSetDiag", SYMBOL_MATRIXSETDIAG));
	m_SymbolList.insert(std::pair<std::string, int>("#MirrorPad", SYMBOL_MIRRORPAD));
	m_SymbolList.insert(std::pair<std::string, int>("#OneHot", SYMBOL_ONEHOT));
	m_SymbolList.insert(std::pair<std::string, int>("#OnesLike", SYMBOL_ONESLIKE));
	m_SymbolList.insert(std::pair<std::string, int>("#Pad", SYMBOL_PAD));
	m_SymbolList.insert(std::pair<std::string, int>("#ParallelConcat", SYMBOL_PARALLELCONCAT));
	m_SymbolList.insert(std::pair<std::string, int>("#Placeholder", SYMBOL_PLACEHOLDER));
	m_SymbolList.insert(std::pair<std::string, int>("#PlaceholderV2", SYMBOL_PLACEHOLDERV2));
	m_SymbolList.insert(std::pair<std::string, int>("#PlaceholderWithDefault", SYMBOL_PLACEHOLDERWITHDEFAULT));
	m_SymbolList.insert(std::pair<std::string, int>("#PreventGradient", SYMBOL_PREVENTGRADIENT));
	m_SymbolList.insert(std::pair<std::string, int>("#QuantizeAndDequantizeV2", SYMBOL_QUANTIZEANDDEQUANTIZEV2));
	m_SymbolList.insert(std::pair<std::string, int>("#QuantizeV2", SYMBOL_QUANTIZEV2));
	m_SymbolList.insert(std::pair<std::string, int>("#QuantizedConcat", SYMBOL_QUANTIZEDCONCAT));
	m_SymbolList.insert(std::pair<std::string, int>("#QuantizedInstanceNorm", SYMBOL_QUANTIZEDINSTANCENORM));
	m_SymbolList.insert(std::pair<std::string, int>("#QuantizedReshape", SYMBOL_QUANTIZEDRESHAPE));
	m_SymbolList.insert(std::pair<std::string, int>("#Rank", SYMBOL_RANK));
	m_SymbolList.insert(std::pair<std::string, int>("#Reshape", SYMBOL_RESHAPE));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceStridedSliceAssign", SYMBOL_RESOURCESTRIDEDSLICEASSIGN));
	m_SymbolList.insert(std::pair<std::string, int>("#Reverse", SYMBOL_REVERSE));
	m_SymbolList.insert(std::pair<std::string, int>("#ReverseSequence", SYMBOL_REVERSESEQUENCE));
	m_SymbolList.insert(std::pair<std::string, int>("#ScatterNd", SYMBOL_SCATTERND));
	m_SymbolList.insert(std::pair<std::string, int>("#SetDiff1D", SYMBOL_SETDIFF1D));
	m_SymbolList.insert(std::pair<std::string, int>("#Shape", SYMBOL_SHAPE));
	m_SymbolList.insert(std::pair<std::string, int>("#ShapeN", SYMBOL_SHAPEN));
	m_SymbolList.insert(std::pair<std::string, int>("#Size", SYMBOL_SIZE));
	m_SymbolList.insert(std::pair<std::string, int>("#Slice", SYMBOL_SLICE));
	m_SymbolList.insert(std::pair<std::string, int>("#SpaceToBatch", SYMBOL_SPACETOBATCH));
	m_SymbolList.insert(std::pair<std::string, int>("#SpaceToBatchND", SYMBOL_SPACETOBATCHND));
	m_SymbolList.insert(std::pair<std::string, int>("#SpaceToDepth", SYMBOL_SPACETODEPTH));
	m_SymbolList.insert(std::pair<std::string, int>("#Split", SYMBOL_SPLIT));
	m_SymbolList.insert(std::pair<std::string, int>("#SplitV", SYMBOL_SPLITV));
	m_SymbolList.insert(std::pair<std::string, int>("#Squeeze", SYMBOL_SQUEEZE));
	m_SymbolList.insert(std::pair<std::string, int>("#Stack", SYMBOL_STACK));
	m_SymbolList.insert(std::pair<std::string, int>("#StopGradient", SYMBOL_STOPGRADIENT));
	m_SymbolList.insert(std::pair<std::string, int>("#StridedSlice", SYMBOL_STRIDEDSLICE));
	m_SymbolList.insert(std::pair<std::string, int>("#StridedSliceAssign", SYMBOL_STRIDEDSLICEASSIGN));
	m_SymbolList.insert(std::pair<std::string, int>("#StridedSliceGrad", SYMBOL_STRIDEDSLICEGRAD));
	m_SymbolList.insert(std::pair<std::string, int>("#Tile", SYMBOL_TILE));
	m_SymbolList.insert(std::pair<std::string, int>("#Transpose", SYMBOL_TRANSPOSE));
	m_SymbolList.insert(std::pair<std::string, int>("#Unique", SYMBOL_UNIQUE));
	m_SymbolList.insert(std::pair<std::string, int>("#UniqueWithCounts", SYMBOL_UNIQUEWITHCOUNTS));
	m_SymbolList.insert(std::pair<std::string, int>("#Unstack", SYMBOL_UNSTACK));
	m_SymbolList.insert(std::pair<std::string, int>("#Where", SYMBOL_WHERE));
	m_SymbolList.insert(std::pair<std::string, int>("#ZerosLike", SYMBOL_ZEROSLIKE));
	m_SymbolList.insert(std::pair<std::string, int>("#AllCandidateSampler", SYMBOL_ALLCANDIDATESAMPLER));
	m_SymbolList.insert(std::pair<std::string, int>("#ComputeAccidentalHits", SYMBOL_COMPUTEACCIDENTALHITS));
	m_SymbolList.insert(std::pair<std::string, int>("#FixedUnigramCandidateSampler", SYMBOL_FIXEDUNIGRAMCANDIDATESAMPLER));
	m_SymbolList.insert(std::pair<std::string, int>("#LearnedUnigramCandidateSampler", SYMBOL_LEARNEDUNIGRAMCANDIDATESAMPLER));
	m_SymbolList.insert(std::pair<std::string, int>("#LogUniformCandidateSampler", SYMBOL_LOGUNIFORMCANDIDATESAMPLER));
	m_SymbolList.insert(std::pair<std::string, int>("#UniformCandidateSampler", SYMBOL_UNIFORMCANDIDATESAMPLER));
	m_SymbolList.insert(std::pair<std::string, int>("#Abort", SYMBOL_ABORT));
	m_SymbolList.insert(std::pair<std::string, int>("#ControlTrigger", SYMBOL_CONTROLTRIGGER));
	m_SymbolList.insert(std::pair<std::string, int>("#LoopCond", SYMBOL_LOOPCOND));
	m_SymbolList.insert(std::pair<std::string, int>("#Merge", SYMBOL_MERGE));
	m_SymbolList.insert(std::pair<std::string, int>("#NextIteration", SYMBOL_NEXTITERATION));
	m_SymbolList.insert(std::pair<std::string, int>("#RefNextIteration", SYMBOL_REFNEXTITERATION));
	m_SymbolList.insert(std::pair<std::string, int>("#RefSelect", SYMBOL_REFSELECT));
	m_SymbolList.insert(std::pair<std::string, int>("#RefSwitch", SYMBOL_REFSWITCH));
	m_SymbolList.insert(std::pair<std::string, int>("#Switch", SYMBOL_SWITCH));
	m_SymbolList.insert(std::pair<std::string, int>("#ClientSession", SYMBOL_CLIENTSESSION));
	m_SymbolList.insert(std::pair<std::string, int>("#Input", SYMBOL_INPUT));
	m_SymbolList.insert(std::pair<std::string, int>("#Input_Initializer", SYMBOL_INPUT_INITIALIZER));
	m_SymbolList.insert(std::pair<std::string, int>("#InputList", SYMBOL_INPUTLIST));
	m_SymbolList.insert(std::pair<std::string, int>("#Operation", SYMBOL_OPERATION));
	m_SymbolList.insert(std::pair<std::string, int>("#Output", SYMBOL_OUTPUT));
	m_SymbolList.insert(std::pair<std::string, int>("#Scope", SYMBOL_SCOPE));
	m_SymbolList.insert(std::pair<std::string, int>("#Status", SYMBOL_STATUS));
	m_SymbolList.insert(std::pair<std::string, int>("#Tensor", SYMBOL_TENSOR));
	m_SymbolList.insert(std::pair<std::string, int>("#AccumulatorApplyGradient", SYMBOL_ACCUMULATORAPPLYGRADIENT));
	m_SymbolList.insert(std::pair<std::string, int>("#AccumulatorNumAccumulated", SYMBOL_ACCUMULATORNUMACCUMULATED));
	m_SymbolList.insert(std::pair<std::string, int>("#AccumulatorSetGlobalStep", SYMBOL_ACCUMULATORSETGLOBALSTEP));
	m_SymbolList.insert(std::pair<std::string, int>("#AccumulatorTakeGradient", SYMBOL_ACCUMULATORTAKEGRADIENT));
	m_SymbolList.insert(std::pair<std::string, int>("#Barrier", SYMBOL_BARRIER));
	m_SymbolList.insert(std::pair<std::string, int>("#BarrierClose", SYMBOL_BARRIERCLOSE));
	m_SymbolList.insert(std::pair<std::string, int>("#BarrierIncompleteSize", SYMBOL_BARRIERINCOMPLETESIZE));
	m_SymbolList.insert(std::pair<std::string, int>("#BarrierInsertMany", SYMBOL_BARRIERINSERTMANY));
	m_SymbolList.insert(std::pair<std::string, int>("#BarrierReadySize", SYMBOL_BARRIERREADYSIZE));
	m_SymbolList.insert(std::pair<std::string, int>("#BarrierTakeMany", SYMBOL_BARRIERTAKEMANY));
	m_SymbolList.insert(std::pair<std::string, int>("#ConditionalAccumulator", SYMBOL_CONDITIONALACCUMULATOR));
	m_SymbolList.insert(std::pair<std::string, int>("#DeleteSessionTensor", SYMBOL_DELETESESSIONTENSOR));
	m_SymbolList.insert(std::pair<std::string, int>("#DynamicPartition", SYMBOL_DYNAMICPARTITION));
	m_SymbolList.insert(std::pair<std::string, int>("#DynamicStitch", SYMBOL_DYNAMICSTITCH));
	m_SymbolList.insert(std::pair<std::string, int>("#FIFOQueue", SYMBOL_FIFOQUEUE));
	m_SymbolList.insert(std::pair<std::string, int>("#GetSessionHandle", SYMBOL_GETSESSIONHANDLE));
	m_SymbolList.insert(std::pair<std::string, int>("#GetSessionHandleV2", SYMBOL_GETSESSIONHANDLEV2));
	m_SymbolList.insert(std::pair<std::string, int>("#GetSessionTensor", SYMBOL_GETSESSIONTENSOR));
	m_SymbolList.insert(std::pair<std::string, int>("#PaddingFIFOQueue", SYMBOL_PADDINGFIFOQUEUE));
	m_SymbolList.insert(std::pair<std::string, int>("#PriorityQueue", SYMBOL_PRIORITYQUEUE));
	m_SymbolList.insert(std::pair<std::string, int>("#QueueClose", SYMBOL_QUEUECLOSE));
	m_SymbolList.insert(std::pair<std::string, int>("#QueueDequeue", SYMBOL_QUEUEDEQUEUE));
	m_SymbolList.insert(std::pair<std::string, int>("#QueueDequeueMany", SYMBOL_QUEUEDEQUEUEMANY));
	m_SymbolList.insert(std::pair<std::string, int>("#QueueDequeueUpTo", SYMBOL_QUEUEDEQUEUEUPTO));
	m_SymbolList.insert(std::pair<std::string, int>("#QueueEnqueue", SYMBOL_QUEUEENQUEUE));
	m_SymbolList.insert(std::pair<std::string, int>("#QueueEnqueueMany", SYMBOL_QUEUEENQUEUEMANY));
	m_SymbolList.insert(std::pair<std::string, int>("#QueueSize", SYMBOL_QUEUESIZE));
	m_SymbolList.insert(std::pair<std::string, int>("#RandomShuffleQueue", SYMBOL_RANDOMSHUFFLEQUEUE));
	m_SymbolList.insert(std::pair<std::string, int>("#RecordInput", SYMBOL_RECORDINPUT));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseAccumulatorApplyGradient", SYMBOL_SPARSEACCUMULATORAPPLYGRADIENT));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseAccumulatorTakeGradient", SYMBOL_SPARSEACCUMULATORTAKEGRADIENT));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseConditionalAccumulator", SYMBOL_SPARSECONDITIONALACCUMULATOR));
	m_SymbolList.insert(std::pair<std::string, int>("#Stage", SYMBOL_STAGE));
	m_SymbolList.insert(std::pair<std::string, int>("#TensorArray", SYMBOL_TENSORARRAY));
	m_SymbolList.insert(std::pair<std::string, int>("#TensorArrayClose", SYMBOL_TENSORARRAYCLOSE));
	m_SymbolList.insert(std::pair<std::string, int>("#TensorArrayConcat", SYMBOL_TENSORARRAYCONCAT));
	m_SymbolList.insert(std::pair<std::string, int>("#TensorArrayGather", SYMBOL_TENSORARRAYGATHER));
	m_SymbolList.insert(std::pair<std::string, int>("#TensorArrayGrad", SYMBOL_TENSORARRAYGRAD));
	m_SymbolList.insert(std::pair<std::string, int>("#TensorArrayRead", SYMBOL_TENSORARRAYREAD));
	m_SymbolList.insert(std::pair<std::string, int>("#TensorArrayScatter", SYMBOL_TENSORARRAYSCATTER));
	m_SymbolList.insert(std::pair<std::string, int>("#TensorArraySize", SYMBOL_TENSORARRAYSIZE));
	m_SymbolList.insert(std::pair<std::string, int>("#TensorArraySplit", SYMBOL_TENSORARRAYSPLIT));
	m_SymbolList.insert(std::pair<std::string, int>("#TensorArrayWrite", SYMBOL_TENSORARRAYWRITE));
	m_SymbolList.insert(std::pair<std::string, int>("#Unstage", SYMBOL_UNSTAGE));
	m_SymbolList.insert(std::pair<std::string, int>("#AdjustContrast", SYMBOL_ADJUSTCONTRAST));
	m_SymbolList.insert(std::pair<std::string, int>("#AdjustHue", SYMBOL_ADJUSTHUE));
	m_SymbolList.insert(std::pair<std::string, int>("#AdjustSaturation", SYMBOL_ADJUSTSATURATION));
	m_SymbolList.insert(std::pair<std::string, int>("#CropAndResize", SYMBOL_CROPANDRESIZE));
	m_SymbolList.insert(std::pair<std::string, int>("#CropAndResizeGradBoxes", SYMBOL_CROPANDRESIZEGRADBOXES));
	m_SymbolList.insert(std::pair<std::string, int>("#CropAndResizeGradImage", SYMBOL_CROPANDRESIZEGRADIMAGE));
	m_SymbolList.insert(std::pair<std::string, int>("#DecodeGif", SYMBOL_DECODEGIF));
	m_SymbolList.insert(std::pair<std::string, int>("#DecodeJpeg", SYMBOL_DECODEJPEG));
	m_SymbolList.insert(std::pair<std::string, int>("#DecodePng", SYMBOL_DECODEPNG));
	m_SymbolList.insert(std::pair<std::string, int>("#DrawBoundingBoxes", SYMBOL_DRAWBOUNDINGBOXES));
	m_SymbolList.insert(std::pair<std::string, int>("#EncodeJpeg", SYMBOL_ENCODEJPEG));
	m_SymbolList.insert(std::pair<std::string, int>("#EncodePng", SYMBOL_ENCODEPNG));
	m_SymbolList.insert(std::pair<std::string, int>("#ExtractGlimpse", SYMBOL_EXTRACTGLIMPSE));
	m_SymbolList.insert(std::pair<std::string, int>("#HSVToRGB", SYMBOL_HSVTORGB));
	m_SymbolList.insert(std::pair<std::string, int>("#NonMaxSuppression", SYMBOL_NONMAXSUPPRESSION));
	m_SymbolList.insert(std::pair<std::string, int>("#RGBToHSV", SYMBOL_RGBTOHSV));
	m_SymbolList.insert(std::pair<std::string, int>("#ResizeArea", SYMBOL_RESIZEAREA));
	m_SymbolList.insert(std::pair<std::string, int>("#ResizeBicubic", SYMBOL_RESIZEBICUBIC));
	m_SymbolList.insert(std::pair<std::string, int>("#ResizeBilinear", SYMBOL_RESIZEBILINEAR));
	m_SymbolList.insert(std::pair<std::string, int>("#ResizeNearestNeighbor", SYMBOL_RESIZENEARESTNEIGHBOR));
	m_SymbolList.insert(std::pair<std::string, int>("#SampleDistortedBoundingBox", SYMBOL_SAMPLEDISTORTEDBOUNDINGBOX));
	m_SymbolList.insert(std::pair<std::string, int>("#FixedLengthRecordReader", SYMBOL_FIXEDLENGTHRECORDREADER));
	m_SymbolList.insert(std::pair<std::string, int>("#IdentityReader", SYMBOL_IDENTITYREADER));
	m_SymbolList.insert(std::pair<std::string, int>("#MatchingFiles", SYMBOL_MATCHINGFILES));
	m_SymbolList.insert(std::pair<std::string, int>("#MergeV2Checkpoints", SYMBOL_MERGEV2CHECKPOINTS));
	m_SymbolList.insert(std::pair<std::string, int>("#ReadFile", SYMBOL_READFILE));
	m_SymbolList.insert(std::pair<std::string, int>("#ReaderNumRecordsProduced", SYMBOL_READERNUMRECORDSPRODUCED));
	m_SymbolList.insert(std::pair<std::string, int>("#ReaderNumWorkUnitsCompleted", SYMBOL_READERNUMWORKUNITSCOMPLETED));
	m_SymbolList.insert(std::pair<std::string, int>("#ReaderRead", SYMBOL_READERREAD));
	m_SymbolList.insert(std::pair<std::string, int>("#ReaderReadUpTo", SYMBOL_READERREADUPTO));
	m_SymbolList.insert(std::pair<std::string, int>("#ReaderReset", SYMBOL_READERRESET));
	m_SymbolList.insert(std::pair<std::string, int>("#ReaderRestoreState", SYMBOL_READERRESTORESTATE));
	m_SymbolList.insert(std::pair<std::string, int>("#ReaderSerializeState", SYMBOL_READERSERIALIZESTATE));
	m_SymbolList.insert(std::pair<std::string, int>("#Restore", SYMBOL_RESTORE));
	m_SymbolList.insert(std::pair<std::string, int>("#RestoreSlice", SYMBOL_RESTORESLICE));
	m_SymbolList.insert(std::pair<std::string, int>("#RestoreV2", SYMBOL_RESTOREV2));
	m_SymbolList.insert(std::pair<std::string, int>("#Save", SYMBOL_SAVE));
	m_SymbolList.insert(std::pair<std::string, int>("#SaveSlices", SYMBOL_SAVESLICES));
	m_SymbolList.insert(std::pair<std::string, int>("#SaveV2", SYMBOL_SAVEV2));
	m_SymbolList.insert(std::pair<std::string, int>("#ShardedFilename", SYMBOL_SHARDEDFILENAME));
	m_SymbolList.insert(std::pair<std::string, int>("#ShardedFilespec", SYMBOL_SHARDEDFILESPEC));
	m_SymbolList.insert(std::pair<std::string, int>("#TFRecordReader", SYMBOL_TFRECORDREADER));
	m_SymbolList.insert(std::pair<std::string, int>("#TextLineReader", SYMBOL_TEXTLINEREADER));
	m_SymbolList.insert(std::pair<std::string, int>("#WholeFileReader", SYMBOL_WHOLEFILEREADER));
	m_SymbolList.insert(std::pair<std::string, int>("#WriteFile", SYMBOL_WRITEFILE));
	m_SymbolList.insert(std::pair<std::string, int>("#Assert", SYMBOL_ASSERT));
	m_SymbolList.insert(std::pair<std::string, int>("#HistogramSummary", SYMBOL_HISTOGRAMSUMMARY));
	m_SymbolList.insert(std::pair<std::string, int>("#MergeSummary", SYMBOL_MERGESUMMARY));
	m_SymbolList.insert(std::pair<std::string, int>("#Print", SYMBOL_PRINT));
	m_SymbolList.insert(std::pair<std::string, int>("#ScalarSummary", SYMBOL_SCALARSUMMARY));
	m_SymbolList.insert(std::pair<std::string, int>("#TensorSummary", SYMBOL_TENSORSUMMARY));
	m_SymbolList.insert(std::pair<std::string, int>("#Abs", SYMBOL_ABS));
	m_SymbolList.insert(std::pair<std::string, int>("#Acos", SYMBOL_ACOS));
	m_SymbolList.insert(std::pair<std::string, int>("#Add", SYMBOL_ADD));
	m_SymbolList.insert(std::pair<std::string, int>("#AddN", SYMBOL_ADDN));
	m_SymbolList.insert(std::pair<std::string, int>("#All", SYMBOL_ALL));
	m_SymbolList.insert(std::pair<std::string, int>("#Any", SYMBOL_ANY));
	m_SymbolList.insert(std::pair<std::string, int>("#ApproximateEqual", SYMBOL_APPROXIMATEEQUAL));
	m_SymbolList.insert(std::pair<std::string, int>("#ArgMax", SYMBOL_ARGMAX));
	m_SymbolList.insert(std::pair<std::string, int>("#ArgMin", SYMBOL_ARGMIN));
	m_SymbolList.insert(std::pair<std::string, int>("#Asin", SYMBOL_ASIN));
	m_SymbolList.insert(std::pair<std::string, int>("#Atan", SYMBOL_ATAN));
	m_SymbolList.insert(std::pair<std::string, int>("#Atan2", SYMBOL_ATAN2));
	m_SymbolList.insert(std::pair<std::string, int>("#BatchMatMul", SYMBOL_BATCHMATMUL));
	m_SymbolList.insert(std::pair<std::string, int>("#Betainc", SYMBOL_BETAINC));
	m_SymbolList.insert(std::pair<std::string, int>("#Bincount", SYMBOL_BINCOUNT));
	m_SymbolList.insert(std::pair<std::string, int>("#Bucketize", SYMBOL_BUCKETIZE));
	m_SymbolList.insert(std::pair<std::string, int>("#Cast", SYMBOL_CAST));
	m_SymbolList.insert(std::pair<std::string, int>("#Ceil", SYMBOL_CEIL));
	m_SymbolList.insert(std::pair<std::string, int>("#Complex", SYMBOL_COMPLEX));
	m_SymbolList.insert(std::pair<std::string, int>("#ComplexAbs", SYMBOL_COMPLEXABS));
	m_SymbolList.insert(std::pair<std::string, int>("#Conj", SYMBOL_CONJ));
	m_SymbolList.insert(std::pair<std::string, int>("#Cos", SYMBOL_COS));
	m_SymbolList.insert(std::pair<std::string, int>("#Cross", SYMBOL_CROSS));
	m_SymbolList.insert(std::pair<std::string, int>("#Cumprod", SYMBOL_CUMPROD));
	m_SymbolList.insert(std::pair<std::string, int>("#Cumsum", SYMBOL_CUMSUM));
	m_SymbolList.insert(std::pair<std::string, int>("#Digamma", SYMBOL_DIGAMMA));
	m_SymbolList.insert(std::pair<std::string, int>("#Div", SYMBOL_DIV));
	m_SymbolList.insert(std::pair<std::string, int>("#Equal", SYMBOL_EQUAL));
	m_SymbolList.insert(std::pair<std::string, int>("#Erf", SYMBOL_ERF));
	m_SymbolList.insert(std::pair<std::string, int>("#Erfc", SYMBOL_ERFC));
	m_SymbolList.insert(std::pair<std::string, int>("#Exp", SYMBOL_EXP));
	m_SymbolList.insert(std::pair<std::string, int>("#Expm1", SYMBOL_EXPM1));
	m_SymbolList.insert(std::pair<std::string, int>("#Floor", SYMBOL_FLOOR));
	m_SymbolList.insert(std::pair<std::string, int>("#FloorDiv", SYMBOL_FLOORDIV));
	m_SymbolList.insert(std::pair<std::string, int>("#FloorMod", SYMBOL_FLOORMOD));
	m_SymbolList.insert(std::pair<std::string, int>("#Greater", SYMBOL_GREATER));
	m_SymbolList.insert(std::pair<std::string, int>("#GreaterEqual", SYMBOL_GREATEREQUAL));
	m_SymbolList.insert(std::pair<std::string, int>("#Igamma", SYMBOL_IGAMMA));
	m_SymbolList.insert(std::pair<std::string, int>("#Igammac", SYMBOL_IGAMMAC));
	m_SymbolList.insert(std::pair<std::string, int>("#Imag", SYMBOL_IMAG));
	m_SymbolList.insert(std::pair<std::string, int>("#IsInf", SYMBOL_ISINF));
	m_SymbolList.insert(std::pair<std::string, int>("#IsNan", SYMBOL_ISNAN));
	m_SymbolList.insert(std::pair<std::string, int>("#Less", SYMBOL_LESS));
	m_SymbolList.insert(std::pair<std::string, int>("#LessEqual", SYMBOL_LESSEQUAL));
	m_SymbolList.insert(std::pair<std::string, int>("#Lgamma", SYMBOL_LGAMMA));
	m_SymbolList.insert(std::pair<std::string, int>("#LinSpace", SYMBOL_LINSPACE));
	m_SymbolList.insert(std::pair<std::string, int>("#Log", SYMBOL_LOG));
	m_SymbolList.insert(std::pair<std::string, int>("#Log1p", SYMBOL_LOG1P));
	m_SymbolList.insert(std::pair<std::string, int>("#LogicalAnd", SYMBOL_LOGICALAND));
	m_SymbolList.insert(std::pair<std::string, int>("#LogicalNot", SYMBOL_LOGICALNOT));
	m_SymbolList.insert(std::pair<std::string, int>("#LogicalOr", SYMBOL_LOGICALOR));
	m_SymbolList.insert(std::pair<std::string, int>("#MatMul", SYMBOL_MATMUL));
	m_SymbolList.insert(std::pair<std::string, int>("#Max", SYMBOL_MAX));
	m_SymbolList.insert(std::pair<std::string, int>("#Maximum", SYMBOL_MAXIMUM));
	m_SymbolList.insert(std::pair<std::string, int>("#Mean", SYMBOL_MEAN));
	m_SymbolList.insert(std::pair<std::string, int>("#Min", SYMBOL_MIN));
	m_SymbolList.insert(std::pair<std::string, int>("#Minimum", SYMBOL_MINIMUM));
	m_SymbolList.insert(std::pair<std::string, int>("#Mod", SYMBOL_MOD));
	m_SymbolList.insert(std::pair<std::string, int>("#Multiply", SYMBOL_MULTIPLY));
	m_SymbolList.insert(std::pair<std::string, int>("#Negate", SYMBOL_NEGATE));
	m_SymbolList.insert(std::pair<std::string, int>("#NotEqual", SYMBOL_NOTEQUAL));
	m_SymbolList.insert(std::pair<std::string, int>("#Polygamma", SYMBOL_POLYGAMMA));
	m_SymbolList.insert(std::pair<std::string, int>("#Pow", SYMBOL_POW));
	m_SymbolList.insert(std::pair<std::string, int>("#Prod", SYMBOL_PROD));
	m_SymbolList.insert(std::pair<std::string, int>("#QuantizeDownAndShrinkRange", SYMBOL_QUANTIZEDOWNANDSHRINKRANGE));
	m_SymbolList.insert(std::pair<std::string, int>("#QuantizedMatMul", SYMBOL_QUANTIZEDMATMUL));
	m_SymbolList.insert(std::pair<std::string, int>("#QuantizedMul", SYMBOL_QUANTIZEDMUL));
	m_SymbolList.insert(std::pair<std::string, int>("#Range", SYMBOL_RANGE));
	m_SymbolList.insert(std::pair<std::string, int>("#Real", SYMBOL_REAL));
	m_SymbolList.insert(std::pair<std::string, int>("#RealDiv", SYMBOL_REALDIV));
	m_SymbolList.insert(std::pair<std::string, int>("#Reciprocal", SYMBOL_RECIPROCAL));
	m_SymbolList.insert(std::pair<std::string, int>("#RequantizationRange", SYMBOL_REQUANTIZATIONRANGE));
	m_SymbolList.insert(std::pair<std::string, int>("#Requantize", SYMBOL_REQUANTIZE));
	m_SymbolList.insert(std::pair<std::string, int>("#Rint", SYMBOL_RINT));
	m_SymbolList.insert(std::pair<std::string, int>("#Round", SYMBOL_ROUND));
	m_SymbolList.insert(std::pair<std::string, int>("#Rsqrt", SYMBOL_RSQRT));
	m_SymbolList.insert(std::pair<std::string, int>("#SegmentMax", SYMBOL_SEGMENTMAX));
	m_SymbolList.insert(std::pair<std::string, int>("#SegmentMean", SYMBOL_SEGMENTMEAN));
	m_SymbolList.insert(std::pair<std::string, int>("#SegmentMin", SYMBOL_SEGMENTMIN));
	m_SymbolList.insert(std::pair<std::string, int>("#SegmentProd", SYMBOL_SEGMENTPROD));
	m_SymbolList.insert(std::pair<std::string, int>("#SegmentSum", SYMBOL_SEGMENTSUM));
	m_SymbolList.insert(std::pair<std::string, int>("#Sigmoid", SYMBOL_SIGMOID));
	m_SymbolList.insert(std::pair<std::string, int>("#Sign", SYMBOL_SIGN));
	m_SymbolList.insert(std::pair<std::string, int>("#Sin", SYMBOL_SIN));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseMatMul", SYMBOL_SPARSEMATMUL));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseSegmentMean", SYMBOL_SPARSESEGMENTMEAN));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseSegmentMeanGrad", SYMBOL_SPARSESEGMENTMEANGRAD));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseSegmentSqrtN", SYMBOL_SPARSESEGMENTSQRTN));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseSegmentSqrtNGrad", SYMBOL_SPARSESEGMENTSQRTNGRAD));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseSegmentSum", SYMBOL_SPARSESEGMENTSUM));
	m_SymbolList.insert(std::pair<std::string, int>("#Sqrt", SYMBOL_SQRT));
	m_SymbolList.insert(std::pair<std::string, int>("#Square", SYMBOL_SQUARE));
	m_SymbolList.insert(std::pair<std::string, int>("#SquaredDifference", SYMBOL_SQUAREDDIFFERENCE));
	m_SymbolList.insert(std::pair<std::string, int>("#Subtract", SYMBOL_SUBTRACT));
	m_SymbolList.insert(std::pair<std::string, int>("#Sum", SYMBOL_SUM));
	m_SymbolList.insert(std::pair<std::string, int>("#Tan", SYMBOL_TAN));
	m_SymbolList.insert(std::pair<std::string, int>("#Tanh", SYMBOL_TANH));
	m_SymbolList.insert(std::pair<std::string, int>("#TruncateDiv", SYMBOL_TRUNCATEDIV));
	m_SymbolList.insert(std::pair<std::string, int>("#TruncateMod", SYMBOL_TRUNCATEMOD));
	m_SymbolList.insert(std::pair<std::string, int>("#UnsortedSegmentMax", SYMBOL_UNSORTEDSEGMENTMAX));
	m_SymbolList.insert(std::pair<std::string, int>("#UnsortedSegmentSum", SYMBOL_UNSORTEDSEGMENTSUM));
	m_SymbolList.insert(std::pair<std::string, int>("#Where3", SYMBOL_WHERE3));
	m_SymbolList.insert(std::pair<std::string, int>("#Zeta", SYMBOL_ZETA));
	m_SymbolList.insert(std::pair<std::string, int>("#AvgPool", SYMBOL_AVGPOOL));
	m_SymbolList.insert(std::pair<std::string, int>("#AvgPool3D", SYMBOL_AVGPOOL3D));
	m_SymbolList.insert(std::pair<std::string, int>("#AvgPool3DGrad", SYMBOL_AVGPOOL3DGRAD));
	m_SymbolList.insert(std::pair<std::string, int>("#BiasAdd", SYMBOL_BIASADD));
	m_SymbolList.insert(std::pair<std::string, int>("#BiasAddGrad", SYMBOL_BIASADDGRAD));
	m_SymbolList.insert(std::pair<std::string, int>("#Conv2D", SYMBOL_CONV2D));
	m_SymbolList.insert(std::pair<std::string, int>("#Conv2DBackpropFilter", SYMBOL_CONV2DBACKPROPFILTER));
	m_SymbolList.insert(std::pair<std::string, int>("#Conv2DBackpropInput", SYMBOL_CONV2DBACKPROPINPUT));
	m_SymbolList.insert(std::pair<std::string, int>("#Conv3D", SYMBOL_CONV3D));
	m_SymbolList.insert(std::pair<std::string, int>("#Conv3DBackpropFilterV2", SYMBOL_CONV3DBACKPROPFILTERV2));
	m_SymbolList.insert(std::pair<std::string, int>("#Conv3DBackpropInputV2", SYMBOL_CONV3DBACKPROPINPUTV2));
	m_SymbolList.insert(std::pair<std::string, int>("#DepthwiseConv2dNative", SYMBOL_DEPTHWISECONV2DNATIVE));
	m_SymbolList.insert(std::pair<std::string, int>("#DepthwiseConv2dNativeBackpropFilter", SYMBOL_DEPTHWISECONV2DNATIVEBACKPROPFILTER));
	m_SymbolList.insert(std::pair<std::string, int>("#DepthwiseConv2dNativeBackpropInput", SYMBOL_DEPTHWISECONV2DNATIVEBACKPROPINPUT));
	m_SymbolList.insert(std::pair<std::string, int>("#Dilation2D", SYMBOL_DILATION2D));
	m_SymbolList.insert(std::pair<std::string, int>("#Dilation2DBackpropFilter", SYMBOL_DILATION2DBACKPROPFILTER));
	m_SymbolList.insert(std::pair<std::string, int>("#Dilation2DBackpropInput", SYMBOL_DILATION2DBACKPROPINPUT));
	m_SymbolList.insert(std::pair<std::string, int>("#Elu", SYMBOL_ELU));
	m_SymbolList.insert(std::pair<std::string, int>("#FractionalAvgPool", SYMBOL_FRACTIONALAVGPOOL));
	m_SymbolList.insert(std::pair<std::string, int>("#FractionalMaxPool", SYMBOL_FRACTIONALMAXPOOL));
	m_SymbolList.insert(std::pair<std::string, int>("#FusedBatchNorm", SYMBOL_FUSEDBATCHNORM));
	m_SymbolList.insert(std::pair<std::string, int>("#FusedBatchNormGrad", SYMBOL_FUSEDBATCHNORMGRAD));
	m_SymbolList.insert(std::pair<std::string, int>("#FusedPadConv2D", SYMBOL_FUSEDPADCONV2D));
	m_SymbolList.insert(std::pair<std::string, int>("#FusedResizeAndPadConv2D", SYMBOL_FUSEDRESIZEANDPADCONV2D));
	m_SymbolList.insert(std::pair<std::string, int>("#InTopK", SYMBOL_INTOPK));
	m_SymbolList.insert(std::pair<std::string, int>("#L2Loss", SYMBOL_L2LOSS));
	m_SymbolList.insert(std::pair<std::string, int>("#LRN", SYMBOL_LRN));
	m_SymbolList.insert(std::pair<std::string, int>("#LogSoftmax", SYMBOL_LOGSOFTMAX));
	m_SymbolList.insert(std::pair<std::string, int>("#MaxPool", SYMBOL_MAXPOOL));
	m_SymbolList.insert(std::pair<std::string, int>("#MaxPool3D", SYMBOL_MAXPOOL3D));
	m_SymbolList.insert(std::pair<std::string, int>("#MaxPool3DGrad", SYMBOL_MAXPOOL3DGRAD));
	m_SymbolList.insert(std::pair<std::string, int>("#MaxPool3DGradGrad", SYMBOL_MAXPOOL3DGRADGRAD));
	m_SymbolList.insert(std::pair<std::string, int>("#MaxPoolGradGrad", SYMBOL_MAXPOOLGRADGRAD));
	m_SymbolList.insert(std::pair<std::string, int>("#MaxPoolGradGradWithArgmax", SYMBOL_MAXPOOLGRADGRADWITHARGMAX));
	m_SymbolList.insert(std::pair<std::string, int>("#MaxPoolWithArgmax", SYMBOL_MAXPOOLWITHARGMAX));
	m_SymbolList.insert(std::pair<std::string, int>("#QuantizedAvgPool", SYMBOL_QUANTIZEDAVGPOOL));
	m_SymbolList.insert(std::pair<std::string, int>("#QuantizedBatchNormWithGlobalNormalization", SYMBOL_QUANTIZEDBATCHNORMWITHGLOBALNORMALIZATION));
	m_SymbolList.insert(std::pair<std::string, int>("#QuantizedBiasAdd", SYMBOL_QUANTIZEDBIASADD));
	m_SymbolList.insert(std::pair<std::string, int>("#QuantizedConv2D", SYMBOL_QUANTIZEDCONV2D));
	m_SymbolList.insert(std::pair<std::string, int>("#QuantizedMaxPool", SYMBOL_QUANTIZEDMAXPOOL));
	m_SymbolList.insert(std::pair<std::string, int>("#QuantizedRelu", SYMBOL_QUANTIZEDRELU));
	m_SymbolList.insert(std::pair<std::string, int>("#QuantizedRelu6", SYMBOL_QUANTIZEDRELU6));
	m_SymbolList.insert(std::pair<std::string, int>("#QuantizedReluX", SYMBOL_QUANTIZEDRELUX));
	m_SymbolList.insert(std::pair<std::string, int>("#Relu", SYMBOL_RELU));
	m_SymbolList.insert(std::pair<std::string, int>("#Relu6", SYMBOL_RELU6));
	m_SymbolList.insert(std::pair<std::string, int>("#Softmax", SYMBOL_SOFTMAX));
	m_SymbolList.insert(std::pair<std::string, int>("#SoftmaxCrossEntropyWithLogits", SYMBOL_SOFTMAXCROSSENTROPYWITHLOGITS));
	m_SymbolList.insert(std::pair<std::string, int>("#Softplus", SYMBOL_SOFTPLUS));
	m_SymbolList.insert(std::pair<std::string, int>("#Softsign", SYMBOL_SOFTSIGN));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseSoftmaxCrossEntropyWithLogits", SYMBOL_SPARSESOFTMAXCROSSENTROPYWITHLOGITS));
	m_SymbolList.insert(std::pair<std::string, int>("#TopK", SYMBOL_TOPK));
	m_SymbolList.insert(std::pair<std::string, int>("#NoOp", SYMBOL_NOOP));
	m_SymbolList.insert(std::pair<std::string, int>("#DecodeCSV", SYMBOL_DECODECSV));
	m_SymbolList.insert(std::pair<std::string, int>("#DecodeJSONExample", SYMBOL_DECODEJSONEXAMPLE));
	m_SymbolList.insert(std::pair<std::string, int>("#DecodeRaw", SYMBOL_DECODERAW));
	m_SymbolList.insert(std::pair<std::string, int>("#ParseExample", SYMBOL_PARSEEXAMPLE));
	m_SymbolList.insert(std::pair<std::string, int>("#ParseSingleSequenceExample", SYMBOL_PARSESINGLESEQUENCEEXAMPLE));
	m_SymbolList.insert(std::pair<std::string, int>("#ParseTensor", SYMBOL_PARSETENSOR));
	m_SymbolList.insert(std::pair<std::string, int>("#StringToNumber", SYMBOL_STRINGTONUMBER));
	m_SymbolList.insert(std::pair<std::string, int>("#Multinomial", SYMBOL_MULTINOMIAL));
	m_SymbolList.insert(std::pair<std::string, int>("#ParameterizedTruncatedNormal", SYMBOL_PARAMETERIZEDTRUNCATEDNORMAL));
	m_SymbolList.insert(std::pair<std::string, int>("#RandomGamma", SYMBOL_RANDOMGAMMA));
	m_SymbolList.insert(std::pair<std::string, int>("#RandomNormal", SYMBOL_RANDOMNORMAL));
	m_SymbolList.insert(std::pair<std::string, int>("#RandomPoisson", SYMBOL_RANDOMPOISSON));
	m_SymbolList.insert(std::pair<std::string, int>("#RandomShuffle", SYMBOL_RANDOMSHUFFLE));
	m_SymbolList.insert(std::pair<std::string, int>("#RandomUniform", SYMBOL_RANDOMUNIFORM));
	m_SymbolList.insert(std::pair<std::string, int>("#RandomUniformInt", SYMBOL_RANDOMUNIFORMINT));
	m_SymbolList.insert(std::pair<std::string, int>("#TruncatedNormal", SYMBOL_TRUNCATEDNORMAL));
	m_SymbolList.insert(std::pair<std::string, int>("#AddManySparseToTensorsMap", SYMBOL_ADDMANYSPARSETOTENSORSMAP));
	m_SymbolList.insert(std::pair<std::string, int>("#AddSparseToTensorsMap", SYMBOL_ADDSPARSETOTENSORSMAP));
	m_SymbolList.insert(std::pair<std::string, int>("#DeserializeManySparse", SYMBOL_DESERIALIZEMANYSPARSE));
	m_SymbolList.insert(std::pair<std::string, int>("#SerializeManySparse", SYMBOL_SERIALIZEMANYSPARSE));
	m_SymbolList.insert(std::pair<std::string, int>("#SerializeSparse", SYMBOL_SERIALIZESPARSE));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseAdd", SYMBOL_SPARSEADD));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseAddGrad", SYMBOL_SPARSEADDGRAD));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseConcat", SYMBOL_SPARSECONCAT));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseCross", SYMBOL_SPARSECROSS));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseDenseCwiseAdd", SYMBOL_SPARSEDENSECWISEADD));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseDenseCwiseDiv", SYMBOL_SPARSEDENSECWISEDIV));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseDenseCwiseMul", SYMBOL_SPARSEDENSECWISEMUL));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseReduceSum", SYMBOL_SPARSEREDUCESUM));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseReduceSumSparse", SYMBOL_SPARSEREDUCESUMSPARSE));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseReorder", SYMBOL_SPARSEREORDER));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseReshape", SYMBOL_SPARSERESHAPE));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseSoftmax", SYMBOL_SPARSESOFTMAX));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseSparseMaximum", SYMBOL_SPARSESPARSEMAXIMUM));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseSparseMinimum", SYMBOL_SPARSESPARSEMINIMUM));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseSplit", SYMBOL_SPARSESPLIT));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseTensorDenseAdd", SYMBOL_SPARSETENSORDENSEADD));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseTensorDenseMatMul", SYMBOL_SPARSETENSORDENSEMATMUL));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseToDense", SYMBOL_SPARSETODENSE));
	m_SymbolList.insert(std::pair<std::string, int>("#TakeManySparseFromTensorsMap", SYMBOL_TAKEMANYSPARSEFROMTENSORSMAP));
	m_SymbolList.insert(std::pair<std::string, int>("#Assign", SYMBOL_ASSIGN));
	m_SymbolList.insert(std::pair<std::string, int>("#AssignAdd", SYMBOL_ASSIGNADD));
	m_SymbolList.insert(std::pair<std::string, int>("#AssignSub", SYMBOL_ASSIGNSUB));
	m_SymbolList.insert(std::pair<std::string, int>("#CountUpTo", SYMBOL_COUNTUPTO));
	m_SymbolList.insert(std::pair<std::string, int>("#DestroyTemporaryVariable", SYMBOL_DESTROYTEMPORARYVARIABLE));
	m_SymbolList.insert(std::pair<std::string, int>("#IsVariableInitialized", SYMBOL_ISVARIABLEINITIALIZED));
	m_SymbolList.insert(std::pair<std::string, int>("#ScatterAdd", SYMBOL_SCATTERADD));
	m_SymbolList.insert(std::pair<std::string, int>("#ScatterDiv", SYMBOL_SCATTERDIV));
	m_SymbolList.insert(std::pair<std::string, int>("#ScatterMul", SYMBOL_SCATTERMUL));
	m_SymbolList.insert(std::pair<std::string, int>("#ScatterNdAdd", SYMBOL_SCATTERNDADD));
	m_SymbolList.insert(std::pair<std::string, int>("#ScatterNdSub", SYMBOL_SCATTERNDSUB));
	m_SymbolList.insert(std::pair<std::string, int>("#ScatterNdUpdate", SYMBOL_SCATTERNDUPDATE));
	m_SymbolList.insert(std::pair<std::string, int>("#ScatterSub", SYMBOL_SCATTERSUB));
	m_SymbolList.insert(std::pair<std::string, int>("#ScatterUpdate", SYMBOL_SCATTERUPDATE));
	m_SymbolList.insert(std::pair<std::string, int>("#TemporaryVariable", SYMBOL_TEMPORARYVARIABLE));
	m_SymbolList.insert(std::pair<std::string, int>("#Variable", SYMBOL_VARIABLE));
	m_SymbolList.insert(std::pair<std::string, int>("#AsString", SYMBOL_ASSTRING));
	m_SymbolList.insert(std::pair<std::string, int>("#DecodeBase64", SYMBOL_DECODEBASE64));
	m_SymbolList.insert(std::pair<std::string, int>("#EncodeBase64", SYMBOL_ENCODEBASE64));
	m_SymbolList.insert(std::pair<std::string, int>("#ReduceJoin", SYMBOL_REDUCEJOIN));
	m_SymbolList.insert(std::pair<std::string, int>("#StringJoin", SYMBOL_STRINGJOIN));
	m_SymbolList.insert(std::pair<std::string, int>("#StringSplit", SYMBOL_STRINGSPLIT));
	m_SymbolList.insert(std::pair<std::string, int>("#StringToHashBucket", SYMBOL_STRINGTOHASHBUCKET));
	m_SymbolList.insert(std::pair<std::string, int>("#StringToHashBucketFast", SYMBOL_STRINGTOHASHBUCKETFAST));
	m_SymbolList.insert(std::pair<std::string, int>("#StringToHashBucketStrong", SYMBOL_STRINGTOHASHBUCKETSTRONG));
	m_SymbolList.insert(std::pair<std::string, int>("#Substr", SYMBOL_SUBSTR));
	m_SymbolList.insert(std::pair<std::string, int>("#ApplyAdadelta", SYMBOL_APPLYADADELTA));
	m_SymbolList.insert(std::pair<std::string, int>("#ApplyAdagrad", SYMBOL_APPLYADAGRAD));
	m_SymbolList.insert(std::pair<std::string, int>("#ApplyAdagradDA", SYMBOL_APPLYADAGRADDA));
	m_SymbolList.insert(std::pair<std::string, int>("#ApplyAdam", SYMBOL_APPLYADAM));
	m_SymbolList.insert(std::pair<std::string, int>("#ApplyCenteredRMSProp", SYMBOL_APPLYCENTEREDRMSPROP));
	m_SymbolList.insert(std::pair<std::string, int>("#ApplyFtrl", SYMBOL_APPLYFTRL));
	m_SymbolList.insert(std::pair<std::string, int>("#ApplyGradientDescent", SYMBOL_APPLYGRADIENTDESCENT));
	m_SymbolList.insert(std::pair<std::string, int>("#ApplyMomentum", SYMBOL_APPLYMOMENTUM));
	m_SymbolList.insert(std::pair<std::string, int>("#ApplyProximalAdagrad", SYMBOL_APPLYPROXIMALADAGRAD));
	m_SymbolList.insert(std::pair<std::string, int>("#ApplyProximalGradientDescent", SYMBOL_APPLYPROXIMALGRADIENTDESCENT));
	m_SymbolList.insert(std::pair<std::string, int>("#ApplyRMSProp", SYMBOL_APPLYRMSPROP));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceApplyAdadelta", SYMBOL_RESOURCEAPPLYADADELTA));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceApplyAdagrad", SYMBOL_RESOURCEAPPLYADAGRAD));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceApplyAdagradDA", SYMBOL_RESOURCEAPPLYADAGRADDA));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceApplyAdam", SYMBOL_RESOURCEAPPLYADAM));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceApplyCenteredRMSProp", SYMBOL_RESOURCEAPPLYCENTEREDRMSPROP));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceApplyFtrl", SYMBOL_RESOURCEAPPLYFTRL));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceApplyGradientDescent", SYMBOL_RESOURCEAPPLYGRADIENTDESCENT));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceApplyMomentum", SYMBOL_RESOURCEAPPLYMOMENTUM));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceApplyProximalAdagrad", SYMBOL_RESOURCEAPPLYPROXIMALADAGRAD));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceApplyProximalGradientDescent", SYMBOL_RESOURCEAPPLYPROXIMALGRADIENTDESCENT));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceApplyRMSProp", SYMBOL_RESOURCEAPPLYRMSPROP));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceSparseApplyAdadelta", SYMBOL_RESOURCESPARSEAPPLYADADELTA));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceSparseApplyAdagrad", SYMBOL_RESOURCESPARSEAPPLYADAGRAD));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceSparseApplyAdagradDA", SYMBOL_RESOURCESPARSEAPPLYADAGRADDA));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceSparseApplyCenteredRMSProp", SYMBOL_RESOURCESPARSEAPPLYCENTEREDRMSPROP));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceSparseApplyFtrl", SYMBOL_RESOURCESPARSEAPPLYFTRL));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceSparseApplyMomentum", SYMBOL_RESOURCESPARSEAPPLYMOMENTUM));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceSparseApplyProximalAdagrad", SYMBOL_RESOURCESPARSEAPPLYPROXIMALADAGRAD));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceSparseApplyProximalGradientDescent", SYMBOL_RESOURCESPARSEAPPLYPROXIMALGRADIENTDESCENT));
	m_SymbolList.insert(std::pair<std::string, int>("#ResourceSparseApplyRMSProp", SYMBOL_RESOURCESPARSEAPPLYRMSPROP));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseApplyAdadelta", SYMBOL_SPARSEAPPLYADADELTA));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseApplyAdagrad", SYMBOL_SPARSEAPPLYADAGRAD));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseApplyAdagradDA", SYMBOL_SPARSEAPPLYADAGRADDA));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseApplyCenteredRMSProp", SYMBOL_SPARSEAPPLYCENTEREDRMSPROP));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseApplyFtrl", SYMBOL_SPARSEAPPLYFTRL));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseApplyMomentum", SYMBOL_SPARSEAPPLYMOMENTUM));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseApplyProximalAdagrad", SYMBOL_SPARSEAPPLYPROXIMALADAGRAD));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseApplyProximalGradientDescent", SYMBOL_SPARSEAPPLYPROXIMALGRADIENTDESCENT));
	m_SymbolList.insert(std::pair<std::string, int>("#SparseApplyRMSProp", SYMBOL_SPARSEAPPLYRMSPROP));
	m_SymbolList.insert(std::pair<std::string, int>("#Fact", SYMBOL_FACT));
	m_SymbolList.insert(std::pair<std::string, int>("#Const", SYMBOL_CONST));
	m_SymbolList.insert(std::pair<std::string, int>("#Input_ex", SYMBOL_INPUT_EX));
	m_SymbolList.insert(std::pair<std::string, int>("#RandomNormal_ex", SYMBOL_RANDOMNORMAL_EX));
}

int GetSymbolType(std::string strSymbolName)
{
	int iType = SYMBOL_NONE;
	const std::map<std::string, int>::const_iterator aLookup = m_SymbolList.find(strSymbolName);
	const bool bExists = aLookup != m_SymbolList.end();
	if (bExists)
	{
		iType = aLookup->second;
		return iType;
	}
	else
	{
		return iType;
	}
}

void* Create_Symbol(int iSymbol, std::string id, Json::Value pInputItem)
{
	void* pCreate = nullptr;
	switch (iSymbol)
	{
	case SYMBOL_BATCHTOSPACE: {		pCreate = Create_BatchToSpace(id, pInputItem);	break;	}
	case SYMBOL_BATCHTOSPACEND: {		pCreate = Create_BatchToSpaceND(id, pInputItem);	break;	}
	case SYMBOL_BITCAST: {		pCreate = Create_Bitcast(id, pInputItem);	break;	}
	case SYMBOL_BROADCASTDYNAMICSHAPE: {		pCreate = Create_BroadcastDynamicShape(id, pInputItem);	break;	}
	case SYMBOL_CHECKNUMERICS: {		pCreate = Create_CheckNumerics(id, pInputItem);	break;	}
	case SYMBOL_CONCAT: {		pCreate = Create_Concat(id, pInputItem);	break;	}
	case SYMBOL_DEPTHTOSPACE: {		pCreate = Create_DepthToSpace(id, pInputItem);	break;	}
	case SYMBOL_DEQUANTIZE: {		pCreate = Create_Dequantize(id, pInputItem);	break;	}
	case SYMBOL_DIAG: {		pCreate = Create_Diag(id, pInputItem);	break;	}
	case SYMBOL_DIAGPART: {		pCreate = Create_DiagPart(id, pInputItem);	break;	}
	case SYMBOL_EDITDISTANCE: {		pCreate = Create_EditDistance(id, pInputItem);	break;	}
	case SYMBOL_EXPANDDIMS: {		pCreate = Create_ExpandDims(id, pInputItem);	break;	}
	case SYMBOL_EXTRACTIMAGEPATCHES: {		pCreate = Create_ExtractImagePatches(id, pInputItem);	break;	}
	case SYMBOL_FAKEQUANTWITHMINMAXARGS: {		pCreate = Create_FakeQuantWithMinMaxArgs(id, pInputItem);	break;	}
	case SYMBOL_FAKEQUANTWITHMINMAXARGSGRADIENT: {		pCreate = Create_FakeQuantWithMinMaxArgsGradient(id, pInputItem);	break;	}
	case SYMBOL_FAKEQUANTWITHMINMAXVARS: {		pCreate = Create_FakeQuantWithMinMaxVars(id, pInputItem);	break;	}
	case SYMBOL_FAKEQUANTWITHMINMAXVARSGRADIENT: {		pCreate = Create_FakeQuantWithMinMaxVarsGradient(id, pInputItem);	break;	}
	case SYMBOL_FAKEQUANTWITHMINMAXVARSPERCHANNEL: {		pCreate = Create_FakeQuantWithMinMaxVarsPerChannel(id, pInputItem);	break;	}
	case SYMBOL_FAKEQUANTWITHMINMAXVARSPERCHANNELGRADIENT: {		pCreate = Create_FakeQuantWithMinMaxVarsPerChannelGradient(id, pInputItem);	break;	}
	case SYMBOL_FILL: {		pCreate = Create_Fill(id, pInputItem);	break;	}
	case SYMBOL_GATHER: {		pCreate = Create_Gather(id, pInputItem);	break;	}
	case SYMBOL_GATHERND: {		pCreate = Create_GatherNd(id, pInputItem);	break;	}
	case SYMBOL_IDENTITY: {		pCreate = Create_Identity(id, pInputItem);	break;	}
	case SYMBOL_IMMUTABLECONST: {		pCreate = Create_ImmutableConst(id, pInputItem);	break;	}
	case SYMBOL_INVERTPERMUTATION: {		pCreate = Create_InvertPermutation(id, pInputItem);	break;	}
	case SYMBOL_MATRIXBANDPART: {		pCreate = Create_MatrixBandPart(id, pInputItem);	break;	}
	case SYMBOL_MATRIXDIAG: {		pCreate = Create_MatrixDiag(id, pInputItem);	break;	}
	case SYMBOL_MATRIXDIAGPART: {		pCreate = Create_MatrixDiagPart(id, pInputItem);	break;	}
	case SYMBOL_MATRIXSETDIAG: {		pCreate = Create_MatrixSetDiag(id, pInputItem);	break;	}
	case SYMBOL_MIRRORPAD: {		pCreate = Create_MirrorPad(id, pInputItem);	break;	}
	case SYMBOL_ONEHOT: {		pCreate = Create_OneHot(id, pInputItem);	break;	}
	case SYMBOL_ONESLIKE: {		pCreate = Create_OnesLike(id, pInputItem);	break;	}
	case SYMBOL_PAD: {		pCreate = Create_Pad(id, pInputItem);	break;	}
	case SYMBOL_PARALLELCONCAT: {		pCreate = Create_ParallelConcat(id, pInputItem);	break;	}
	case SYMBOL_PLACEHOLDER: {		pCreate = Create_Placeholder(id, pInputItem);	break;	}
	case SYMBOL_PLACEHOLDERV2: {		pCreate = Create_PlaceholderV2(id, pInputItem);	break;	}
	case SYMBOL_PLACEHOLDERWITHDEFAULT: {		pCreate = Create_PlaceholderWithDefault(id, pInputItem);	break;	}
	case SYMBOL_PREVENTGRADIENT: {		pCreate = Create_PreventGradient(id, pInputItem);	break;	}
	case SYMBOL_QUANTIZEANDDEQUANTIZEV2: {		pCreate = Create_QuantizeAndDequantizeV2(id, pInputItem);	break;	}
	case SYMBOL_QUANTIZEV2: {		pCreate = Create_QuantizeV2(id, pInputItem);	break;	}
	case SYMBOL_QUANTIZEDCONCAT: {		pCreate = Create_QuantizedConcat(id, pInputItem);	break;	}
	case SYMBOL_QUANTIZEDINSTANCENORM: {		pCreate = Create_QuantizedInstanceNorm(id, pInputItem);	break;	}
	case SYMBOL_QUANTIZEDRESHAPE: {		pCreate = Create_QuantizedReshape(id, pInputItem);	break;	}
	case SYMBOL_RANK: {		pCreate = Create_Rank(id, pInputItem);	break;	}
	case SYMBOL_RESHAPE: {		pCreate = Create_Reshape(id, pInputItem);	break;	}
	case SYMBOL_RESOURCESTRIDEDSLICEASSIGN: {		pCreate = Create_ResourceStridedSliceAssign(id, pInputItem);	break;	}
	case SYMBOL_REVERSE: {		pCreate = Create_Reverse(id, pInputItem);	break;	}
	case SYMBOL_REVERSESEQUENCE: {		pCreate = Create_ReverseSequence(id, pInputItem);	break;	}
	case SYMBOL_SCATTERND: {		pCreate = Create_ScatterNd(id, pInputItem);	break;	}
	case SYMBOL_SETDIFF1D: {		pCreate = Create_SetDiff1D(id, pInputItem);	break;	}
	case SYMBOL_SHAPE: {		pCreate = Create_Shape(id, pInputItem);	break;	}
	case SYMBOL_SHAPEN: {		pCreate = Create_ShapeN(id, pInputItem);	break;	}
	case SYMBOL_SIZE: {		pCreate = Create_Size(id, pInputItem);	break;	}
	case SYMBOL_SLICE: {		pCreate = Create_Slice(id, pInputItem);	break;	}
	case SYMBOL_SPACETOBATCH: {		pCreate = Create_SpaceToBatch(id, pInputItem);	break;	}
	case SYMBOL_SPACETOBATCHND: {		pCreate = Create_SpaceToBatchND(id, pInputItem);	break;	}
	case SYMBOL_SPACETODEPTH: {		pCreate = Create_SpaceToDepth(id, pInputItem);	break;	}
	case SYMBOL_SPLIT: {		pCreate = Create_Split(id, pInputItem);	break;	}
	case SYMBOL_SPLITV: {		pCreate = Create_SplitV(id, pInputItem);	break;	}
	case SYMBOL_SQUEEZE: {		pCreate = Create_Squeeze(id, pInputItem);	break;	}
	case SYMBOL_STACK: {		pCreate = Create_Stack(id, pInputItem);	break;	}
	case SYMBOL_STOPGRADIENT: {		pCreate = Create_StopGradient(id, pInputItem);	break;	}
	case SYMBOL_STRIDEDSLICE: {		pCreate = Create_StridedSlice(id, pInputItem);	break;	}
	case SYMBOL_STRIDEDSLICEASSIGN: {		pCreate = Create_StridedSliceAssign(id, pInputItem);	break;	}
	case SYMBOL_STRIDEDSLICEGRAD: {		pCreate = Create_StridedSliceGrad(id, pInputItem);	break;	}
	case SYMBOL_TILE: {		pCreate = Create_Tile(id, pInputItem);	break;	}
	case SYMBOL_TRANSPOSE: {		pCreate = Create_Transpose(id, pInputItem);	break;	}
	case SYMBOL_UNIQUE: {		pCreate = Create_Unique(id, pInputItem);	break;	}
	case SYMBOL_UNIQUEWITHCOUNTS: {		pCreate = Create_UniqueWithCounts(id, pInputItem);	break;	}
	case SYMBOL_UNSTACK: {		pCreate = Create_Unstack(id, pInputItem);	break;	}
	case SYMBOL_WHERE: {		pCreate = Create_Where(id, pInputItem);	break;	}
	case SYMBOL_ZEROSLIKE: {		pCreate = Create_ZerosLike(id, pInputItem);	break;	}
	case SYMBOL_ALLCANDIDATESAMPLER: {		pCreate = Create_AllCandidateSampler(id, pInputItem);	break;	}
	case SYMBOL_COMPUTEACCIDENTALHITS: {		pCreate = Create_ComputeAccidentalHits(id, pInputItem);	break;	}
	case SYMBOL_FIXEDUNIGRAMCANDIDATESAMPLER: {		pCreate = Create_FixedUnigramCandidateSampler(id, pInputItem);	break;	}
	case SYMBOL_LEARNEDUNIGRAMCANDIDATESAMPLER: {		pCreate = Create_LearnedUnigramCandidateSampler(id, pInputItem);	break;	}
	case SYMBOL_LOGUNIFORMCANDIDATESAMPLER: {		pCreate = Create_LogUniformCandidateSampler(id, pInputItem);	break;	}
	case SYMBOL_UNIFORMCANDIDATESAMPLER: {		pCreate = Create_UniformCandidateSampler(id, pInputItem);	break;	}
	case SYMBOL_ABORT: {		pCreate = Create_Abort(id, pInputItem);	break;	}
	case SYMBOL_CONTROLTRIGGER: {		pCreate = Create_ControlTrigger(id, pInputItem);	break;	}
	case SYMBOL_LOOPCOND: {		pCreate = Create_LoopCond(id, pInputItem);	break;	}
	case SYMBOL_MERGE: {		pCreate = Create_Merge(id, pInputItem);	break;	}
	case SYMBOL_NEXTITERATION: {		pCreate = Create_NextIteration(id, pInputItem);	break;	}
	case SYMBOL_REFNEXTITERATION: {		pCreate = Create_RefNextIteration(id, pInputItem);	break;	}
	case SYMBOL_REFSELECT: {		pCreate = Create_RefSelect(id, pInputItem);	break;	}
	case SYMBOL_REFSWITCH: {		pCreate = Create_RefSwitch(id, pInputItem);	break;	}
	case SYMBOL_SWITCH: {		pCreate = Create_Switch(id, pInputItem);	break;	}
	case SYMBOL_CLIENTSESSION: {		pCreate = Create_ClientSession(id, pInputItem);	break;	}
	case SYMBOL_INPUT: {		pCreate = Create_Input(id, pInputItem);	break;	}
	case SYMBOL_INPUT_INITIALIZER: {		pCreate = Create_Input_Initializer(id, pInputItem);	break;	}
	case SYMBOL_INPUTLIST: {		pCreate = Create_InputList(id, pInputItem);	break;	}
	case SYMBOL_OPERATION: {		pCreate = Create_Operation(id, pInputItem);	break;	}
	case SYMBOL_OUTPUT: {		pCreate = Create_Output(id, pInputItem);	break;	}
	case SYMBOL_SCOPE: {		pCreate = Create_Scope(id, pInputItem);	break;	}
	case SYMBOL_STATUS: {		pCreate = Create_Status(id, pInputItem);	break;	}
	case SYMBOL_TENSOR: {		pCreate = Create_Tensor(id, pInputItem);	break;	}
	case SYMBOL_ACCUMULATORAPPLYGRADIENT: {		pCreate = Create_AccumulatorApplyGradient(id, pInputItem);	break;	}
	case SYMBOL_ACCUMULATORNUMACCUMULATED: {		pCreate = Create_AccumulatorNumAccumulated(id, pInputItem);	break;	}
	case SYMBOL_ACCUMULATORSETGLOBALSTEP: {		pCreate = Create_AccumulatorSetGlobalStep(id, pInputItem);	break;	}
	case SYMBOL_ACCUMULATORTAKEGRADIENT: {		pCreate = Create_AccumulatorTakeGradient(id, pInputItem);	break;	}
	case SYMBOL_BARRIER: {		pCreate = Create_Barrier(id, pInputItem);	break;	}
	case SYMBOL_BARRIERCLOSE: {		pCreate = Create_BarrierClose(id, pInputItem);	break;	}
	case SYMBOL_BARRIERINCOMPLETESIZE: {		pCreate = Create_BarrierIncompleteSize(id, pInputItem);	break;	}
	case SYMBOL_BARRIERINSERTMANY: {		pCreate = Create_BarrierInsertMany(id, pInputItem);	break;	}
	case SYMBOL_BARRIERREADYSIZE: {		pCreate = Create_BarrierReadySize(id, pInputItem);	break;	}
	case SYMBOL_BARRIERTAKEMANY: {		pCreate = Create_BarrierTakeMany(id, pInputItem);	break;	}
	case SYMBOL_CONDITIONALACCUMULATOR: {		pCreate = Create_ConditionalAccumulator(id, pInputItem);	break;	}
	case SYMBOL_DELETESESSIONTENSOR: {		pCreate = Create_DeleteSessionTensor(id, pInputItem);	break;	}
	case SYMBOL_DYNAMICPARTITION: {		pCreate = Create_DynamicPartition(id, pInputItem);	break;	}
	case SYMBOL_DYNAMICSTITCH: {		pCreate = Create_DynamicStitch(id, pInputItem);	break;	}
	case SYMBOL_FIFOQUEUE: {		pCreate = Create_FIFOQueue(id, pInputItem);	break;	}
	case SYMBOL_GETSESSIONHANDLE: {		pCreate = Create_GetSessionHandle(id, pInputItem);	break;	}
	case SYMBOL_GETSESSIONHANDLEV2: {		pCreate = Create_GetSessionHandleV2(id, pInputItem);	break;	}
	case SYMBOL_GETSESSIONTENSOR: {		pCreate = Create_GetSessionTensor(id, pInputItem);	break;	}
	case SYMBOL_PADDINGFIFOQUEUE: {		pCreate = Create_PaddingFIFOQueue(id, pInputItem);	break;	}
	case SYMBOL_PRIORITYQUEUE: {		pCreate = Create_PriorityQueue(id, pInputItem);	break;	}
	case SYMBOL_QUEUECLOSE: {		pCreate = Create_QueueClose(id, pInputItem);	break;	}
	case SYMBOL_QUEUEDEQUEUE: {		pCreate = Create_QueueDequeue(id, pInputItem);	break;	}
	case SYMBOL_QUEUEDEQUEUEMANY: {		pCreate = Create_QueueDequeueMany(id, pInputItem);	break;	}
	case SYMBOL_QUEUEDEQUEUEUPTO: {		pCreate = Create_QueueDequeueUpTo(id, pInputItem);	break;	}
	case SYMBOL_QUEUEENQUEUE: {		pCreate = Create_QueueEnqueue(id, pInputItem);	break;	}
	case SYMBOL_QUEUEENQUEUEMANY: {		pCreate = Create_QueueEnqueueMany(id, pInputItem);	break;	}
	case SYMBOL_QUEUESIZE: {		pCreate = Create_QueueSize(id, pInputItem);	break;	}
	case SYMBOL_RANDOMSHUFFLEQUEUE: {		pCreate = Create_RandomShuffleQueue(id, pInputItem);	break;	}
	case SYMBOL_RECORDINPUT: {		pCreate = Create_RecordInput(id, pInputItem);	break;	}
	case SYMBOL_SPARSEACCUMULATORAPPLYGRADIENT: {		pCreate = Create_SparseAccumulatorApplyGradient(id, pInputItem);	break;	}
	case SYMBOL_SPARSEACCUMULATORTAKEGRADIENT: {		pCreate = Create_SparseAccumulatorTakeGradient(id, pInputItem);	break;	}
	case SYMBOL_SPARSECONDITIONALACCUMULATOR: {		pCreate = Create_SparseConditionalAccumulator(id, pInputItem);	break;	}
	case SYMBOL_STAGE: {		pCreate = Create_Stage(id, pInputItem);	break;	}
	case SYMBOL_TENSORARRAY: {		pCreate = Create_TensorArray(id, pInputItem);	break;	}
	case SYMBOL_TENSORARRAYCLOSE: {		pCreate = Create_TensorArrayClose(id, pInputItem);	break;	}
	case SYMBOL_TENSORARRAYCONCAT: {		pCreate = Create_TensorArrayConcat(id, pInputItem);	break;	}
	case SYMBOL_TENSORARRAYGATHER: {		pCreate = Create_TensorArrayGather(id, pInputItem);	break;	}
	case SYMBOL_TENSORARRAYGRAD: {		pCreate = Create_TensorArrayGrad(id, pInputItem);	break;	}
	case SYMBOL_TENSORARRAYREAD: {		pCreate = Create_TensorArrayRead(id, pInputItem);	break;	}
	case SYMBOL_TENSORARRAYSCATTER: {		pCreate = Create_TensorArrayScatter(id, pInputItem);	break;	}
	case SYMBOL_TENSORARRAYSIZE: {		pCreate = Create_TensorArraySize(id, pInputItem);	break;	}
	case SYMBOL_TENSORARRAYSPLIT: {		pCreate = Create_TensorArraySplit(id, pInputItem);	break;	}
	case SYMBOL_TENSORARRAYWRITE: {		pCreate = Create_TensorArrayWrite(id, pInputItem);	break;	}
	case SYMBOL_UNSTAGE: {		pCreate = Create_Unstage(id, pInputItem);	break;	}
	case SYMBOL_ADJUSTCONTRAST: {		pCreate = Create_AdjustContrast(id, pInputItem);	break;	}
	case SYMBOL_ADJUSTHUE: {		pCreate = Create_AdjustHue(id, pInputItem);	break;	}
	case SYMBOL_ADJUSTSATURATION: {		pCreate = Create_AdjustSaturation(id, pInputItem);	break;	}
	case SYMBOL_CROPANDRESIZE: {		pCreate = Create_CropAndResize(id, pInputItem);	break;	}
	case SYMBOL_CROPANDRESIZEGRADBOXES: {		pCreate = Create_CropAndResizeGradBoxes(id, pInputItem);	break;	}
	case SYMBOL_CROPANDRESIZEGRADIMAGE: {		pCreate = Create_CropAndResizeGradImage(id, pInputItem);	break;	}
	case SYMBOL_DECODEGIF: {		pCreate = Create_DecodeGif(id, pInputItem);	break;	}
	case SYMBOL_DECODEJPEG: {		pCreate = Create_DecodeJpeg(id, pInputItem);	break;	}
	case SYMBOL_DECODEPNG: {		pCreate = Create_DecodePng(id, pInputItem);	break;	}
	case SYMBOL_DRAWBOUNDINGBOXES: {		pCreate = Create_DrawBoundingBoxes(id, pInputItem);	break;	}
	case SYMBOL_ENCODEJPEG: {		pCreate = Create_EncodeJpeg(id, pInputItem);	break;	}
	case SYMBOL_ENCODEPNG: {		pCreate = Create_EncodePng(id, pInputItem);	break;	}
	case SYMBOL_EXTRACTGLIMPSE: {		pCreate = Create_ExtractGlimpse(id, pInputItem);	break;	}
	case SYMBOL_HSVTORGB: {		pCreate = Create_HSVToRGB(id, pInputItem);	break;	}
	case SYMBOL_NONMAXSUPPRESSION: {		pCreate = Create_NonMaxSuppression(id, pInputItem);	break;	}
	case SYMBOL_RGBTOHSV: {		pCreate = Create_RGBToHSV(id, pInputItem);	break;	}
	case SYMBOL_RESIZEAREA: {		pCreate = Create_ResizeArea(id, pInputItem);	break;	}
	case SYMBOL_RESIZEBICUBIC: {		pCreate = Create_ResizeBicubic(id, pInputItem);	break;	}
	case SYMBOL_RESIZEBILINEAR: {		pCreate = Create_ResizeBilinear(id, pInputItem);	break;	}
	case SYMBOL_RESIZENEARESTNEIGHBOR: {		pCreate = Create_ResizeNearestNeighbor(id, pInputItem);	break;	}
	case SYMBOL_SAMPLEDISTORTEDBOUNDINGBOX: {		pCreate = Create_SampleDistortedBoundingBox(id, pInputItem);	break;	}
	case SYMBOL_FIXEDLENGTHRECORDREADER: {		pCreate = Create_FixedLengthRecordReader(id, pInputItem);	break;	}
	case SYMBOL_IDENTITYREADER: {		pCreate = Create_IdentityReader(id, pInputItem);	break;	}
	case SYMBOL_MATCHINGFILES: {		pCreate = Create_MatchingFiles(id, pInputItem);	break;	}
	case SYMBOL_MERGEV2CHECKPOINTS: {		pCreate = Create_MergeV2Checkpoints(id, pInputItem);	break;	}
	case SYMBOL_READFILE: {		pCreate = Create_ReadFile(id, pInputItem);	break;	}
	case SYMBOL_READERNUMRECORDSPRODUCED: {		pCreate = Create_ReaderNumRecordsProduced(id, pInputItem);	break;	}
	case SYMBOL_READERNUMWORKUNITSCOMPLETED: {		pCreate = Create_ReaderNumWorkUnitsCompleted(id, pInputItem);	break;	}
	case SYMBOL_READERREAD: {		pCreate = Create_ReaderRead(id, pInputItem);	break;	}
	case SYMBOL_READERREADUPTO: {		pCreate = Create_ReaderReadUpTo(id, pInputItem);	break;	}
	case SYMBOL_READERRESET: {		pCreate = Create_ReaderReset(id, pInputItem);	break;	}
	case SYMBOL_READERRESTORESTATE: {		pCreate = Create_ReaderRestoreState(id, pInputItem);	break;	}
	case SYMBOL_READERSERIALIZESTATE: {		pCreate = Create_ReaderSerializeState(id, pInputItem);	break;	}
	case SYMBOL_RESTORE: {		pCreate = Create_Restore(id, pInputItem);	break;	}
	case SYMBOL_RESTORESLICE: {		pCreate = Create_RestoreSlice(id, pInputItem);	break;	}
	case SYMBOL_RESTOREV2: {		pCreate = Create_RestoreV2(id, pInputItem);	break;	}
	case SYMBOL_SAVE: {		pCreate = Create_Save(id, pInputItem);	break;	}
	case SYMBOL_SAVESLICES: {		pCreate = Create_SaveSlices(id, pInputItem);	break;	}
	case SYMBOL_SAVEV2: {		pCreate = Create_SaveV2(id, pInputItem);	break;	}
	case SYMBOL_SHARDEDFILENAME: {		pCreate = Create_ShardedFilename(id, pInputItem);	break;	}
	case SYMBOL_SHARDEDFILESPEC: {		pCreate = Create_ShardedFilespec(id, pInputItem);	break;	}
	case SYMBOL_TFRECORDREADER: {		pCreate = Create_TFRecordReader(id, pInputItem);	break;	}
	case SYMBOL_TEXTLINEREADER: {		pCreate = Create_TextLineReader(id, pInputItem);	break;	}
	case SYMBOL_WHOLEFILEREADER: {		pCreate = Create_WholeFileReader(id, pInputItem);	break;	}
	case SYMBOL_WRITEFILE: {		pCreate = Create_WriteFile(id, pInputItem);	break;	}
	case SYMBOL_ASSERT: {		pCreate = Create_Assert(id, pInputItem);	break;	}
	case SYMBOL_HISTOGRAMSUMMARY: {		pCreate = Create_HistogramSummary(id, pInputItem);	break;	}
	case SYMBOL_MERGESUMMARY: {		pCreate = Create_MergeSummary(id, pInputItem);	break;	}
	case SYMBOL_PRINT: {		pCreate = Create_Print(id, pInputItem);	break;	}
	case SYMBOL_SCALARSUMMARY: {		pCreate = Create_ScalarSummary(id, pInputItem);	break;	}
	case SYMBOL_TENSORSUMMARY: {		pCreate = Create_TensorSummary(id, pInputItem);	break;	}
	case SYMBOL_ABS: {		pCreate = Create_Abs(id, pInputItem);	break;	}
	case SYMBOL_ACOS: {		pCreate = Create_Acos(id, pInputItem);	break;	}
	case SYMBOL_ADD: {		pCreate = Create_Add(id, pInputItem);	break;	}
	case SYMBOL_ADDN: {		pCreate = Create_AddN(id, pInputItem);	break;	}
	case SYMBOL_ALL: {		pCreate = Create_All(id, pInputItem);	break;	}
	case SYMBOL_ANY: {		pCreate = Create_Any(id, pInputItem);	break;	}
	case SYMBOL_APPROXIMATEEQUAL: {		pCreate = Create_ApproximateEqual(id, pInputItem);	break;	}
	case SYMBOL_ARGMAX: {		pCreate = Create_ArgMax(id, pInputItem);	break;	}
	case SYMBOL_ARGMIN: {		pCreate = Create_ArgMin(id, pInputItem);	break;	}
	case SYMBOL_ASIN: {		pCreate = Create_Asin(id, pInputItem);	break;	}
	case SYMBOL_ATAN: {		pCreate = Create_Atan(id, pInputItem);	break;	}
	case SYMBOL_ATAN2: {		pCreate = Create_Atan2(id, pInputItem);	break;	}
	case SYMBOL_BATCHMATMUL: {		pCreate = Create_BatchMatMul(id, pInputItem);	break;	}
	case SYMBOL_BETAINC: {		pCreate = Create_Betainc(id, pInputItem);	break;	}
	case SYMBOL_BINCOUNT: {		pCreate = Create_Bincount(id, pInputItem);	break;	}
	case SYMBOL_BUCKETIZE: {		pCreate = Create_Bucketize(id, pInputItem);	break;	}
	case SYMBOL_CAST: {		pCreate = Create_Cast(id, pInputItem);	break;	}
	case SYMBOL_CEIL: {		pCreate = Create_Ceil(id, pInputItem);	break;	}
	case SYMBOL_COMPLEX: {		pCreate = Create_Complex(id, pInputItem);	break;	}
	case SYMBOL_COMPLEXABS: {		pCreate = Create_ComplexAbs(id, pInputItem);	break;	}
	case SYMBOL_CONJ: {		pCreate = Create_Conj(id, pInputItem);	break;	}
	case SYMBOL_COS: {		pCreate = Create_Cos(id, pInputItem);	break;	}
	case SYMBOL_CROSS: {		pCreate = Create_Cross(id, pInputItem);	break;	}
	case SYMBOL_CUMPROD: {		pCreate = Create_Cumprod(id, pInputItem);	break;	}
	case SYMBOL_CUMSUM: {		pCreate = Create_Cumsum(id, pInputItem);	break;	}
	case SYMBOL_DIGAMMA: {		pCreate = Create_Digamma(id, pInputItem);	break;	}
	case SYMBOL_DIV: {		pCreate = Create_Div(id, pInputItem);	break;	}
	case SYMBOL_EQUAL: {		pCreate = Create_Equal(id, pInputItem);	break;	}
	case SYMBOL_ERF: {		pCreate = Create_Erf(id, pInputItem);	break;	}
	case SYMBOL_ERFC: {		pCreate = Create_Erfc(id, pInputItem);	break;	}
	case SYMBOL_EXP: {		pCreate = Create_Exp(id, pInputItem);	break;	}
	case SYMBOL_EXPM1: {		pCreate = Create_Expm1(id, pInputItem);	break;	}
	case SYMBOL_FLOOR: {		pCreate = Create_Floor(id, pInputItem);	break;	}
	case SYMBOL_FLOORDIV: {		pCreate = Create_FloorDiv(id, pInputItem);	break;	}
	case SYMBOL_FLOORMOD: {		pCreate = Create_FloorMod(id, pInputItem);	break;	}
	case SYMBOL_GREATER: {		pCreate = Create_Greater(id, pInputItem);	break;	}
	case SYMBOL_GREATEREQUAL: {		pCreate = Create_GreaterEqual(id, pInputItem);	break;	}
	case SYMBOL_IGAMMA: {		pCreate = Create_Igamma(id, pInputItem);	break;	}
	case SYMBOL_IGAMMAC: {		pCreate = Create_Igammac(id, pInputItem);	break;	}
	case SYMBOL_IMAG: {		pCreate = Create_Imag(id, pInputItem);	break;	}
	case SYMBOL_ISINF: {		pCreate = Create_IsInf(id, pInputItem);	break;	}
	case SYMBOL_ISNAN: {		pCreate = Create_IsNan(id, pInputItem);	break;	}
	case SYMBOL_LESS: {		pCreate = Create_Less(id, pInputItem);	break;	}
	case SYMBOL_LESSEQUAL: {		pCreate = Create_LessEqual(id, pInputItem);	break;	}
	case SYMBOL_LGAMMA: {		pCreate = Create_Lgamma(id, pInputItem);	break;	}
	case SYMBOL_LINSPACE: {		pCreate = Create_LinSpace(id, pInputItem);	break;	}
	case SYMBOL_LOG: {		pCreate = Create_Log(id, pInputItem);	break;	}
	case SYMBOL_LOG1P: {		pCreate = Create_Log1p(id, pInputItem);	break;	}
	case SYMBOL_LOGICALAND: {		pCreate = Create_LogicalAnd(id, pInputItem);	break;	}
	case SYMBOL_LOGICALNOT: {		pCreate = Create_LogicalNot(id, pInputItem);	break;	}
	case SYMBOL_LOGICALOR: {		pCreate = Create_LogicalOr(id, pInputItem);	break;	}
	case SYMBOL_MATMUL: {		pCreate = Create_MatMul(id, pInputItem);	break;	}
	case SYMBOL_MAX: {		pCreate = Create_Max(id, pInputItem);	break;	}
	case SYMBOL_MAXIMUM: {		pCreate = Create_Maximum(id, pInputItem);	break;	}
	case SYMBOL_MEAN: {		pCreate = Create_Mean(id, pInputItem);	break;	}
	case SYMBOL_MIN: {		pCreate = Create_Min(id, pInputItem);	break;	}
	case SYMBOL_MINIMUM: {		pCreate = Create_Minimum(id, pInputItem);	break;	}
	case SYMBOL_MOD: {		pCreate = Create_Mod(id, pInputItem);	break;	}
	case SYMBOL_MULTIPLY: {		pCreate = Create_Multiply(id, pInputItem);	break;	}
	case SYMBOL_NEGATE: {		pCreate = Create_Negate(id, pInputItem);	break;	}
	case SYMBOL_NOTEQUAL: {		pCreate = Create_NotEqual(id, pInputItem);	break;	}
	case SYMBOL_POLYGAMMA: {		pCreate = Create_Polygamma(id, pInputItem);	break;	}
	case SYMBOL_POW: {		pCreate = Create_Pow(id, pInputItem);	break;	}
	case SYMBOL_PROD: {		pCreate = Create_Prod(id, pInputItem);	break;	}
	case SYMBOL_QUANTIZEDOWNANDSHRINKRANGE: {		pCreate = Create_QuantizeDownAndShrinkRange(id, pInputItem);	break;	}
	case SYMBOL_QUANTIZEDMATMUL: {		pCreate = Create_QuantizedMatMul(id, pInputItem);	break;	}
	case SYMBOL_QUANTIZEDMUL: {		pCreate = Create_QuantizedMul(id, pInputItem);	break;	}
	case SYMBOL_RANGE: {		pCreate = Create_Range(id, pInputItem);	break;	}
	case SYMBOL_REAL: {		pCreate = Create_Real(id, pInputItem);	break;	}
	case SYMBOL_REALDIV: {		pCreate = Create_RealDiv(id, pInputItem);	break;	}
	case SYMBOL_RECIPROCAL: {		pCreate = Create_Reciprocal(id, pInputItem);	break;	}
	case SYMBOL_REQUANTIZATIONRANGE: {		pCreate = Create_RequantizationRange(id, pInputItem);	break;	}
	case SYMBOL_REQUANTIZE: {		pCreate = Create_Requantize(id, pInputItem);	break;	}
	case SYMBOL_RINT: {		pCreate = Create_Rint(id, pInputItem);	break;	}
	case SYMBOL_ROUND: {		pCreate = Create_Round(id, pInputItem);	break;	}
	case SYMBOL_RSQRT: {		pCreate = Create_Rsqrt(id, pInputItem);	break;	}
	case SYMBOL_SEGMENTMAX: {		pCreate = Create_SegmentMax(id, pInputItem);	break;	}
	case SYMBOL_SEGMENTMEAN: {		pCreate = Create_SegmentMean(id, pInputItem);	break;	}
	case SYMBOL_SEGMENTMIN: {		pCreate = Create_SegmentMin(id, pInputItem);	break;	}
	case SYMBOL_SEGMENTPROD: {		pCreate = Create_SegmentProd(id, pInputItem);	break;	}
	case SYMBOL_SEGMENTSUM: {		pCreate = Create_SegmentSum(id, pInputItem);	break;	}
	case SYMBOL_SIGMOID: {		pCreate = Create_Sigmoid(id, pInputItem);	break;	}
	case SYMBOL_SIGN: {		pCreate = Create_Sign(id, pInputItem);	break;	}
	case SYMBOL_SIN: {		pCreate = Create_Sin(id, pInputItem);	break;	}
	case SYMBOL_SPARSEMATMUL: {		pCreate = Create_SparseMatMul(id, pInputItem);	break;	}
	case SYMBOL_SPARSESEGMENTMEAN: {		pCreate = Create_SparseSegmentMean(id, pInputItem);	break;	}
	case SYMBOL_SPARSESEGMENTMEANGRAD: {		pCreate = Create_SparseSegmentMeanGrad(id, pInputItem);	break;	}
	case SYMBOL_SPARSESEGMENTSQRTN: {		pCreate = Create_SparseSegmentSqrtN(id, pInputItem);	break;	}
	case SYMBOL_SPARSESEGMENTSQRTNGRAD: {		pCreate = Create_SparseSegmentSqrtNGrad(id, pInputItem);	break;	}
	case SYMBOL_SPARSESEGMENTSUM: {		pCreate = Create_SparseSegmentSum(id, pInputItem);	break;	}
	case SYMBOL_SQRT: {		pCreate = Create_Sqrt(id, pInputItem);	break;	}
	case SYMBOL_SQUARE: {		pCreate = Create_Square(id, pInputItem);	break;	}
	case SYMBOL_SQUAREDDIFFERENCE: {		pCreate = Create_SquaredDifference(id, pInputItem);	break;	}
	case SYMBOL_SUBTRACT: {		pCreate = Create_Subtract(id, pInputItem);	break;	}
	case SYMBOL_SUM: {		pCreate = Create_Sum(id, pInputItem);	break;	}
	case SYMBOL_TAN: {		pCreate = Create_Tan(id, pInputItem);	break;	}
	case SYMBOL_TANH: {		pCreate = Create_Tanh(id, pInputItem);	break;	}
	case SYMBOL_TRUNCATEDIV: {		pCreate = Create_TruncateDiv(id, pInputItem);	break;	}
	case SYMBOL_TRUNCATEMOD: {		pCreate = Create_TruncateMod(id, pInputItem);	break;	}
	case SYMBOL_UNSORTEDSEGMENTMAX: {		pCreate = Create_UnsortedSegmentMax(id, pInputItem);	break;	}
	case SYMBOL_UNSORTEDSEGMENTSUM: {		pCreate = Create_UnsortedSegmentSum(id, pInputItem);	break;	}
	case SYMBOL_WHERE3: {		pCreate = Create_Where3(id, pInputItem);	break;	}
	case SYMBOL_ZETA: {		pCreate = Create_Zeta(id, pInputItem);	break;	}
	case SYMBOL_AVGPOOL: {		pCreate = Create_AvgPool(id, pInputItem);	break;	}
	case SYMBOL_AVGPOOL3D: {		pCreate = Create_AvgPool3D(id, pInputItem);	break;	}
	case SYMBOL_AVGPOOL3DGRAD: {		pCreate = Create_AvgPool3DGrad(id, pInputItem);	break;	}
	case SYMBOL_BIASADD: {		pCreate = Create_BiasAdd(id, pInputItem);	break;	}
	case SYMBOL_BIASADDGRAD: {		pCreate = Create_BiasAddGrad(id, pInputItem);	break;	}
	case SYMBOL_CONV2D: {		pCreate = Create_Conv2D(id, pInputItem);	break;	}
	case SYMBOL_CONV2DBACKPROPFILTER: {		pCreate = Create_Conv2DBackpropFilter(id, pInputItem);	break;	}
	case SYMBOL_CONV2DBACKPROPINPUT: {		pCreate = Create_Conv2DBackpropInput(id, pInputItem);	break;	}
	case SYMBOL_CONV3D: {		pCreate = Create_Conv3D(id, pInputItem);	break;	}
	case SYMBOL_CONV3DBACKPROPFILTERV2: {		pCreate = Create_Conv3DBackpropFilterV2(id, pInputItem);	break;	}
	case SYMBOL_CONV3DBACKPROPINPUTV2: {		pCreate = Create_Conv3DBackpropInputV2(id, pInputItem);	break;	}
	case SYMBOL_DEPTHWISECONV2DNATIVE: {		pCreate = Create_DepthwiseConv2dNative(id, pInputItem);	break;	}
	case SYMBOL_DEPTHWISECONV2DNATIVEBACKPROPFILTER: {		pCreate = Create_DepthwiseConv2dNativeBackpropFilter(id, pInputItem);	break;	}
	case SYMBOL_DEPTHWISECONV2DNATIVEBACKPROPINPUT: {		pCreate = Create_DepthwiseConv2dNativeBackpropInput(id, pInputItem);	break;	}
	case SYMBOL_DILATION2D: {		pCreate = Create_Dilation2D(id, pInputItem);	break;	}
	case SYMBOL_DILATION2DBACKPROPFILTER: {		pCreate = Create_Dilation2DBackpropFilter(id, pInputItem);	break;	}
	case SYMBOL_DILATION2DBACKPROPINPUT: {		pCreate = Create_Dilation2DBackpropInput(id, pInputItem);	break;	}
	case SYMBOL_ELU: {		pCreate = Create_Elu(id, pInputItem);	break;	}
	case SYMBOL_FRACTIONALAVGPOOL: {		pCreate = Create_FractionalAvgPool(id, pInputItem);	break;	}
	case SYMBOL_FRACTIONALMAXPOOL: {		pCreate = Create_FractionalMaxPool(id, pInputItem);	break;	}
	case SYMBOL_FUSEDBATCHNORM: {		pCreate = Create_FusedBatchNorm(id, pInputItem);	break;	}
	case SYMBOL_FUSEDBATCHNORMGRAD: {		pCreate = Create_FusedBatchNormGrad(id, pInputItem);	break;	}
	case SYMBOL_FUSEDPADCONV2D: {		pCreate = Create_FusedPadConv2D(id, pInputItem);	break;	}
	case SYMBOL_FUSEDRESIZEANDPADCONV2D: {		pCreate = Create_FusedResizeAndPadConv2D(id, pInputItem);	break;	}
	case SYMBOL_INTOPK: {		pCreate = Create_InTopK(id, pInputItem);	break;	}
	case SYMBOL_L2LOSS: {		pCreate = Create_L2Loss(id, pInputItem);	break;	}
	case SYMBOL_LRN: {		pCreate = Create_LRN(id, pInputItem);	break;	}
	case SYMBOL_LOGSOFTMAX: {		pCreate = Create_LogSoftmax(id, pInputItem);	break;	}
	case SYMBOL_MAXPOOL: {		pCreate = Create_MaxPool(id, pInputItem);	break;	}
	case SYMBOL_MAXPOOL3D: {		pCreate = Create_MaxPool3D(id, pInputItem);	break;	}
	case SYMBOL_MAXPOOL3DGRAD: {		pCreate = Create_MaxPool3DGrad(id, pInputItem);	break;	}
	case SYMBOL_MAXPOOL3DGRADGRAD: {		pCreate = Create_MaxPool3DGradGrad(id, pInputItem);	break;	}
	case SYMBOL_MAXPOOLGRADGRAD: {		pCreate = Create_MaxPoolGradGrad(id, pInputItem);	break;	}
	case SYMBOL_MAXPOOLGRADGRADWITHARGMAX: {		pCreate = Create_MaxPoolGradGradWithArgmax(id, pInputItem);	break;	}
	case SYMBOL_MAXPOOLWITHARGMAX: {		pCreate = Create_MaxPoolWithArgmax(id, pInputItem);	break;	}
	case SYMBOL_QUANTIZEDAVGPOOL: {		pCreate = Create_QuantizedAvgPool(id, pInputItem);	break;	}
	case SYMBOL_QUANTIZEDBATCHNORMWITHGLOBALNORMALIZATION: {		pCreate = Create_QuantizedBatchNormWithGlobalNormalization(id, pInputItem);	break;	}
	case SYMBOL_QUANTIZEDBIASADD: {		pCreate = Create_QuantizedBiasAdd(id, pInputItem);	break;	}
	case SYMBOL_QUANTIZEDCONV2D: {		pCreate = Create_QuantizedConv2D(id, pInputItem);	break;	}
	case SYMBOL_QUANTIZEDMAXPOOL: {		pCreate = Create_QuantizedMaxPool(id, pInputItem);	break;	}
	case SYMBOL_QUANTIZEDRELU: {		pCreate = Create_QuantizedRelu(id, pInputItem);	break;	}
	case SYMBOL_QUANTIZEDRELU6: {		pCreate = Create_QuantizedRelu6(id, pInputItem);	break;	}
	case SYMBOL_QUANTIZEDRELUX: {		pCreate = Create_QuantizedReluX(id, pInputItem);	break;	}
	case SYMBOL_RELU: {		pCreate = Create_Relu(id, pInputItem);	break;	}
	case SYMBOL_RELU6: {		pCreate = Create_Relu6(id, pInputItem);	break;	}
	case SYMBOL_SOFTMAX: {		pCreate = Create_Softmax(id, pInputItem);	break;	}
	case SYMBOL_SOFTMAXCROSSENTROPYWITHLOGITS: {		pCreate = Create_SoftmaxCrossEntropyWithLogits(id, pInputItem);	break;	}
	case SYMBOL_SOFTPLUS: {		pCreate = Create_Softplus(id, pInputItem);	break;	}
	case SYMBOL_SOFTSIGN: {		pCreate = Create_Softsign(id, pInputItem);	break;	}
	case SYMBOL_SPARSESOFTMAXCROSSENTROPYWITHLOGITS: {		pCreate = Create_SparseSoftmaxCrossEntropyWithLogits(id, pInputItem);	break;	}
	case SYMBOL_TOPK: {		pCreate = Create_TopK(id, pInputItem);	break;	}
	case SYMBOL_NOOP: {		pCreate = Create_NoOp(id, pInputItem);	break;	}
	case SYMBOL_DECODECSV: {		pCreate = Create_DecodeCSV(id, pInputItem);	break;	}
	case SYMBOL_DECODEJSONEXAMPLE: {		pCreate = Create_DecodeJSONExample(id, pInputItem);	break;	}
	case SYMBOL_DECODERAW: {		pCreate = Create_DecodeRaw(id, pInputItem);	break;	}
	case SYMBOL_PARSEEXAMPLE: {		pCreate = Create_ParseExample(id, pInputItem);	break;	}
	case SYMBOL_PARSESINGLESEQUENCEEXAMPLE: {		pCreate = Create_ParseSingleSequenceExample(id, pInputItem);	break;	}
	case SYMBOL_PARSETENSOR: {		pCreate = Create_ParseTensor(id, pInputItem);	break;	}
	case SYMBOL_STRINGTONUMBER: {		pCreate = Create_StringToNumber(id, pInputItem);	break;	}
	case SYMBOL_MULTINOMIAL: {		pCreate = Create_Multinomial(id, pInputItem);	break;	}
	case SYMBOL_PARAMETERIZEDTRUNCATEDNORMAL: {		pCreate = Create_ParameterizedTruncatedNormal(id, pInputItem);	break;	}
	case SYMBOL_RANDOMGAMMA: {		pCreate = Create_RandomGamma(id, pInputItem);	break;	}
	case SYMBOL_RANDOMNORMAL: {		pCreate = Create_RandomNormal(id, pInputItem);	break;	}
	case SYMBOL_RANDOMPOISSON: {		pCreate = Create_RandomPoisson(id, pInputItem);	break;	}
	case SYMBOL_RANDOMSHUFFLE: {		pCreate = Create_RandomShuffle(id, pInputItem);	break;	}
	case SYMBOL_RANDOMUNIFORM: {		pCreate = Create_RandomUniform(id, pInputItem);	break;	}
	case SYMBOL_RANDOMUNIFORMINT: {		pCreate = Create_RandomUniformInt(id, pInputItem);	break;	}
	case SYMBOL_TRUNCATEDNORMAL: {		pCreate = Create_TruncatedNormal(id, pInputItem);	break;	}
	case SYMBOL_ADDMANYSPARSETOTENSORSMAP: {		pCreate = Create_AddManySparseToTensorsMap(id, pInputItem);	break;	}
	case SYMBOL_ADDSPARSETOTENSORSMAP: {		pCreate = Create_AddSparseToTensorsMap(id, pInputItem);	break;	}
	case SYMBOL_DESERIALIZEMANYSPARSE: {		pCreate = Create_DeserializeManySparse(id, pInputItem);	break;	}
	case SYMBOL_SERIALIZEMANYSPARSE: {		pCreate = Create_SerializeManySparse(id, pInputItem);	break;	}
	case SYMBOL_SERIALIZESPARSE: {		pCreate = Create_SerializeSparse(id, pInputItem);	break;	}
	case SYMBOL_SPARSEADD: {		pCreate = Create_SparseAdd(id, pInputItem);	break;	}
	case SYMBOL_SPARSEADDGRAD: {		pCreate = Create_SparseAddGrad(id, pInputItem);	break;	}
	case SYMBOL_SPARSECONCAT: {		pCreate = Create_SparseConcat(id, pInputItem);	break;	}
	case SYMBOL_SPARSECROSS: {		pCreate = Create_SparseCross(id, pInputItem);	break;	}
	case SYMBOL_SPARSEDENSECWISEADD: {		pCreate = Create_SparseDenseCwiseAdd(id, pInputItem);	break;	}
	case SYMBOL_SPARSEDENSECWISEDIV: {		pCreate = Create_SparseDenseCwiseDiv(id, pInputItem);	break;	}
	case SYMBOL_SPARSEDENSECWISEMUL: {		pCreate = Create_SparseDenseCwiseMul(id, pInputItem);	break;	}
	case SYMBOL_SPARSEREDUCESUM: {		pCreate = Create_SparseReduceSum(id, pInputItem);	break;	}
	case SYMBOL_SPARSEREDUCESUMSPARSE: {		pCreate = Create_SparseReduceSumSparse(id, pInputItem);	break;	}
	case SYMBOL_SPARSEREORDER: {		pCreate = Create_SparseReorder(id, pInputItem);	break;	}
	case SYMBOL_SPARSERESHAPE: {		pCreate = Create_SparseReshape(id, pInputItem);	break;	}
	case SYMBOL_SPARSESOFTMAX: {		pCreate = Create_SparseSoftmax(id, pInputItem);	break;	}
	case SYMBOL_SPARSESPARSEMAXIMUM: {		pCreate = Create_SparseSparseMaximum(id, pInputItem);	break;	}
	case SYMBOL_SPARSESPARSEMINIMUM: {		pCreate = Create_SparseSparseMinimum(id, pInputItem);	break;	}
	case SYMBOL_SPARSESPLIT: {		pCreate = Create_SparseSplit(id, pInputItem);	break;	}
	case SYMBOL_SPARSETENSORDENSEADD: {		pCreate = Create_SparseTensorDenseAdd(id, pInputItem);	break;	}
	case SYMBOL_SPARSETENSORDENSEMATMUL: {		pCreate = Create_SparseTensorDenseMatMul(id, pInputItem);	break;	}
	case SYMBOL_SPARSETODENSE: {		pCreate = Create_SparseToDense(id, pInputItem);	break;	}
	case SYMBOL_TAKEMANYSPARSEFROMTENSORSMAP: {		pCreate = Create_TakeManySparseFromTensorsMap(id, pInputItem);	break;	}
	case SYMBOL_ASSIGN: {		pCreate = Create_Assign(id, pInputItem);	break;	}
	case SYMBOL_ASSIGNADD: {		pCreate = Create_AssignAdd(id, pInputItem);	break;	}
	case SYMBOL_ASSIGNSUB: {		pCreate = Create_AssignSub(id, pInputItem);	break;	}
	case SYMBOL_COUNTUPTO: {		pCreate = Create_CountUpTo(id, pInputItem);	break;	}
	case SYMBOL_DESTROYTEMPORARYVARIABLE: {		pCreate = Create_DestroyTemporaryVariable(id, pInputItem);	break;	}
	case SYMBOL_ISVARIABLEINITIALIZED: {		pCreate = Create_IsVariableInitialized(id, pInputItem);	break;	}
	case SYMBOL_SCATTERADD: {		pCreate = Create_ScatterAdd(id, pInputItem);	break;	}
	case SYMBOL_SCATTERDIV: {		pCreate = Create_ScatterDiv(id, pInputItem);	break;	}
	case SYMBOL_SCATTERMUL: {		pCreate = Create_ScatterMul(id, pInputItem);	break;	}
	case SYMBOL_SCATTERNDADD: {		pCreate = Create_ScatterNdAdd(id, pInputItem);	break;	}
	case SYMBOL_SCATTERNDSUB: {		pCreate = Create_ScatterNdSub(id, pInputItem);	break;	}
	case SYMBOL_SCATTERNDUPDATE: {		pCreate = Create_ScatterNdUpdate(id, pInputItem);	break;	}
	case SYMBOL_SCATTERSUB: {		pCreate = Create_ScatterSub(id, pInputItem);	break;	}
	case SYMBOL_SCATTERUPDATE: {		pCreate = Create_ScatterUpdate(id, pInputItem);	break;	}
	case SYMBOL_TEMPORARYVARIABLE: {		pCreate = Create_TemporaryVariable(id, pInputItem);	break;	}
	case SYMBOL_VARIABLE: {		pCreate = Create_Variable(id, pInputItem);	break;	}
	case SYMBOL_ASSTRING: {		pCreate = Create_AsString(id, pInputItem);	break;	}
	case SYMBOL_DECODEBASE64: {		pCreate = Create_DecodeBase64(id, pInputItem);	break;	}
	case SYMBOL_ENCODEBASE64: {		pCreate = Create_EncodeBase64(id, pInputItem);	break;	}
	case SYMBOL_REDUCEJOIN: {		pCreate = Create_ReduceJoin(id, pInputItem);	break;	}
	case SYMBOL_STRINGJOIN: {		pCreate = Create_StringJoin(id, pInputItem);	break;	}
	case SYMBOL_STRINGSPLIT: {		pCreate = Create_StringSplit(id, pInputItem);	break;	}
	case SYMBOL_STRINGTOHASHBUCKET: {		pCreate = Create_StringToHashBucket(id, pInputItem);	break;	}
	case SYMBOL_STRINGTOHASHBUCKETFAST: {		pCreate = Create_StringToHashBucketFast(id, pInputItem);	break;	}
	case SYMBOL_STRINGTOHASHBUCKETSTRONG: {		pCreate = Create_StringToHashBucketStrong(id, pInputItem);	break;	}
	case SYMBOL_SUBSTR: {		pCreate = Create_Substr(id, pInputItem);	break;	}
	case SYMBOL_APPLYADADELTA: {		pCreate = Create_ApplyAdadelta(id, pInputItem);	break;	}
	case SYMBOL_APPLYADAGRAD: {		pCreate = Create_ApplyAdagrad(id, pInputItem);	break;	}
	case SYMBOL_APPLYADAGRADDA: {		pCreate = Create_ApplyAdagradDA(id, pInputItem);	break;	}
	case SYMBOL_APPLYADAM: {		pCreate = Create_ApplyAdam(id, pInputItem);	break;	}
	case SYMBOL_APPLYCENTEREDRMSPROP: {		pCreate = Create_ApplyCenteredRMSProp(id, pInputItem);	break;	}
	case SYMBOL_APPLYFTRL: {		pCreate = Create_ApplyFtrl(id, pInputItem);	break;	}
	case SYMBOL_APPLYGRADIENTDESCENT: {		pCreate = Create_ApplyGradientDescent(id, pInputItem);	break;	}
	case SYMBOL_APPLYMOMENTUM: {		pCreate = Create_ApplyMomentum(id, pInputItem);	break;	}
	case SYMBOL_APPLYPROXIMALADAGRAD: {		pCreate = Create_ApplyProximalAdagrad(id, pInputItem);	break;	}
	case SYMBOL_APPLYPROXIMALGRADIENTDESCENT: {		pCreate = Create_ApplyProximalGradientDescent(id, pInputItem);	break;	}
	case SYMBOL_APPLYRMSPROP: {		pCreate = Create_ApplyRMSProp(id, pInputItem);	break;	}
	case SYMBOL_RESOURCEAPPLYADADELTA: {		pCreate = Create_ResourceApplyAdadelta(id, pInputItem);	break;	}
	case SYMBOL_RESOURCEAPPLYADAGRAD: {		pCreate = Create_ResourceApplyAdagrad(id, pInputItem);	break;	}
	case SYMBOL_RESOURCEAPPLYADAGRADDA: {		pCreate = Create_ResourceApplyAdagradDA(id, pInputItem);	break;	}
	case SYMBOL_RESOURCEAPPLYADAM: {		pCreate = Create_ResourceApplyAdam(id, pInputItem);	break;	}
	case SYMBOL_RESOURCEAPPLYCENTEREDRMSPROP: {		pCreate = Create_ResourceApplyCenteredRMSProp(id, pInputItem);	break;	}
	case SYMBOL_RESOURCEAPPLYFTRL: {		pCreate = Create_ResourceApplyFtrl(id, pInputItem);	break;	}
	case SYMBOL_RESOURCEAPPLYGRADIENTDESCENT: {		pCreate = Create_ResourceApplyGradientDescent(id, pInputItem);	break;	}
	case SYMBOL_RESOURCEAPPLYMOMENTUM: {		pCreate = Create_ResourceApplyMomentum(id, pInputItem);	break;	}
	case SYMBOL_RESOURCEAPPLYPROXIMALADAGRAD: {		pCreate = Create_ResourceApplyProximalAdagrad(id, pInputItem);	break;	}
	case SYMBOL_RESOURCEAPPLYPROXIMALGRADIENTDESCENT: {		pCreate = Create_ResourceApplyProximalGradientDescent(id, pInputItem);	break;	}
	case SYMBOL_RESOURCEAPPLYRMSPROP: {		pCreate = Create_ResourceApplyRMSProp(id, pInputItem);	break;	}
	case SYMBOL_RESOURCESPARSEAPPLYADADELTA: {		pCreate = Create_ResourceSparseApplyAdadelta(id, pInputItem);	break;	}
	case SYMBOL_RESOURCESPARSEAPPLYADAGRAD: {		pCreate = Create_ResourceSparseApplyAdagrad(id, pInputItem);	break;	}
	case SYMBOL_RESOURCESPARSEAPPLYADAGRADDA: {		pCreate = Create_ResourceSparseApplyAdagradDA(id, pInputItem);	break;	}
	case SYMBOL_RESOURCESPARSEAPPLYCENTEREDRMSPROP: {		pCreate = Create_ResourceSparseApplyCenteredRMSProp(id, pInputItem);	break;	}
	case SYMBOL_RESOURCESPARSEAPPLYFTRL: {		pCreate = Create_ResourceSparseApplyFtrl(id, pInputItem);	break;	}
	case SYMBOL_RESOURCESPARSEAPPLYMOMENTUM: {		pCreate = Create_ResourceSparseApplyMomentum(id, pInputItem);	break;	}
	case SYMBOL_RESOURCESPARSEAPPLYPROXIMALADAGRAD: {		pCreate = Create_ResourceSparseApplyProximalAdagrad(id, pInputItem);	break;	}
	case SYMBOL_RESOURCESPARSEAPPLYPROXIMALGRADIENTDESCENT: {		pCreate = Create_ResourceSparseApplyProximalGradientDescent(id, pInputItem);	break;	}
	case SYMBOL_RESOURCESPARSEAPPLYRMSPROP: {		pCreate = Create_ResourceSparseApplyRMSProp(id, pInputItem);	break;	}
	case SYMBOL_SPARSEAPPLYADADELTA: {		pCreate = Create_SparseApplyAdadelta(id, pInputItem);	break;	}
	case SYMBOL_SPARSEAPPLYADAGRAD: {		pCreate = Create_SparseApplyAdagrad(id, pInputItem);	break;	}
	case SYMBOL_SPARSEAPPLYADAGRADDA: {		pCreate = Create_SparseApplyAdagradDA(id, pInputItem);	break;	}
	case SYMBOL_SPARSEAPPLYCENTEREDRMSPROP: {		pCreate = Create_SparseApplyCenteredRMSProp(id, pInputItem);	break;	}
	case SYMBOL_SPARSEAPPLYFTRL: {		pCreate = Create_SparseApplyFtrl(id, pInputItem);	break;	}
	case SYMBOL_SPARSEAPPLYMOMENTUM: {		pCreate = Create_SparseApplyMomentum(id, pInputItem);	break;	}
	case SYMBOL_SPARSEAPPLYPROXIMALADAGRAD: {		pCreate = Create_SparseApplyProximalAdagrad(id, pInputItem);	break;	}
	case SYMBOL_SPARSEAPPLYPROXIMALGRADIENTDESCENT: {		pCreate = Create_SparseApplyProximalGradientDescent(id, pInputItem);	break;	}
	case SYMBOL_SPARSEAPPLYRMSPROP: {		pCreate = Create_SparseApplyRMSProp(id, pInputItem);	break;	}
	case SYMBOL_CONST: {		pCreate = Create_Const(id, pInputItem);	break;	}
	case SYMBOL_INPUT_EX: {		pCreate = Create_Input_ex(id, pInputItem);	break;	}
	case SYMBOL_RANDOMNORMAL_EX: {		pCreate = Create_RandomNormal_ex(id, pInputItem);	break;	}
	}
	return pCreate;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// dynamic map interface functions
ObjectInfo* LookupFromObjectMap(std::string strid)
{
	ObjectInfo* pObj = nullptr;
	const std::map<std::string, ObjectInfo*>::const_iterator aLookup = m_ObjectMapList.find(strid);
	const bool bExists = aLookup != m_ObjectMapList.end();
	if (bExists)
	{
		pObj = aLookup->second;
		return pObj;
	}
	else
	{
		std::string msg = string_format("warning : Could not find object(%s) in object map.", strid.c_str());
		PrintMessage(msg);
	}
	return pObj;
}

ObjectInfo* AddObjectMap(void* pCreate, std::string id, int iSymbol, std::string type_name, Json::Value pInputItem)
{
	if (pCreate)
	{
		ObjectInfo* pObj = nullptr;
		const std::map<std::string, ObjectInfo*>::const_iterator aLookup = m_ObjectMapList.find(id);
		const bool bExists = aLookup != m_ObjectMapList.end();
		if (bExists == false)
		{
			ObjectInfo* pCreateObj = new ObjectInfo;
			pCreateObj->id = id;
			pCreateObj->pObject = pCreate;
			pCreateObj->type = iSymbol;
			pCreateObj->type_name = type_name;
			pCreateObj->param = pInputItem;

			m_ObjectMapList.insert(std::pair<std::string, ObjectInfo*>(id, pCreateObj));
			return pCreateObj;
		}
	}
	std::string msg = string_format("warning : Could not add to list(all object). existed id(%s).", id.c_str());
	PrintMessage(msg);
	return nullptr;
}

FetchInfo* AddRunObjectMap(ObjectInfo* pRunObj)
{
	if (pRunObj)
	{
		const std::map<std::string, FetchInfo*>::const_iterator aLookup = m_RunMapList.find(pRunObj->id);
		const bool bExists = aLookup != m_RunMapList.end();
		if (bExists == false)
		{
			FetchInfo* pCreateObj = new FetchInfo;
			pCreateObj->pSession = pRunObj;
			m_RunMapList.insert(std::pair<std::string, FetchInfo*>(pRunObj->id, pCreateObj));
			return pCreateObj;
		}
	}
	std::string msg = string_format("warning : Could not add to list(run list). existed id(%s).", pRunObj->id.c_str());
	PrintMessage(msg);
	return nullptr;
}
 
void ObjectMapClear()
{
	std::map<std::string, ObjectInfo*>::iterator vit;
	for (vit = m_ObjectMapList.begin(); vit != m_ObjectMapList.end(); ++vit)
	{
		ObjectInfo* pTar = vit->second;
		if (pTar)
		{
			std::map<std::string, OutputInfo*>::iterator vitOutput;
			for (vitOutput = pTar->pMapOutputs.begin(); vitOutput != pTar->pMapOutputs.end(); ++vitOutput)
			{
				OutputInfo* pTarObject = vitOutput->second;
				delete pTarObject;
			}

			if (pTar->pObject)
			{
				delete pTar->pObject;
			}
			pTar->pObject = nullptr;
		}
		delete pTar;
		pTar = nullptr;
	}
	m_ObjectMapList.clear();

	// RUN object 리스트 제거.
	std::map<std::string, FetchInfo*>::iterator vitRun;
	for (vitRun = m_RunMapList.begin(); vitRun != m_RunMapList.end(); ++vitRun)
	{
		FetchInfo* pTar = vitRun->second;
		delete pTar;
		pTar = nullptr;
	}
	m_RunMapList.clear();
}

bool AddOutputInfo(ObjectInfo* pObjectInfo, void* pOutput, int iType, std::string strname)
{
	if (pObjectInfo)
	{
		const std::map<std::string, OutputInfo*>::const_iterator aLookup = pObjectInfo->pMapOutputs.find(strname);
		const bool bExists = aLookup != pObjectInfo->pMapOutputs.end();
		if (bExists)
		{
			std::string msg = string_format("warning : output object existed(%s).", strname.c_str());
			PrintMessage(msg);
			return false;
		}
		else
		{
			OutputInfo* pOutputInfo = new OutputInfo;
			pOutputInfo->type = iType;
			pOutputInfo->pOutput = pOutput;
			pObjectInfo->pMapOutputs.insert(std::pair<std::string, OutputInfo*>(strname, pOutputInfo));
			return true;
		}
	}
	std::string msg = string_format("error : Could not add to output list. null object info(%s).", strname.c_str());
	PrintMessage(msg);
	return false;
}

OutputInfo* LookupFromOutputMap(ObjectInfo* pObjectInfo, std::string strname)
{
	if (pObjectInfo)
	{
		OutputInfo* pOutputObj = nullptr;
		const std::map<std::string, OutputInfo*>::const_iterator aLookup = pObjectInfo->pMapOutputs.find(strname);
		const bool bExists = aLookup != pObjectInfo->pMapOutputs.end();
		if (bExists)
		{
			pOutputObj = aLookup->second;
			return pOutputObj;
		}
		else
		{
			return nullptr;
		}
	}
	return nullptr;
}
//////////////////////////////////////////////////////////////////////////////////////////////
