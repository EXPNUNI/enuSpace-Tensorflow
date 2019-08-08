// Logic.h : Logic DLL�� �⺻ ��� �����Դϴ�.
//

#pragma once

#include "stdafx.h"
#include "GlobalHeader.h"

#ifndef __AFXWIN_H__
	#error "PCH�� ���� �� ������ �����ϱ� ���� 'stdafx.h'�� �����մϴ�."
#endif

#include "EnuObj.h"


#pragma pack(push, 1) 

struct tfLink : TRANSFER
{

};

struct tfObject : EnuObject 
{

};

struct _BatchToSpace : tfObject {};
struct _BatchToSpaceND : tfObject {};
struct _Bitcast : tfObject {};
struct _BroadcastDynamicShape : tfObject {};
struct _CheckNumerics : tfObject {};
struct _Concat : tfObject {};
struct _DebugGradientIdentity : tfObject {};
struct _DepthToSpace : tfObject {};
struct _Dequantize : tfObject {};
struct _Diag : tfObject {};
struct _DiagPart : tfObject {};
struct _EditDistance : tfObject {};
struct _ExpandDims : tfObject {};
struct _ExtractImagePatches : tfObject {};
struct _FakeQuantWithMinMaxArgs : tfObject {};
struct _FakeQuantWithMinMaxArgsGradient : tfObject {};
struct _FakeQuantWithMinMaxVars : tfObject {};
struct _FakeQuantWithMinMaxVarsGradient : tfObject {};
struct _FakeQuantWithMinMaxVarsPerChannel : tfObject {};
struct _FakeQuantWithMinMaxVarsPerChannelGradient : tfObject {};
struct _Fill : tfObject {};
struct _Gather : tfObject {};
struct _GatherV2 : tfObject {};
struct _GatherNd : tfObject {};
struct _Identity : tfObject {};
struct _ImmutableConst : tfObject {};
struct _InvertPermutation : tfObject {};
struct _MatrixBandPart : tfObject {};
struct _MatrixDiag : tfObject {};
struct _MatrixDiagPart : tfObject {};
struct _MatrixSetDiag : tfObject {};
struct _MirrorPad : tfObject {};
struct _OneHot : tfObject {};
struct _OnesLike : tfObject {};
struct _Pad : tfObject {};
struct _PadV2 : tfObject {};
struct _ParallelConcat : tfObject {};
struct _Placeholder : tfObject {};
//struct PlaceholderV2	 : tfObject {};
struct _PlaceholderWithDefault : tfObject {};
struct _PreventGradient : tfObject {};
struct _QuantizeAndDequantizeV2 : tfObject {};
struct _QuantizeV2 : tfObject {};
struct _QuantizedConcat : tfObject {};
struct _QuantizedInstanceNorm : tfObject {};
struct _QuantizedReshape : tfObject {};
struct _Rank : tfObject {};
struct _Reshape : tfObject {};
struct _ResourceStridedSliceAssign : tfObject {};
struct _Reverse : tfObject {};
struct _ReverseSequence : tfObject {};
struct _ScatterNd : tfObject {};
struct _ScatterNdNonAliasingAdd : tfObject {};
struct _SetDiff1D : tfObject {};
struct _Shape : tfObject {};
struct _ShapeN : tfObject {};
struct _Size : tfObject {};
struct _Slice : tfObject {};
struct _SpaceToBatch : tfObject {};
struct _SpaceToBatchND : tfObject {};
struct _SpaceToDepth : tfObject {};
struct _Split : tfObject {};
struct _SplitV : tfObject {};
struct _Squeeze : tfObject {};
struct _Stack : tfObject {};
struct _StopGradient : tfObject {};
struct _StridedSlice : tfObject {};
struct _StridedSliceAssign : tfObject {};
struct _StridedSliceGrad : tfObject {};
struct _Tile : tfObject {};
struct _Transpose : tfObject {};
struct _Unique : tfObject {};
struct _UniqueWithCounts : tfObject {};
struct _Unstack : tfObject {};
struct _Where : tfObject {};
struct _ZerosLike : tfObject {};
struct _AllCandidateSampler : tfObject {};
struct _ComputeAccidentalHits : tfObject {};
struct _FixedUnigramCandidateSampler : tfObject {};
struct _LearnedUnigramCandidateSampler : tfObject {};
struct _LogUniformCandidateSampler : tfObject {};
struct _UniformCandidateSampler : tfObject {};
struct _Abort : tfObject {};
struct _ControlTrigger : tfObject {};
struct _LoopCond : tfObject {};
struct _Merge : tfObject {};
struct _NextIteration : tfObject {};
struct _RefNextIteration : tfObject {};
struct _RefSelect : tfObject {};
struct _RefSwitch : tfObject {};
struct _Switch : tfObject {};
struct _ClientSession : tfObject {};
struct _Input : tfObject {};
struct _Input_Initializer : tfObject {};
struct _InputList : tfObject {};
struct _Operation : tfObject {};
struct _Output : tfObject {};
struct _Scope : tfObject {};
struct _Status : tfObject {};
struct _Tensor : tfObject {};
struct _FeedType : tfObject {};
struct _AccumulatorApplyGradient : tfObject {};
struct _AccumulatorNumAccumulated : tfObject {};
struct _AccumulatorSetGlobalStep : tfObject {};
struct _AccumulatorTakeGradient : tfObject {};
struct _Barrier : tfObject {};
struct _BarrierClose : tfObject {};
struct _BarrierIncompleteSize : tfObject {};
struct _BarrierInsertMany : tfObject {};
struct _BarrierReadySize : tfObject {};
struct _BarrierTakeMany : tfObject {};
struct _ConditionalAccumulator : tfObject {};
struct _DeleteSessionTensor : tfObject {};
struct _DynamicPartition : tfObject {};
struct _DynamicStitch : tfObject {};
struct _FIFOQueue : tfObject {};
struct _GetSessionHandle : tfObject {};
struct _GetSessionHandleV2 : tfObject {};
struct _GetSessionTensor : tfObject {};

struct _MapClear : tfObject {};
struct _MapIncompleteSize : tfObject {};
struct _MapPeek : tfObject {};
struct _MapSize : tfObject {};
struct _MapStage : tfObject {};
struct _MapUnstage : tfObject {};
struct _MapUnstageNoKey : tfObject {};
struct _OrderedMapClear : tfObject {};
struct _OrderedMapIncompleteSize : tfObject {};
struct _OrderedMapPeek : tfObject {};
struct _OrderedMapSize : tfObject {};
struct _OrderedMapStage : tfObject {};
struct _OrderedMapUnstage : tfObject {};
struct _OrderedMapUnstageNoKey : tfObject {};

struct _PaddingFIFOQueue : tfObject {};
struct _PriorityQueue : tfObject {};
struct _QueueClose : tfObject {};
struct _QueueDequeue : tfObject {};
struct _QueueDequeueMany : tfObject {};
struct _QueueDequeueUpTo : tfObject {};
struct _QueueEnqueue : tfObject {};
struct _QueueEnqueueMany : tfObject {};
struct _QueueSize : tfObject {};
struct _RandomShuffleQueue : tfObject {};
struct _RecordInput : tfObject {};
struct _SparseAccumulatorApplyGradient : tfObject {};
struct _SparseAccumulatorTakeGradient : tfObject {};
struct _SparseConditionalAccumulator : tfObject {};
struct _Stage : tfObject {};

struct _StageClear : tfObject {};
struct _StagePeek : tfObject {};
struct _StageSize : tfObject {};

struct _TensorArray : tfObject {};
struct _TensorArrayClose : tfObject {};
struct _TensorArrayConcat : tfObject {};
struct _TensorArrayGather : tfObject {};
struct _TensorArrayGrad : tfObject {};
struct _TensorArrayRead : tfObject {};
struct _TensorArrayScatter : tfObject {};
struct _TensorArraySize : tfObject {};
struct _TensorArraySplit : tfObject {};
struct _TensorArrayWrite : tfObject {};
struct _Unstage : tfObject {};
struct _AdjustContrast : tfObject {};
struct _AdjustHue : tfObject {};
struct _AdjustSaturation : tfObject {};
struct _CropAndResize : tfObject {};
struct _CropAndResizeGradBoxes : tfObject {};
struct _CropAndResizeGradImage : tfObject {};
struct _DecodeBmp : tfObject {};
struct _DecodeGif : tfObject {};
struct _DecodeJpeg : tfObject {};
struct _DecodePng : tfObject {};
struct _DrawBoundingBoxes : tfObject {};
struct _EncodeJpeg : tfObject {};
struct _EncodePng : tfObject {};
struct _ExtractGlimpse : tfObject {};
struct _HSVToRGB : tfObject {};
struct _NonMaxSuppression : tfObject {};
struct _QuantizedResizeBilinear : tfObject {};
struct _RGBToHSV : tfObject {};
struct _ResizeArea : tfObject {};
struct _ResizeBicubic : tfObject {};
struct _ResizeBilinear : tfObject {};
struct _ResizeNearestNeighbor : tfObject {};
struct _SampleDistortedBoundingBoxV2 : tfObject {};
struct _SampleDistortedBoundingBox : tfObject {};
struct _FixedLengthRecordReader : tfObject {};
struct _IdentityReader : tfObject {};
struct _LMDBReader : tfObject {};
struct _MatchingFiles : tfObject {};
struct _MergeV2Checkpoints : tfObject {};
struct _ReadFile : tfObject {};
struct _ReaderNumRecordsProduced : tfObject {};
struct _ReaderNumWorkUnitsCompleted : tfObject {};
struct _ReaderRead : tfObject {};
struct _ReaderReadUpTo : tfObject {};
struct _ReaderReset : tfObject {};
struct _ReaderRestoreState : tfObject {};
struct _ReaderSerializeState : tfObject {};
struct _Restore : tfObject {};
struct _RestoreSlice : tfObject {};
struct _RestoreV2 : tfObject {};
struct _Save : tfObject {};
struct _SaveSlices : tfObject {};
struct _SaveV2 : tfObject {};
struct _ShardedFilename : tfObject {};
struct _ShardedFilespec : tfObject {};
struct _TFRecordReader : tfObject {};
struct _TextLineReader : tfObject {};
struct _WholeFileReader : tfObject {};
struct _WriteFile : tfObject {};
struct _Assert : tfObject {};
struct _HistogramSummary : tfObject {};
struct _MergeSummary : tfObject {};
struct _Print : tfObject {};
struct _ScalarSummary : tfObject {};
struct _TensorSummary : tfObject {};
struct _Abs : tfObject {};
struct _Acos : tfObject {};
struct _Add : tfObject {};
struct _AddN : tfObject {};
struct _All : tfObject {};
struct _Any : tfObject {};
struct _ApproximateEqual : tfObject {};
struct _ArgMax : tfObject {};
struct _ArgMin : tfObject {};
struct _Asin : tfObject {};
struct _Atan : tfObject {};
struct _Atan2 : tfObject {};
struct _BatchMatMul : tfObject {};
struct _Betainc : tfObject {};
struct _Bincount : tfObject {};
struct _Bucketize : tfObject {};
struct _Cast : tfObject {};
struct _Ceil : tfObject {};
struct _Complex : tfObject {};
struct _ComplexAbs : tfObject {};
struct _Conj : tfObject {};
struct _Cos : tfObject {};
struct _Cosh : tfObject {};
struct _Cross : tfObject {};
struct _Cumprod : tfObject {};
struct _Cumsum : tfObject {};
struct _Digamma : tfObject {};
struct _Div : tfObject {};
struct _Equal : tfObject {};
struct _Erf : tfObject {};
struct _Erfc : tfObject {};
struct _Exp : tfObject {};
struct _Expm1 : tfObject {};
struct _Floor : tfObject {};
struct _FloorDiv : tfObject {};
struct _FloorMod : tfObject {};
struct _Greater : tfObject {};
struct _GreaterEqual : tfObject {};
struct _Igamma : tfObject {};
struct _Igammac : tfObject {};
struct _Imag : tfObject {};
struct _IsFinite : tfObject {};
struct _IsInf : tfObject {};
struct _IsNan : tfObject {};
struct _Less : tfObject {};
struct _LessEqual : tfObject {};
struct _Lgamma : tfObject {};
struct _LinSpace : tfObject {};
struct _Log : tfObject {};
struct _Log1p : tfObject {};
struct _LogicalAnd : tfObject {};
struct _LogicalNot : tfObject {};
struct _LogicalOr : tfObject {};
struct _MatMul : tfObject {};
struct _Max : tfObject {};
struct _Maximum : tfObject {};
struct _Mean : tfObject {};
struct _Min : tfObject {};
struct _Minimum : tfObject {};
struct _Mod : tfObject {};
struct _Multiply : tfObject {};
struct _Negate : tfObject {};
struct _NotEqual : tfObject {};
struct _Polygamma : tfObject {};
struct _Pow : tfObject {};
struct _Prod : tfObject {};
struct _QuantizeDownAndShrinkRange : tfObject {};
struct _QuantizedMatMul : tfObject {};
struct _QuantizedMul : tfObject {};
struct _Range : tfObject {};
struct _Real : tfObject {};
struct _RealDiv : tfObject {};
struct _Reciprocal : tfObject {};
struct _RequantizationRange : tfObject {};
struct _Requantize : tfObject {};
struct _Rint : tfObject {};
struct _Round : tfObject {};
struct _Rsqrt : tfObject {};
struct _SegmentMax : tfObject {};
struct _SegmentMean : tfObject {};
struct _SegmentMin : tfObject {};
struct _SegmentProd : tfObject {};
struct _SegmentSum : tfObject {};
struct _Sigmoid : tfObject {};
struct _Sign : tfObject {};
struct _Sin : tfObject {};
struct _Sinh : tfObject {};
struct _SparseMatMul : tfObject {};
struct _SparseSegmentMean : tfObject {};
struct _SparseSegmentMeanGrad : tfObject {};
struct _SparseSegmentSqrtN : tfObject {};
struct _SparseSegmentSqrtNGrad : tfObject {};
struct _SparseSegmentSum : tfObject {};
struct _Sqrt : tfObject {};
struct _Square : tfObject {};
struct _SquaredDifference : tfObject {};
struct _Subtract : tfObject {};
struct _Sum : tfObject {};
struct _Tan : tfObject {};
struct _Tanh : tfObject {};
struct _TruncateDiv : tfObject {};
struct _TruncateMod : tfObject {};
struct _UnsortedSegmentMax : tfObject {};
struct _UnsortedSegmentSum : tfObject {};
struct _Where3 : tfObject {};
struct _Zeta : tfObject {};
struct _AvgPool : tfObject {};
struct _AvgPool3D : tfObject {};
struct _AvgPool3DGrad : tfObject {};
struct _BiasAdd : tfObject {};
struct _BiasAddGrad : tfObject {};
struct _Conv2D : tfObject {};
struct _Conv2DBackpropFilter : tfObject {};
struct _Conv2DBackpropInput : tfObject {};
struct _Conv3D : tfObject {};
struct _Conv3DBackpropFilterV2 : tfObject {};
struct _Conv3DBackpropInputV2 : tfObject {};
struct _DepthwiseConv2dNative : tfObject {};
struct _DepthwiseConv2dNativeBackpropFilter : tfObject {};
struct _DepthwiseConv2dNativeBackpropInput : tfObject {};
struct _Dilation2D : tfObject {};
struct _Dilation2DBackpropFilter : tfObject {};
struct _Dilation2DBackpropInput : tfObject {};
struct _Elu : tfObject {};
struct _FractionalAvgPool : tfObject {};
struct _FractionalMaxPool : tfObject {};
struct _FusedBatchNorm : tfObject {};
struct _FusedBatchNormGrad : tfObject {};
struct _FusedPadConv2D : tfObject {};
struct _FusedResizeAndPadConv2D : tfObject {};
struct _InTopK : tfObject {};
struct _L2Loss : tfObject {};
struct _LRN : tfObject {};
struct _LogSoftmax : tfObject {};
struct _MaxPool : tfObject {};
struct _MaxPool3D : tfObject {};
struct _MaxPool3DGrad : tfObject {};
struct _MaxPool3DGradGrad : tfObject {};
struct _MaxPoolGradGrad : tfObject {};
struct _MaxPoolGradGradWithArgmax : tfObject {};
struct _MaxPoolWithArgmax : tfObject {};
struct _QuantizedAvgPool : tfObject {};
struct _QuantizedBatchNormWithGlobalNormalization : tfObject {};
struct _QuantizedBiasAdd : tfObject {};
struct _QuantizedConv2D : tfObject {};
struct _QuantizedMaxPool : tfObject {};
struct _QuantizedRelu : tfObject {};
struct _QuantizedRelu6 : tfObject {};
struct _QuantizedReluX : tfObject {};
struct _Relu : tfObject {};
struct _Relu6 : tfObject {};
struct _Softmax : tfObject {};
struct _SoftmaxCrossEntropyWithLogits : tfObject {};
struct _Softplus : tfObject {};
struct _Softsign : tfObject {};
struct _SparseSoftmaxCrossEntropyWithLogits : tfObject {};
struct _TopK : tfObject {};
struct _NoOp : tfObject {};
struct _DecodeCSV : tfObject {};
struct _DecodeJSONExample : tfObject {};
struct _DecodeRaw : tfObject {};
struct _ParseExample : tfObject {};
struct _ParseSingleSequenceExample : tfObject {};
struct _ParseTensor : tfObject {};
struct _StringToNumber : tfObject {};
struct _Multinomial : tfObject {};
struct _ParameterizedTruncatedNormal : tfObject {};
struct _RandomGamma : tfObject {};
struct _RandomNormal : tfObject {};
struct _RandomPoisson : tfObject {};
struct _RandomShuffle : tfObject {};
struct _RandomUniform : tfObject {};
struct _RandomUniformInt : tfObject {};
struct _TruncatedNormal : tfObject {};
struct _AddManySparseToTensorsMap : tfObject {};
struct _AddSparseToTensorsMap : tfObject {};
struct _DeserializeManySparse : tfObject {};
struct _SerializeManySparse : tfObject {};
struct _SerializeSparse : tfObject {};
struct _SparseAdd : tfObject {};
struct _SparseAddGrad : tfObject {};
struct _SparseConcat : tfObject {};
struct _SparseCross : tfObject {};
struct _SparseDenseCwiseAdd : tfObject {};
struct _SparseDenseCwiseDiv : tfObject {};
struct _SparseDenseCwiseMul : tfObject {};
struct _SparseReduceSum : tfObject {};
struct _SparseReduceSumSparse : tfObject {};
struct _SparseReorder : tfObject {};
struct _SparseReshape : tfObject {};
struct _SparseSoftmax : tfObject {};
struct _SparseSparseMaximum : tfObject {};
struct _SparseSparseMinimum : tfObject {};
struct _SparseSplit : tfObject {};
struct _SparseTensorDenseAdd : tfObject {};
struct _SparseTensorDenseMatMul : tfObject {};
struct _SparseToDense : tfObject {};
struct _TakeManySparseFromTensorsMap : tfObject {};
struct _Assign : tfObject {};
struct _AssignAdd : tfObject {};
struct _AssignSub : tfObject {};
struct _CountUpTo : tfObject {};
struct _DestroyTemporaryVariable : tfObject {};
struct _IsVariableInitialized : tfObject {};
struct _ScatterAdd : tfObject {};
struct _ScatterDiv : tfObject {};
struct _ScatterMul : tfObject {};
struct _ScatterNdAdd : tfObject {};
struct _ScatterNdSub : tfObject {};
struct _ScatterNdUpdate : tfObject {};
struct _ScatterSub : tfObject {};
struct _ScatterUpdate : tfObject {};
struct _TemporaryVariable : tfObject {};
struct _Variable : tfObject {};
struct _AsString : tfObject {};
struct _DecodeBase64 : tfObject {};
struct _EncodeBase64 : tfObject {};
struct _ReduceJoin : tfObject {};
struct _StringJoin : tfObject {};
struct _StringSplit : tfObject {};
struct _StringToHashBucket : tfObject {};
struct _StringToHashBucketFast : tfObject {};
struct _StringToHashBucketStrong : tfObject {};
struct _Substr : tfObject {};
struct _ApplyAdadelta : tfObject {};
struct _ApplyAdagrad : tfObject {};
struct _ApplyAdagradDA : tfObject {};
struct _ApplyAdam : tfObject {};
struct _ApplyCenteredRMSProp : tfObject {};
struct _ApplyFtrl : tfObject {};
struct _ApplyGradientDescent : tfObject {};
struct _ApplyMomentum : tfObject {};
struct _ApplyProximalAdagrad : tfObject {};
struct _ApplyProximalGradientDescent : tfObject {};
struct _ApplyRMSProp : tfObject {};
struct _ResourceApplyAdadelta : tfObject {};
struct _ResourceApplyAdagrad : tfObject {};
struct _ResourceApplyAdagradDA : tfObject {};
struct _ResourceApplyAdam : tfObject {};
struct _ResourceApplyCenteredRMSProp : tfObject {};
struct _ResourceApplyFtrl : tfObject {};
struct _ResourceApplyGradientDescent : tfObject {};
struct _ResourceApplyMomentum : tfObject {};
struct _ResourceApplyProximalAdagrad : tfObject {};
struct _ResourceApplyProximalGradientDescent : tfObject {};
struct _ResourceApplyRMSProp : tfObject {};
struct _ResourceSparseApplyAdadelta : tfObject {};
struct _ResourceSparseApplyAdagrad : tfObject {};
struct _ResourceSparseApplyAdagradDA : tfObject {};
struct _ResourceSparseApplyCenteredRMSProp : tfObject {};
struct _ResourceSparseApplyFtrl : tfObject {};
struct _ResourceSparseApplyMomentum : tfObject {};
struct _ResourceSparseApplyProximalAdagrad : tfObject {};
struct _ResourceSparseApplyProximalGradientDescent : tfObject {};
struct _ResourceSparseApplyRMSProp : tfObject {};
struct _SparseApplyAdadelta : tfObject {};
struct _SparseApplyAdagrad : tfObject {};
struct _SparseApplyAdagradDA : tfObject {};
struct _SparseApplyCenteredRMSProp : tfObject {};
struct _SparseApplyFtrl : tfObject {};
struct _SparseApplyMomentum : tfObject {};
struct _SparseApplyProximalAdagrad : tfObject {};
struct _SparseApplyProximalGradientDescent : tfObject {};
struct _SparseApplyRMSProp : tfObject {};
struct _Fact : tfObject {};
struct _Const : tfObject {};
struct _RandomNormal_ex : tfObject {};
struct _Const_ex : tfObject {};
struct _SparseSlice : tfObject {};
struct _SparseFillEmptyRows : tfObject {};
struct _SparseFillEmptyRowsGrad : tfObject {};
struct _SparseReduceMax : tfObject {};
struct _SparseReduceMaxSparse : tfObject {};
struct _AddSymbolicGradients : tfObject {};



#pragma pack(pop) 
