#include "stdafx.h"
#include "tf_image_ops.h"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "include/json/json.h"

#include "GlobalHeader.h"
#include "tensorflow.h"
#include "utility_functions.h"
#include "enuSpaceToTensorflow.h"

void* Create_AdjustContrast(std::string id, Json::Value pInputItem) {
	AdjustContrast* pAdjustContrast = nullptr;
	Scope* pScope = nullptr;
	return pAdjustContrast;
}

void* Create_AdjustHue(std::string id, Json::Value pInputItem) {
	AdjustHue* pAdjustHue = nullptr;
	Scope* pScope = nullptr;
	return pAdjustHue;
}

void* Create_AdjustSaturation(std::string id, Json::Value pInputItem) {
	AdjustSaturation* pAdjustSaturation = nullptr;
	Scope* pScope = nullptr;
	return pAdjustSaturation;
}

void* Create_CropAndResize(std::string id, Json::Value pInputItem) {
	CropAndResize* pCropAndResize = nullptr;
	Scope* pScope = nullptr;
	return pCropAndResize;
}

void* Create_CropAndResizeGradBoxes(std::string id, Json::Value pInputItem) {
	CropAndResizeGradBoxes* pCropAndResizeGradBoxes = nullptr;
	Scope* pScope = nullptr;
	return pCropAndResizeGradBoxes;
}

void* Create_CropAndResizeGradImage(std::string id, Json::Value pInputItem) {
	CropAndResizeGradImage* pCropAndResizeGradImage = nullptr;
	Scope* pScope = nullptr;
	return pCropAndResizeGradImage;
}

void* Create_DecodeGif(std::string id, Json::Value pInputItem) {
	DecodeGif* pDecodeGif = nullptr;
	Scope* pScope = nullptr;
	return pDecodeGif;
}

void* Create_DecodeJpeg(std::string id, Json::Value pInputItem) {
	DecodeJpeg* pDecodeJpeg = nullptr;
	Scope* pScope = nullptr;
	return pDecodeJpeg;
}

void* Create_DecodePng(std::string id, Json::Value pInputItem) {
	DecodePng* pDecodePng = nullptr;
	Scope* pScope = nullptr;
	return pDecodePng;
}

void* Create_DrawBoundingBoxes(std::string id, Json::Value pInputItem) {
	DrawBoundingBoxes* pDrawBoundingBoxes = nullptr;
	Scope* pScope = nullptr;
	return pDrawBoundingBoxes;
}

void* Create_EncodeJpeg(std::string id, Json::Value pInputItem) {
	EncodeJpeg* pEncodeJpeg = nullptr;
	Scope* pScope = nullptr;
	return pEncodeJpeg;
}

void* Create_EncodePng(std::string id, Json::Value pInputItem) {
	EncodePng* pEncodePng = nullptr;
	Scope* pScope = nullptr;
	return pEncodePng;
}

void* Create_ExtractGlimpse(std::string id, Json::Value pInputItem) {
	ExtractGlimpse* pExtractGlimpse = nullptr;
	Scope* pScope = nullptr;
	return pExtractGlimpse;
}

void* Create_HSVToRGB(std::string id, Json::Value pInputItem) {
	HSVToRGB* pHSVToRGB = nullptr;
	Scope* pScope = nullptr;
	return pHSVToRGB;
}

void* Create_NonMaxSuppression(std::string id, Json::Value pInputItem) {
	NonMaxSuppression* pNonMaxSuppression = nullptr;
	Scope* pScope = nullptr;
	return pNonMaxSuppression;
}

void* Create_RGBToHSV(std::string id, Json::Value pInputItem) {
	RGBToHSV* pRGBToHSV = nullptr;
	Scope* pScope = nullptr;
	return pRGBToHSV;
}

void* Create_ResizeArea(std::string id, Json::Value pInputItem) {
	ResizeArea* pResizeArea = nullptr;
	Scope* pScope = nullptr;
	return pResizeArea;
}

void* Create_ResizeBicubic(std::string id, Json::Value pInputItem) {
	ResizeBicubic* pResizeBicubic = nullptr;
	Scope* pScope = nullptr;
	return pResizeBicubic;
}

void* Create_ResizeBilinear(std::string id, Json::Value pInputItem) {
	ResizeBilinear* pResizeBilinear = nullptr;
	Scope* pScope = nullptr;
	return pResizeBilinear;
}

void* Create_ResizeNearestNeighbor(std::string id, Json::Value pInputItem) {
	ResizeNearestNeighbor* pResizeNearestNeighbor = nullptr;
	Scope* pScope = nullptr;
	return pResizeNearestNeighbor;
}

void* Create_SampleDistortedBoundingBox(std::string id, Json::Value pInputItem) {
	SampleDistortedBoundingBox* pSampleDistortedBoundingBox = nullptr;
	Scope* pScope = nullptr;
	return pSampleDistortedBoundingBox;
}
