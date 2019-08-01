#pragma once

#ifndef _TF_IMAGE_OPS_HEADER_
#define _TF_IMAGE_OPS_HEADER_

#include <string>
#include "include/json/json.h"


void* Create_AdjustContrast(std::string id, Json::Value pInputItem);
void* Create_AdjustHue(std::string id, Json::Value pInputItem);
void* Create_AdjustSaturation(std::string id, Json::Value pInputItem);
void* Create_CropAndResize(std::string id, Json::Value pInputItem);
void* Create_CropAndResizeGradBoxes(std::string id, Json::Value pInputItem);
void* Create_CropAndResizeGradImage(std::string id, Json::Value pInputItem);
void* Create_DecodeBmp(std::string id, Json::Value pInputItem);
void* Create_DecodeGif(std::string id, Json::Value pInputItem);
void* Create_DecodeJpeg(std::string id, Json::Value pInputItem);
void* Create_DecodePng(std::string id, Json::Value pInputItem);
void* Create_DrawBoundingBoxes(std::string id, Json::Value pInputItem);
void* Create_EncodeJpeg(std::string id, Json::Value pInputItem);
void* Create_EncodePng(std::string id, Json::Value pInputItem);
void* Create_ExtractGlimpse(std::string id, Json::Value pInputItem);
void* Create_HSVToRGB(std::string id, Json::Value pInputItem);
void* Create_NonMaxSuppression(std::string id, Json::Value pInputItem);
void* Create_QuantizedResizeBilinear(std::string id, Json::Value pInputItem);
void* Create_RGBToHSV(std::string id, Json::Value pInputItem);
void* Create_ResizeArea(std::string id, Json::Value pInputItem);
void* Create_ResizeBicubic(std::string id, Json::Value pInputItem);
void* Create_ResizeBilinear(std::string id, Json::Value pInputItem);
void* Create_ResizeNearestNeighbor(std::string id, Json::Value pInputItem);
void* Create_SampleDistortedBoundingBox(std::string id, Json::Value pInputItem);
void* Create_SampleDistortedBoundingBoxV2(std::string id, Json::Value pInputItem);

#endif