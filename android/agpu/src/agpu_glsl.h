#pragma once
#include "agpu.h"
namespace agpu {
extern const char* KO4C4HW_to_tex_glsl;
extern const char* addmm_glsl;
extern const char* binary_add_glsl;
extern const char* convDW_buf_IKnchw_glsl;
extern const char* convDW_buf_IKnhwc_glsl;
extern const char* convDW_buf_Inhwc_Knchw_glsl;
extern const char* conv_buf_IKnchw_KrO4C4HW_glsl;
extern const char* conv_buf_IKnchw_KrO4HWC_glsl;
extern const char* conv_buf_IKnchw_SIKOnc4hw_KrO4C4HW_glsl;
extern const char* conv_buf_IKnchw_SIKOnc4hw_KrO4HWC_glsl;
extern const char* conv_buf_IKnchw_SIKnc4hw_SOnchw_glsl;
extern const char* conv_buf_IKnchw_SKnc4hw_KrO4C4HW_glsl;
extern const char* conv_buf_IKnhwc_glsl;
extern const char* conv_buf_IKnhwc_KrO4C4HW_glsl;
extern const char* conv_buf_IKnhwc_KrO4HWC_glsl;
extern const char* conv_buf_Inhwc_Knchw_KrO4C4HW_glsl;
extern const char* conv_tex_IKnc4hw_glsl;
extern const char* gemm_glsl;
extern const char* gemm16x16_glsl;
extern const char* nc4hw4_buf_to_tex_glsl;
extern const char* nc4hw_buf_to_nchw_buf_glsl;
extern const char* nchw_buf_to_nc4hw_buf_glsl;
extern const char* nchw_buf_to_tex_glsl;
extern const char* nhwc_buf_to_tex_glsl;
extern const char* normalization_glsl;
extern const char* tex_to_nc4hw4_buf_glsl;
extern const char* tex_to_nchw_buf_glsl;
extern const char* threshold_glsl;
extern const char* upsampleNearest2d_glsl;


enum class AConv : int32_t {
  convDW_buf_IKnchw = 0,
  convDW_buf_IKnhwc = 10,
  convDW_buf_Inhwc_Knchw = 20,
  conv_buf_IKnchw_KrO4C4HW = 30,
  conv_buf_IKnchw_KrO4HWC = 40,
  conv_buf_IKnchw_SIKOnc4hw_KrO4C4HW = 50,
  conv_buf_IKnchw_SIKOnc4hw_KrO4HWC = 60,
  conv_buf_IKnchw_SIKnc4hw_SOnchw = 70,
  conv_buf_IKnchw_SKnc4hw_KrO4C4HW = 80,
  conv_buf_IKnhwc = 90,
  conv_buf_IKnhwc_KrO4C4HW = 100,
  conv_buf_IKnhwc_KrO4HWC = 110,
  conv_buf_Inhwc_Knchw_KrO4C4HW = 120,
  conv_tex_IKnc4hw = 130,
};

using fp_agpu_conv_t = decltype(&::agpu::conv_);
fp_agpu_conv_t aConvToFun(AConv aconv);
AConv aConvByCode(int64_t code);
} //namespace agpu
