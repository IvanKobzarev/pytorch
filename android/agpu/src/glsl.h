#pragma once
#include <ATen/native/vulkan/glsl.h>
namespace at { namespace native { namespace vulkan { namespace gl {
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
extern const char* maxpool2d_glsl;
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

} } } } //namespace at::native::vulkan::gl
