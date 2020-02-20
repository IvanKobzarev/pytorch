#pragma once

#define AGPU_VERBOSE false

#include <stdint.h>
#include <stdio.h>
#include "agpu_gl_header.h"

#ifdef __ANDROID__
#include <android/log.h>
#define AGPU_ERROR(format, ...) \
  __android_log_print(ANDROID_LOG_ERROR, "AGPU", format, ##__VA_ARGS__)
#define APRINT(format, ...) \
  if (AGPU_VERBOSE)         \
  __android_log_print(ANDROID_LOG_INFO, "AGPU", format, ##__VA_ARGS__)
#else
#define APRINT(format, ...) printf(format, ##__VA_ARGS__)
#define AGPU_ERROR(format, ...) printf(format, ##__VA_ARGS__)
#endif

#define FUNC_PRINT(x) APRINT(#x "=%d in %s, %d \n", x, __func__, __LINE__);
#define FUNC_PRINT_ALL(x, type) \
  APRINT(#x "=" #type " %" #type " in %s, %d \n", x, __func__, __LINE__);

#define AGL_CHECK_ERROR \
  { \
    GLenum error = glGetError(); \
    if (GL_NO_ERROR != error) { \
      APRINT(\
          "File = %s Line = %d Func=%s\n", __FILE__, __LINE__, __FUNCTION__); \
      FUNC_PRINT_ALL(error, 0x); \
    } \
    assert(GL_NO_ERROR == error); \
  }
namespace agpu {

void agpu_conv2d(
    const float* input,
    uint32_t input_n,
    uint32_t input_c,
    uint32_t input_h,
    uint32_t input_w,
    const float* weights,
    uint32_t kernel_c,
    uint32_t kernel_h,
    uint32_t kernel_w,
    const float* bias,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t input_padding_h,
    uint32_t input_padding_w,
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint32_t groups,
    float* output,
    int64_t mod = 0);

void agpu_conv2d_sTextures(
    const float* input,
    uint32_t input_n,
    uint32_t input_c,
    uint32_t input_h,
    uint32_t input_w,
    const float* weights,
    uint32_t kernel_c,
    uint32_t kernel_h,
    uint32_t kernel_w,
    const float* bias,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t input_padding_h,
    uint32_t input_padding_w,
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint32_t groups,
    float* output,
    int64_t mod = 0);

void agpu_conv2d_buffers_sOutNc4nc(
    const float* input,
    uint32_t input_n,
    uint32_t input_c,
    uint32_t input_h,
    uint32_t input_w,
    const float* weights,
    uint32_t kernel_c,
    uint32_t kernel_h,
    uint32_t kernel_w,
    const float* bias,
    uint32_t stride_y,
    uint32_t stride_x,
    uint32_t input_padding_y,
    uint32_t input_padding_x,
    uint32_t dilation_y,
    uint32_t dilation_x,
    uint32_t groups,
    float* output,
    int64_t mod = 0);

void agpu_conv2d_buffers_sOutNchw(
    const float* input,
    uint32_t input_n,
    uint32_t input_c,
    uint32_t input_h,
    uint32_t input_w,
    const float* weights,
    uint32_t kernel_c,
    uint32_t kernel_h,
    uint32_t kernel_w,
    const float* bias,
    uint32_t stride_y,
    uint32_t stride_x,
    uint32_t input_padding_y,
    uint32_t input_padding_x,
    uint32_t dilation_y,
    uint32_t dilation_x,
    uint32_t groups,
    float* output,
    int64_t mod = 0);

void agpu_conv2d_buffers_sInOutNchw(
    const float* input,
    uint32_t input_n,
    uint32_t input_c,
    uint32_t input_h,
    uint32_t input_w,
    const float* weights,
    uint32_t kernel_c,
    uint32_t kernel_h,
    uint32_t kernel_w,
    const float* bias,
    uint32_t stride_y,
    uint32_t stride_x,
    uint32_t input_padding_y,
    uint32_t input_padding_x,
    uint32_t dilation_y,
    uint32_t dilation_x,
    uint32_t groups,
    float* output,
    int64_t mod = 0);

void agpu_add2t(
    const float* input0,
    const float* input1,
    uint32_t n,
    uint32_t c,
    uint32_t h,
    uint32_t w,
    float* output);

void agpu_threshold(
    const float* input,
    uint32_t n,
    uint32_t c,
    uint32_t h,
    uint32_t w,
    float threshold,
    float value,
    float* output);

void agpu_batch_norm(
    const float* input,
    uint32_t n,
    uint32_t c,
    uint32_t h,
    uint32_t w,
    const float* weight,
    const float* bias,
    const float* mean,
    const float* variance,
    const float eps,
    float* output);

void agpu_print(const char* m, const float* t, uint32_t rank, uint32_t* dims);

void agpu_print4d(
    const char* m,
    const float* data,
    uint32_t d0,
    uint32_t d1,
    uint32_t d2,
    uint32_t d3);
} // namespace agpu
