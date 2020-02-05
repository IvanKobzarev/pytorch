#pragma once

#include <stdint.h>
#include <stdio.h>
#include "agpu_gl_header.h"

#ifdef __ANDROID__
#include <android/log.h>
#define AGPU_ERROR(format, ...) \
  __android_log_print(ANDROID_LOG_ERROR, "AGPU", format, ##__VA_ARGS__)
#define AGPU_PRINT(format, ...) \
  __android_log_print(ANDROID_LOG_INFO, "AGPU", format, ##__VA_ARGS__)
#else
#define AGPU_PRINT(format, ...) printf(format, ##__VA_ARGS__)
#define AGPU_ERROR(format, ...) printf(format, ##__VA_ARGS__)
#endif

#define FUNC_PRINT(x) AGPU_PRINT(#x "=%d in %s, %d \n", x, __func__, __LINE__);
#define FUNC_PRINT_ALL(x, type) \
  AGPU_PRINT(#x "=" #type " %" #type " in %s, %d \n", x, __func__, __LINE__);

#define AGL_CHECK_ERROR                                                       \
  {                                                                           \
    GLenum error = glGetError();                                              \
    if (GL_NO_ERROR != error) {                                               \
      AGPU_PRINT(                                                             \
          "File = %s Line = %d Func=%s\n", __FILE__, __LINE__, __FUNCTION__); \
      FUNC_PRINT_ALL(error, 0x);                                              \
    }                                                                         \
    assert(GL_NO_ERROR == error);                                             \
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
    float* output);

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
    float* output);

} // namespace agpu
