#pragma once

#include <stdint.h>
#include <stdio.h>

#ifdef __ANDROID__

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>

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

const char* agpu_test();

void agpu_conv2d(
    const float* input,
    uint32_t input_height,
    uint32_t input_width,
    const float* kernel,
    uint32_t kernel_height,
    uint32_t kernel_width,
    const float* bias,
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,

    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,

    float* output);
