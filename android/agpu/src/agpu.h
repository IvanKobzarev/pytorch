#pragma once

#define AGPU_VERBOSE true
#define AGPU_VERBOSE_VIP true

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
#define APRINTVIP(format, ...) \
  if (AGPU_VERBOSE_VIP)        \
  __android_log_print(ANDROID_LOG_INFO, "AGPU", format, ##__VA_ARGS__)
#else
#define APRINT(format, ...) printf(format, ##__VA_ARGS__)
#define AGPU_ERROR(format, ...) printf(format, ##__VA_ARGS__)
#endif

#define FUNC_PRINT(x) APRINT(#x "=%d in %s, %d \n", x, __func__, __LINE__);
#define FUNC_PRINT_ALL(x, type) \
  APRINT(#x "=" #type " %" #type " in %s, %d \n", x, __func__, __LINE__);

#define AGL_CHECK_ERROR_ENABLED

#ifdef AGL_CHECK_ERROR_ENABLED
#define AGL_CHECK_ERROR                                                   \
  {                                                                       \
    GLenum error = glGetError();                                          \
    if (GL_NO_ERROR != error) {                                           \
      APRINT("File:%s Line:%d F:%s\n", __FILE__, __LINE__, __FUNCTION__); \
      FUNC_PRINT_ALL(error, 0x);                                          \
    }                                                                     \
    assert(GL_NO_ERROR == error);                                         \
  }
#else
#define AGL_CHECK_ERROR
#endif
namespace agpu {

struct AResult {
  AResult()
      : cpu_kernel_repackO4I4_time(0.),
        gpu_shader_conv_time(0.),
        gpu_shader_hkernel_to_dtex_time(0.),
        gpu_shader_dtex_to_hchw_time(0.),
        gpu_shader_hchw_to_dtex_time(0.),
        gpu_shader_hchw_to_dc4hw_time(0.),
        gpu_shader_dc4hw_to_hchw_time(0.) {}

  double gpu_shader_conv_time;
  double cpu_kernel_repackO4I4_time;
  double gpu_shader_hkernel_to_dtex_time;
  double gpu_shader_dtex_to_hchw_time;
  double gpu_shader_hchw_to_dtex_time;
  double gpu_shader_hchw_to_dc4hw_time;
  double gpu_shader_dc4hw_to_hchw_time;

  double gpu_shader_total_time() {
    return gpu_shader_conv_time + gpu_shader_hkernel_to_dtex_time +
        gpu_shader_dtex_to_hchw_time + gpu_shader_hchw_to_dtex_time +
        gpu_shader_hchw_to_dc4hw_time + gpu_shader_dc4hw_to_hchw_time;
  }
};

AResult agpu_conv2d(
    const float* input,
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    const float* weights,
    uint32_t KC,
    uint32_t KH,
    uint32_t KW,
    const float* bias,
    uint32_t SY,
    uint32_t SX,
    uint32_t PY,
    uint32_t PX,
    uint32_t DY,
    uint32_t DX,
    uint32_t G,
    float* output);

#define CONV_DECL(fname)    \
  AResult fname(            \
      const float* input,   \
      uint32_t N,           \
      uint32_t C,           \
      uint32_t H,           \
      uint32_t W,           \
      const float* weights, \
      uint32_t OC,          \
      uint32_t KH,          \
      uint32_t KW,          \
      const float* bias,    \
      uint32_t SY,          \
      uint32_t SX,          \
      uint32_t PY,          \
      uint32_t PX,          \
      uint32_t DY,          \
      uint32_t DX,          \
      uint32_t G,           \
      float* output,        \
      int64_t mod = 0);

CONV_DECL(agpu_conv2d_);

// Tex
CONV_DECL(agpu_conv2d_tex_IKnc4hw); // benchId: [0, 10)
//---

// Buf NCHW
CONV_DECL(agpu_conv2d_buf_IKnchw_SIKOnc4hw_KrO4C4HW); // [10, 20)
CONV_DECL(agpu_conv2d_buf_IKnchw_SIKOnc4hw_KrO4HWC); // [20, 30)

CONV_DECL(agpu_conv2d_buf_IKnchw_SIKnc4hw_SOnchw); // [30, 40)
CONV_DECL(agpu_conv2d_buf_IKnchw_SKnc4hw); // [50, 60)
//---

// NHWC
CONV_DECL(agpu_conv2d_buf_Inhwc_Knchw);
CONV_DECL(agpu_conv2d_buf_IKnhwc);

CONV_DECL(agpu_conv2d_buf_IKnhwc_KrO4C4HW);
CONV_DECL(agpu_conv2d_buf_IKnhwc_KrO4HWC);
//---

// Depthwise
CONV_DECL(agpu_conv2dDW_buf_IKnchw);
CONV_DECL(agpu_conv2dDW_buf_Inhwc_Knchw);
CONV_DECL(agpu_conv2dDW_buf_IKnhwc);

// benchId: [40, 50)
CONV_DECL(agpu_conv2d_kernel_repack_);

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
