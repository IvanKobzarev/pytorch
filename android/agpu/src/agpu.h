#pragma once

#define AGPU_VERBOSE true
#define AGPU_VERBOSE_VIP true
// TODO:? why does not work GL time #define AGPU_USE_GL_TIME
//#define AGPU_USE_GL_TIME

#include <stdint.h>
#include <stdio.h>
#include <string>

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

std::string getGLInfo();

struct AResult {
  AResult()
      : cpu_kernel_CHW_repack_O4C4HW_time(0.),
        cpu_kernel_CHW_repack_O4HWC_time(0.),

        cpu_kernel_HWC_repack_O4C4HW_time(0.),
        cpu_kernel_HWC_repack_O4HWC_time(0.),

        gpu_shader_conv_time(0.),
        gpu_shader_hkernel_to_dtex_time(0.),
        gpu_shader_dtex_to_hchw_time(0.),
        gpu_shader_hchw_to_dtex_time(0.),
        gpu_shader_hchw_to_dc4hw_time(0.),
        gpu_shader_dc4hw_to_hchw_time(0.) {}

  double cpu_kernel_CHW_repack_O4C4HW_time;
  double cpu_kernel_CHW_repack_O4HWC_time;
  double cpu_kernel_HWC_repack_O4C4HW_time;
  double cpu_kernel_HWC_repack_O4HWC_time;

  double gpu_shader_conv_time;

  double gpu_shader_hkernel_to_dtex_time;
  double gpu_shader_dtex_to_hchw_time;
  double gpu_shader_hchw_to_dtex_time;
  double gpu_shader_hchw_to_dc4hw_time;
  double gpu_shader_dc4hw_to_hchw_time;

  bool isInvalid() {
    return gpu_shader_conv_time < 0 || gpu_shader_hkernel_to_dtex_time < 0 ||
        gpu_shader_dtex_to_hchw_time < 0 || gpu_shader_hchw_to_dtex_time < 0 ||
        gpu_shader_hchw_to_dc4hw_time < 0 || gpu_shader_dc4hw_to_hchw_time < 0;
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

// inline void agpu_conv2d(
//    const float* input,
//    uint32_t N,
//    uint32_t C,
//    uint32_t H,
//    uint32_t W,
//    const float* weights,
//    uint32_t KC,
//    uint32_t KH,
//    uint32_t KW,
//    const float* bias,
//    uint32_t SY,
//    uint32_t SX,
//    uint32_t PY,
//    uint32_t PX,
//    uint32_t DY,
//    uint32_t DX,
//    uint32_t G,
//    float* output){
//  return agpu_conv2d(
//        input,
//        N,
//        C,
//        H,
//        W,
//        weights,
//        KC,
//        KH,
//        KW,
//        bias,
//        SY,
//        SX,
//        PY,
//        PX,
//        DY,
//        DX,
//        G,
//        output,
//        0);
//}

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

#define CONV_DECL_MOD(modfname, basefname) \
  inline AResult modfname(                 \
      const float* input,                  \
      uint32_t N,                          \
      uint32_t C,                          \
      uint32_t H,                          \
      uint32_t W,                          \
      const float* weights,                \
      uint32_t OC,                         \
      uint32_t KH,                         \
      uint32_t KW,                         \
      const float* bias,                   \
      uint32_t SY,                         \
      uint32_t SX,                         \
      uint32_t PY,                         \
      uint32_t PX,                         \
      uint32_t DY,                         \
      uint32_t DX,                         \
      uint32_t G,                          \
      float* output,                       \
      int64_t mod = 0) {                   \
    return basefname(                      \
        input,                             \
        N,                                 \
        C,                                 \
        H,                                 \
        W,                                 \
        weights,                           \
        OC,                                \
        KH,                                \
        KW,                                \
        bias,                              \
        SY,                                \
        SX,                                \
        PY,                                \
        PX,                                \
        DY,                                \
        DX,                                \
        G,                                 \
        output,                            \
        mod);                              \
  }

CONV_DECL(conv_);

// Tex
CONV_DECL(conv_tex_IKnc4hw);
//---

// Buf NCHW
CONV_DECL(conv_buf_IKnchw_KrO4C4HW);
CONV_DECL(conv_buf_IKnchw_KrO4HWC);

CONV_DECL(conv_buf_IKnchw_SIKOnc4hw_KrO4C4HW);
CONV_DECL(conv_buf_IKnchw_SIKOnc4hw_KrO4HWC);

CONV_DECL(conv_buf_IKnchw_SIKnc4hw_SOnchw);

CONV_DECL(conv_buf_IKnchw_SKnc4hw_KrO4C4HW);

// CONV_DECL_MOD(
//    conv_buf_IKnchw_SKnc4hw_KrO4C4HW_1,
//    conv_buf_IKnchw_SKnc4hw_KrO4C4HW);
//---

// NHWC
CONV_DECL(conv_buf_Inhwc_Knchw_KrO4C4HW);
CONV_DECL(conv_buf_IKnhwc);

CONV_DECL(conv_buf_IKnhwc_KrO4C4HW);
CONV_DECL(conv_buf_IKnhwc_KrO4HWC);
//---

// Depthwise
CONV_DECL(convDW_buf_IKnchw);
CONV_DECL(convDW_buf_Inhwc_Knchw);
CONV_DECL(convDW_buf_IKnhwc);

CONV_DECL(conv2d_kernel_repack_);

void agpu_addmm(
    const float* m1,
    uint32_t m1dim,
    uint32_t* m1sizes,
    const float* m2,
    uint32_t m2dim,
    uint32_t* m2sizes,
    float beta,
    float alpha,
    const float* t,
    uint32_t tdim,
    uint32_t* tsizes,

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
