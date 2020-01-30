#include "agpu.h"

const char* agpu_test() {
    return "GPUGPUGPU";
}

void agpu_conv2d(
    const float* input,
    uint32_t input_n,
    uint32_t input_c,
    uint32_t input_h,
    uint32_t input_w,

    const float* kernel,
    uint32_t kernel_c,
    uint32_t kernel_h,
    uint32_t kernel_width,

    const float* bias,
    
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t input_padding_h,
    uint32_t input_padding_w,
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint32_t groups,
    float* output) {

      uint32_t totalWeightSize = kernel_c * input_c * kernel_h * kernel_w;
      glGenBuffers(





}
