#pragma once
#include <stdint.h>

//#include <GLES2/gl2.h>
//#include <GLES2/gl2ext.h>
//#include <GLES3/gl31.h>

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

    float* output
);
