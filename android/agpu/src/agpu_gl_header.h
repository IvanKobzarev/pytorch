#pragma once

#ifdef __ANDROID__

#include <EGL/egl.h>
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
