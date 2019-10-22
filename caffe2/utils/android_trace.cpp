#include <caffe2/utils/android_trace.h>
#include <android/log.h>

#define ALOGI(...)                                                             \
    __android_log_print(ANDROID_LOG_INFO, "pytorch-ATrace", __VA_ARGS__)

namespace caffe2 {

bool ATrace::is_initialized_ = false;

ATrace::ATrace(const char* name) {
  if (!ATrace::is_initialized_) {
    ATrace::init();
    ATrace::is_initialized_ = true;
  }

  ALOGI("ATrace_beginSection(%s)", name);
  ATrace_beginSection(name);
}

ATrace::~ATrace() {
  ALOGI("ATrace_endSection()");
  ATrace_endSection();
}

void ATrace::init() {

  void *lib = dlopen("libandroid.so", RTLD_NOW || RTLD_LOCAL);

  if (lib != NULL) {
    ATrace_beginSection = reinterpret_cast<fp_ATrace_beginSection>(
        dlsym(lib, "ATrace_beginSection"));
    ATrace_endSection = reinterpret_cast<fp_ATrace_endSection>(
        dlsym(lib, "ATrace_endSection"));
  }
}

} // namespace caffe2
