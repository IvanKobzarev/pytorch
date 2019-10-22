#pragma once

#include <android/trace.h>
#include <dlfcn.h>

#define ATRACE_NAME(name) ATrace ___tracer(name)

#define ATRACE_CALL() ATRACE_NAME(__FUNCTION__)

namespace caffe2 {

typedef void *(*fp_ATrace_beginSection) (const char* sectionName);
typedef void *(*fp_ATrace_endSection) (void);

static void *(*ATrace_beginSection)(const char *sectionName);
static void *(*ATrace_endSection)(void);

class ATrace {
public:
  ATrace(const char *name);
  ~ATrace();

private:
  static void init();
  static bool is_initialized_;
};

} // namespace caffe2
