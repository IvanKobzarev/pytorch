#include <unistd.h>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "agpu.h"
#include "shader.h"

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x) ROUND_UP((x), 4)

#define DEBUG_PRINT_TENSOR false

namespace agpu {

bool isVerbose() {
  return AGPU_VERBOSE;
}

void agpu_print(const char* m, const float* t, uint32_t rank, uint32_t* dims) {
  if (!isVerbose()) {
    return;
  }
  static const char* kFloatFormat = "%4.4f";
  std::cout << m;
  std::cout << " dims:(";
  for (uint32_t i = 0; i < rank; i++) {
    std::cout << dims[i] << " ";
  }
  std::cout << ")" << std::endl;

  if (rank == 0) {
    std::cout << *t;
  } else if (rank == 1 || rank == 2) {
    char fbuf[12];
    uint32_t rows = rank == 1 ? 1 : dims[0];
    uint32_t cols = rank == 1 ? dims[0] : dims[1];
    for (uint32_t i = 0; i < rows; ++i) {
      std::cout << "\n";
      for (uint32_t j = 0; j < cols; ++j) {
        sprintf(fbuf, kFloatFormat, t[i * cols + j]);
        std::cout << fbuf << " ";
      }
    }
  } else if (rank == 3) {
    uint32_t d0 = dims[0];
    uint32_t d12size = dims[1] * dims[2];
    for (uint32_t i = 0; i < d0; i++) {
      char s[80];
      sprintf(s, "[%d, *, *]", i);
      agpu_print(s, t + i * d12size, 2, dims + 1);
    }
  } else if (rank == 4) {
    uint32_t d0 = dims[0];
    uint32_t d1 = dims[1];
    uint32_t d23size = dims[2] * dims[3];
    for (uint32_t i = 0; i < d0; ++i) {
      for (uint32_t j = 0; j < d1; ++j) {
        char s[80];
        sprintf(s, "[%d, %d, *, *] offset:%d", i, j, (i * d0 + j) * d23size);
        agpu_print(s, t + i * d1 * d23size + j * d23size, 2, dims + 2);
      }
    }
  }
  std::cout << std::endl;
}

void agpu_print4d(
    const char* m,
    const float* data,
    uint32_t d0,
    uint32_t d1,
    uint32_t d2,
    uint32_t d3) {
  uint32_t dims[4] = {d0, d1, d2, d3};
  agpu_print(m, data, 4, dims);
}

#ifndef __ANDROID__
// not android

ares agpu_conv2d(
    const float* input,
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    const float* weights,
    uint32_t OC,
    uint32_t KH,
    uint32_t KW,
    const float* bias,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t input_padding_h,
    uint32_t input_padding_w,
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint32_t groups,
    float* output) {}

void agpu_add2t(
    const float* input0,
    const float* input1,
    uint32_t n,
    uint32_t c,
    uint32_t h,
    uint32_t w,
    float* output) {}

void agpu_threshold(
    const float* input,
    uint32_t n,
    uint32_t c,
    uint32_t h,
    uint32_t w,
    float threshold,
    float value,
    float* output) {}

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
    float* output) {}

void agpu_bench() {}
#else

class AGLShader;

class AGLContext {
 public:
  AGLContext() {
    if (!(eglGetCurrentContext() != EGL_NO_CONTEXT)) {
      display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
      if (display_ == EGL_NO_DISPLAY) {
        APRINT("eglGetDisplay error");
        isCreateError_ = true;
      }
      int majorVersion;
      int minorVersion;
      eglInitialize(display_, &majorVersion, &minorVersion);
      APRINT("GLContext version major:%d minor:%d", majorVersion, minorVersion);
      EGLint numConfigs;
      static const EGLint configAttribs[] = {EGL_SURFACE_TYPE,
                                             EGL_PBUFFER_BIT,
                                             EGL_RENDERABLE_TYPE,
                                             EGL_OPENGL_ES2_BIT,
                                             EGL_RED_SIZE,
                                             8,
                                             EGL_GREEN_SIZE,
                                             8,
                                             EGL_BLUE_SIZE,
                                             8,
                                             EGL_ALPHA_SIZE,
                                             8,
                                             EGL_NONE};

      EGLConfig surfaceConfig;
      if (!eglChooseConfig(
              display_, configAttribs, &surfaceConfig, 1, &numConfigs)) {
        eglMakeCurrent(
            display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglTerminate(display_);
        display_ = EGL_NO_DISPLAY;
        APRINT("eglChooseConfig error !!!");
        isCreateError_ = true;
      }

      static const EGLint contextAttribs[] = {
          EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
      context_ =
          eglCreateContext(display_, surfaceConfig, NULL, contextAttribs);
      static const EGLint surfaceAttribs[] = {
          EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
      surface_ =
          eglCreatePbufferSurface(display_, surfaceConfig, surfaceAttribs);
      eglMakeCurrent(display_, surface_, surface_, context_);
      eglBindAPI(EGL_OPENGL_ES_API);
      int major;
      glGetIntegerv(GL_MAJOR_VERSION, &major);
      int minor;
      glGetIntegerv(GL_MINOR_VERSION, &minor);
      APRINTVIP(
          "GLContext: GL_MAJOR_VERSION:%d GL_MINOR_VERSION:%d", major, minor);
      APRINTVIP(
          "GLContext: GL_SHADING_LANGUAGE_VERSION:%s",
          (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

      int maxShaderStorageBlockSize;
      glGetIntegerv(
          GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxShaderStorageBlockSize);
      APRINTVIP(
          "GLContext: GL_MAX_SHADER_STORAGE_BLOCK_SIZE:%d",
          maxShaderStorageBlockSize);

      GLint maxCompGroupSizeX, maxCompGroupSizeY, maxCompGroupSizeZ;
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maxCompGroupSizeX);
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &maxCompGroupSizeY);
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &maxCompGroupSizeZ);
      APRINTVIP(
          "GLContext: GL_MAX_COMPUTE_WORK_GROUP_SIZE: %d,%d,%d",
          maxCompGroupSizeX,
          maxCompGroupSizeY,
          maxCompGroupSizeZ);

      GLint maxCompGroupCountX, maxCompGroupCountY, maxCompGroupCountZ;
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &maxCompGroupCountX);
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &maxCompGroupCountY);
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &maxCompGroupCountZ);
      APRINTVIP(
          "GLContext: GL_MAX_COMPUTE_WORK_GROUP_COUNT: %d,%d,%d",
          maxCompGroupCountX,
          maxCompGroupCountY,
          maxCompGroupCountZ);

      GLint maxCompGroupInvocations;
      glGetIntegerv(
          GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &maxCompGroupInvocations);
      APRINTVIP(
          "GLContext: GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS: %d",
          maxCompGroupInvocations);

      GLint maxCompUniformBlocks;
      glGetIntegerv(GL_MAX_COMPUTE_UNIFORM_BLOCKS, &maxCompUniformBlocks);
      APRINTVIP(
          "GLContext: GL_MAX_COMPUTE_UNIFORM_BLOCKS: %d", maxCompUniformBlocks);

      GLint maxCompSharedMemorySize;
      glGetIntegerv(
          GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &maxCompSharedMemorySize);
      APRINTVIP(
          "GLContext: GL_MAX_COMPUTE_SHARED_MEMORY_SIZE: %d",
          maxCompSharedMemorySize);

      int extNum;
      glGetIntegerv(GL_NUM_EXTENSIONS, &extNum);
      for (int i = 0; i < extNum; i++) {
        const GLubyte* extName = glGetStringi(GL_EXTENSIONS, i);
        APRINTVIP("GLContext ext %3d: %s", i, extName);
      }

      if (major < 3) {
        isCreateError_ = true;
      }
    } else {
      context_ = EGL_NO_CONTEXT;
      APRINT("eglGetCurrentContext() != EGL_NO_CONTEXT");
      isCreateError_ = true;
    }
  }

  ~AGLContext() {
    if (display_ != EGL_NO_DISPLAY) {
      if (context_ != EGL_NO_CONTEXT) {
        eglDestroyContext(display_, context_);
        context_ = EGL_NO_CONTEXT;
      }
      if (surface_ != EGL_NO_SURFACE) {
        eglDestroySurface(display_, surface_);
        surface_ = EGL_NO_SURFACE;
      }
      eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
      eglTerminate(display_);
      display_ = EGL_NO_DISPLAY;
    }
    eglReleaseThread();
  }

  bool isCreateError() const {
    return isCreateError_;
  }

  std::map<std::string, std::shared_ptr<AGLShader>> shaderCache_;

 private:
  EGLContext context_;
  EGLDisplay display_;
  EGLSurface surface_;
  bool isCreateError_{false};
};

static std::unique_ptr<AGLContext> glContext;

void initAGLContextOnce() {
  static const int once = []() {
    APRINT("Creating GLContext...");
    glContext = std::make_unique<AGLContext>();
    if (!glContext) {
      APRINT("ERROR Failed to create GLContext");
      assert(false);
    }
    APRINT("GLContext created ok");
    return 0;
  }();
  ((void)once);
}

using buffer_size_t = GLsizeiptr;

class AGLSSBuffer {
 public:
  AGLSSBuffer(buffer_size_t size, GLenum type = GL_SHADER_STORAGE_BUFFER) {
    type_ = type;
    assert(size > 0);
    glGenBuffers(1, &id_);
    AGL_CHECK_ERROR;
    glBindBuffer(type_, id_);
    AGL_CHECK_ERROR;
    assert(id_ > 0);
    glBufferData(type_, size, NULL, GL_DYNAMIC_DRAW);
    AGL_CHECK_ERROR;
    size_ = size;
  }

  ~AGLSSBuffer() {
    glDeleteBuffers(1, &id_);
    AGL_CHECK_ERROR;
  }

  void* map(GLbitfield bufMask) {
    glBindBuffer(type_, id_);
    AGL_CHECK_ERROR;
    auto p = glMapBufferRange(type_, 0, size_, bufMask);
    AGL_CHECK_ERROR;
    return p;
  }

  void unmap() {
    glBindBuffer(type_, id_);
    glUnmapBuffer(type_);
    AGL_CHECK_ERROR;
  }

  buffer_size_t size() const {
    return size_;
  }

  void bindInProgram(int binding) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, id_);
    AGL_CHECK_ERROR;
  }

  std::unique_ptr<AGLSSBuffer> static from(
      const float* data,
      GLsizeiptr size,
      size_t sizeCopy) {
    auto buffer = std::make_unique<AGLSSBuffer>(size);
    float* bufferDataPtr =
        (float*)(buffer->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
    if (!bufferDataPtr) {
      assert(false);
    }
    memset(bufferDataPtr, 0, size);
    memcpy(bufferDataPtr, data, sizeCopy);
    buffer->unmap();
    return buffer;
  }

  std::unique_ptr<AGLSSBuffer> static from(
      const float* data,
      buffer_size_t size) {
    return from(data, size, size);
  }

  auto copyToHostVec() {
    uint32_t n = size_ / sizeof(float);
    std::vector<float> ret(n);
    float* retDataPtr = ret.data();
    std::cout << "copyToHostVec size:" << size_ << " n:" << n << std::endl;
    const float* bufferDataPtr = (const float*)map(GL_MAP_READ_BIT);
    if (!bufferDataPtr) {
      assert(false);
    }
    memset(retDataPtr, 0, n);
    memcpy(retDataPtr, bufferDataPtr, size_);
    unmap();
    return ret;
  }

  void copyToHost(float* outputDataPtr, size_t sizeCopy) {
    const float* bufferDataPtr = (const float*)map(GL_MAP_READ_BIT);
    if (!bufferDataPtr) {
      assert(false);
    }
    memcpy(outputDataPtr, bufferDataPtr, sizeCopy);
    unmap();
  }

 private:
  GLuint id_ = 0;
  buffer_size_t size_;
  GLenum type_;
}; // class AGLSSBuffer

class AGLTexture {
 public:
  AGLTexture(
      int w,
      int h,
      int d,
      GLenum texFormat,
      GLenum target = GL_TEXTURE_3D,
      bool HWC4 = true) {
    texFormat_ = texFormat;
    if (target == GL_TEXTURE_3D) {
      assert(w > 0 && h > 0 && d > 0);
      target_ = target;
      glGenTextures(1, &id_);
      AGL_CHECK_ERROR;
      glBindTexture(target_, id_);
      AGL_CHECK_ERROR;
      glTexParameteri(target_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      AGL_CHECK_ERROR;
      glTexParameteri(target_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      AGL_CHECK_ERROR;
      glTexParameteri(target_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      AGL_CHECK_ERROR;
      glTexParameteri(target_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      AGL_CHECK_ERROR;
      glTexParameteri(target_, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
      AGL_CHECK_ERROR;

      int realW = w;
      int realH = h;
      int realD = d;
      if (HWC4) {
        realD = UP_DIV(d, 4);
        realH = h;
        realW = w;
      }
      glTexStorage3D(target_, 1, texFormat_, realW, realH, realD);
      AGL_CHECK_ERROR;
    } else if (target == GL_TEXTURE_2D) {
      assert(w > 0 && h > 0);
      target_ = target;
      glGenTextures(1, &id_);
      AGL_CHECK_ERROR;
      glBindTexture(target_, id_);
      AGL_CHECK_ERROR;
      glTexParameteri(target_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      AGL_CHECK_ERROR;
      glTexParameteri(target_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      AGL_CHECK_ERROR;
      glTexParameteri(target_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      AGL_CHECK_ERROR;
      glTexParameteri(target_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      AGL_CHECK_ERROR;
      glTexParameteri(target_, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
      AGL_CHECK_ERROR;

      int realW = w;
      int realH = h;
      glTexStorage2D(target_, 1, texFormat_, realW, realH);
      AGL_CHECK_ERROR;
    }
  }

  ~AGLTexture() {
    glDeleteTextures(1, &id_);
    AGL_CHECK_ERROR;
  }
  unsigned int id() const {
    return id_;
  }

  void read(GLuint unit) {
    glBindImageTexture(unit, id_, 0, GL_TRUE, 0, GL_READ_ONLY, texFormat_);
    AGL_CHECK_ERROR;
  }

  void write(GLuint unit) {
    glBindImageTexture(unit, id_, 0, GL_TRUE, 0, GL_WRITE_ONLY, texFormat_);
    AGL_CHECK_ERROR;
  }

  void sample(GLuint unit, GLuint texId) {
    glActiveTexture(GL_TEXTURE0 + texId);
    glUniform1i(unit, texId);
    glBindTexture(target_, id_);
    AGL_CHECK_ERROR;
  }

  void bindInProgram(int programTexId, int binding) {
    glActiveTexture(GL_TEXTURE0 + programTexId);
    glUniform1i(binding, programTexId);
    glBindTexture(GL_TEXTURE_3D, id());
    AGL_CHECK_ERROR;
  }

 private:
  unsigned int id_;
  GLenum target_;
  GLenum texFormat_{GL_RGBA32F};
}; // class AGLTexture

enum AGLPrecision { highp = 0, mediump = 1, lowp = 2, count = 3 };

class AGLShader {
 public:
  AGLShader(const std::string& computeShader) {
    shaderId_ = glCreateShader(GL_COMPUTE_SHADER);
    AGL_CHECK_ERROR;
    const char* _ver[1];
    _ver[0] = computeShader.c_str();
    glShaderSource(shaderId_, 1, _ver, NULL);
    AGL_CHECK_ERROR;

    bool res = compileShader(shaderId_);
    assert(res);

    programId_ = glCreateProgram();
    AGL_CHECK_ERROR;
    glAttachShader(programId_, shaderId_);
    AGL_CHECK_ERROR;
    glLinkProgram(programId_);
    AGL_CHECK_ERROR;
    GLint linked;
    glGetProgramiv(programId_, GL_LINK_STATUS, &linked);
    if (!linked) {
      GLsizei len;
      glGetProgramiv(programId_, GL_INFO_LOG_LENGTH, &len);
      if (len <= 0) {
        glGetProgramInfoLog(programId_, 0, &len, NULL);
      }
      if (len > 0) {
        char* buffer = new char[len + 1];
        buffer[len] = '\0';
        glGetProgramInfoLog(programId_, len, NULL, buffer);
        FUNC_PRINT_ALL(buffer, s);
        delete[] buffer;
      }
    }
  }

  ~AGLShader() {
    glDeleteShader(shaderId_);
    glDeleteProgram(programId_);
    AGL_CHECK_ERROR;
  }

  unsigned int getProgramId() const {
    return programId_;
  }

  static std::string getHead(std::string imageFormat, std::string precision) {
    std::ostringstream headOs;
    headOs << "#version 310 es\n";
    headOs << "#define PRECISION " << precision << "\n";
    headOs << "precision PRECISION float;\n";
    headOs << "#define FORMAT " << imageFormat << "\n";
    return headOs.str();
  }

  void useProgram() {
    glUseProgram(programId_);
    AGL_CHECK_ERROR;
  }

  int getAttribLocation(const char* name) const {
    assert(NULL != name && 0 != programId_);
    return glGetAttribLocation(programId_, name);
  }

  int getUniformLocation(const char* name) const {
    assert(NULL != name && 0 != programId_);
    return glGetUniformLocation(programId_, name);
  }

 private:
  bool compileShader(GLuint s) {
    GLint status;
    glCompileShader(s);
    glGetShaderiv(s, GL_COMPILE_STATUS, &status);
    if (!status) {
      int len;
      glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
      if (0 >= len) {
        glGetShaderInfoLog(s, 0, &len, NULL);
      }
      char* buffer = new char[len + 1];
      glGetShaderInfoLog(s, len, NULL, buffer);
      buffer[len] = 0;
      FUNC_PRINT_ALL(buffer, s);
      delete[] buffer;
      return false;
    }
    return true;
  }

  unsigned int shaderId_ = 0;
  unsigned int programId_ = 0;
}; // class AGLShader

inline auto atime_now() {
  return std::chrono::high_resolution_clock::now();
}

inline double atime_duration(
    std::chrono::high_resolution_clock::time_point tp0,
    std::chrono::high_resolution_clock::time_point tp1) {
  std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(tp1 - tp0);
  return time_span.count();
}

inline double atime_duration_to_now(
    std::chrono::high_resolution_clock::time_point tp0) {
  auto tp1 = std::chrono::high_resolution_clock::now();
  return atime_duration(tp0, tp1);
}

GLenum getTexFormat() {
  return GL_RGBA32F;
}

std::string getImageFormat() {
  return "rgba32f";
}

static int64_t gBenchId = 0;
void setBenchId(int64_t bid) {
  gBenchId = bid;
}

static AGLPrecision gPrecision = highp;
void setPrecision(AGLPrecision p) {
  gPrecision = p;
}

void wait() {
#ifdef USE_GL_FINISH
  glFinish();
#else
  glFlush();
#endif
}

void compute(int dim0, int dim1, int dim2) {
  glDispatchCompute(dim0, dim1, dim2);
}

double computeStdTime(
    int dim0,
    int dim1,
    int dim2,
    const char* log,
    int compGroupSize0,
    int compGroupSize1,
    int compGroupSize2) {
  auto tp = atime_now();
  compute(dim0, dim1, dim2);
  glFinish();
  return atime_duration_to_now(tp);
}

double computeGLTime(
    int dim0,
    int dim1,
    int dim2,
    const char* log,
    int compGroupSize0,
    int compGroupSize1,
    int compGroupSize2) {
  GLint gpu_ts0 = 0;
  GLint gpu_ts1 = 0;
  double gpu_time_ts = 0;

  glGetIntegerv(GL_TIMESTAMP_EXT, &gpu_ts0);

  compute(dim0, dim1, dim2);

  glGetIntegerv(GL_TIMESTAMP_EXT, &gpu_ts1);

  gpu_time_ts = gpu_ts1 - gpu_ts0;

  APRINTVIP(
      "B{%ld}compute %16s(%3d,%3d,%3d)xCG(%3d,%3d,%3d) %.3fus",
      gBenchId,
      log,
      dim0,
      dim1,
      dim2,
      compGroupSize0,
      compGroupSize1,
      compGroupSize2,
      (gpu_time_ts / 1e3));

  return gpu_time_ts / 1e9;
}

// double computeGLTime2(int dim0, int dim1, int dim2, const char* log) {
//  //GLuint q;
//  //glGenQueries(1, &q);
//  //AGL_CHECK_ERROR;
//  //glBeginQuery(GL_TIME_ELAPSED_EXT, q);
//  GLint gpu_ts0 = 0;
//  GLint gpu_ts1 = 0;
//  auto tp = atime_now();
//  glGetIntegerv(GL_TIMESTAMP_EXT, &gpu_ts0);
//
//
//  compute(dim0, dim1, dim2);
//  //glFinish();
//
//
//  glGetIntegerv(GL_TIMESTAMP_EXT, &gpu_ts1);
//
//  double cpu_time = atime_duration_to_now(tp);
//  //glEndQuery(GL_TIME_ELAPSED_EXT);
//  //AGL_CHECK_ERROR;
//
//  //GLuint done = GL_FALSE;
//  //while (done == GL_FALSE) {
//  //  glGetQueryObjectuiv(q, GL_QUERY_RESULT_AVAILABLE_EXT, &done);
//  //}
//  AGL_CHECK_ERROR;
//
//  GLuint elapsed_time = 0;
//  //glGetQueryObjectuiv(q, GL_QUERY_RESULT_EXT, &elapsed_time);
//  //AGL_CHECK_ERROR;
//  //glDeleteQueries(1, &q);
//  //AGL_CHECK_ERROR;
//
//  double gpu_time = elapsed_time;
//  double gpu_time_ts = gpu_ts1 - gpu_ts0;
//
//  std::cout
//      << "compute " << log
//      << "(" << dim0 << "," << dim1 << "," << dim2 << ")"
//      << " cpu_t:" << (1e6 * cpu_time) << "us"
//      //<< " gpu_t:" << (gpu_time) << "ns "
//      << " gpu_t_ts:" << (gpu_time_ts/1e3) << "us"
//      << std::endl;
//
//  return gpu_time;
//}

typedef double (
    *fp_gComputeWithTime)(int, int, int, const char*, int, int, int);
static fp_gComputeWithTime gComputeWithTime = &computeGLTime;

std::string getPrecision() {
  static const char* precisionStr[AGLPrecision::count] = {
      "highp", "mediump", "lowp"};
  return precisionStr[gPrecision];
}

void printShaderCode(const std::string& s) {
  std::string token;
  std::istringstream tokenStream(s);
  int i = 0;
  while (std::getline(tokenStream, token, '\n')) {
    APRINT("%3d %s", i++, token.c_str());
  }
}

std::unique_ptr<AGLShader> createShader(
    const char* content,
    const std::vector<std::string>& prefix = {}) {
  std::ostringstream tc;
  tc << AGLShader::getHead(getImageFormat(), getPrecision());
  for (auto& s : prefix) {
    tc << s << "\n";
  }
  tc << content;

  auto shaderCode = tc.str();
  APRINT("createShader prefix length:%lu", prefix.size());
  printShaderCode(shaderCode);

  return std::make_unique<AGLShader>(tc.str());
}

std::shared_ptr<AGLShader> getShader(
    const std::string& key,
    const char* content,
    const std::vector<std::string>& prefix = {}) {
  initAGLContextOnce();

  std::ostringstream newKey;
  for (auto s : prefix) {
    newKey << s;
  }
  newKey << key;
  newKey << gBenchId;
  auto newKeyStr = newKey.str();

  auto it = glContext->shaderCache_.find(newKeyStr);
  if (it != glContext->shaderCache_.end()) {
    return it->second;
  }

  std::shared_ptr<AGLShader> shader{createShader(content, prefix)};
  glContext->shaderCache_.insert(std::make_pair(newKeyStr, shader));
  return shader;
}

void addCompGroupSizeDefines(
    std::vector<std::string>& header,
    int* compGroupSize,
    int compGroupSizeX,
    int compGroupSizeY,
    int compGroupSizeZ) {
  GLint maxCompGroupSizeX, maxCompGroupSizeY, maxCompGroupSizeZ;
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maxCompGroupSizeX);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &maxCompGroupSizeY);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &maxCompGroupSizeZ);

  compGroupSize[0] =
      compGroupSizeX < maxCompGroupSizeX ? compGroupSizeX : maxCompGroupSizeX;
  compGroupSize[1] =
      compGroupSizeY < maxCompGroupSizeY ? compGroupSizeY : maxCompGroupSizeY;
  compGroupSize[2] =
      compGroupSizeZ < maxCompGroupSizeZ ? compGroupSizeZ : maxCompGroupSizeZ;

  header.push_back(
      std::string{"#define WORKGROUP_X "} + std::to_string(compGroupSize[0]));
  header.push_back(
      std::string{"#define WORKGROUP_Y "} + std::to_string(compGroupSize[1]));
  header.push_back(
      std::string{"#define WORKGROUP_Z "} + std::to_string(compGroupSize[2]));
}

double deviceTex2hostCHW(
    GLuint texId,
    float* outputData,
    int d0,
    int d1,
    int d2,
    bool outputC4HW) {
  APRINT("deviceTex2hostCHW(%d %d %d align %d)", d0, d1, d2, outputC4HW);
  wait();
  auto d2_4 = UP_DIV(d2, 4);
  auto size = d2_4 * 4 * d0 * d1 * sizeof(float);
  auto buffer = std::make_unique<AGLSSBuffer>(size);
  auto program = outputC4HW
      ? getShader("image_to_nc4hw4_buffer_glsl", image_to_nc4hw4_buffer_glsl)
      : getShader("image_to_nchw_buffer_glsl", image_to_nchw_buffer_glsl);
  program->useProgram();

  glBindImageTexture(0, texId, 0, GL_TRUE, 0, GL_READ_ONLY, getTexFormat());
  AGL_CHECK_ERROR;
  buffer->bindInProgram(1);

  glUniform1i(2, d0);
  glUniform1i(3, d1);
  AGL_CHECK_ERROR;

  double shaderTime = gComputeWithTime(
      UP_DIV(d0, 8), UP_DIV(d1, 8), d2_4, "dTex2hCHW", 8, 8, 1);
  AGL_CHECK_ERROR;

  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  AGL_CHECK_ERROR;

  auto dOutputData = buffer->map(GL_MAP_READ_BIT);
  if (dOutputData) {
    if (outputC4HW) {
      ::memcpy(outputData, dOutputData, size);
    } else {
      ::memcpy(outputData, dOutputData, d0 * d1 * d2 * sizeof(float));
    }
  }
  buffer->unmap();
  return shaderTime;
}

auto hostCHW_to_deviceC4HWBuffer(
    const float* inputData,
    const int c,
    const int h,
    const int w,
    AResult& ares) {
  APRINT("hostCHW_to_deviceC4HWBuffer(c %d h %d w %d)", c, h, w);

  const int c_4 = UP_DIV(c, 4);
  buffer_size_t size = ROUND_UP(c, 4) * w * h * sizeof(float);
  auto src = AGLSSBuffer::from(inputData, size, c * h * w * sizeof(float));
  auto dst = std::make_unique<AGLSSBuffer>(size);

  int workGroupSize[3];
  std::vector<std::string> header;
  addCompGroupSizeDefines(header, workGroupSize, 8, 8, 4);

  auto shader = getShader(
      "nchw_buffer_to_nc4hw_buffer_glsl",
      nchw_buffer_to_nc4hw_buffer_glsl,
      header);
  shader->useProgram();
  AGL_CHECK_ERROR;
  dst->bindInProgram(0);
  AGL_CHECK_ERROR;
  src->bindInProgram(1);
  AGL_CHECK_ERROR;
  glUniform1i(2, w);
  glUniform1i(3, h);
  AGL_CHECK_ERROR;

  ares.gpu_shader_hchw_to_dc4hw_time = gComputeWithTime(
      UP_DIV(w, workGroupSize[0]),
      UP_DIV(h, workGroupSize[1]),
      UP_DIV(c, workGroupSize[2]),
      "hCHW2dC4HW",
      workGroupSize[0],
      workGroupSize[1],
      workGroupSize[2]);
  AGL_CHECK_ERROR;
  return dst;
}

void hostCHW2deviceTex(
    GLuint texId,
    const float* inputData,
    const int c,
    const int h,
    const int w,
    const bool inputC4HW,
    AResult& ares) {
  APRINT("hostCHW2deviceTex(c %d h %d w %d align %d)", c, h, w, inputC4HW);

  const int c_4 = UP_DIV(c, 4);
  GLsizeiptr size = ROUND_UP(c, 4) * w * h * sizeof(float);
  auto buffer = AGLSSBuffer::from(
      inputData, size, inputC4HW ? size : c * h * w * sizeof(float));

  auto shader = inputC4HW
      ? getShader("nc4hw4_buffer_to_image_glsl", nc4hw4_buffer_to_image_glsl)
      : getShader("nchw_buffer_to_image_glsl", nchw_buffer_to_image_glsl);
  shader->useProgram();

  glBindImageTexture(0, texId, 0, GL_TRUE, 0, GL_WRITE_ONLY, getTexFormat());
  AGL_CHECK_ERROR;

  buffer->bindInProgram(1);
  glUniform1i(2, w);
  glUniform1i(3, h);
  AGL_CHECK_ERROR;

  ares.gpu_shader_hchw_to_dtex_time =
      gComputeWithTime(UP_DIV(w, 8), UP_DIV(h, 8), c_4, "hCHW2dTex", 8, 8, 1);
  AGL_CHECK_ERROR;
}

auto kernelCHWBufferPackO4I4(
    const float* weights,
    const int OC,
    const int C,
    const int KH,
    const int KW,
    AResult& ares) {
  auto tp0 = atime_now();
  const uint32_t KBuffSize = ALIGN_UP4(OC) * ALIGN_UP4(C) * KH * KW;
  auto kernelBuff = std::make_unique<AGLSSBuffer>(sizeof(float) * KBuffSize);
  const int alignedKOCSize = UP_DIV(C, 4) * KW * KH * 16;
  float* kernelPtr = (float*)(kernelBuff->map(
      GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
  if (kernelPtr) {
    memset(kernelPtr, 0, sizeof(float) * KBuffSize);
    const float* src = weights;
    float* dst = kernelPtr;
    int ridx = 0;
    for (int oc = 0; oc < OC; ++oc) {
      int oc_4 = oc / 4;
      int oc_4_i = oc % 4;
      float* dst_oc = dst + oc_4 * alignedKOCSize;
      for (int ic = 0; ic < C; ++ic) {
        int ic_4 = ic / 4;
        int ic_4_i = ic % 4;
        float* dst_ic = dst_oc + ic_4 * KW * KH * 16;
        for (int ky = 0; ky < KH; ++ky) {
          float* dst_ky = dst_ic + ky * KW * 16;
          for (int kx = 0; kx < KW; ++kx) {
            float* dst_kx = dst_ky + kx * 16;
            dst_kx[4 * ic_4_i + oc_4_i] = src[ridx++];
          }
        }
      }
    }
  }
  kernelBuff->unmap();

  ares.cpu_kernel_repackO4I4_time = atime_duration_to_now(tp0);
  return kernelBuff;
}

AResult agpu_conv2d_buf_IKnchw_SIKOnc4hw(
    const float* input,
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    const float* weights,
    uint32_t OC,
    uint32_t KH,
    uint32_t KW,
    const float* bias,
    uint32_t SY,
    uint32_t SX,
    uint32_t PY,
    uint32_t PX,
    uint32_t DY,
    uint32_t DX,
    uint32_t groups,
    float* output,
    int64_t mod) {
  setBenchId(mod);
  setPrecision(highp);
  AResult ares;
  initAGLContextOnce();
  static const bool debugPrintTensors = DEBUG_PRINT_TENSOR;
  if (debugPrintTensors) {
    agpu_print4d("input:", input, N, C, H, W);
    agpu_print4d("kernel:", weights, OC, C, KH, KW);
    uint32_t bdims[1] = {OC};
    agpu_print("bias:", bias, 1, bdims);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;

  auto biasBuff = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);
  auto kernelBuff = kernelCHWBufferPackO4I4(weights, OC, C, KH, KW, ares);

  auto inputBuff = hostCHW_to_deviceC4HWBuffer(input, C, H, W, ares);

  if (debugPrintTensors) {
    wait();
    std::vector<float> inputBuffDebugVec = inputBuff->copyToHostVec();
    agpu_print4d(
        "input buffer nc4hw:\n", inputBuffDebugVec.data(), C_4, H, W, 4);
  }
  const int convOutC4BufferSize = sizeof(float) * 4 * OC_4 * OW * OH;
  auto convOutC4Buffer = std::make_unique<AGLSSBuffer>(convOutC4BufferSize);

  static const char* modeShaderKey[2] = {"conv_buf_IKnchw_SIKOnc4hw_glsl",
                                         "conv_buf_IKnchw_SIKOnc4hw_glsl"};
  static const char* modeShaderCode[2] = {conv_buf_IKnchw_SIKOnc4hw_glsl,
                                          conv_buf_IKnchw_SIKOnc4hw_glsl};

  {
    int workGroupSize[3];
    std::vector<std::string> header;
    addCompGroupSizeDefines(header, workGroupSize, 1, 1, OC_4);

    auto shader = getShader(modeShaderKey[mod], modeShaderCode[mod], header);

    shader->useProgram();
    AGL_CHECK_ERROR;

    convOutC4Buffer->bindInProgram(0);
    inputBuff->bindInProgram(1);
    kernelBuff->bindInProgram(2);
    biasBuff->bindInProgram(3);

    glUniform2i(4, PX, PY);
    glUniform2i(5, KW, KH);
    glUniform2i(6, SX, SY);
    glUniform2i(7, DX, DY);
    AGL_CHECK_ERROR;

    glUniform3i(8, OW, OH, OC_4);
    glUniform3i(9, W, H, C_4);
    AGL_CHECK_ERROR;

    ares.gpu_shader_conv_time = gComputeWithTime(
        UP_DIV(OW, 4 * workGroupSize[0]),
        UP_DIV(OH, workGroupSize[1]),
        UP_DIV(OC_4, workGroupSize[2]),
        modeShaderKey[mod],
        workGroupSize[0],
        workGroupSize[1],
        workGroupSize[2]);
    AGL_CHECK_ERROR;

    if (debugPrintTensors) {
      wait();
      std::vector<float> convOutC4BufferDebugVec =
          convOutC4Buffer->copyToHostVec();
      agpu_print4d(
          modeShaderKey[mod], convOutC4BufferDebugVec.data(), OC_4, OH, OW, 4);
    }
  }

  auto shaderKey = modeShaderKey[mod];

  const int outBufferSize = sizeof(float) * ALIGN_UP4(OC) * OH * OW;
  auto outBuffer = std::make_unique<AGLSSBuffer>(outBufferSize);
  {
    int workGroupSize[3];
    std::vector<std::string> header;
    addCompGroupSizeDefines(header, workGroupSize, 8, 8, OC_4);

    auto shader = getShader(
        "nc4hw_buffer_to_nchw_buffer_glsl",
        nc4hw_buffer_to_nchw_buffer_glsl,
        header);
    shader->useProgram();
    AGL_CHECK_ERROR;

    outBuffer->bindInProgram(0);
    convOutC4Buffer->bindInProgram(1);
    AGL_CHECK_ERROR;

    glUniform2i(2, OW, OH);
    AGL_CHECK_ERROR;

    ares.gpu_shader_dc4hw_to_hchw_time = gComputeWithTime(
        UP_DIV(OW, workGroupSize[0]),
        UP_DIV(OH, workGroupSize[1]),
        UP_DIV(OC_4, workGroupSize[2]),
        "dC4HW2hCHW",
        workGroupSize[0],
        workGroupSize[1],
        workGroupSize[2]);
    AGL_CHECK_ERROR;
  }

  outBuffer->copyToHost(output, sizeof(float) * OC * OH * OW);
  if (debugPrintTensors) {
    agpu_print4d(shaderKey, output, N, OC, OH, OW);
  }
  return ares;
}

AResult agpu_conv2d_buf_IKnchw_SIKnc4hw_SOnchw(
    const float* input,
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    const float* weights,
    uint32_t OC,
    uint32_t KH,
    uint32_t KW,
    const float* bias,
    uint32_t SY,
    uint32_t SX,
    uint32_t PY,
    uint32_t PX,
    uint32_t DY,
    uint32_t DX,
    uint32_t groups,
    float* output,
    int64_t mod) {
  setBenchId(mod);
  setPrecision(highp);
  initAGLContextOnce();
  AResult ares;
  static const bool debugPrintTensors = DEBUG_PRINT_TENSOR;
  if (debugPrintTensors) {
    agpu_print4d("input:", input, N, C, H, W);
    agpu_print4d("kernel:", weights, OC, C, KH, KW);
    uint32_t bdims[1] = {OC};
    agpu_print("bias:", bias, 1, bdims);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;

  auto biasBuff = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);

  auto kernelBuff = kernelCHWBufferPackO4I4(weights, OC, C, KH, KW, ares);
  auto inputBuff = hostCHW_to_deviceC4HWBuffer(input, C, H, W, ares);

  if (debugPrintTensors) {
    wait();
    std::vector<float> inputBuffDebugVec = inputBuff->copyToHostVec();
    agpu_print4d(
        "input buffer nc4hw:\n", inputBuffDebugVec.data(), C_4, H, W, 4);
  }

  const int outBufferSize = sizeof(float) * ALIGN_UP4(OC) * OH * OW;
  auto outBuffer = std::make_unique<AGLSSBuffer>(outBufferSize);

  static auto shaderKey = "conv_buf_IKnchw_SIKnc4hw_SOnchw_glsl";
  {
    int workGroupSize[3];
    std::vector<std::string> header;
    addCompGroupSizeDefines(header, workGroupSize, 1, 1, OC_4);
    auto shader =
        getShader(shaderKey, conv_buf_IKnchw_SIKnc4hw_SOnchw_glsl, header);

    shader->useProgram();
    AGL_CHECK_ERROR;

    outBuffer->bindInProgram(0);
    inputBuff->bindInProgram(1);
    kernelBuff->bindInProgram(2);
    biasBuff->bindInProgram(3);

    glUniform2i(4, PX, PY);
    glUniform2i(5, KW, KH);
    glUniform2i(6, SX, SY);
    glUniform2i(7, DX, DY);
    AGL_CHECK_ERROR;

    glUniform3i(8, OW, OH, OC_4);
    glUniform3i(9, W, H, C_4);
    AGL_CHECK_ERROR;

    ares.gpu_shader_conv_time = gComputeWithTime(
        UP_DIV(OW, 4 * workGroupSize[0]),
        UP_DIV(OH, workGroupSize[1]),
        UP_DIV(OC_4, workGroupSize[2]),
        shaderKey,
        workGroupSize[0],
        workGroupSize[1],
        workGroupSize[2]);
    AGL_CHECK_ERROR;

    if (debugPrintTensors) {
      wait();
      std::vector<float> outBufferDebugVec = outBuffer->copyToHostVec();
      agpu_print4d(shaderKey, outBufferDebugVec.data(), 1, OC, OH, OW);
    }
  }

  outBuffer->copyToHost(output, sizeof(float) * OC * OH * OW);
  if (debugPrintTensors) {
    agpu_print4d(shaderKey, output, N, OC, OH, OW);
  }
  return ares;
}

AResult agpu_conv2d_kernel_repack_(
    const float* input,
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    const float* weights,
    uint32_t OC,
    uint32_t KH,
    uint32_t KW,
    const float* bias,
    uint32_t SY,
    uint32_t SX,
    uint32_t PY,
    uint32_t PX,
    uint32_t DY,
    uint32_t DX,
    uint32_t groups,
    float* output,
    int64_t mod) {
  setBenchId(mod);
  setPrecision(highp);
  initAGLContextOnce();
  AResult ares;

  static const bool debugPrintTensors = DEBUG_PRINT_TENSOR;
  if (debugPrintTensors) {
    agpu_print4d("input:", input, N, C, H, W);
    agpu_print4d("kernel:", weights, OC, C, KH, KW);
    uint32_t bdims[1] = {OC};
    agpu_print("bias:", bias, 1, bdims);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int C_4 = UP_DIV(C, 4);

  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  APRINT("KHE KWE (%d,%d)", KHE, KWE);

  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;
  APRINT("OH OW (%d,%d)", OH, OW);

  auto biasBuff = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);
  auto kernelBuff = kernelCHWBufferPackO4I4(weights, OC, C, KH, KW, ares);

  return ares;
}

AResult agpu_conv2d_buf_IKnchw_SKnc4hw(
    const float* input,
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    const float* weights,
    uint32_t OC,
    uint32_t KH,
    uint32_t KW,
    const float* bias,
    uint32_t SY,
    uint32_t SX,
    uint32_t PY,
    uint32_t PX,
    uint32_t DY,
    uint32_t DX,
    uint32_t groups,
    float* output,
    int64_t mod) {
  setBenchId(mod);
  setPrecision(highp);
  initAGLContextOnce();
  AResult ares;
  static const bool debugPrintTensors = DEBUG_PRINT_TENSOR;
  if (debugPrintTensors) {
    agpu_print4d("input:", input, N, C, H, W);
    agpu_print4d("kernel:", weights, OC, C, KH, KW);
    uint32_t bdims[1] = {OC};
    agpu_print("bias:", bias, 1, bdims);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;
  auto biasBuff = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);

  auto kernelBuff = kernelCHWBufferPackO4I4(weights, OC, C, KH, KW, ares);

  auto inputBuff = AGLSSBuffer::from(input, sizeof(float) * C * H * W);
  if (debugPrintTensors) {
    std::vector<float> inputBuffDebugVec = inputBuff->copyToHostVec();
    agpu_print4d("input buffer nchw:\n", inputBuffDebugVec.data(), 1, C, H, W);
  }

  static const char* modeShaderKey[3] = {
      "conv_buf_IKnchw_SKnc4hw_glsl",
      "conv_buf_IKnchw_SKnc4hw_1_glsl",
      "conv_buf_IKnchw_SKnc4hw_2_glsl",
  };
  auto shaderKey = modeShaderKey[mod];
  static const char* modeShaderCode[3] = {
      conv_buf_IKnchw_SKnc4hw_glsl,
      conv_buf_IKnchw_SKnc4hw_1_glsl,
      conv_buf_IKnchw_SKnc4hw_2_glsl,
  };

  const int outBufferSize = sizeof(float) * ALIGN_UP4(OC) * OH * OW;
  auto outBuffer = std::make_unique<AGLSSBuffer>(outBufferSize);
  {
    int workGroupSize[3];
    std::vector<std::string> header;
    addCompGroupSizeDefines(header, workGroupSize, 1, 1, OC_4);

    auto shader = getShader(modeShaderKey[mod], modeShaderCode[mod], header);

    shader->useProgram();
    AGL_CHECK_ERROR;

    outBuffer->bindInProgram(0);
    inputBuff->bindInProgram(1);
    kernelBuff->bindInProgram(2);
    biasBuff->bindInProgram(3);

    glUniform2i(4, PX, PY);
    glUniform2i(5, KW, KH);
    glUniform2i(6, SX, SY);
    glUniform2i(7, DX, DY);
    glUniform3i(8, OW, OH, OC_4);
    glUniform3i(9, W, H, C_4);
    AGL_CHECK_ERROR;

    ares.gpu_shader_conv_time = gComputeWithTime(
        UP_DIV(OW, 4 * workGroupSize[0]),
        UP_DIV(OH, workGroupSize[1]),
        UP_DIV(OC_4, workGroupSize[2]),
        shaderKey,
        workGroupSize[0],
        workGroupSize[1],
        workGroupSize[2]);
    AGL_CHECK_ERROR;
  }

  outBuffer->copyToHost(output, sizeof(float) * OC * OH * OW);
  if (debugPrintTensors) {
    agpu_print4d(shaderKey, output, N, OC, OH, OW);
  }

  return ares;
}

AResult agpu_conv2d_buf_Inhwc_Knchw(
    const float* input,
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    const float* weights,
    uint32_t OC,
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
    float* output,
    int64_t mod) {
  setBenchId(mod);
  setPrecision(highp);
  initAGLContextOnce();
  AResult ares;
  static const bool debugPrintTensors = true; // DEBUG_PRINT_TENSOR;
  if (debugPrintTensors) {
    agpu_print4d("input:", input, N, H, W, C);
    agpu_print4d("kernel:", weights, OC, C, KH, KW);
    uint32_t bdims[1] = {OC};
    agpu_print("bias:", bias, 1, bdims);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;
  auto biasBuff = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);

  auto kernelBuff = kernelCHWBufferPackO4I4(weights, OC, C, KH, KW, ares);

  auto inputBuff = AGLSSBuffer::from(input, sizeof(float) * H * W * C);

  const char* modeShaderKey[1] = {
      "conv_buf_Inhwc_Knchw_glsl",
  };
  auto shaderKey = modeShaderKey[0];
  const char* modeShaderCode[1] = {
      conv_buf_Inhwc_Knchw_glsl,
  };
  const int outBufferSize = sizeof(float) * OH * OW * OC;

  auto outBuffer = std::make_unique<AGLSSBuffer>(outBufferSize);
  {
    int workGroupSize[3];
    std::vector<std::string> header;
    addCompGroupSizeDefines(header, workGroupSize, 1, 1, OC_4);

    auto shader = getShader(modeShaderKey[mod], modeShaderCode[mod], header);

    shader->useProgram();
    AGL_CHECK_ERROR;

    outBuffer->bindInProgram(0);
    inputBuff->bindInProgram(1);
    kernelBuff->bindInProgram(2);
    biasBuff->bindInProgram(3);

    glUniform2i(4, PX, PY);
    glUniform2i(5, KW, KH);
    glUniform2i(6, SX, SY);
    glUniform2i(7, DX, DY);
    glUniform3i(8, OW, OH, OC);
    glUniform3i(9, W, H, C);
    AGL_CHECK_ERROR;

    ares.gpu_shader_conv_time = gComputeWithTime(
        UP_DIV(OW, 4 * workGroupSize[0]),
        UP_DIV(OH, workGroupSize[1]),
        UP_DIV(OC_4, workGroupSize[2]),
        shaderKey,
        workGroupSize[0],
        workGroupSize[1],
        workGroupSize[2]);

    AGL_CHECK_ERROR;
  }

  outBuffer->copyToHost(output, sizeof(float) * OH * OW * OC);
  if (debugPrintTensors) {
    agpu_print4d(shaderKey, output, N, OH, OW, OC);
  }

  return ares;
}

AResult agpu_conv2d_buf_IKnhwc(
    const float* input,
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    const float* weights,
    uint32_t OC,
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
    float* output,
    int64_t mod) {
  assert(false); // TODO: Not implemented
  setBenchId(mod);
  setPrecision(highp);
  initAGLContextOnce();
  AResult ares;
  static const bool debugPrintTensors = true; // DEBUG_PRINT_TENSOR;
  if (debugPrintTensors) {
    agpu_print4d("input:", input, N, H, W, C);
    agpu_print4d("kernel:", weights, OC, KH, KW, C);
    uint32_t bdims[1] = {OC};
    agpu_print("bias:", bias, 1, bdims);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;
  auto biasBuff = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);

  auto kernelBuff = kernelCHWBufferPackO4I4(weights, OC, C, KH, KW, ares);

  auto inputBuff = AGLSSBuffer::from(input, sizeof(float) * H * W * C);

  const char* modeShaderKey[1] = {
      "conv_buf_IKnhwc_glsl",
  };
  auto shaderKey = modeShaderKey[0];
  const char* modeShaderCode[1] = {
      conv_buf_IKnhwc_glsl,
  };
  const int outBufferSize = sizeof(float) * OH * OW * OC;

  auto outBuffer = std::make_unique<AGLSSBuffer>(outBufferSize);
  {
    int workGroupSize[3];
    std::vector<std::string> header;
    addCompGroupSizeDefines(header, workGroupSize, 1, 1, OC_4);

    auto shader = getShader(modeShaderKey[mod], modeShaderCode[mod], header);

    shader->useProgram();
    AGL_CHECK_ERROR;

    outBuffer->bindInProgram(0);
    inputBuff->bindInProgram(1);
    kernelBuff->bindInProgram(2);
    biasBuff->bindInProgram(3);

    glUniform2i(4, PX, PY);
    glUniform2i(5, KW, KH);
    glUniform2i(6, SX, SY);
    glUniform2i(7, DX, DY);
    glUniform3i(8, OW, OH, OC);
    glUniform3i(9, W, H, C);
    AGL_CHECK_ERROR;

    ares.gpu_shader_conv_time = gComputeWithTime(
        UP_DIV(OW, 4 * workGroupSize[0]),
        UP_DIV(OH, workGroupSize[1]),
        UP_DIV(OC_4, workGroupSize[2]),
        shaderKey,
        workGroupSize[0],
        workGroupSize[1],
        workGroupSize[2]);

    AGL_CHECK_ERROR;
  }

  outBuffer->copyToHost(output, sizeof(float) * OH * OW * OC);
  if (debugPrintTensors) {
    agpu_print4d(shaderKey, output, N, OH, OW, OC);
  }

  return ares;
}

// Depthwise
AResult agpu_conv2dDW_buf_IKnchw(
    const float* input,
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    const float* weights,
    uint32_t OC,
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
    float* output,
    int64_t mod) {
  assert(G == C);

  setBenchId(mod);
  setPrecision(highp);
  initAGLContextOnce();
  AResult ares;
  static const bool debugPrintTensors = DEBUG_PRINT_TENSOR;
  if (debugPrintTensors) {
    agpu_print4d("input:", input, N, C, H, W);
    agpu_print4d("kernel:", weights, OC, C, KH, KW);
    uint32_t bdims[1] = {OC};
    agpu_print("bias:", bias, 1, bdims);
  }
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;
  const uint32_t GC = G * C;
  const uint32_t GOC = G * OC;
  const uint32_t OC_4 = UP_DIV(OC, 4);

  auto biasBuff = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);
  auto kernelBuff = AGLSSBuffer::from(
      weights,
      sizeof(float) * ALIGN_UP4(GOC) * KH * KW,
      sizeof(float) * GOC * KH * KW);
  auto inputBuff = AGLSSBuffer::from(
      input, sizeof(float) * ALIGN_UP4(GC) * H * W, sizeof(float) * GC * H * W);

  if (debugPrintTensors) {
    std::vector<float> inputBuffDebugVec = inputBuff->copyToHostVec();
    agpu_print4d("input buffer nchw:\n", inputBuffDebugVec.data(), 1, C, H, W);
  }

  const char* modeShaderKey[1] = {
      "convDW_buf_IKnchw_glsl",
  };
  auto shaderKey = modeShaderKey[0];
  const char* modeShaderCode[1] = {
      convDW_buf_IKnchw_glsl,
  };
  const int outBufferSize = sizeof(float) * ALIGN_UP4(GOC) * OH * OW;
  auto outBuffer = std::make_unique<AGLSSBuffer>(outBufferSize);
  {
    int workGroupSize[3];
    std::vector<std::string> header;
    addCompGroupSizeDefines(header, workGroupSize, 8, 8, 1);

    auto shader = getShader(modeShaderKey[mod], modeShaderCode[mod], header);

    shader->useProgram();
    AGL_CHECK_ERROR;

    outBuffer->bindInProgram(0);
    inputBuff->bindInProgram(1);
    kernelBuff->bindInProgram(2);
    biasBuff->bindInProgram(3);

    glUniform2i(4, PX, PY);
    glUniform2i(5, KW, KH);
    glUniform2i(6, SX, SY);
    glUniform2i(7, DX, DY);
    glUniform3i(8, OW, OH, OC_4);
    glUniform3i(9, W, H, C);
    AGL_CHECK_ERROR;

    ares.gpu_shader_conv_time = gComputeWithTime(
        UP_DIV(OW, workGroupSize[0]),
        UP_DIV(OH, workGroupSize[1]),
        UP_DIV(OC_4, workGroupSize[2]),
        shaderKey,
        workGroupSize[0],
        workGroupSize[1],
        workGroupSize[2]);

    AGL_CHECK_ERROR;
  }

  outBuffer->copyToHost(output, sizeof(float) * OC * OH * OW);
  if (debugPrintTensors) {
    agpu_print4d(shaderKey, output, N, OC, OH, OW);
  }

  return ares;
}

AResult agpu_conv2dDW_buf_Inhwc_Knchw(
    const float* input,
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    const float* weights,
    uint32_t OC,
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
    float* output,
    int64_t mod) {
  setBenchId(mod);
  setPrecision(highp);
  initAGLContextOnce();
  AResult ares;
  static const bool debugPrintTensors = true; // DEBUG_PRINT_TENSOR;
  if (debugPrintTensors) {
    agpu_print4d("input:", input, N, H, W, C);
    agpu_print4d("kernel:", weights, OC, C, KH, KW);
    uint32_t bdims[1] = {OC};
    agpu_print("bias:", bias, 1, bdims);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;

  const uint32_t GC = G * C;
  const uint32_t GOC = G * OC;

  auto biasBuff = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);
  auto kernelBuff = AGLSSBuffer::from(
      weights,
      sizeof(float) * ALIGN_UP4(GOC) * KH * KW,
      sizeof(float) * GOC * KH * KW);
  auto inputBuff = AGLSSBuffer::from(input, sizeof(float) * H * W * C);

  const char* modeShaderKey[1] = {
      "convDW_buf_Inhwc_Knchw_glsl",
  };
  const char* modeShaderCode[1] = {
      convDW_buf_Inhwc_Knchw_glsl,
  };
  auto shaderKey = modeShaderKey[0];
  const int outBufferSize = sizeof(float) * OH * OW * OC;

  auto outBuffer = std::make_unique<AGLSSBuffer>(outBufferSize);
  {
    int workGroupSize[3];
    std::vector<std::string> header;
    addCompGroupSizeDefines(header, workGroupSize, 8, 8, 1);

    auto shader = getShader(modeShaderKey[mod], modeShaderCode[mod], header);

    shader->useProgram();
    AGL_CHECK_ERROR;

    outBuffer->bindInProgram(0);
    inputBuff->bindInProgram(1);
    kernelBuff->bindInProgram(2);
    biasBuff->bindInProgram(3);

    glUniform2i(4, PX, PY);
    glUniform2i(5, KW, KH);
    glUniform2i(6, SX, SY);
    glUniform2i(7, DX, DY);
    glUniform3i(8, OW, OH, OC);
    glUniform3i(9, W, H, C);
    AGL_CHECK_ERROR;

    ares.gpu_shader_conv_time = gComputeWithTime(
        UP_DIV(OW, workGroupSize[0]),
        UP_DIV(OH, workGroupSize[1]),
        UP_DIV(OC_4, workGroupSize[2]),
        shaderKey,
        workGroupSize[0],
        workGroupSize[1],
        workGroupSize[2]);

    AGL_CHECK_ERROR;
  }

  outBuffer->copyToHost(output, sizeof(float) * OH * OW * OC);
  if (debugPrintTensors) {
    agpu_print4d(shaderKey, output, N, OH, OW, OC);
  }
  return ares;
}

AResult agpu_conv2dDW_buf_IKnhwc(
    const float* input,
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    const float* weights,
    uint32_t OC,
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
    float* output,
    int64_t mod) {
  setBenchId(mod);
  setPrecision(highp);
  initAGLContextOnce();
  AResult ares;
  static const bool debugPrintTensors = true; // DEBUG_PRINT_TENSOR;
  if (debugPrintTensors) {
    agpu_print4d("input:", input, N, H, W, C);
    agpu_print4d("kernel:", weights, OC, KH, KW, C);
    uint32_t bdims[1] = {OC};
    agpu_print("bias:", bias, 1, bdims);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;
  auto biasBuff = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);

  auto kernelBuff = kernelCHWBufferPackO4I4(weights, OC, C, KH, KW, ares);

  auto inputBuff = AGLSSBuffer::from(input, sizeof(float) * H * W * C);

  const char* modeShaderKey[1] = {
      "convDW_buf_IKnhwc_glsl",
  };
  const char* modeShaderCode[1] = {
      convDW_buf_IKnhwc_glsl,
  };
  auto shaderKey = modeShaderKey[0];

  const int outBufferSize = sizeof(float) * OH * OW * OC;
  auto outBuffer = std::make_unique<AGLSSBuffer>(outBufferSize);
  {
    int workGroupSize[3];
    std::vector<std::string> header;
    addCompGroupSizeDefines(header, workGroupSize, 1, 1, OC_4);

    auto shader = getShader(modeShaderKey[mod], modeShaderCode[mod], header);

    shader->useProgram();
    AGL_CHECK_ERROR;

    outBuffer->bindInProgram(0);
    inputBuff->bindInProgram(1);
    kernelBuff->bindInProgram(2);
    biasBuff->bindInProgram(3);

    glUniform2i(4, PX, PY);
    glUniform2i(5, KW, KH);
    glUniform2i(6, SX, SY);
    glUniform2i(7, DX, DY);
    AGL_CHECK_ERROR;

    glUniform3i(8, OW, OH, OC);
    glUniform3i(9, W, H, C);
    AGL_CHECK_ERROR;

    ares.gpu_shader_conv_time = gComputeWithTime(
        UP_DIV(OW, workGroupSize[0]),
        UP_DIV(OH, workGroupSize[1]),
        UP_DIV(OC_4, workGroupSize[2]),
        modeShaderCode[mod],
        workGroupSize[0],
        workGroupSize[1],
        workGroupSize[2]);

    AGL_CHECK_ERROR;
  }

  outBuffer->copyToHost(output, sizeof(float) * OH * OW * OC);
  if (debugPrintTensors) {
    agpu_print4d(shaderKey, output, N, OH, OW, OC);
  }

  return ares;
}

AResult agpu_conv2d(
    const float* input,
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    const float* weights,
    uint32_t OC,
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
    float* output) {
  return agpu_conv2d_(
      input,
      N,
      C,
      H,
      W,
      weights,
      OC,
      KH,
      KW,
      bias,
      SY,
      SX,
      PY,
      PX,
      DY,
      DX,
      G,
      output,
      0);
}

AResult agpu_conv2d_(
    const float* input,
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    const float* weights,
    uint32_t OC,
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
    float* output,
    int64_t mod) {
  return agpu_conv2d_tex_IKnc4hw(
      input,
      N,
      C,
      H,
      W,
      weights,
      OC,
      KH,
      KW,
      bias,
      SY,
      SX,
      PY,
      PX,
      DY,
      DX,
      G,
      output,
      mod);
}

AResult agpu_conv2d_tex_IKnc4hw(
    const float* input,
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    const float* weights,
    uint32_t OC,
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
    float* output,
    int64_t mod) {
  setBenchId(mod);
  setPrecision(highp);
  initAGLContextOnce();
  AResult ares;

  if (mod == 2) {
    setPrecision(lowp);
  } else if (mod == 1) {
    setPrecision(mediump);
  } else {
    setPrecision(highp);
  }

  static const bool debugPrintTensors = DEBUG_PRINT_TENSOR;
  static const int unit = 4;
  const int unit2 = unit * unit;
  const int OC_4 = UP_DIV(OC, 4);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;

  if (debugPrintTensors) {
    agpu_print4d("input:", input, N, C, H, W);
    agpu_print4d("kernel:", weights, OC, C, KH, KW);
    uint32_t bdims[1] = {OC};
    agpu_print("bias:", bias, 1, bdims);
  }

  auto biasBuff = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);
  auto kernelBuff = kernelCHWBufferPackO4I4(weights, OC, C, KH, KW, ares);

  APRINT("kernelTex(%d, %d, %d)", C_4 * unit, OC_4, KH * KW);
  auto kernelTex = std::make_unique<AGLTexture>(
      C_4 * unit, OC_4, KH * KW, getTexFormat(), GL_TEXTURE_3D, false);

  auto kernel2ImageShader =
      getShader("kernel2image_adreno_glsl", kernel2image_adreno_glsl);

  kernel2ImageShader->useProgram();
  // binding kernel2Image {
  glBindImageTexture(
      0, kernelTex->id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, getTexFormat());
  kernelBuff->bindInProgram(2);
  glUniform1i(3, KW * KH);
  glUniform1i(4, C_4);
  AGL_CHECK_ERROR;
  // binding kernel2Image }

  ares.gpu_shader_hkernel_to_dtex_time =
      gComputeWithTime(C_4, OC_4, KH * KW, "hK2dTex", 1, 1, 1);
  AGL_CHECK_ERROR;
  // kernelTex done

  auto inputTex = std::make_unique<AGLTexture>(
      W, H, C_4, getTexFormat(), GL_TEXTURE_3D, false);
  hostCHW2deviceTex(inputTex->id(), input, C, H, W, false /* isC4HW */, ares);

  int compGroupSize[3];
  std::vector<std::string> header;
  addCompGroupSizeDefines(header, compGroupSize, 1, 1, OC_4);

  auto shaderKey = "conv_tex_IKnc4hw_glsl";
  auto convProgram = getShader(shaderKey, conv_tex_IKnc4hw_glsl, header);

  auto outputTex = std::make_unique<AGLTexture>(
      W, H, OC_4, getTexFormat(), GL_TEXTURE_3D, false);

  convProgram->useProgram();

  // binding convolution {
  glBindImageTexture(
      0, outputTex->id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, getTexFormat());

  inputTex->bindInProgram(0, 1);
  kernelTex->bindInProgram(1, 2);
  biasBuff->bindInProgram(3);
  glUniform2i(4, PX, PY);
  glUniform2i(5, KW, KH);
  glUniform2i(6, SX, SY);
  glUniform2i(7, DX, DY);
  glUniform1i(8, unit);
  AGL_CHECK_ERROR;

  glUniform3i(10, OW, OH, OC_4);
  glUniform3i(11, W, H, C_4);
  AGL_CHECK_ERROR;
  // binding convolution }

  ares.gpu_shader_conv_time = gComputeWithTime(
      UP_DIV(OW, unit * compGroupSize[0]),
      UP_DIV(OH, compGroupSize[1]),
      UP_DIV(OC_4, compGroupSize[2]),
      "conv_tex",
      compGroupSize[0],
      compGroupSize[1],
      compGroupSize[2]);
  AGL_CHECK_ERROR;

  ares.gpu_shader_dtex_to_hchw_time =
      deviceTex2hostCHW(outputTex->id(), output, OW, OH, OC, false /* C4 */);

  if (debugPrintTensors) {
    agpu_print4d(shaderKey, output, N, OC, OH, OW);
  }
  return ares;
}

// region not_convolution
void agpu_add2t(
    const float* input0,
    const float* input1,
    uint32_t n,
    uint32_t c,
    uint32_t h,
    uint32_t w,
    float* output) {
  agpu_print4d("agpu_add2t input0:\n", input0, n, c, h, w);
  agpu_print4d("agpu_add2t input1:\n", input1, n, c, h, w);

  initAGLContextOnce();
  AResult ares;

  int c_4 = UP_DIV(c, 4);

  auto input0Tex = std::make_unique<AGLTexture>(
      w, h, c_4, getTexFormat(), GL_TEXTURE_3D, false);
  hostCHW2deviceTex(input0Tex->id(), input0, c, h, w, false /* align */, ares);
  auto input1Tex = std::make_unique<AGLTexture>(
      w, h, c_4, getTexFormat(), GL_TEXTURE_3D, false);
  hostCHW2deviceTex(input1Tex->id(), input1, c, h, w, false /* align */, ares);

  auto outputTex = std::make_unique<AGLTexture>(
      w, h, c_4, getTexFormat(), GL_TEXTURE_3D, false);

  int compGroupSize[3];
  std::vector<std::string> prefix;
  addCompGroupSizeDefines(prefix, compGroupSize, 8, 8, 1);

  auto binAddProgram = getShader("binary_add_glsl", binary_add_glsl, prefix);
  binAddProgram->useProgram();
  // binding binary_add_glsl {
  glBindImageTexture(
      0, outputTex->id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, getTexFormat());
  input0Tex->bindInProgram(0, 1);
  input1Tex->bindInProgram(1, 2);
  glUniform4i(3, w, h, c_4, 1);
  AGL_CHECK_ERROR;
  // binding binary_add_glsl }

  compute(
      UP_DIV(w, compGroupSize[0]),
      UP_DIV(h, compGroupSize[1]),
      UP_DIV(c_4, compGroupSize[2]));
  deviceTex2hostCHW(outputTex->id(), output, w, h, c, false /* align */);

  agpu_print4d("agpu_add2t output:\n", output, n, c, h, w);
}

void agpu_threshold(
    const float* input,
    uint32_t n,
    uint32_t c,
    uint32_t h,
    uint32_t w,
    float threshold,
    float value,
    float* output) {
  agpu_print4d("agpu_threshold input:\n", input, n, c, h, w);

  initAGLContextOnce();
  AResult ares;

  const int c_4 = UP_DIV(c, 4);
  auto inputTex = std::make_unique<AGLTexture>(
      w, h, c_4, getTexFormat(), GL_TEXTURE_3D, false);

  hostCHW2deviceTex(inputTex->id(), input, c, h, w, false /* align */, ares);

  auto outputTex = std::make_unique<AGLTexture>(
      w, h, c_4, getTexFormat(), GL_TEXTURE_3D, false);

  int compGroupSize[3];
  std::vector<std::string> prefix;
  addCompGroupSizeDefines(prefix, compGroupSize, 8, 8, 1);

  auto program = getShader("threshold_glsl", threshold_glsl, prefix);
  program->useProgram();
  // binding {
  glBindImageTexture(
      0, outputTex->id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, getTexFormat());
  inputTex->bindInProgram(0, 1);

  glUniform4i(2, w, h, c_4, 1);
  glUniform1f(3, threshold);
  glUniform1f(4, value);
  AGL_CHECK_ERROR;
  // binding }
  compute(
      UP_DIV(w, compGroupSize[0]),
      UP_DIV(h, compGroupSize[1]),
      UP_DIV(c_4, compGroupSize[2]));

  deviceTex2hostCHW(outputTex->id(), output, w, h, c, false /* align */);

  agpu_print4d("agpu_threshold output:\n", input, n, c, h, w);
}

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
    float* output) {
  agpu_print4d("agpu_batch_norm input:\n", input, n, c, h, w);
  initAGLContextOnce();
  AResult ares;

  int c_4 = UP_DIV(c, 4);

  auto inputTex = std::make_unique<AGLTexture>(
      w, h, c_4, getTexFormat(), GL_TEXTURE_3D, false);
  hostCHW2deviceTex(inputTex->id(), input, c, h, w, false /* align */, ares);

  auto outputTex = std::make_unique<AGLTexture>(
      w, h, c_4, getTexFormat(), GL_TEXTURE_3D, false);

  GLsizeiptr bufferSize = sizeof(float) * ALIGN_UP4(c);
  auto weightBuffer = AGLSSBuffer::from(weight, bufferSize);
  auto biasBuff = AGLSSBuffer::from(bias, bufferSize);
  auto meanBuffer = AGLSSBuffer::from(mean, bufferSize);
  auto varianceBuffer = AGLSSBuffer::from(variance, bufferSize);

  // computation work group
  int compGroupSize[3];
  std::vector<std::string> prefix;
  addCompGroupSizeDefines(prefix, compGroupSize, 8, 8, 1);

  auto program = getShader("normalization_glsl", normalization_glsl, prefix);
  program->useProgram();

  // binding normalization_glsl {
  glBindImageTexture(
      0, outputTex->id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, getTexFormat());

  inputTex->bindInProgram(0, 1 /* binding */);
  weightBuffer->bindInProgram(3);
  biasBuff->bindInProgram(4);
  meanBuffer->bindInProgram(5);
  varianceBuffer->bindInProgram(6);

  glUniform1f(7, eps);
  AGL_CHECK_ERROR;
  // binding normalization_glsl }

  compute(
      UP_DIV(w, compGroupSize[0]),
      UP_DIV(h, compGroupSize[1]),
      UP_DIV(c_4, compGroupSize[2]));

  deviceTex2hostCHW(outputTex->id(), output, w, h, c, false /* align */);
  agpu_print4d("agpu_batch_norm output:\n", output, n, c, h, w);
}
// endregion notconvolution
#endif
} // namespace agpu
