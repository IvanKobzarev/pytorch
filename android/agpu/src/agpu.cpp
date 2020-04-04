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
#include "agpu_glsl.h"
#include "streamline_annotate.h"

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x) ROUND_UP((x), 4)

#define DEBUG_PRINT_TENSOR true

namespace agpu {

std::string gGLInfo;
std::string getGLInfo() {
  return gGLInfo;
}

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
        sprintf(s, "[%d, %d, *, *]", i, j);
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

void agpu_print1d(const char* m, const float* data, uint32_t d0) {
  uint32_t dims[1] = {d0};
  agpu_print(m, data, 1, dims);
}

void agpu_print2d(const char* m, const float* data, uint32_t d0, uint32_t d1) {
  uint32_t dims[2] = {d0, d1};
  agpu_print(m, data, 2, dims);
}

void agpu_print3d(const char* m, const float* data, uint32_t d0, uint32_t d1, uint32_t d2) {
  uint32_t dims[3] = {d0, d1, d2};
  agpu_print(m, data, 3, dims);
}

void agpu_print_NCHW(
    const float* input,
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    const float* weights,
    uint32_t OC,
    uint32_t KH,
    uint32_t KW,
    const float* bias) {
  agpu_print4d("input:", input, N, C, H, W);
  agpu_print4d("kernel:", weights, OC, C, KH, KW);
  agpu_print1d("bias:", bias, OC);
}

void agpu_print_NHWC(
    const float* input,
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    const float* weights,
    uint32_t OC,
    uint32_t KH,
    uint32_t KW,
    const float* bias) {
  agpu_print4d("input:", input, N, H, W, C);
  agpu_print4d("kernel:", weights, OC, KH, KW, C);
  agpu_print1d("bias:", bias, OC);
}

#ifndef __ANDROID__
// not android

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
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t input_padding_h,
    uint32_t input_padding_w,
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint32_t groups,
    float* output) {}

void add2t(
    const float* input0,
    const float* input1,
    uint32_t n,
    uint32_t c,
    uint32_t h,
    uint32_t w,
    float* output) {}

void threshold(
    const float* input,
    uint32_t n,
    uint32_t c,
    uint32_t h,
    uint32_t w,
    float threshold,
    float value,
    float* output) {}

void batch_norm(
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

void bench() {}
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

      std::string vendor{(const char*)glGetString(GL_VENDOR)};
      std::string renderer{(const char*)glGetString(GL_RENDERER)};
      APRINTVIP("GLContext: GL_VENDOR:%s", vendor.c_str());
      APRINTVIP("GLContext: GL_RENDERER:%s", renderer.c_str());

      std::string s;
      s.append(vendor);
      s.append(" ");
      s.append(renderer);
      gGLInfo = s;
      APRINTVIP("GLContext gGLInfo: %s", gGLInfo.c_str());

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
      glTexParameteri(target_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(target_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(target_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(target_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexParameteri(target_, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
      AGL_CHECK_ERROR;
      int realW = w;
      int realH = h;
      int realD = d;
      if (HWC4) {
        realH = h;
        realW = w;
        realD = UP_DIV(d, 4);
      }
      glTexStorage3D(target_, 1 /* level */, texFormat_, realW, realH, realD);
      AGL_CHECK_ERROR;
    } else if (target == GL_TEXTURE_2D) {
      assert(w > 0 && h > 0);
      target_ = target;
      glGenTextures(1, &id_);
      AGL_CHECK_ERROR;
      glBindTexture(target_, id_);
      AGL_CHECK_ERROR;
      glTexParameteri(target_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(target_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(target_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(target_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexParameteri(target_, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
      AGL_CHECK_ERROR;
      glTexStorage2D(target_, 1 /* level */, texFormat_, w, h);
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
    glBindImageTexture(unit, id_, 0 /* level */, GL_TRUE /* layered */, 0 /* layer */, GL_READ_ONLY, texFormat_);
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
    std::cout << "bindInProgram(progTexId:" << programTexId
        << ", binding:" << binding << ")" << std::endl;

    glActiveTexture(GL_TEXTURE0 + programTexId);
    AGL_CHECK_ERROR;
    glUniform1i(binding, programTexId);
    AGL_CHECK_ERROR;
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

void compute(GLuint dim0, GLuint dim1, GLuint dim2) {
  glDispatchCompute(dim0, dim1, dim2);
}

double computeStdTime(
    GLuint dim0,
    GLuint dim1,
    GLuint dim2,
    const char* log,
    int compGroupSize0,
    int compGroupSize1,
    int compGroupSize2) {
  glFlush();
  glFinish();

  ANNOTATE(log);
  ANNOTATE_COLOR(ANNOTATE_PURPLE, log);

  auto tp = atime_now();
  compute(dim0, dim1, dim2);
  glFinish();
  auto ret = atime_duration_to_now(tp);

  APRINTVIP(
      "B{%ld}compute %16s(%3d,%3d,%3d)xCG(%3d,%3d,%3d) cpuTime:%.6fs",
      gBenchId,
      log,
      dim0,
      dim1,
      dim2,
      compGroupSize0,
      compGroupSize1,
      compGroupSize2,
      ret);

  auto s = std::string{log} + std::to_string(ret);
  ANNOTATE_MARKER_COLOR_STR(ANNOTATE_YELLOW, s.c_str());

  ANNOTATE_END();
  return ret;
}

template <typename T>
void logPfngl(T pfngl) {}

double computeGLTime(
    GLuint dim0,
    GLuint dim1,
    GLuint dim2,
    const char* log,
    int compGroupSize0,
    int compGroupSize1,
    int compGroupSize2) {

  ANNOTATE(log);
  ANNOTATE_COLOR(ANNOTATE_PURPLE, log);

  static auto _glGenQueriesEXT =
      (PFNGLGENQUERIESEXTPROC)eglGetProcAddress("glGenQueriesEXT");
  static auto _glDeleteQueriesEXT =
      (PFNGLDELETEQUERIESEXTPROC)eglGetProcAddress("glDeleteQueriesEXT");
  static auto _glIsQueryEXT =
      (PFNGLISQUERYEXTPROC)eglGetProcAddress("glIsQueryEXT");
  static auto _glBeginQueryEXT =
      (PFNGLBEGINQUERYEXTPROC)eglGetProcAddress("glBeginQueryEXT");
  static auto _glEndQueryEXT =
      (PFNGLENDQUERYEXTPROC)eglGetProcAddress("glEndQueryEXT");
  static auto _glQueryCounterEXT =
      (PFNGLQUERYCOUNTEREXTPROC)eglGetProcAddress("glQueryCounterEXT");
  static auto _glGetQueryivEXT =
      (PFNGLGETQUERYIVEXTPROC)eglGetProcAddress("glGetQueryivEXT");
  static auto _glGetQueryObjectivEXT =
      (PFNGLGETQUERYOBJECTIVEXTPROC)eglGetProcAddress("glGetQueryObjectivEXT");
  static auto _glGetQueryObjectuivEXT =
      (PFNGLGETQUERYOBJECTUIVEXTPROC)eglGetProcAddress(
          "glGetQueryObjectuivEXT");
  static auto _glGetQueryObjecti64vEXT =
      (PFNGLGETQUERYOBJECTI64VEXTPROC)eglGetProcAddress(
          "glGetQueryObjecti64vEXT");
  static auto _glGetQueryObjectui64vEXT =
      (PFNGLGETQUERYOBJECTUI64VEXTPROC)eglGetProcAddress(
          "glGetQueryObjectui64vEXT");

  glFlush();
  glFinish();

  GLint gpu_ts0 = 0;
  GLint gpu_ts1 = 0;

  double gpu_time_ts = -1;
  double gpu_time_q = -1;

  GLuint q[2];
  GLint disjointOccurred = 0;

  _glGenQueriesEXT(2, q);
  AGL_CHECK_ERROR;

  static int N = 0;
  static int disjointN = 0;
  glGetIntegerv(GL_GPU_DISJOINT_EXT, &disjointOccurred);
  AGL_CHECK_ERROR;

  glGetIntegerv(GL_TIMESTAMP_EXT, &gpu_ts0);
  AGL_CHECK_ERROR;

  _glQueryCounterEXT(q[0], GL_TIMESTAMP_EXT);
  AGL_CHECK_ERROR;

  compute(dim0, dim1, dim2);

  _glQueryCounterEXT(q[1], GL_TIMESTAMP_EXT);
  AGL_CHECK_ERROR;

  glGetIntegerv(GL_TIMESTAMP_EXT, &gpu_ts1);
  AGL_CHECK_ERROR;

  GLuint done = GL_FALSE;
  while (done == GL_FALSE) {
    glGetQueryObjectuiv(q[1], GL_QUERY_RESULT_AVAILABLE_EXT, &done);
  }
  AGL_CHECK_ERROR;

  glGetIntegerv(GL_GPU_DISJOINT_EXT, &disjointOccurred);

  N++;
  if (disjointOccurred) {
    disjointN++;
  }
  APRINTVIP(
      "GL_GPU_DISJOINT=%d ratio:%.3f(%5d/%5d)",
      disjointOccurred,
      ((float)disjointN / N),
      disjointN,
      N);

  GLuint64 timeQStart;
  GLuint64 timeQStop;

  _glGetQueryObjectui64vEXT(q[0], GL_QUERY_RESULT, &timeQStart);
  _glGetQueryObjectui64vEXT(q[1], GL_QUERY_RESULT, &timeQStop);
  AGL_CHECK_ERROR;

  GLuint64 timeQ = timeQStop - timeQStart;

  gpu_time_q = timeQ;
  gpu_time_ts = gpu_ts1 - gpu_ts0;

  _glDeleteQueriesEXT(2, q);
  AGL_CHECK_ERROR;

  APRINTVIP(
      "B{%ld}compute %16s(%3d,%3d,%3d)xCG(%3d,%3d,%3d) glTimeTs:%9.0f glTimeQ:%9.0f",
      gBenchId,
      log,
      dim0,
      dim1,
      dim2,
      compGroupSize0,
      compGroupSize1,
      compGroupSize2,
      gpu_time_ts,
      gpu_time_q);
  auto s = std::string{log} + std::to_string(gpu_time_q);
  ANNOTATE_MARKER_COLOR_STR(ANNOTATE_YELLOW, s.c_str());
  ANNOTATE_END();
  return disjointOccurred ? -1.f : (gpu_time_q / 1e9);
}

typedef double (
    *fp_gComputeWithTime)(GLuint, GLuint, GLuint, const char*, int, int, int);

#ifdef AGPU_USE_GL_TIME
static fp_gComputeWithTime gComputeWithTime = &computeGLTime;
#else
static fp_gComputeWithTime gComputeWithTime = &computeStdTime;
#endif

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
  if (AGPU_VERBOSE) {
    printShaderCode(shaderCode);
  }

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
  if (AGPU_VERBOSE) {
    APRINT("getShader %s", key.c_str());
  }

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
  static GLint maxCompGroupSizeX, maxCompGroupSizeY, maxCompGroupSizeZ,
      maxCompGroupInvocations;
  static const int once = []() {
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maxCompGroupSizeX);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &maxCompGroupSizeY);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &maxCompGroupSizeZ);
    glGetIntegerv(
        GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &maxCompGroupInvocations);
    return 0;
  }();
  ((void)once);

  compGroupSize[0] =
      compGroupSizeX < maxCompGroupSizeX ? compGroupSizeX : maxCompGroupSizeX;
  compGroupSize[1] =
      compGroupSizeY < maxCompGroupSizeY ? compGroupSizeY : maxCompGroupSizeY;
  compGroupSize[2] =
      compGroupSizeZ < maxCompGroupSizeZ ? compGroupSizeZ : maxCompGroupSizeZ;

  const int compGroupInvocations =
      compGroupSize[0] * compGroupSize[1] * compGroupSize[2];
  if (compGroupInvocations > maxCompGroupInvocations) {
    int oldCompGroupSizeZ = compGroupSize[2];
    compGroupSize[2] =
        maxCompGroupInvocations / (compGroupSize[0] * compGroupSize[1]);
    APRINTVIP(
        "compGroupSize(%3d, %3d, %3d) compGroupInvocations:%4d > GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS:%4d => changeZto (%3d, %3d, %3d)",
        compGroupSize[0],
        compGroupSize[1],
        oldCompGroupSizeZ,
        compGroupInvocations,
        maxCompGroupInvocations,
        compGroupSize[0],
        compGroupSize[1],
        compGroupSize[2]);
  }

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
    int d2) {
  auto d2_4 = UP_DIV(d2, 4);
  auto size = d2_4 * 4 * d0 * d1 * sizeof(float);
  auto buffer = std::make_unique<AGLSSBuffer>(size);
  auto program = getShader("tex_to_nchw_buf_glsl", tex_to_nchw_buf_glsl);
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
    ::memcpy(outputData, dOutputData, d0 * d1 * d2 * sizeof(float));
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
  const int c_4 = UP_DIV(c, 4);
  buffer_size_t size = ROUND_UP(c, 4) * w * h * sizeof(float);
  auto src = AGLSSBuffer::from(inputData, size, c * h * w * sizeof(float));
  auto dst = std::make_unique<AGLSSBuffer>(size);

  int workGroupSize[3];
  std::vector<std::string> header;
  addCompGroupSizeDefines(header, workGroupSize, 8, 8, 4);

  auto shader = getShader(
      "nchw_buf_to_nc4hw_buf_glsl", nchw_buf_to_nc4hw_buf_glsl, header);
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

void hostCHW_to_deviceTex(
    GLuint texId,
    const float* inputData,
    const int C,
    const int H,
    const int W,
    AResult& ares) {
  const int C_4 = UP_DIV(C, 4);
  GLsizeiptr size = ROUND_UP(C, 4) * W * H * sizeof(float);
  std::cout << "AGPU hostCHW_to_deviceTex(C:"<< C
      << " H:" << H << " W:" << W << ")" << std::endl;

  std::cout << "AGPU size:" << size << std::endl;
  auto buffer = AGLSSBuffer::from(inputData, size, C * H * W * sizeof(float));

  auto shader = getShader("nchw_buf_to_tex_glsl", nchw_buf_to_tex_glsl);
  shader->useProgram();

  glBindImageTexture(0, texId, 0, GL_TRUE, 0, GL_WRITE_ONLY, getTexFormat());
  AGL_CHECK_ERROR;

  buffer->bindInProgram(1);
  glUniform1i(2, W);
  glUniform1i(3, H);
  AGL_CHECK_ERROR;

  ares.gpu_shader_hchw_to_dtex_time =
      gComputeWithTime(UP_DIV(W, 8), UP_DIV(H, 8), C_4, "hCHW2dTex", 8, 8, 1);
  AGL_CHECK_ERROR;
}

void hostHWC_to_deviceTex(
    GLuint texId,
    const float* inputData,
    const int C,
    const int H,
    const int W,
    AResult& ares) {
  const int C_4 = UP_DIV(C, 4);
  GLsizeiptr size = ROUND_UP(C, 4) * W * H * sizeof(float);
  auto buffer = AGLSSBuffer::from(inputData, size, C * H * W * sizeof(float));

  auto shader = getShader("nhwc_buf_to_tex_glsl", nhwc_buf_to_tex_glsl);
  shader->useProgram();

  glBindImageTexture(0, texId, 0, GL_TRUE, 0, GL_WRITE_ONLY, getTexFormat());
  AGL_CHECK_ERROR;

  buffer->bindInProgram(1);
  glUniform4i(2, W, H, C, C_4);
  AGL_CHECK_ERROR;

  //TODO: uncomment
  //ares.gpu_shader_hhwc_to_dtex_time =
      gComputeWithTime(
        UP_DIV(W, 8),
        UP_DIV(H, 8),
        C_4,
        "hHWC2dTex",
        8, 8, 1);
  AGL_CHECK_ERROR;
}

// Kernel repack CHW -> ... {
auto kernelNCHW_OCHW_repack_O4C4HWi4o4(
    const float* weights,
    const int OC,
    const int C,
    const int KH,
    const int KW,
    AResult& ares) {
  auto tp0 = atime_now();
  const uint32_t kBufSizeNumel = ALIGN_UP4(OC) * ALIGN_UP4(C) * KH * KW;
  auto kernelBuf = std::make_unique<AGLSSBuffer>(sizeof(float) * kBufSizeNumel);
  const int oc_4SizeNumel = UP_DIV(C, 4) * KW * KH * 16;
  float* kernelPtr =
      (float*)(kernelBuf->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
  if (kernelPtr) {
    memset(kernelPtr, 0, sizeof(float) * kBufSizeNumel);
    const float* src = weights;
    float* dst = kernelPtr;
    int ridx = 0;
    for (int oc = 0; oc < OC; ++oc) {
      int oc_4 = oc / 4;
      int oc_4_i = oc % 4;
      float* dst_oc = dst + oc_4 * oc_4SizeNumel;
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
  kernelBuf->unmap();

  ares.cpu_kernel_CHW_repack_O4C4HW_time = atime_duration_to_now(tp0);
  return kernelBuf;
}

auto kernelNCHW_OCHW_repack_O4HWC(
    const float* weights,
    const int OC,
    const int C,
    const int KH,
    const int KW,
    AResult& ares) {
  auto tp0 = atime_now();
  const uint32_t Cau4 = ALIGN_UP4(C);
  const uint32_t C_4 = UP_DIV(C, 4);
  const uint32_t kBufSizeNumel = ALIGN_UP4(OC) * Cau4 * KH * KW;

  auto kernelBuf = std::make_unique<AGLSSBuffer>(sizeof(float) * kBufSizeNumel);
  const int oc_4SizeNumel = KW * KH * Cau4 * 4;

  float* kernelPtr =
      (float*)(kernelBuf->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));

  if (kernelPtr) {
    memset(kernelPtr, 0, sizeof(float) * kBufSizeNumel);
    const float* src = weights;
    float* dst = kernelPtr;
    int ridx = 0;
    for (int oc = 0; oc < OC; ++oc) {
      int oc_4 = oc / 4;
      int oc_4_i = oc % 4;
      float* dst_oc = dst + oc_4 * oc_4SizeNumel;
      for (int ic = 0; ic < C; ++ic) {
        for (int ky = 0; ky < KH; ++ky) {
          float* dst_oc_ky = dst_oc + ky * KW * Cau4 * 4;
          for (int kx = 0; kx < KW; ++kx) {
            dst_oc_ky[kx * Cau4 * 4 + ic * 4 + oc_4_i] = src[ridx++];
          }
        }
      }
    }
  }
  kernelBuf->unmap();

  ares.cpu_kernel_CHW_repack_O4HWC_time = atime_duration_to_now(tp0);
  return kernelBuf;
}

// } Kernel repack CHW -> ...

// Kernel repack HWC -> ...
auto kernelNHWC_OHWC_repack_O4C4HWi4o4(
    const float* weights,
    const int OC,
    const int C,
    const int KH,
    const int KW,
    AResult& ares) {
  auto tp0 = atime_now();
  const uint32_t kBufSizeNumel = ALIGN_UP4(OC) * ALIGN_UP4(C) * KH * KW;
  auto kernelBuf = std::make_unique<AGLSSBuffer>(sizeof(float) * kBufSizeNumel);
  const int oc_4SizeNumel = UP_DIV(C, 4) * KW * KH * 16;
  float* kernelPtr =
      (float*)(kernelBuf->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
  if (kernelPtr) {
    memset(kernelPtr, 0, sizeof(float) * kBufSizeNumel);
    const float* src = weights;
    float* dst = kernelPtr;

    int ridx = 0;
    for (int oc = 0; oc < OC; ++oc) {
      int oc_4 = oc / 4;
      int oc_4_i = oc % 4;
      float* dst_oc = dst + oc_4 * oc_4SizeNumel;
      for (int ky = 0; ky < KH; ++ky) {
        float* dst_ky = dst_oc + ky * KW * 16;
        for (int kx = 0; kx < KW; ++kx) {
          float* dst_kx = dst_ky + kx * 16;
          for (int ic = 0; ic < C; ++ic) {
            int ic_4 = ic / 4;
            int ic_4_i = ic % 4;
            dst_kx[ic_4 * (16 * KH * KW) + (4 * ic_4_i + oc_4_i)] = src[ridx++];
          }
        }
      }
    }
  }
  kernelBuf->unmap();

  ares.cpu_kernel_HWC_repack_O4C4HW_time = atime_duration_to_now(tp0);
  return kernelBuf;
}

auto kernelNHWC_OHWC_repack_O4HWC(
    const float* weights,
    const int OC,
    const int C,
    const int KH,
    const int KW,
    AResult& ares) {
  auto tp0 = atime_now();
  const uint32_t CAU4 = ALIGN_UP4(C);
  const uint32_t C_4 = UP_DIV(C, 4);
  const uint32_t KBufSizeNumel = ALIGN_UP4(OC) * CAU4 * KH * KW;

  auto kernelBuf = std::make_unique<AGLSSBuffer>(sizeof(float) * KBufSizeNumel);
  const int oc_4SizeNumel = KH * KW * CAU4 * 4;
  float* kernelPtr =
      (float*)(kernelBuf->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
  if (kernelPtr) {
    memset(kernelPtr, 0, sizeof(float) * KBufSizeNumel);
    const float* src = weights;
    float* dst = kernelPtr;

    int ridx = 0;
    for (int oc = 0; oc < OC; ++oc) {
      int oc_4 = oc / 4;
      int oc_4_i = oc % 4;
      float* dst_oc_4 = dst + oc_4 * oc_4SizeNumel;
      for (int ky = 0; ky < KH; ++ky) {
        for (int kx = 0; kx < KW; ++kx) {
          float* dst_kxy = dst_oc_4 + (ky * KW + kx) * CAU4 * 4;
          for (int ic = 0; ic < C; ++ic) {
            dst_kxy[ic * 4 + oc_4_i] = src[ridx++];
          }
        }
      }
    }
  }
  kernelBuf->unmap();

  ares.cpu_kernel_HWC_repack_O4HWC_time = atime_duration_to_now(tp0);
  return kernelBuf;
}
// Kernel repack HWC -> ...

AResult conv_buf_IKnchw_SIKOnc4hw_KrO4C4HW(
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
    agpu_print_NCHW(input, N, C, H, W, weights, OC, KH, KW, bias);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;

  auto biasBuf = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);
  auto kernelBuf =
      kernelNCHW_OCHW_repack_O4C4HWi4o4(weights, OC, C, KH, KW, ares);
  auto inputBuf = hostCHW_to_deviceC4HWBuffer(input, C, H, W, ares);
  auto convOutC4Buf =
      std::make_unique<AGLSSBuffer>(sizeof(float) * 4 * OC_4 * OW * OH);

  static const char* modeShaderKey[1] = {
      "conv_buf_IKnchw_SIKOnc4hw_KrO4C4HW_glsl",
  };
  static const char* modeShaderCode[1] = {
      conv_buf_IKnchw_SIKOnc4hw_KrO4C4HW_glsl,
  };
  {
    int workGroupSize[3];
    std::vector<std::string> header;
    addCompGroupSizeDefines(header, workGroupSize, 1, 1, OC_4);

    auto shader = getShader(modeShaderKey[mod], modeShaderCode[mod], header);

    shader->useProgram();
    AGL_CHECK_ERROR;

    convOutC4Buf->bindInProgram(0);
    inputBuf->bindInProgram(1);
    kernelBuf->bindInProgram(2);
    biasBuf->bindInProgram(3);

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
        modeShaderKey[mod],
        workGroupSize[0],
        workGroupSize[1],
        workGroupSize[2]);
    AGL_CHECK_ERROR;
  }

  auto shaderKey = modeShaderKey[mod];

  const int outBufferSize = sizeof(float) * ALIGN_UP4(OC) * OH * OW;
  auto outBuffer = std::make_unique<AGLSSBuffer>(outBufferSize);
  {
    int workGroupSize[3];
    std::vector<std::string> header;
    addCompGroupSizeDefines(header, workGroupSize, 8, 8, OC_4);

    auto shader = getShader(
        "nc4hw_buf_to_nchw_buf_glsl", nc4hw_buf_to_nchw_buf_glsl, header);
    shader->useProgram();
    AGL_CHECK_ERROR;

    outBuffer->bindInProgram(0);
    convOutC4Buf->bindInProgram(1);
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

AResult conv_buf_IKnchw_SIKOnc4hw_KrO4HWC(
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
    agpu_print_NCHW(input, N, C, H, W, weights, OC, KH, KW, bias);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;

  auto biasBuf = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);
  auto kernelBuf = kernelNCHW_OCHW_repack_O4HWC(weights, OC, C, KH, KW, ares);
  if (debugPrintTensors) {
    agpu_print4d(
        "kernelNCHW_OCHW_repack_O4HWC:",
        kernelBuf->copyToHostVec().data(),
        KH,
        KW,
        4,
        4);
  }
  auto inputBuf = hostCHW_to_deviceC4HWBuffer(input, C, H, W, ares);
  auto convOutC4Buf =
      std::make_unique<AGLSSBuffer>(sizeof(float) * 4 * OC_4 * OW * OH);

  static const char* modeShaderKey[1] = {
      "conv_buf_IKnchw_SIKOnc4hw_KrO4HWC_glsl",
  };
  static const char* modeShaderCode[1] = {
      conv_buf_IKnchw_SIKOnc4hw_KrO4HWC_glsl,
  };
  {
    int workGroupSize[3];
    std::vector<std::string> header;
    addCompGroupSizeDefines(header, workGroupSize, 1, 1, OC_4);

    auto shader = getShader(modeShaderKey[mod], modeShaderCode[mod], header);
    shader->useProgram();
    AGL_CHECK_ERROR;

    convOutC4Buf->bindInProgram(0);
    inputBuf->bindInProgram(1);
    kernelBuf->bindInProgram(2);
    biasBuf->bindInProgram(3);

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
        modeShaderKey[mod],
        workGroupSize[0],
        workGroupSize[1],
        workGroupSize[2]);
    AGL_CHECK_ERROR;
  }

  auto shaderKey = modeShaderKey[mod];
  auto outBuffer =
      std::make_unique<AGLSSBuffer>(sizeof(float) * ALIGN_UP4(OC) * OH * OW);
  {
    int workGroupSize[3];
    std::vector<std::string> header;
    addCompGroupSizeDefines(header, workGroupSize, 8, 8, OC_4);

    auto shader = getShader(
        "nc4hw_buf_to_nchw_buf_glsl", nc4hw_buf_to_nchw_buf_glsl, header);
    shader->useProgram();
    AGL_CHECK_ERROR;

    outBuffer->bindInProgram(0);
    convOutC4Buf->bindInProgram(1);
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

AResult conv_buf_IKnchw_SIKnc4hw_SOnchw(
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
    agpu_print_NCHW(input, N, C, H, W, weights, OC, KH, KW, bias);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;

  auto biasBuf = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);

  auto kernelBuf =
      kernelNCHW_OCHW_repack_O4C4HWi4o4(weights, OC, C, KH, KW, ares);
  auto inputBuf = hostCHW_to_deviceC4HWBuffer(input, C, H, W, ares);
  auto outBuffer =
      std::make_unique<AGLSSBuffer>(sizeof(float) * ALIGN_UP4(OC) * OH * OW);

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
    inputBuf->bindInProgram(1);
    kernelBuf->bindInProgram(2);
    biasBuf->bindInProgram(3);

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

AResult conv_buf_IKnchw_KrO4C4HW(
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
    agpu_print_NCHW(input, N, C, H, W, weights, OC, KH, KW, bias);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int OCau4 = ALIGN_UP4(OC);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;

  auto biasBuf =
      AGLSSBuffer::from(bias, sizeof(float) * OCau4, sizeof(float) * OC);

  auto kernelBuf =
      kernelNCHW_OCHW_repack_O4C4HWi4o4(weights, OC, C, KH, KW, ares);

  auto inputBuf = AGLSSBuffer::from(input, sizeof(float) * C * H * W);

  auto outBuffer =
      std::make_unique<AGLSSBuffer>(sizeof(float) * OCau4 * OH * OW);

  static auto shaderKey = "conv_buf_IKnchw_KrO4C4HW_glsl";
  {
    int workGroupSize[3];
    std::vector<std::string> header;
    addCompGroupSizeDefines(header, workGroupSize, 1, 1, OC_4);
    auto shader = getShader(shaderKey, conv_buf_IKnchw_KrO4C4HW_glsl, header);

    shader->useProgram();
    AGL_CHECK_ERROR;

    outBuffer->bindInProgram(0);
    inputBuf->bindInProgram(1);
    kernelBuf->bindInProgram(2);
    biasBuf->bindInProgram(3);

    glUniform2i(4, PX, PY);
    glUniform2i(5, KW, KH);
    glUniform2i(6, SX, SY);
    glUniform2i(7, DX, DY);
    glUniform4i(8, OW, OH, OC_4, OC);
    glUniform4i(9, W, H, C_4, C);
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

AResult conv_buf_IKnchw_KrO4HWC(
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
    agpu_print_NCHW(input, N, C, H, W, weights, OC, KH, KW, bias);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int OCau4 = ALIGN_UP4(OC);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;

  auto biasBuf =
      AGLSSBuffer::from(bias, sizeof(float) * OCau4, sizeof(float) * OC);

  auto kernelBuf = kernelNCHW_OCHW_repack_O4HWC(weights, OC, C, KH, KW, ares);

  auto inputBuf = AGLSSBuffer::from(input, sizeof(float) * C * H * W);

  auto outBuffer =
      std::make_unique<AGLSSBuffer>(sizeof(float) * OCau4 * OH * OW);

  static auto shaderKey = "conv_buf_IKnchw_KrO4HWC_glsl";
  {
    int workGroupSize[3];
    std::vector<std::string> header;
    addCompGroupSizeDefines(header, workGroupSize, 1, 1, OC_4);
    auto shader = getShader(shaderKey, conv_buf_IKnchw_KrO4C4HW_glsl, header);

    shader->useProgram();
    AGL_CHECK_ERROR;

    outBuffer->bindInProgram(0);
    inputBuf->bindInProgram(1);
    kernelBuf->bindInProgram(2);
    biasBuf->bindInProgram(3);

    glUniform2i(4, PX, PY);
    glUniform2i(5, KW, KH);
    glUniform2i(6, SX, SY);
    glUniform2i(7, DX, DY);
    glUniform4i(8, OW, OH, OC_4, OC);
    glUniform4i(9, W, H, C_4, C);
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
AResult conv_kernel_repack_(
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
    agpu_print_NCHW(input, N, C, H, W, weights, OC, KH, KW, bias);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;

  auto biasBuf = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);
  auto kernelBuf =
      kernelNCHW_OCHW_repack_O4C4HWi4o4(weights, OC, C, KH, KW, ares);

  return ares;
}

AResult conv_buf_IKnchw_SKnc4hw_KrO4C4HW(
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
    agpu_print_NCHW(input, N, C, H, W, weights, OC, KH, KW, bias);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;
  auto biasBuf = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);

  auto kernelBuf =
      kernelNCHW_OCHW_repack_O4C4HWi4o4(weights, OC, C, KH, KW, ares);
  auto inputBuf = AGLSSBuffer::from(input, sizeof(float) * C * H * W);

  static const char* modeShaderKey[3] = {
      "conv_buf_IKnchw_SKnc4hw_KrO4C4HW_glsl",
  };
  auto shaderKey = modeShaderKey[mod];
  static const char* modeShaderCode[3] = {
      agpu::conv_buf_IKnchw_SKnc4hw_KrO4C4HW_glsl,
  };

  const int outBufferSize = sizeof(float) * ALIGN_UP4(OC) * OH * OW;
  auto outBuffer = std::make_unique<AGLSSBuffer>(outBufferSize);
  int workGroupSize[3];
  std::vector<std::string> header;
  addCompGroupSizeDefines(header, workGroupSize, 1, 1, OC_4);

  auto shader = getShader(modeShaderKey[mod], modeShaderCode[mod], header);

  shader->useProgram();
  AGL_CHECK_ERROR;

  outBuffer->bindInProgram(0);
  inputBuf->bindInProgram(1);
  kernelBuf->bindInProgram(2);
  biasBuf->bindInProgram(3);

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

  outBuffer->copyToHost(output, sizeof(float) * OC * OH * OW);
  if (debugPrintTensors) {
    agpu_print4d(shaderKey, output, N, OC, OH, OW);
  }

  return ares;
}

AResult conv_buf_Inhwc_Knchw_KrO4C4HW(
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
  static const bool debugPrintTensors = DEBUG_PRINT_TENSOR;
  if (debugPrintTensors) {
    agpu_print_NHWC(input, N, C, H, W, weights, OC, KH, KW, bias);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;
  auto biasBuf = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);

  auto kernelBuf =
      kernelNCHW_OCHW_repack_O4C4HWi4o4(weights, OC, C, KH, KW, ares);

  auto inputBuf = AGLSSBuffer::from(input, sizeof(float) * H * W * C);

  const char* modeShaderKey[1] = {
      "conv_buf_Inhwc_Knchw_KrO4C4HW_glsl",
  };
  auto shaderKey = modeShaderKey[0];
  const char* modeShaderCode[1] = {
      conv_buf_Inhwc_Knchw_KrO4C4HW_glsl,
  };
  const int outBufferSize = sizeof(float) * OH * OW * OC;

  auto outBuffer = std::make_unique<AGLSSBuffer>(outBufferSize);
  int workGroupSize[3];
  std::vector<std::string> header;
  addCompGroupSizeDefines(header, workGroupSize, 1, 1, OC_4);

  auto shader = getShader(modeShaderKey[mod], modeShaderCode[mod], header);

  shader->useProgram();
  AGL_CHECK_ERROR;

  outBuffer->bindInProgram(0);
  inputBuf->bindInProgram(1);
  kernelBuf->bindInProgram(2);
  biasBuf->bindInProgram(3);

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

  outBuffer->copyToHost(output, sizeof(float) * OH * OW * OC);
  if (debugPrintTensors) {
    agpu_print4d(shaderKey, output, N, OH, OW, OC);
  }

  return ares;
}

static constexpr const int kDsdim = 16;

AResult conv_buf_IKnhwc(
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
  static const bool debugPrintTensors = DEBUG_PRINT_TENSOR;
  if (debugPrintTensors) {
    agpu_print_NHWC(input, N, C, H, W, weights, OC, KH, KW, bias);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;
  auto biasBuf = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);

  auto kernelBuf = AGLSSBuffer::from(weights, sizeof(float) * OC * KH * KW * C);
  auto inputBuf = AGLSSBuffer::from(input, sizeof(float) * H * W * C);

  const char* modeShaderKey[1] = {
      "conv_buf_IKnhwc_glsl",
  };
  auto shaderKey = modeShaderKey[0];
  const char* modeShaderCode[1] = {
      conv_buf_IKnhwc_glsl,
  };

  auto outBuf = std::make_unique<AGLSSBuffer>(sizeof(float) * OH * OW * OC);

  int workGroupSize[3];
  std::vector<std::string> header;
  addCompGroupSizeDefines(header, workGroupSize, 1, 1, OC_4);

  auto shader = getShader(modeShaderKey[mod], modeShaderCode[mod], header);

  shader->useProgram();
  AGL_CHECK_ERROR;

  outBuf->bindInProgram(0);
  inputBuf->bindInProgram(1);
  kernelBuf->bindInProgram(2);
  biasBuf->bindInProgram(3);

  glUniform2i(4, PX, PY);
  glUniform2i(5, KW, KH);
  glUniform2i(6, SX, SY);
  glUniform2i(7, DX, DY);
  glUniform4i(8, OW, OH, OC_4, OC);
  glUniform4i(9, W, H, C_4, C);
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

  outBuf->copyToHost(output, sizeof(float) * OH * OW * OC);
  if (debugPrintTensors) {
    agpu_print4d(shaderKey, output, N, OH, OW, OC);
  }
  return ares;
}

AResult conv_buf_IKnhwc_KrO4C4HW(
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
  static const bool debugPrintTensors = DEBUG_PRINT_TENSOR;
  if (debugPrintTensors) {
    agpu_print_NHWC(input, N, C, H, W, weights, OC, KH, KW, bias);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;
  auto biasBuf = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);

  auto kernelBuf =
      kernelNHWC_OHWC_repack_O4C4HWi4o4(weights, OC, C, KH, KW, ares);
  if (debugPrintTensors) {
    agpu_print4d(
        "kernel_repack_O4C4HWi4o4:",
        kernelBuf->copyToHostVec().data(),
        KH,
        KW,
        4,
        4);
  }
  auto inputBuf = AGLSSBuffer::from(input, sizeof(float) * H * W * C);

  const char* modeShaderKey[1] = {
      "conv_buf_IKnhwc_KrO4C4HW_glsl",
  };
  auto shaderKey = modeShaderKey[0];
  const char* modeShaderCode[1] = {
      conv_buf_IKnhwc_KrO4C4HW_glsl,
  };

  auto outBuf = std::make_unique<AGLSSBuffer>(sizeof(float) * OH * OW * OC);

  int workGroupSize[3];
  std::vector<std::string> header;
  addCompGroupSizeDefines(header, workGroupSize, 1, 1, OC_4);

  auto shader = getShader(modeShaderKey[mod], modeShaderCode[mod], header);

  shader->useProgram();
  AGL_CHECK_ERROR;

  outBuf->bindInProgram(0);
  inputBuf->bindInProgram(1);
  kernelBuf->bindInProgram(2);
  biasBuf->bindInProgram(3);

  glUniform2i(4, PX, PY);
  glUniform2i(5, KW, KH);
  glUniform2i(6, SX, SY);
  glUniform2i(7, DX, DY);
  glUniform4i(8, OW, OH, OC_4, OC);
  glUniform4i(9, W, H, C_4, C);
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

  outBuf->copyToHost(output, sizeof(float) * OH * OW * OC);
  if (debugPrintTensors) {
    agpu_print4d(shaderKey, output, N, OH, OW, OC);
  }
  return ares;
}

AResult conv_buf_IKnhwc_KrO4HWC(
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
  static const bool debugPrintTensors = DEBUG_PRINT_TENSOR;
  if (debugPrintTensors) {
    agpu_print_NHWC(input, N, C, H, W, weights, OC, KH, KW, bias);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;
  auto biasBuf = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);

  auto kernelBuf = kernelNHWC_OHWC_repack_O4HWC(weights, OC, C, KH, KW, ares);
  if (debugPrintTensors) {
    agpu_print4d(
        "kernel_repack_O4HWC:",
        kernelBuf->copyToHostVec().data(),
        KH,
        KW,
        4,
        4);
  }
  auto inputBuf = AGLSSBuffer::from(input, sizeof(float) * H * W * C);

  const char* modeShaderKey[1] = {
      "conv_buf_IKnhwc_KrO4HWC_glsl",
  };
  auto shaderKey = modeShaderKey[0];
  const char* modeShaderCode[1] = {
      conv_buf_IKnhwc_KrO4HWC_glsl,
  };

  auto outBuf = std::make_unique<AGLSSBuffer>(sizeof(float) * OH * OW * OC);

  int workGroupSize[3];
  std::vector<std::string> header;
  addCompGroupSizeDefines(header, workGroupSize, 1, 1, OC_4);

  auto shader = getShader(modeShaderKey[mod], modeShaderCode[mod], header);

  shader->useProgram();
  AGL_CHECK_ERROR;

  outBuf->bindInProgram(0);
  inputBuf->bindInProgram(1);
  kernelBuf->bindInProgram(2);
  biasBuf->bindInProgram(3);

  glUniform2i(4, PX, PY);
  glUniform2i(5, KW, KH);
  glUniform2i(6, SX, SY);
  glUniform2i(7, DX, DY);
  glUniform4i(8, OW, OH, OC_4, OC);
  glUniform4i(9, W, H, C_4, C);
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

  outBuf->copyToHost(output, sizeof(float) * OH * OW * OC);
  if (debugPrintTensors) {
    agpu_print4d(shaderKey, output, N, OH, OW, OC);
  }
  return ares;
}

// Depthwise
AResult convDW_buf_IKnchw(
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
  assert(G == OC);

  setBenchId(mod);
  setPrecision(highp);
  initAGLContextOnce();
  AResult ares;
  static const bool debugPrintTensors = DEBUG_PRINT_TENSOR;
  if (debugPrintTensors) {
    agpu_print_NCHW(input, N, C, H, W, weights, OC, KH, KW, bias);
  }
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;
  auto biasBuf = AGLSSBuffer::from(bias, sizeof(float) * OC);
  auto kernelBuf = AGLSSBuffer::from(weights, sizeof(float) * OC * KH * KW);
  auto inputBuf = AGLSSBuffer::from(input, sizeof(float) * C * H * W);

  const char* modeShaderKey[1] = {
      "convDW_buf_IKnchw_glsl",
  };
  auto shaderKey = modeShaderKey[0];
  const char* modeShaderCode[1] = {
      convDW_buf_IKnchw_glsl,
  };
  const int outBufferSize = sizeof(float) * OC * OH * OW;
  auto outBuffer = std::make_unique<AGLSSBuffer>(outBufferSize);
  int workGroupSize[3];
  std::vector<std::string> header;
  addCompGroupSizeDefines(header, workGroupSize, 8, 8, 1);

  auto shader = getShader(modeShaderKey[mod], modeShaderCode[mod], header);

  shader->useProgram();
  AGL_CHECK_ERROR;

  outBuffer->bindInProgram(0);
  inputBuf->bindInProgram(1);
  kernelBuf->bindInProgram(2);
  biasBuf->bindInProgram(3);

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
      UP_DIV(OC, workGroupSize[2]),
      shaderKey,
      workGroupSize[0],
      workGroupSize[1],
      workGroupSize[2]);
  AGL_CHECK_ERROR;

  outBuffer->copyToHost(output, sizeof(float) * OC * OH * OW);
  if (debugPrintTensors) {
    agpu_print4d(shaderKey, output, N, OC, OH, OW);
  }
  return ares;
}

AResult convDW_buf_Inhwc_Knchw(
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
  assert(G == OC);

  setBenchId(mod);
  setPrecision(highp);
  initAGLContextOnce();
  AResult ares;
  static const bool debugPrintTensors = DEBUG_PRINT_TENSOR;
  if (debugPrintTensors) {
    agpu_print_NHWC(input, N, C, H, W, weights, OC, KH, KW, bias);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int C_4 = UP_DIV(C, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;
  const uint32_t GC = G * C;
  const uint32_t GOC = G * OC;

  auto biasBuf = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);
  auto kernelBuf = AGLSSBuffer::from(
      weights,
      sizeof(float) * ALIGN_UP4(GOC) * KH * KW,
      sizeof(float) * GOC * KH * KW);
  auto inputBuf = AGLSSBuffer::from(input, sizeof(float) * H * W * C);

  const char* modeShaderKey[1] = {
      "convDW_buf_Inhwc_Knchw_glsl",
  };
  const char* modeShaderCode[1] = {
      convDW_buf_Inhwc_Knchw_glsl,
  };
  auto shaderKey = modeShaderKey[0];
  auto outBuffer = std::make_unique<AGLSSBuffer>(sizeof(float) * OH * OW * OC);
  int workGroupSize[3];
  std::vector<std::string> header;
  addCompGroupSizeDefines(header, workGroupSize, 8, 8, 1);

  auto shader = getShader(modeShaderKey[mod], modeShaderCode[mod], header);

  shader->useProgram();
  AGL_CHECK_ERROR;

  outBuffer->bindInProgram(0);
  inputBuf->bindInProgram(1);
  kernelBuf->bindInProgram(2);
  biasBuf->bindInProgram(3);

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

  outBuffer->copyToHost(output, sizeof(float) * OH * OW * OC);
  if (debugPrintTensors) {
    agpu_print4d(shaderKey, output, N, OH, OW, OC);
  }
  return ares;
}

AResult convDW_buf_IKnhwc(
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
  assert(G == OC);

  setBenchId(mod);
  setPrecision(highp);
  initAGLContextOnce();
  AResult ares;
  static const bool debugPrintTensors = DEBUG_PRINT_TENSOR;
  if (debugPrintTensors) {
    agpu_print_NHWC(input, N, C, H, W, weights, OC, KH, KW, bias);
  }

  const int OC_4 = UP_DIV(OC, 4);
  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const uint32_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const uint32_t OH = ((H - KHE + 2 * PY) / SY) + 1;
  auto biasBuf = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);
  auto kernelBuf = AGLSSBuffer::from(weights, sizeof(float) * OC * KH * KW);
  auto inputBuf = AGLSSBuffer::from(input, sizeof(float) * H * W * C);

  const char* modeShaderKey[1] = {
      "convDW_buf_IKnhwc_glsl",
  };
  const char* modeShaderCode[1] = {
      convDW_buf_IKnhwc_glsl,
  };
  auto shaderKey = modeShaderKey[0];

  auto outBuffer = std::make_unique<AGLSSBuffer>(sizeof(float) * OH * OW * OC);

  int workGroupSize[3];
  std::vector<std::string> header;
  addCompGroupSizeDefines(header, workGroupSize, 1, 1, 1);

  auto shader = getShader(modeShaderKey[mod], modeShaderCode[mod], header);

  shader->useProgram();
  AGL_CHECK_ERROR;

  outBuffer->bindInProgram(0);
  inputBuf->bindInProgram(1);
  kernelBuf->bindInProgram(2);
  biasBuf->bindInProgram(3);

  glUniform2i(4, PX, PY);
  glUniform2i(5, KW, KH);
  glUniform2i(6, SX, SY);
  glUniform2i(7, DX, DY);
  AGL_CHECK_ERROR;

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
  return conv_(
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

AResult conv_(
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
  return conv_tex_IKnc4hw(
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

AResult conv_tex_IKnc4hw(
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
    agpu_print_NCHW(input, N, C, H, W, weights, OC, KH, KW, bias);
  }

  auto biasBuf = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);
  auto kernelBuf =
      kernelNCHW_OCHW_repack_O4C4HWi4o4(weights, OC, C, KH, KW, ares);

  auto kernelTex = std::make_unique<AGLTexture>(
      C_4 * unit, OC_4, KH * KW, getTexFormat(), GL_TEXTURE_3D, false);

  auto kernel2ImageShader =
      getShader("KO4C4HW_to_tex_glsl", KO4C4HW_to_tex_glsl);

  kernel2ImageShader->useProgram();
  // binding kernel2Image {
  glBindImageTexture(
      0, kernelTex->id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, getTexFormat());
  kernelBuf->bindInProgram(2);
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
  hostCHW_to_deviceTex(inputTex->id(), input, C, H, W, ares);

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
      0 /* unit */,
      outputTex->id(),
      0 /* level */,
      GL_TRUE /* layered */,
      0 /* layer */,
      GL_WRITE_ONLY,
      getTexFormat());

  inputTex->bindInProgram(0 /* unit */, 1 /* binding */);
  kernelTex->bindInProgram(1, 2);
  biasBuf->bindInProgram(3);
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

  ares.gpu_shader_dtex_to_hchw_time = deviceTex2hostCHW(outputTex->id(), output, OW, OH, OC);

  if (debugPrintTensors) {
    agpu_print4d(shaderKey, output, N, OC, OH, OW);
  }
  return ares;
}

// region not_convolution

void agpu_addmm(
    const float* m1Data,
    uint32_t m1dim,
    uint32_t* m1sizes,

    const float* m2Data,
    uint32_t m2dim,
    uint32_t* m2sizes,

    float beta,
    float alpha,

    const float* tData,
    uint32_t tdim,
    uint32_t* tsizes,

    float* output) {
  initAGLContextOnce();
  static const bool debugPrintTensors = DEBUG_PRINT_TENSOR;
  AResult ares;

  int64_t* m1sizes64 = (int64_t*) m1sizes;
  uint32_t m1s0 = m1sizes64[0];
  uint32_t m1s1 = m1sizes64[1];
  int64_t* m2sizes64 = (int64_t*) m2sizes;
  uint32_t m2s0 = m2sizes64[0];
  uint32_t m2s1 = m2sizes64[1];
  int64_t* tsizes64 = (int64_t*) tsizes;
  uint32_t ts0 = tsizes64[0];
  uint32_t ts1 = tsizes64[1];

  if (debugPrintTensors) {
    std::cout << "AGPU agpu_addmm()" << std::endl;
    std::cout << "AGPU m1s0:" << m1s0 << " m1s1:" << m1s1 << std::endl;
    agpu_print2d("m1:", m1Data, m1s0, m1s1);
    std::cout << "AGPU m2s0:" << m2s0 << " m2s1:" << m2s1 << std::endl;
    agpu_print2d("m2:", m2Data, m2s0, m2s1);
    std::cout << "AGPU ts0:" << ts0 << " ts1:" << ts1 << std::endl;
    agpu_print2d("t:", tData, ts0, ts1);
    std::cout << "AGPU beta:" << beta << " alpha:" << alpha << std::endl;
  }
  uint32_t m1H = m1s0;
  uint32_t m1W = m1s1;
  uint32_t m1C = 1;
  uint32_t m1C_4 = UP_DIV(m1C, 4);

  uint32_t m2H = m2s0;
  uint32_t m2W = m2s1;
  uint32_t m2C = 1;
  uint32_t m2C_4 = UP_DIV(m2C, 4);

  uint32_t OH = m1s0;
  uint32_t OW = m2s1;

  assert(m1W == m2H);
  assert(m1C == m2C);
  uint32_t C = m1C;
  uint32_t C_4 = UP_DIV(C, 4);
  uint32_t K = m1W;

  uint32_t TH = ts0;
  uint32_t TW = ts1;
  uint32_t TC = 1;


  auto m1Tex = std::make_unique<AGLTexture>(
      m1W, m1H, C_4, getTexFormat(), GL_TEXTURE_3D, false);
  std::cout << "m1Tex hostCHW_to_deviceTex" << std::endl;
  hostCHW_to_deviceTex(m1Tex->id(), m1Data, m1C, m1H, m1W, ares);

  auto m2Tex = std::make_unique<AGLTexture>(
      m2W, m2H, C_4, getTexFormat(), GL_TEXTURE_3D, false);
  std::cout << "m2Tex hostCHW_to_deviceTex" << std::endl;
  hostCHW_to_deviceTex(m2Tex->id(), m2Data, m2C, m2H, m2W, ares);

  auto tTex = std::make_unique<AGLTexture>(
      TW, TH, C_4, getTexFormat(), GL_TEXTURE_3D, false);
  std::cout << "tTex hostCHW_to_deviceTex" << std::endl;
  hostCHW_to_deviceTex(tTex->id(), tData, TC, TH, TW, ares);

  std::cout << "outTex " << std::endl;
  auto outTex = std::make_unique<AGLTexture>(
      OW, OH, C_4, getTexFormat(), GL_TEXTURE_3D, false);

  int compGroupSize[3];
  std::vector<std::string> header;
  addCompGroupSizeDefines(header, compGroupSize, 8, 8, 1);
  auto shaderKey = "addmm_glsl";
  auto addmmProgram = getShader(shaderKey, addmm_glsl, header);

  addmmProgram->useProgram();
  // { binding
  glBindImageTexture(
      0 /* unit */,
      outTex->id(),
      0 /* level */,
      GL_TRUE /* layered */,
      0 /* layer */,
      GL_WRITE_ONLY,
      getTexFormat());

  // IK? outTex and m1 binding to the same texUnit ?
  m1Tex->bindInProgram(0, 1 /* binding */);
  m2Tex->bindInProgram(1, 2 /* binding */);
  tTex->bindInProgram(2, 3 /* binding */);
  glUniform1f(4, beta);
  glUniform1f(5, alpha);
  glUniform3i(6, OW, OH, C_4);
  glUniform1i(7, K);
  AGL_CHECK_ERROR;
  // } binding

  gComputeWithTime(
      UP_DIV(OW, compGroupSize[0]),
      UP_DIV(OH, compGroupSize[1]),
      UP_DIV(C_4, compGroupSize[2]),
      shaderKey,
      compGroupSize[0],
      compGroupSize[1],
      compGroupSize[2]);
  AGL_CHECK_ERROR;

  ares.gpu_shader_dtex_to_hchw_time = deviceTex2hostCHW(outTex->id(), output, OW, OH, C);

  std::cout << "AGPU addmm----------------------------------------" << std::endl;
  if (debugPrintTensors) {
    agpu_print2d(shaderKey, output, OH, OW);
  }
  std::cout << "AGPU addmm----------------------------------------" << std::endl;
}

void agpu_upsample_nearest2d(
    float* output,
    const float* input,
    uint32_t IH,
    uint32_t IW,
    uint32_t OH,
    uint32_t OW,
    uint32_t _N,
    uint32_t _C,
    float scaleH,
    float scaleW) {
  initAGLContextOnce();
  AResult ares;

  static const bool debugPrintTensors = DEBUG_PRINT_TENSOR;

  if (debugPrintTensors) {
    agpu_print4d("agpu_upsample_nearest2d input:\n", input, _N, _C, IH, IW);
    std::cout
        << "AGPU scaleH:"<< scaleH
        << " scaleW:" << scaleW
        << std::endl;
  }
  uint32_t C = _N * _C;
  uint32_t C_4 = UP_DIV(C, 4);
  // inTex
  auto inTex = std::make_unique<AGLTexture>(
      IW, IH, C_4, getTexFormat(), GL_TEXTURE_3D, false);
  hostCHW_to_deviceTex(inTex->id(), input, C, IH, IW, ares);
  // outTex
  auto outTex = std::make_unique<AGLTexture>(
      OW, OH, C_4, getTexFormat(), GL_TEXTURE_3D, false);

  int compGroupSize[3];
  std::vector<std::string> header;
  addCompGroupSizeDefines(header, compGroupSize, 8, 8, 1);
  auto shaderKey = "upsampleNearest2d_glsl";
  auto program = getShader(shaderKey, upsampleNearest2d_glsl, header);

  program->useProgram();
  // { binding
  glBindImageTexture(
      0 /* unit */,
      outTex->id(),
      0 /* level */,
      GL_TRUE /* layered */,
      0 /* layer */,
      GL_WRITE_ONLY,
      getTexFormat());

  inTex->bindInProgram(0, 1 /* binding */);

  glUniform3i(2, IW, IH, C_4);
  glUniform3i(3, OW, OH, C_4);

  glUniform1f(4, scaleW);
  glUniform1f(5, scaleH);
  AGL_CHECK_ERROR;
  // } binding

  gComputeWithTime(
      UP_DIV(OW, compGroupSize[0]),
      UP_DIV(OH, compGroupSize[1]),
      UP_DIV(C_4, compGroupSize[2]),
      shaderKey,
      compGroupSize[0],
      compGroupSize[1],
      compGroupSize[2]);
  AGL_CHECK_ERROR;

  ares.gpu_shader_dtex_to_hchw_time = deviceTex2hostCHW(outTex->id(), output, OW, OH, C);


  std::cout << "AGPU upsampleN2d ----------------------------------------" << std::endl;
  if (debugPrintTensors) {
    agpu_print4d(shaderKey, output, _N, _C, OH, OW);
  }
  std::cout << "AGPU upsampleN2d ----------------------------------------" << std::endl;
}

void agpu_max_pool2d(
    float* output,
    float* input,
    int64_t* ind,

    uint32_t N,
    uint32_t IW,
    uint32_t IH,
    uint32_t OW,
    uint32_t OH,

    uint32_t KW,
    uint32_t KH,

    uint32_t SW,
    uint32_t SH,
   
    uint32_t PW,
    uint32_t PH,

    uint32_t DW,
    uint32_t DH) {
  std::cout << "AAA agpu_max_pool2d()" << std::endl;
  initAGLContextOnce();
  AResult ares;
  static const bool debugPrintTensors = DEBUG_PRINT_TENSOR;

  if (debugPrintTensors) {
    agpu_print3d("agpu_max_pool2d input:\n", input, N, IH, IW);
    std::cout
        << "AGPU KH:"<< KH
        << " KW:" << KW
        << std::endl;
  }
  uint32_t C = N;
  uint32_t C_4 = UP_DIV(C, 4);

  // inTex
  auto inTex = std::make_unique<AGLTexture>(
      IW, IH, C_4, getTexFormat(), GL_TEXTURE_3D, false);
  hostCHW_to_deviceTex(inTex->id(), input, C, IH, IW, ares);
  // outTex
  auto outTex = std::make_unique<AGLTexture>(
      OW, OH, C_4, getTexFormat(), GL_TEXTURE_3D, false);

  int compGroupSize[3];
  std::vector<std::string> header;
  addCompGroupSizeDefines(header, compGroupSize, 2, 2, 16);
  auto shaderKey = "maxpool2d_glsl";
  auto program = getShader(shaderKey, maxpool2d_glsl, header);

  program->useProgram();
  // { binding
  glBindImageTexture(
      0 /* unit */,
      outTex->id(),
      0 /* level */,
      GL_TRUE /* layered */,
      0 /* layer */,
      GL_WRITE_ONLY,
      getTexFormat());

  glBindImageTexture(
      1 /* unit */,
      inTex->id(),
      0 /* level */,
      GL_TRUE /* layered */,
      0 /* layer */,
      GL_WRITE_ONLY,
      getTexFormat());

  glUniform2i(2, KW, KH);
  glUniform2i(3, SW, SH);
  glUniform2i(4, PW, PH);
  glUniform2i(5, DW, DH);

  glUniform3i(10, OW, OH, C_4);
  glUniform3i(11, IW, IH, C_4);
  AGL_CHECK_ERROR;
  // } binding

  gComputeWithTime(
      UP_DIV(OW, compGroupSize[0]),
      UP_DIV(OH, compGroupSize[1]),
      UP_DIV(C_4, compGroupSize[2]),
      shaderKey,
      compGroupSize[0],
      compGroupSize[1],
      compGroupSize[2]);
  AGL_CHECK_ERROR;

  ares.gpu_shader_dtex_to_hchw_time = deviceTex2hostCHW(outTex->id(), output, OW, OH, C);
  std::cout << "AGPU maxPool2d ----------------------------------------" << std::endl;
  if (debugPrintTensors) {
    agpu_print3d(shaderKey, output, N, OH, OW);
  }
  std::cout << "AGPU maxPool2d ----------------------------------------" << std::endl;
}

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
  hostCHW_to_deviceTex(input0Tex->id(), input0, c, h, w, ares);
  auto input1Tex = std::make_unique<AGLTexture>(
      w, h, c_4, getTexFormat(), GL_TEXTURE_3D, false);
  hostCHW_to_deviceTex(input1Tex->id(), input1, c, h, w, ares);

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
  deviceTex2hostCHW(outputTex->id(), output, w, h, c);

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

  hostCHW_to_deviceTex(inputTex->id(), input, c, h, w, ares);

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

  deviceTex2hostCHW(outputTex->id(), output, w, h, c);

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
  hostCHW_to_deviceTex(inputTex->id(), input, c, h, w, ares);

  auto outputTex = std::make_unique<AGLTexture>(
      w, h, c_4, getTexFormat(), GL_TEXTURE_3D, false);

  GLsizeiptr bufferSize = sizeof(float) * ALIGN_UP4(c);
  auto weightBuffer = AGLSSBuffer::from(weight, bufferSize);
  auto biasBuf = AGLSSBuffer::from(bias, bufferSize);
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
  biasBuf->bindInProgram(4);
  meanBuffer->bindInProgram(5);
  varianceBuffer->bindInProgram(6);

  glUniform1f(7, eps);
  AGL_CHECK_ERROR;
  // binding normalization_glsl }

  compute(
      UP_DIV(w, compGroupSize[0]),
      UP_DIV(h, compGroupSize[1]),
      UP_DIV(c_4, compGroupSize[2]));

  deviceTex2hostCHW(outputTex->id(), output, w, h, c);
  agpu_print4d("agpu_batch_norm output:\n", output, n, c, h, w);
}
// endregion notconvolution
#endif
} // namespace agpu
