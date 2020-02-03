#include <cassert>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "agpu.h"
#include "shader.h"

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x) ROUND_UP((x), 4)

namespace agpu {

const char* agpu_test() {
  return "GPUGPUGPU";
}

void agpu_print(const char* m, const float* t, uint32_t rank, uint32_t* dims) {
  static const char* kFloatFormat = "%8.1f";
  std::cout << m;

  if (rank == 0) {
    std::cout << *t;
  } else if (rank == 1 || rank == 2) {
    char fbuf[12];
    uint32_t rows = rank == 1 ? 1 : dims[0];
    uint32_t cols = rank == 1 ? dims[0] : dims[1];
    for (uint32_t i = 0; i < rows; i++) {
      std::cout << "\n";
      for (uint32_t j = 0; j < cols; j++) {
        sprintf(fbuf, kFloatFormat, t[i * cols + j]);
        std::cout << fbuf << " ";
      }
    }
  } else if (rank == 3) {
    std::cout << " dims:(";
    for (uint32_t i = 0; i < rank; i++) {
      std::cout << dims[i] << " ";
    }
    std::cout << ")";
    uint32_t d0 = dims[0];
    uint32_t d12size = dims[1] * dims[2];
    for (uint32_t i = 0; i < d0; i++) {
      char s[80];
      sprintf(s, "[%d, *, *]", i);
      agpu_print(s, t + i * d12size, 2, dims + 1);
    }
  } else if (rank == 4) {
    std::cout << " dims:(";
    for (uint32_t i = 0; i < rank; i++) {
      std::cout << dims[i] << " ";
    }
    std::cout << ")";
    uint32_t d0 = dims[0];
    uint32_t d1 = dims[1];
    uint32_t d23size = dims[2] * dims[3];
    for (uint32_t i = 0; i < d0; i++) {
      for (uint32_t j = 0; j < d1; j++) {
        char s[80];
        sprintf(s, "[%d, %d, *, *]", i, j);
        agpu_print(s, t + (i * d0 + j) * d23size, 2, dims + 2);
      }
    }
  } else {
    // TODO: support print r > 4
    assert(false);
  }
  std::cout << std::endl;
}

#ifndef __ANDROID__

void agpu_conv2d(
    const float* input,
    uint32_t input_n,
    uint32_t input_c,
    uint32_t input_h,
    uint32_t input_w,
    const float* weights,
    uint32_t kernel_c,
    uint32_t kernel_h,
    uint32_t kernel_w,
    const float* bias,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t input_padding_h,
    uint32_t input_padding_w,
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint32_t groups,
    float* output) {}

#else

class AGLContext {
 public:
  AGLContext() {
    if (!(eglGetCurrentContext() != EGL_NO_CONTEXT)) {
      display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
      if (display_ == EGL_NO_DISPLAY) {
        AGPU_PRINT("eglGetDisplay error !!! \n");
        isCreateError_ = true;
      }
      int majorVersion;
      int minorVersion;
      eglInitialize(display_, &majorVersion, &minorVersion);
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
        AGPU_PRINT("eglChooseConfig error !!! \n");
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
      if (major < 3) {
        isCreateError_ = true;
      }
    } else {
      context_ = EGL_NO_CONTEXT;
      AGPU_PRINT("eglGetCurrentContext() != EGL_NO_CONTEXT \n");
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

 private:
  EGLContext context_;
  EGLDisplay display_;
  EGLSurface surface_;
  bool isCreateError_{false};
};

class AGLBuffer {
 public:
  AGLBuffer(GLsizeiptr size, GLenum type = GL_SHADER_STORAGE_BUFFER) {
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

  ~AGLBuffer() {
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

  GLsizeiptr size() const {
    return size_;
  }

  GLuint getId() const {
    return id_;
  }

 private:
  GLuint id_ = 0;
  GLsizeiptr size_;
  GLenum type_;
}; // class AGLBuffer

class AGLTexture {
 public:
  AGLTexture(
      int w,
      int h,
      int d,
      GLenum textureFormat,
      GLenum target = GL_TEXTURE_3D,
      bool HWC4 = true) {
    textureFormat_ = textureFormat;
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
      glTexStorage3D(target_, 1, textureFormat_, realW, realH, realD);
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
      glTexStorage2D(target_, 1, textureFormat_, realW, realH);
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
    glBindImageTexture(unit, id_, 0, GL_TRUE, 0, GL_READ_ONLY, textureFormat_);
    AGL_CHECK_ERROR;
  }

  void write(GLuint unit) {
    glBindImageTexture(unit, id_, 0, GL_TRUE, 0, GL_WRITE_ONLY, textureFormat_);
    AGL_CHECK_ERROR;
  }

  void sample(GLuint unit, GLuint texId) {
    glActiveTexture(GL_TEXTURE0 + texId);
    glUniform1i(unit, texId);
    glBindTexture(target_, id_);
    AGL_CHECK_ERROR;
  }

 private:
  unsigned int id_;
  GLenum target_;
  GLenum textureFormat_{GL_RGBA32F};
}; // class AGLTexture

class AGLProgram {
 public:
  AGLProgram(const std::string& computeShader) {
    shaderId_ = glCreateShader(GL_COMPUTE_SHADER);
    AGL_CHECK_ERROR;
    const char* _ver[1];
    _ver[0] = computeShader.c_str();
    glShaderSource(shaderId_, 1, _ver, NULL);
    AGL_CHECK_ERROR;

    bool res = compileShader(shaderId_);
    // if (!res) FUNC_PRINT_ALL(mVertex.c_str(), s);
    assert(res);

    /*Create Program*/
    programId_ = glCreateProgram();
    AGL_CHECK_ERROR;
    glAttachShader(programId_, shaderId_);
    AGL_CHECK_ERROR;
    glLinkProgram(programId_);
    AGL_CHECK_ERROR;
    GLint linked;
    glGetProgramiv(programId_, GL_LINK_STATUS, &linked);
    if (!linked) {
      //        FUNC_PRINT(linked);
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

  ~AGLProgram() {
    glDeleteShader(shaderId_);
    glDeleteProgram(programId_);
    AGL_CHECK_ERROR;
  }

  unsigned int getProgramId() const {
    return programId_;
  }

  static std::string getHead(std::string imageFormat) {
    std::ostringstream headOs;
    headOs << "#version 310 es\n";
    headOs << "#define PRECISION mediump\n";
    headOs << "precision PRECISION float;\n";
    headOs << "#define FORMAT " << imageFormat << "\n";
    return headOs.str();
  }

  /*These API must be called in openGL context Thread*/
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
}; // class AGLProgram

GLenum getTextureFormat() {
  return GL_RGBA32F;
}

std::string getImageFormat() {
  return "rgba32f";
}

std::unique_ptr<AGLProgram> getProgramWithPrefix(
    const char* content,
    const std::vector<std::string>& prefix) {
  std::ostringstream tc;
  tc << AGLProgram::getHead(getImageFormat());
  for (auto& s : prefix) {
    tc << s << "\n";
  }
  tc << content;
  return std::make_unique<AGLProgram>(tc.str());
}

std::unique_ptr<AGLProgram> getProgram(const char* content) {
  std::ostringstream tc;
  tc << AGLProgram::getHead(getImageFormat()) << content;
  return std::make_unique<AGLProgram>(tc.str());
}

std::unique_ptr<AGLProgram> getProgram(
    const std::string&,
    const char* content) {
  return getProgram(content);
}

std::unique_ptr<AGLProgram> getProgram(
    const std::string&,
    const char* content,
    const std::vector<std::string>& prefix) {
  return getProgramWithPrefix(content, prefix);
}

void wait() {
#ifdef USE_GL_FINISH
  glFinish();
#else
  glFlush();
#endif
}

void compute(int dim1, int dim2, int dim3) {
  AGPU_PRINT("compute(%d %d %d)", dim1, dim2, dim3);
  wait();
  glDispatchCompute(dim1, dim2, dim3);
}

void device2host(
    GLuint textureId,
    float* outputData,
    int d1,
    int d2,
    int d3,
    bool outputAlign4) {
  AGPU_PRINT("device2host(%d %d %d align %d)", d1, d2, d3, outputAlign4);
  wait();
  auto depthQuad = UP_DIV(d3, 4);
  auto size = depthQuad * 4 * d1 * d2 * sizeof(float);
  auto buffer = std::make_unique<AGLBuffer>(size);

  auto program = outputAlign4
      ? getProgram(
            "glsl_image_to_nc4hw4_buffer_glsl",
            glsl_image_to_nc4hw4_buffer_glsl)
      : getProgram(
            "glsl_image_to_nchw_buffer_glsl", glsl_image_to_nchw_buffer_glsl);
  program->useProgram();

  glBindImageTexture(
      0, textureId, 0, GL_TRUE, 0, GL_READ_ONLY, getTextureFormat());
  AGL_CHECK_ERROR;
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffer->getId());
  AGL_CHECK_ERROR;
  glUniform1i(2, d1);
  glUniform1i(3, d2);
  AGL_CHECK_ERROR;

  compute(UP_DIV(d1, 8), UP_DIV(d2, 8), depthQuad);
  AGL_CHECK_ERROR;

  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  AGL_CHECK_ERROR;

  auto d_output = buffer->map(GL_MAP_READ_BIT);
  if (d_output != nullptr) {
    if (outputAlign4) {
      ::memcpy(outputData, d_output, size);
    } else {
      ::memcpy(outputData, d_output, d1 * d2 * d3 * sizeof(float));
    }
  }
  buffer->unmap();
}

void host2device(
    GLuint textureId,
    const float* inputData,
    int c,
    int h,
    int w,
    bool inputData4Aligned) {
  AGPU_PRINT(
      "device2host(c %d h %d w %d align %d)", c, h, w, inputData4Aligned);

  int c_4 = UP_DIV(c, 4);
  auto size = ROUND_UP(c, 4) * w * h * sizeof(float);
  AGPU_PRINT("device2host c_4:%d", c_4);
  AGPU_PRINT("device2host ROUND_UP(c, 4):%d", ROUND_UP(c, 4));
  auto buffer = std::make_unique<AGLBuffer>(sizeof(float) * size);

  auto d_output = buffer->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

  if (d_output != nullptr) {
    if (inputData4Aligned) {
      ::memcpy(d_output, inputData, size);
    } else {
      ::memcpy(d_output, inputData, c * h * w * sizeof(float));
    }
  }
  buffer->unmap();

  auto program = inputData4Aligned
      ? getProgram(
            "glsl_nc4hw4_buffer_to_image_glsl",
            glsl_nc4hw4_buffer_to_image_glsl)
      : getProgram(
            "glsl_nchw_buffer_to_image_glsl", glsl_nchw_buffer_to_image_glsl);
  program->useProgram();

  glBindImageTexture(
      0, textureId, 0, GL_TRUE, 0, GL_WRITE_ONLY, getTextureFormat());
  AGL_CHECK_ERROR;
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffer->getId());
  AGL_CHECK_ERROR;
  glUniform1i(2, w);
  glUniform1i(3, h);
  AGL_CHECK_ERROR;
  compute(UP_DIV(w, 8), UP_DIV(h, 8), c_4);
  AGL_CHECK_ERROR;
}

static std::unique_ptr<AGLContext> glContext;

void agpu_conv2d(
    const float* input,
    uint32_t input_n,
    uint32_t input_c,
    uint32_t input_h,
    uint32_t input_w,
    const float* weights,
    uint32_t kernel_c,
    uint32_t kernel_h,
    uint32_t kernel_w,
    const float* bias,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t input_padding_h,
    uint32_t input_padding_w,
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint32_t groups,
    float* output) {
  AGPU_PRINT(
      "%s",
      std::string{R"(
            .-"""-.
           /       \
           \       /
    .-"""-.-`.-.-.<  _
   /      _,-\ ()()_/:)
   \     / ,  `     `|
    '-..-| \-.,___,  /
          \ `-.__/  /
     jgs / `-.__.-\`
        / /|    ___\
       ( ( |.-"`   `'\
        \ \/    {}{}  |
         \|           /
          \        , /
          ( __`;-;'__`)
          `//'`   `||`
         _//       ||
 .-"-._,(__)     .(__).-""-.
/          \    /           \
\          /    \           /
 `'-------`      `--------'`
)"}
          .c_str());
  AGPU_PRINT(
      "agpu_conv2d(input nchw %d %d %d %d kernel chw %d %d %d stride hw %d %d i_pad hw %d %d dilation %d %d groups %d",
      input_n,
      input_c,
      input_h,
      input_w,
      kernel_c,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      input_padding_h,
      input_padding_w,
      dilation_h,
      dilation_w,
      groups);

  static const int once = []() {
    AGPU_PRINT("Creating GLContext...");
    glContext = std::make_unique<AGLContext>();
    if (!glContext) {
      AGPU_PRINT("ERROR Failed to create GLContext");
      assert(false);
    }
    AGPU_PRINT("GLContext created ok");
    return 0;
  }();
  ((void)once);

  uint32_t idims[4] = {input_n, input_c, input_h, input_w};
  agpu_print("input:", input, 4, idims);
  uint32_t kdims[4] = {kernel_c, input_c, kernel_h, kernel_w};
  agpu_print("kernel:", weights, 4, kdims);
  uint32_t bdims[1] = {kernel_c};
  agpu_print("bias:", bias, 1, bdims);

  uint32_t totalWeightSize =
      ALIGN_UP4(kernel_c) * ALIGN_UP4(input_c) * kernel_h * kernel_w;
  AGPU_PRINT("totalWeightSize:%d", totalWeightSize);

  auto biasBuffer =
      std::make_unique<AGLBuffer>(sizeof(float) * ALIGN_UP4(kernel_c));
  float* biasPtr = (float*)(biasBuffer->map(
      GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
  if (biasPtr) {
    memset(biasPtr, 0, sizeof(float) * ALIGN_UP4(kernel_c));
    memcpy(biasPtr, bias, sizeof(float) * ALIGN_UP4(kernel_c));
  }
  biasBuffer->unmap();

  auto kernelBuffer =
      std::make_unique<AGLBuffer>(sizeof(float) * totalWeightSize);
  const int unit = 4;
  const int unit2 = unit * unit;

  const int alignedWeightSize =
      UP_DIV(input_c, unit) * kernel_w * kernel_h * unit2;
  const int oc_4 = UP_DIV(kernel_c, unit);
  AGPU_PRINT("alignedWeightSize:%d", alignedWeightSize);
  AGPU_PRINT("oc_4:%d", oc_4);

  float* kernelPtr = (float*)(kernelBuffer->map(
      GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));

  if (kernelPtr) {
    memset(kernelPtr, 0, sizeof(float) * totalWeightSize);
    const float* src = weights;
    float* dst = kernelPtr;
    int cur = 0;
    // (oc, ic, h, w) -> (oc/4, ic/4, ky kx ic4 oc4)
    for (int b = 0; b < kernel_c; ++b) {
      int b_4 = b / unit;
      float* dst_b = dst + b_4 * alignedWeightSize;
      int mx = b % unit;
      for (int d = 0; d < input_c; ++d) {
        int my = d % unit;
        int d_4 = d / unit;
        float* dst_d = dst_b + d_4 * kernel_w * kernel_h * unit2;
        for (int y = 0; y < kernel_h; ++y) {
          float* dst_y = dst_d + y * kernel_w * unit2;
          for (int x = 0; x < kernel_w; ++x) {
            float* dst_x = dst_y + x * unit2;
            dst_x[unit * my + mx] = src[cur++];
          }
        }
      }
    }
  }
  kernelBuffer->unmap();

  uint32_t rkdims[4] = {kernel_h, kernel_w, 4, 4};
  agpu_print("repacked kernel", kernelPtr, 4, rkdims);

  int ic_4 = UP_DIV(input_c, unit);
  AGPU_PRINT("ic_4:%d", ic_4);

  AGPU_PRINT(
      "kernelTexture(%d, %d, %d)", ic_4 * unit, oc_4, kernel_w * kernel_h);
  auto kernelTexture = std::make_unique<AGLTexture>(
      ic_4 * unit,
      oc_4,
      kernel_w * kernel_h,
      getTextureFormat(),
      GL_TEXTURE_3D,
      false);

  auto transform = getProgram(
      "glsl_kernel2image_adreno_glsl", glsl_kernel2image_adreno_glsl);
  transform->useProgram();
  glBindImageTexture(
      0, kernelTexture->id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, getTextureFormat());
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, kernelBuffer->getId());
  AGL_CHECK_ERROR;
  glUniform1i(3, kernel_w * kernel_h);
  glUniform1i(4, ic_4);
  AGL_CHECK_ERROR;

  compute(ic_4, oc_4, kernel_w * kernel_h);
  AGL_CHECK_ERROR;
  // kernelTexture done

  auto inputTexture = std::make_unique<AGLTexture>(
      input_w, input_h, ic_4, getTextureFormat(), GL_TEXTURE_3D, false);
  host2device(
      inputTexture->id(),
      input,
      input_c,
      input_h,
      input_w,
      false /* input not aligned */);
  // inputTexture done

  // onResize
  std::vector<std::string> prefix;
  // prefix.push_back("#define RELU");

  auto dstDepthQuad = UP_DIV(kernel_c, 4);

  int localSize[3];

  int setLocalSizeX = 1;
  int setLocalSizeY = 1;
  int setLocalSizeZ = dstDepthQuad;

  GLint maxLocalSizeX, maxLocalSizeY, maxLocalSizeZ;

  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maxLocalSizeX);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &maxLocalSizeY);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &maxLocalSizeZ);

  localSize[0] = setLocalSizeX < maxLocalSizeX ? setLocalSizeX : maxLocalSizeX;
  localSize[1] = setLocalSizeY < maxLocalSizeY ? setLocalSizeY : maxLocalSizeY;
  localSize[2] = setLocalSizeZ < maxLocalSizeZ ? setLocalSizeZ : maxLocalSizeZ;
  {
    std::ostringstream os;
    os << "#define XLOCAL " << localSize[0];
    prefix.push_back(os.str());
  }
  {
    std::ostringstream os;
    os << "#define YLOCAL " << localSize[1];
    prefix.push_back(os.str());
  }
  {
    std::ostringstream os;
    os << "#define ZLOCAL " << localSize[2];
    prefix.push_back(os.str());
  }
  AGPU_PRINT("localSize:%d %d %d", localSize[0], localSize[1], localSize[2]);

  // general convolution, no 1x1 separation
  auto program = getProgram("convolution", glsl_convolution_glsl, prefix);

  // onExecute
  uint32_t output_w = ((input_w - kernel_w + input_padding_w) / stride_w) + 1;
  uint32_t output_h = ((input_h - kernel_h + input_padding_h) / stride_h) + 1;
  AGPU_PRINT(
      "output size: output_w:%d output_h:%d oc_4:%d kernel_c:%d",
      output_w,
      output_h,
      oc_4,
      kernel_c);
  auto outputTexture = std::make_unique<AGLTexture>(
      output_w, output_h, oc_4, getTextureFormat(), GL_TEXTURE_3D, false);

  program->useProgram();
  glBindImageTexture(
      0, outputTexture->id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, getTextureFormat());
  {
    int texId = 0;
    glActiveTexture(GL_TEXTURE0 + texId);
    glUniform1i(1, texId);
    glBindTexture(GL_TEXTURE_3D, inputTexture->id());
    AGL_CHECK_ERROR;
  }
  {
    int texId = 1;
    glActiveTexture(GL_TEXTURE0 + texId);
    AGL_CHECK_ERROR;
    glUniform1i(2, texId);
    AGL_CHECK_ERROR;
    glBindTexture(GL_TEXTURE_3D, kernelTexture->id());
    AGL_CHECK_ERROR;
  }

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, biasBuffer->getId());
  glUniform2i(4, input_padding_w, input_padding_h);
  glUniform2i(5, kernel_w, kernel_h);
  glUniform2i(6, stride_w, stride_h);
  glUniform2i(7, dilation_w, dilation_h);

  AGL_CHECK_ERROR;
  glUniform3i(10, output_w, output_h, UP_DIV(kernel_c, 4));
  glUniform3i(11, input_w, input_h, UP_DIV(input_c, 4));

  glUniform1i(8, unit);
  AGL_CHECK_ERROR;

  AGPU_PRINT("convolution compute()");
  compute(
      UP_DIV(output_w, unit * localSize[0]),
      UP_DIV(output_h, localSize[1]),
      UP_DIV(oc_4, localSize[2]));
  AGL_CHECK_ERROR;

  device2host(
      outputTexture->id(),
      output,
      output_w,
      output_h,
      kernel_c,
      false /* align */);
  AGPU_PRINT("out h:%d w:%d", output_h, output_w);
  uint32_t odims[4] = {input_n, kernel_c, output_h, output_w};
  agpu_print("output:", output, 4, odims);
}
#endif
} // namespace agpu
