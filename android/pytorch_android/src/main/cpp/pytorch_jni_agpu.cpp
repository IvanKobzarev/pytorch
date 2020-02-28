#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <typeinfo>

#include <ATen/AgpuUtils.h>
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "agpu.h"
#include "pytorch_jni_agpu.h"

#ifdef __ANDROID__
#include <android/log.h>
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, "XXX", __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, "XXX", __VA_ARGS__)
#endif

namespace pytorch_jni_agpu {

template <typename T>
void log(const char* m, T t) {
  std::ostringstream os;
  os << t << std::endl;
  ALOGI("%s %s", m, os.str().c_str());
}

void agpu_print(const char* m, const float* t, uint32_t rank, uint32_t* dims) {
  static const char* kFloatFormat = "%12.12f";
  std::cout << m << std::endl;

  std::cout << " dims:(";
  for (uint32_t i = 0; i < rank; i++) {
    std::cout << dims[i] << " ";
  }
  std::cout << ")";

  if (rank == 0) {
    std::cout << *t;
  } else if ((rank == 1) || (rank == 2)) {
    char fbuf[12];
    uint32_t rows = rank == 1 ? 1 : dims[0];
    uint32_t cols = rank == 1 ? dims[0] : dims[1];
    auto rrange = std::min(rows, 100u);
    auto crange = std::min(cols, 100u);
    for (uint32_t i = 0; i < rrange; i++) {
      std::cout << "\n";
      for (uint32_t j = 0; j < crange; j++) {
        sprintf(fbuf, kFloatFormat, t[i * cols + j]);
        std::cout << "(" << i << "," << j << ")" << fbuf << std::endl;
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
    // assert(false);
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

void print_tensor(const char* m, const at::Tensor& t) {
  auto ts = t.sizes();
  std::cout << "print_tensor ts:" << ts << std::endl;
  auto r = ts.size();
  uint32_t* dims = new uint32_t[r];
  for (uint32_t i = 0; i < r; ++i) {
    dims[i] = ts[i];
    std::cout << "print_tensor dims:" << dims[i] << std::endl;
  }
  agpu_print(m, t.data_ptr<float>(), r, dims);
  delete[] dims;
}

static bool almostEqual(
    const at::Tensor& t,
    const at::Tensor& expected,
    bool logOnFalse = false) {
  double rtol = 0.001;
  double atol = 0.0001;
  bool ret = torch::allclose(t, expected, rtol, atol, true);
  if (logOnFalse && !ret) {
    auto diff = (t - expected).abs();
    auto diff_max = diff.max().item<float>();
    ALOGI("almostEquals abs_diff_max:%12.8f", diff_max);

    double rtoli = 0.1;
    int i = 1;
    while (i < 5) {
      ALOGI(
          "almostEquals allClose(%d rtoli:%12.8f):%d",
          i++,
          rtoli,
          torch::allclose(t, expected, rtoli, 0.00001, true));
      rtoli *= 0.1;
    }
    print_tensor("diff:", diff);
    print_tensor("almostEquals t:", t);
    print_tensor("almostEquals expected:", expected);
  }
  return ret;
}

static void agpuOff() {
  at::setUseAgpuNorm(false);
  at::setUseAgpuAdd(false);
  at::setUseAgpuRelu(false);
  at::setUseAgpuConv(false);
}

static void test_conv_(
    int64_t n,
    int64_t h,
    int64_t w,
    int64_t kh,
    int64_t kw,
    int64_t py,
    int64_t px,
    int64_t s,
    int64_t d,
    int64_t g,
    int64_t gcin,
    int64_t gcout,
    bool log = false) {
  ALOGI(
      "test_conv_ nhw (%" PRId64 " %" PRId64 " %" PRId64 ") kh kw (%" PRId64
      " %" PRId64 ") pad yx (%" PRId64 " %" PRId64 ") s:%" PRId64 " d:%" PRId64
      " g:%" PRId64 " cin:%" PRId64 " cout:%" PRId64,
      n,
      h,
      w,
      kh,
      kw,
      py,
      px,
      s,
      d,
      g,
      gcin,
      gcout);

  auto input = torch::randn({n, gcin, h, w}, torch::kFloat);
  auto weight = torch::randn({gcout, gcin, kh, kw}, torch::kFloat);
  auto bias = torch::randn({gcout}, torch::kFloat);

  agpuOff();
  auto outputC = at::conv2d(
      input,
      weight,
      bias,
      c10::IntArrayRef{s}, // stride
      c10::IntArrayRef{py, px}, // padding
      c10::IntArrayRef{d}, // dilation
      1);

  at::setUseAgpuConv(true);
  auto outputT = at::conv2d(
      input,
      weight,
      bias,
      c10::IntArrayRef{s}, // stride
      c10::IntArrayRef{py, px}, // padding
      c10::IntArrayRef{d}, // dilation
      1);
  agpuOff();
  assert(almostEqual(outputC, outputT, log));
}

TEST(conv, xxs) {
  test_conv_(
      /* n */ 1,
      /* h */ 3,
      /* w */ 3,
      /* kh */ 3,
      /* kw */ 3,
      /* ph */ 0,
      /* pw */ 0,
      /* s */ 1,
      /* d */ 1,
      /* g */ 1,
      /* gcin */ 3,
      /* gcout */ 3,
      /* log */ false);
}

TEST(conv, xxs_padding) {
  test_conv_(
      /* n */ 1,
      /* h */ 3,
      /* w */ 3,
      /* kh */ 3,
      /* kw */ 3,
      /* ph */ 1,
      /* pw */ 1,
      /* s */ 1,
      /* d */ 1,
      /* g */ 1,
      /* gcin */ 3,
      /* gcout */ 3,
      /* log */ true);
}

TEST(conv, xs_padding) {
  test_conv_(
      /* n */ 1,
      /* h */ 3,
      /* w */ 3,
      /* kh */ 3,
      /* kw */ 3,
      /* py */ 1,
      /* px */ 1,
      /* s */ 1,
      /* d */ 1,
      /* g */ 1,
      /* gcin */ 3,
      /* gcout */ 3,
      /* log */ true);
}

TEST(conv, xxs_stride) {
  test_conv_(
      /* n */ 1,
      /* h */ 3,
      /* w */ 3,
      /* kh */ 3,
      /* kw */ 3,
      /* ph */ 0,
      /* pw */ 0,
      /* s */ 2,
      /* d */ 1,
      /* g */ 1,
      /* gcin */ 12,
      /* gcout */ 12,
      /* log */ true);
}

TEST(conv, xxs_padding2) {
  test_conv_(
      /* n */ 1,
      /* h */ 3,
      /* w */ 3,
      /* kh */ 3,
      /* kw */ 3,
      /* ph */ 2,
      /* pw */ 2,
      /* s */ 1,
      /* d */ 1,
      /* g */ 1,
      /* gcin */ 3,
      /* gcout */ 3,
      /* log */ true);
}

TEST(conv, mn2) {
  //         n,   h,   w kh,kw,py,px, s, d,   g,  cin, cout
  test_conv_(1, 224, 224, 3, 3, 1, 1, 2, 1, 1, 3, 32);
  test_conv_(1, 112, 112, 3, 3, 1, 1, 1, 1, 32, 1, 1);
  test_conv_(1, 112, 112, 1, 1, 0, 0, 1, 1, 1, 32, 16);
  test_conv_(1, 112, 112, 1, 1, 0, 0, 1, 1, 1, 16, 96);
  test_conv_(1, 112, 112, 3, 3, 1, 1, 2, 1, 96, 1, 1);
  test_conv_(1, 56, 56, 1, 1, 0, 0, 1, 1, 1, 96, 24);
  test_conv_(1, 56, 56, 1, 1, 0, 0, 1, 1, 1, 24, 144);
  test_conv_(1, 56, 56, 3, 3, 1, 1, 1, 1, 144, 1, 1);
  test_conv_(1, 56, 56, 1, 1, 0, 0, 1, 1, 1, 144, 24);
  test_conv_(1, 56, 56, 3, 3, 1, 1, 2, 1, 144, 1, 1);
  test_conv_(1, 28, 28, 1, 1, 0, 0, 1, 1, 1, 144, 32);
  test_conv_(1, 28, 28, 1, 1, 0, 0, 1, 1, 1, 32, 192);
  test_conv_(1, 28, 28, 3, 3, 1, 1, 1, 1, 192, 1, 1);
  test_conv_(1, 28, 28, 1, 1, 0, 0, 1, 1, 1, 192, 32);
  test_conv_(1, 28, 28, 3, 3, 1, 1, 2, 1, 192, 1, 1);
  test_conv_(1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 192, 64);
  test_conv_(1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 64, 384);
  test_conv_(1, 14, 14, 3, 3, 1, 1, 1, 1, 384, 1, 1);
  test_conv_(1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 384, 64);
  test_conv_(1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 384, 96);
  test_conv_(1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 96, 576);
  test_conv_(1, 14, 14, 3, 3, 1, 1, 1, 1, 576, 1, 1);
  test_conv_(1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 576, 96);
  test_conv_(1, 14, 14, 3, 3, 1, 1, 2, 1, 576, 1, 1);
  test_conv_(1, 7, 7, 1, 1, 0, 0, 1, 1, 1, 576, 160);
  test_conv_(1, 7, 7, 1, 1, 0, 0, 1, 1, 1, 160, 960);
  test_conv_(1, 7, 7, 3, 3, 1, 1, 1, 1, 960, 1, 1);
  test_conv_(1, 7, 7, 1, 1, 0, 0, 1, 1, 1, 960, 160);
  test_conv_(1, 7, 7, 1, 1, 0, 0, 1, 1, 1, 960, 320);
  test_conv_(1, 7, 7, 1, 1, 0, 0, 1, 1, 1, 320, 1280);
  test_conv_(1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1280, 1000);
}

TEST(add, xxs) {
  int64_t n = 1;
  int64_t ih = 3;
  int64_t iw = 3;
  int64_t kc = 3;
  auto tina = torch::randn({n, kc, ih, iw}, torch::kFloat);
  auto tinb = torch::randn({n, kc, ih, iw}, torch::kFloat);

  agpuOff();
  auto toutC = torch::add(tina, tinb);
  at::setUseAgpuAdd(true);
  auto toutT = torch::add(tina, tinb);
  agpuOff();
  assert(almostEqual(toutC, toutT));
}

TEST(threshold, xxs) {
  int64_t n = 1;
  int64_t ih = 3;
  int64_t iw = 3;
  int64_t kc = 3;
  auto tin = torch::randn({n, kc, ih, iw}, torch::kFloat);
  agpuOff();
  auto toutC = at::relu(tin);
  at::setUseAgpuRelu(true);
  auto toutT = at::relu(tin);
  agpuOff();
  assert(almostEqual(toutC, toutT));
}

TEST(norm, xxs) {
  int64_t n = 1;
  int64_t ic = 3;
  int64_t ih = 3;
  int64_t iw = 3;
  auto tin = torch::randn({n, ic, ih, iw}, torch::kFloat);
  auto weight = torch::randn({ic}, torch::kFloat);
  auto bias = torch::randn({ic}, torch::kFloat);
  auto mean = torch::ones({ic}, torch::kFloat);
  auto var = torch::ones({ic}, torch::kFloat);

  agpuOff();
  auto toutC =
      at::batch_norm(tin, weight, bias, mean, var, false, 0.1, 0.00001, false);
  at::setUseAgpuNorm(true);
  auto toutT =
      at::batch_norm(tin, weight, bias, mean, var, false, 0.1, 0.00001, false);
  agpuOff();
  assert(almostEqual(toutC, toutT));
}

struct ArgsCV {
  ArgsCV(const std::string& args) {
    std::istringstream iss(args);
    std::vector<std::string> argsVec(
        std::istream_iterator<std::string>{iss},
        std::istream_iterator<std::string>());

    int argc = argsVec.size();
    char** argv = new char*[argc];
    for (size_t i = 0; i < argc; i++) {
      argv[i] = new char[argsVec[i].size() + 1];
      std::strcpy(argv[i], argsVec[i].c_str());
      ALOGI("%s", argsVec[i].c_str());
    }
    c = argc;
    v = argv;
  }

  ArgsCV(int c, char** v) : c(c), v(v) {}

  ~ArgsCV() {
    for (size_t i = 0; i < c; i++) {
      delete[] v[i];
    }
    delete[] v;
  }

  int c;
  char** v;
};

void gtest_main(const std::string& args) {
  ALOGI("pytorch_android_agpu::test(%s)", args.c_str());
  auto argscv = ArgsCV{args};
  auto _ = RUN_ALL_TESTS();
  ((void)_);
  ::testing::InitGoogleTest(&(argscv.c), argscv.v);
}

static void BM_conv(benchmark::State& state, const char* name) {
  for (auto _ : state) {
    state.PauseTiming();
    const int64_t useAgpu = state.range(0);

    const int64_t n = state.range(1);
    const int64_t h = state.range(2);
    const int64_t w = state.range(3);

    const int64_t kh = state.range(4);
    const int64_t kw = state.range(5);

    const int64_t py = state.range(6);
    const int64_t px = state.range(7);

    const int64_t stride = state.range(8);
    const int64_t dilation = state.range(9);

    const int64_t groups = state.range(10);
    const int64_t groups_c_in = state.range(11);
    const int64_t groups_c_out = state.range(12);

    const int64_t c_in = groups_c_in;
    const int64_t c_out = groups_c_out;
    agpuOff();
    at::setUseAgpuConv((useAgpu != 0));

    auto tin = torch::randn({n, c_in, h, w}, torch::kFloat);
    auto tw = torch::randn({c_out, c_in, kh, kw}, torch::kFloat);
    auto tb = torch::randn({c_out}, torch::kFloat);
    int64_t g = 1;
    torch::nn::functional::Conv2dFuncOptions o =
        torch::nn::functional::Conv2dFuncOptions().stride(stride).padding(0);
    state.ResumeTiming();
    auto tout = at::conv2d(
        tin,
        tw,
        tb,
        c10::IntArrayRef{stride}, // stride
        c10::IntArrayRef{py, px}, // padding
        c10::IntArrayRef{dilation}, // dilation
        g);
  }
}

static void BM_convArgs(
    benchmark::internal::Benchmark* b,
    int64_t n,
    int64_t h,
    int64_t w,
    int64_t kh,
    int64_t kw,
    int64_t py,
    int64_t px,
    int64_t s,
    int64_t d,
    int64_t g,
    int64_t gcin,
    int64_t gcout) {
  b->Args({0, n, h, w, kh, kw, py, px, s, d, g, gcin, gcout});
  b->Args({1, n, h, w, kh, kw, py, px, s, d, g, gcin, gcout});
}

static void BM_conv_args_base(benchmark::internal::Benchmark* b) {
  b->ArgNames({"AGPU",
               "N",
               "H",
               "W",
               "KH",
               "KW",
               "py",
               "px",
               "S",
               "D",
               "G",
               "GCin",
               "GCout"});
  /*             N   H    W   KH  KW  py  px  S  D    G  GCin  GCout */
  BM_convArgs(b, 1, 224, 224, 3, 3, 2, 1, 1, 1, 1, 3, 32);
  BM_convArgs(b, 1, 112, 112, 3, 3, 2, 1, 1, 1, 96, 1, 1);
  BM_convArgs(b, 1, 56, 56, 1, 1, 0, 0, 1, 1, 1, 144, 24);
  BM_convArgs(b, 1, 28, 28, 3, 3, 1, 1, 2, 1, 192, 1, 1);
  BM_convArgs(b, 1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 384, 96);
  BM_convArgs(b, 1, 7, 7, 3, 3, 1, 1, 1, 1, 960, 1, 1);
}

static void BM_conv_args_MobileNetV2(benchmark::internal::Benchmark* b) {
  b->ArgNames({"AGPU",
               "N",
               "H",
               "W",
               "KH",
               "KW",
               "py",
               "px",
               "S",
               "D",
               "G",
               "GCin",
               "GCout"});

  /*             N   H    W   KH  KW  py  px  S  D    G  GCin  GCout */
  BM_convArgs(b, 1, 224, 224, 3, 3, 1, 1, 2, 1, 1, 3, 32);
  BM_convArgs(b, 1, 112, 112, 3, 3, 1, 1, 1, 1, 32, 1, 1);
  BM_convArgs(b, 1, 112, 112, 1, 1, 0, 0, 1, 1, 1, 32, 16);
  BM_convArgs(b, 1, 112, 112, 1, 1, 0, 0, 1, 1, 1, 16, 96);
  BM_convArgs(b, 1, 112, 112, 3, 3, 1, 1, 2, 1, 96, 1, 1);
  BM_convArgs(b, 1, 56, 56, 1, 1, 0, 0, 1, 1, 1, 96, 24);
  BM_convArgs(b, 1, 56, 56, 1, 1, 0, 0, 1, 1, 1, 24, 144);
  BM_convArgs(b, 1, 56, 56, 3, 3, 1, 1, 1, 1, 144, 1, 1);
  BM_convArgs(b, 1, 56, 56, 1, 1, 0, 0, 1, 1, 1, 144, 24);
  BM_convArgs(b, 1, 56, 56, 3, 3, 1, 1, 2, 1, 144, 1, 1);
  BM_convArgs(b, 1, 28, 28, 1, 1, 0, 0, 1, 1, 1, 144, 32);
  BM_convArgs(b, 1, 28, 28, 1, 1, 0, 0, 1, 1, 1, 32, 192);
  BM_convArgs(b, 1, 28, 28, 3, 3, 1, 1, 1, 1, 192, 1, 1);
  BM_convArgs(b, 1, 28, 28, 1, 1, 0, 0, 1, 1, 1, 192, 32);
  BM_convArgs(b, 1, 28, 28, 3, 3, 1, 1, 2, 1, 192, 1, 1);
  BM_convArgs(b, 1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 192, 64);
  BM_convArgs(b, 1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 64, 384);
  BM_convArgs(b, 1, 14, 14, 3, 3, 1, 1, 1, 1, 384, 1, 1);
  BM_convArgs(b, 1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 384, 64);
  BM_convArgs(b, 1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 384, 96);
  BM_convArgs(b, 1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 96, 576);
  BM_convArgs(b, 1, 14, 14, 3, 3, 1, 1, 1, 1, 576, 1, 1);
  BM_convArgs(b, 1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 576, 96);
  BM_convArgs(b, 1, 14, 14, 3, 3, 1, 1, 2, 1, 576, 1, 1);
  BM_convArgs(b, 1, 7, 7, 1, 1, 0, 0, 1, 1, 1, 576, 160);
  BM_convArgs(b, 1, 7, 7, 1, 1, 0, 0, 1, 1, 1, 160, 960);
  BM_convArgs(b, 1, 7, 7, 3, 3, 1, 1, 1, 1, 960, 1, 1);
  BM_convArgs(b, 1, 7, 7, 1, 1, 0, 0, 1, 1, 1, 960, 160);
  BM_convArgs(b, 1, 7, 7, 1, 1, 0, 0, 1, 1, 1, 960, 320);
  BM_convArgs(b, 1, 7, 7, 1, 1, 0, 0, 1, 1, 1, 320, 1280);
  BM_convArgs(b, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1280, 1000);
}

torch::jit::script::Module module_;

void BM_moduleForward(benchmark::State& state, const char* name) {
  for (auto _ : state) {
    state.PauseTiming();
    torch::autograd::AutoGradMode no_autograd_guard{false};
    torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
    auto tin = torch::randn({1, 3, 224, 224}, torch::kFloat);

    const int64_t useAgpu = state.range(0);
    if (useAgpu == 0) {
      agpuOff();
    } else {
      at::setAgpuVerbose(false);
      at::setUseAgpuConv(true);
      at::setUseAgpuNorm(true);
      at::setUseAgpuRelu(true);
      at::setUseAgpuAdd(true);
    }

    state.ResumeTiming();
    auto out = module_.forward({tin});
    agpuOff();
  }
}

void gbench_module(torch::jit::script::Module module, const std::string& args) {
  ALOGI("pytorch_android_agpu::gbench_module(%s)", args.c_str());
  auto argscv = ArgsCV{args};
  module_ = std::move(module);
  BENCHMARK_CAPTURE(BM_moduleForward, fwdBase, "fwdBase")
      ->Arg(0)
      ->Arg(1)
      //->Iterations(1)
      //->Repetitions(1)
      ->Unit(benchmark::kMillisecond)
      //->ReportAggregatesOnly(true)
      ->UseRealTime();

  benchmark::Initialize(&(argscv.c), argscv.v);
  benchmark::RunSpecifiedBenchmarks();
}

static void BM_conv_agpu_Args(
    benchmark::internal::Benchmark* b,
    int64_t n,
    int64_t h,
    int64_t w,
    int64_t kh,
    int64_t kw,
    int64_t py,
    int64_t px,
    int64_t s,
    int64_t d,
    int64_t g,
    int64_t gcin,
    int64_t gcout) {
  b->Args({0, n, h, w, kh, kw, py, px, s, d, g, gcin, gcout});
  b->Args({1, n, h, w, kh, kw, py, px, s, d, g, gcin, gcout});
  b->Args({2, n, h, w, kh, kw, py, px, s, d, g, gcin, gcout});
  // b->Args({4, n, h, w, kh, kw, py, px, s, d, g, gcin, gcout});
  // b->Args({5, n, h, w, kh, kw, py, px, s, d, g, gcin, gcout});
  b->Args({10, n, h, w, kh, kw, py, px, s, d, g, gcin, gcout});
  // b->Args({10, n, h, w, kh, kw, py, px, s, d, g, gcin, gcout});
  b->Args({11, n, h, w, kh, kw, py, px, s, d, g, gcin, gcout});
  b->Args({20, n, h, w, kh, kw, py, px, s, d, g, gcin, gcout});
  b->Args({30, n, h, w, kh, kw, py, px, s, d, g, gcin, gcout});
  b->Args({31, n, h, w, kh, kw, py, px, s, d, g, gcin, gcout});
  // b->Args({32, n, h, w, kh, kw, py, px, s, d, g, gcin, gcout});
  b->Args({40, n, h, w, kh, kw, py, px, s, d, g, gcin, gcout});
}

static void BM_conv_agpu_args_base(benchmark::internal::Benchmark* b) {
  b->ArgNames({"ACX",
               "N",
               "H",
               "W",
               "KH",
               "KW",
               "py",
               "px",
               "S",
               "D",
               "G",
               "GCin",
               "GCout"});
  /*             N   H    W   KH  KW  py  px  S  D    G  GCin  GCout */
  BM_conv_agpu_Args(b, 1, 224, 224, 3, 3, 2, 1, 1, 1, 1, 3, 32);
  // BM_conv_agpu_Args(b, 1, 112, 112, 3, 3, 2, 1, 1, 1, 96, 1, 1);
  // BM_conv_agpu_Args(b, 1, 56, 56, 1, 1, 0, 0, 1, 1, 1, 144, 24);
  // BM_conv_agpu_Args(b, 1, 28, 28, 3, 3, 1, 1, 2, 1, 192, 1, 1);
  // BM_conv_agpu_Args(b, 1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 384, 96);
  // BM_conv_agpu_Args(b, 1, 7, 7, 3, 3, 1, 1, 1, 1, 960, 1, 1);
}

static void BM_conv_agpu_args_base_once(benchmark::internal::Benchmark* b) {
  b->ArgNames({"ACX",
               "N",
               "H",
               "W",
               "KH",
               "KW",
               "py",
               "px",
               "S",
               "D",
               "G",
               "GCin",
               "GCout"});
  /*             N   H    W   KH  KW  py  px  S  D    G  GCin  GCout */
  BM_conv_agpu_Args(b, 1, 224, 224, 3, 3, 2, 1, 1, 1, 1, 3, 32);
  // BM_conv_agpu_Args(b, 1, 112, 112, 3, 3, 2, 1, 1, 1, 96, 1, 1);
  // BM_conv_agpu_Args(b, 1, 56, 56, 1, 1, 0, 0, 1, 1, 1, 144, 24);
  // BM_conv_agpu_Args(b, 1, 28, 28, 3, 3, 1, 1, 2, 1, 192, 1, 1);
  // BM_conv_agpu_Args(b, 1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 384, 96);
  // BM_conv_agpu_Args(b, 1, 7, 7, 3, 3, 1, 1, 1, 1, 960, 1, 1);
}

uint64_t getCurrentCpuFrequency() {
#ifdef __linux__
  int freq = 0;
  char cpuinfo_name[512];
  int cpu = sched_getcpu();
  snprintf(
      cpuinfo_name,
      sizeof(cpuinfo_name),
      "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq",
      cpu);

  FILE* f = fopen(cpuinfo_name, "r");
  if (f) {
    if (fscanf(f, "%d", &freq)) {
      fclose(f);
      return uint64_t(freq) * 1000;
    }
    fclose(f);
  }
#endif // __linux__
  return 0;
}

static void* wipe_buffer = nullptr;
static size_t wipe_buffer_size = 0;

static pthread_once_t wipe_buffer_guard = PTHREAD_ONCE_INIT;

static void initWipeBuffer() {
  // Default: the largest know cache size (128 MB Intel Crystalwell L4 cache).
  wipe_buffer_size = 128 * 1024 * 1024;
#if defined(__ANDROID__)
  // memalign is obsolete, but it is the only option on Android until API
  // level 17.
  wipe_buffer = memalign(128, wipe_buffer_size);
#else
  (void)posix_memalign((void**)&wipe_buffer, 128, wipe_buffer_size);
#endif
  if (wipe_buffer != nullptr) {
    memset(wipe_buffer, 0xA5, wipe_buffer_size);
  }
}

static uint32_t prefetchToL1(const void* ptr, size_t size) {
  uint32_t step = 16;
  const uint8_t* u8_ptr = static_cast<const uint8_t*>(ptr);
  // Compute and return sum of data to prevent compiler from removing data
  // reads.
  uint32_t sum = 0;
  while (size >= step) {
    sum += uint32_t(*u8_ptr);
    u8_ptr += step;
    size -= step;
  }
  return sum;
}

uint32_t wipeCache() {
  pthread_once(&wipe_buffer_guard, &initWipeBuffer);
  return prefetchToL1(wipe_buffer, wipe_buffer_size);
}

static void BM_conv_agpu(benchmark::State& state, const char* name) {
  const int64_t agpuConvX = state.range(0);

  const int64_t agpuConvMethod = agpuConvX / 10;
  const int64_t agpuConvMethodMod = agpuConvX % 10;

  const int64_t n = state.range(1);
  const int64_t h = state.range(2);
  const int64_t w = state.range(3);

  const int64_t kh = state.range(4);
  const int64_t kw = state.range(5);

  const int64_t py = state.range(6);
  const int64_t px = state.range(7);

  const int64_t stride = state.range(8);
  const int64_t dilation = state.range(9);

  const int64_t groups = state.range(10);
  const int64_t groups_c_in = state.range(11);
  const int64_t groups_c_out = state.range(12);

  const int64_t c_in = groups_c_in;
  const int64_t c_out = groups_c_out;

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng =
      std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), rng);

  std::vector<float> input(n * c_in * h * w);
  std::generate(input.begin(), input.end(), std::ref(f32rng));

  std::vector<float> kernel(c_out * c_in * kh * kw);
  std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));

  std::vector<float> bias(c_out);
  std::generate(bias.begin(), bias.end(), std::ref(f32rng));

  const size_t khe = (kh - 1) * dilation + 1;
  const size_t kwe = (kw - 1) * dilation + 1;
  const size_t output_h = (h + 2 * py - khe) / stride + 1;
  const size_t output_w = (w + 2 * px - kwe) / stride + 1;

  std::vector<float> output(n * c_out * output_w * output_h);
  using fp_conv2d_t = decltype(&agpu::agpu_conv2d_);
  static fp_conv2d_t fa[6] = {
      /*  0 */ agpu::agpu_conv2d_tex_IKnc4hw,

      /* 10 */ agpu::agpu_conv2d_buf_IKnchw_SIKOnc4hw_KrO4C4HW,
      /* 20 */ agpu::agpu_conv2d_buf_IKnchw_SIKOnc4hw_KrO4HWC,

      /* 30 */ agpu::agpu_conv2d_buf_IKnchw_SIKnc4hw_SOnchw,
      /* 40 */ agpu::agpu_conv2d_buf_IKnchw_SKnc4hw,
      /* 50 */ agpu::agpu_conv2d_kernel_repack_};

  for (auto _ : state) {
    state.PauseTiming();
    wipeCache();
    prefetchToL1(bias.data(), sizeof(float) * bias.size());
    prefetchToL1(kernel.data(), sizeof(float) * kernel.size());
    prefetchToL1(input.data(), sizeof(float) * input.size());
    state.ResumeTiming();
    auto aresult = fa[agpuConvMethod](
        input.data(),
        n,
        c_in,
        h,
        w,
        kernel.data(),
        c_out,
        kh,
        kw,
        bias.data(),
        stride,
        stride,
        py,
        px,
        dilation,
        dilation,
        groups,
        output.data(),
        agpuConvMethodMod);
    state.PauseTiming();

    state.SetIterationTime(aresult.gpu_shader_total_time());
    state.counters["\nS_Total     "] = aresult.gpu_shader_total_time();
    state.counters["\nC_K_pack    "] = aresult.cpu_kernel_repackO4I4_time;
    state.counters["\nS_Conv      "] = aresult.gpu_shader_conv_time;
    state.counters["\nS_hK_dTex   "] = aresult.gpu_shader_hkernel_to_dtex_time;
    state.counters["\nS_dTex_hCHW "] = aresult.gpu_shader_dtex_to_hchw_time;
    state.counters["\nS_hCHW_dTex "] = aresult.gpu_shader_hchw_to_dtex_time;
    state.counters["\nS_hCHW_dC4HW"] = aresult.gpu_shader_hchw_to_dc4hw_time;
    state.counters["\nS_dC4HW_hCHW"] = aresult.gpu_shader_dc4hw_to_hchw_time;
    state.ResumeTiming();
  }
  //  state.counters["\nFreq        "] = getCurrentCpuFrequency();
}

static constexpr int64_t kT0_N = 1;
static constexpr int64_t kT0_H = 3;
static constexpr int64_t kT0_W = 3;

static constexpr int64_t kT0_KH = 2;
static constexpr int64_t kT0_KW = 2;

static constexpr int64_t kT0_PY = 0;
static constexpr int64_t kT0_PX = 0;

static constexpr int64_t kT0_S = 1;
static constexpr int64_t kT0_D = 1;

static constexpr size_t kT0_KHE = (kT0_KH - 1) * kT0_D + 1;
static constexpr size_t kT0_KWE = (kT0_KW - 1) * kT0_D + 1;
static constexpr size_t kT0_OH = (kT0_H + 2 * kT0_PY - kT0_KHE) / kT0_S + 1;
static constexpr size_t kT0_OW = (kT0_W + 2 * kT0_PX - kT0_KWE) / kT0_S + 1;

static std::vector<float> kT0_Bias = {0, 0};

static std::vector<float> kT0_InputNCHW = {
    1,    2,    3,    4,    5,    6,    7,    8,    9,
    101,  102,  103,  104,  105,  106,  107,  108,  109,
    1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009};
static std::vector<float> kT0_KernelNCHW = {
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0};
static std::vector<float> kT0_GKernelNCHW =
    {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};

static std::vector<float> kT0_InputNHWC = {
    1, 101, 1001, 2, 102, 1002, 3, 103, 1003, 4, 104, 1004, 5, 105, 1005,
    6, 106, 1006, 7, 107, 1007, 8, 108, 1008, 9, 109, 1009};

static std::vector<float> kT0_KernelNHWC = {
    1,  0, 0, 0, 1,  0, 0, 0, 1,  0, 0, 0,

    -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0};

static std::vector<float> kT0_GKernelNHWC =
    {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};

static std::vector<float> kT0_OutputNCHW =
    {1107, 1110, 1116, 1119, -1107, -1110, -1116, -1119};

static std::vector<float> kT0_OutputNHWC =
    {1107, -1107, 1110, -1110, 1116, -1116, 1119, -1119};

static std::vector<float> kT0_GOutputNCHW =
    {1, 2, 4, 5, 102, 103, 105, 106, 1004, 1005, 1007, 1008};
static std::vector<float> kT0_GOutputNHWC =
    {1, 102, 1004, 2, 103, 1005, 4, 105, 1007, 5, 106, 1008};

static void test0_conv_agpu_IKnchw() {
  const int64_t G = 1;
  const int64_t C = 3;
  const int64_t OC = 2;

  std::vector<float> output(kT0_N * OC * kT0_OH * kT0_OW);
  agpu::agpu_conv2d_buf_IKnchw_SKnc4hw(
      kT0_InputNCHW.data(),
      kT0_N,
      C,
      kT0_H,
      kT0_W,
      kT0_KernelNCHW.data(),
      OC,
      kT0_KH,
      kT0_KW,
      kT0_Bias.data(),
      kT0_S,
      kT0_S,
      kT0_PY,
      kT0_PX,
      kT0_D,
      kT0_D,
      G,
      output.data());

  log("output:", output);
  log("expected_output:", kT0_OutputNCHW);
  assert(kT0_OutputNCHW == output);
}

static void test0_conv_agpu_Inhwc_Knchw() {
  const int64_t G = 1;
  const int64_t C = 3;
  const int64_t OC = 2;

  std::vector<float> output(kT0_N * OC * kT0_OH * kT0_OW);
  agpu::agpu_conv2d_buf_Inhwc_Knchw(
      kT0_InputNHWC.data(),
      kT0_N,
      C,
      kT0_H,
      kT0_W,
      kT0_KernelNCHW.data(),
      OC,
      kT0_KH,
      kT0_KW,
      kT0_Bias.data(),
      kT0_S,
      kT0_S,
      kT0_PY,
      kT0_PX,
      kT0_D,
      kT0_D,
      G,
      output.data());

  log("output:", output);
  log("expected_output:", kT0_OutputNHWC);
  assert(kT0_OutputNHWC == output);
}

static void test0_conv_agpu_IKnhwc() {
  const int64_t G = 1;
  const int64_t C = 3;
  const int64_t OC = 2;

  std::vector<float> output(kT0_N * OC * kT0_OH * kT0_OW);
  agpu::agpu_conv2d_buf_IKnhwc(
      kT0_InputNHWC.data(),
      kT0_N,
      C,
      kT0_H,
      kT0_W,
      kT0_KernelNHWC.data(),
      OC,
      kT0_KH,
      kT0_KW,
      kT0_Bias.data(),
      kT0_S,
      kT0_S,
      kT0_PY,
      kT0_PX,
      kT0_D,
      kT0_D,
      G,
      output.data());

  log("output:", output);
  log("expected_output:", kT0_OutputNHWC);
  assert(kT0_OutputNHWC == output);
}

static void test0_conv_agpu_IKnhwc_KrO4C4HW() {
  const int64_t G = 1;
  const int64_t C = 3;
  const int64_t OC = 2;

  std::vector<float> output(kT0_N * OC * kT0_OH * kT0_OW);
  agpu::agpu_conv2d_buf_IKnhwc_KrO4C4HW(
      kT0_InputNHWC.data(),
      kT0_N,
      C,
      kT0_H,
      kT0_W,
      kT0_KernelNHWC.data(),
      OC,
      kT0_KH,
      kT0_KW,
      kT0_Bias.data(),
      kT0_S,
      kT0_S,
      kT0_PY,
      kT0_PX,
      kT0_D,
      kT0_D,
      G,
      output.data());

  log("output:", output);
  log("expected_output:", kT0_OutputNHWC);
  assert(kT0_OutputNHWC == output);
}

static void test0_conv_agpu_IKnhwc_KrO4HWC() {
  const int64_t G = 1;
  const int64_t C = 3;
  const int64_t OC = 2;

  std::vector<float> output(kT0_N * OC * kT0_OH * kT0_OW);
  agpu::agpu_conv2d_buf_IKnhwc_KrO4HWC(
      kT0_InputNHWC.data(),
      kT0_N,
      C,
      kT0_H,
      kT0_W,
      kT0_KernelNHWC.data(),
      OC,
      kT0_KH,
      kT0_KW,
      kT0_Bias.data(),
      kT0_S,
      kT0_S,
      kT0_PY,
      kT0_PX,
      kT0_D,
      kT0_D,
      G,
      output.data());

  log("output:", output);
  log("expected_output:", kT0_OutputNHWC);
  assert(kT0_OutputNHWC == output);
}

static void test0_conv_agpu_IKnchw_SIKOnc4hw_KrO4C4HW() {
  const int64_t G = 1;
  const int64_t C = 3;
  const int64_t OC = 2;

  std::vector<float> output(kT0_N * OC * kT0_OH * kT0_OW);
  agpu::agpu_conv2d_buf_IKnchw_SIKOnc4hw_KrO4C4HW(
      kT0_InputNCHW.data(),
      kT0_N,
      C,
      kT0_H,
      kT0_W,
      kT0_KernelNCHW.data(),
      OC,
      kT0_KH,
      kT0_KW,
      kT0_Bias.data(),
      kT0_S,
      kT0_S,
      kT0_PY,
      kT0_PX,
      kT0_D,
      kT0_D,
      G,
      output.data());

  log("output:", output);
  log("expected_output:", kT0_OutputNCHW);
  assert(kT0_OutputNCHW == output);
}

static void test0_conv_agpu_IKnchw_SIKOnc4hw_KrO4HWC() {
  const int64_t G = 1;
  const int64_t C = 3;
  const int64_t OC = 2;

  std::vector<float> output(kT0_N * OC * kT0_OH * kT0_OW);
  agpu::agpu_conv2d_buf_IKnchw_SIKOnc4hw_KrO4HWC(
      kT0_InputNCHW.data(),
      kT0_N,
      C,
      kT0_H,
      kT0_W,
      kT0_KernelNCHW.data(),
      OC,
      kT0_KH,
      kT0_KW,
      kT0_Bias.data(),
      kT0_S,
      kT0_S,
      kT0_PY,
      kT0_PX,
      kT0_D,
      kT0_D,
      G,
      output.data());

  log("output:", output);
  log("expected_output:", kT0_OutputNCHW);
  assert(kT0_OutputNCHW == output);
}

// Depthwise
static void test0_convDW_agpu_IKnchw() {
  const int64_t G = 3;
  const int64_t C = 3;
  const int64_t OC = 3;

  std::vector<float> output(kT0_N * OC * kT0_OH * kT0_OW);
  agpu::agpu_conv2dDW_buf_IKnchw(
      kT0_InputNCHW.data(),
      kT0_N,
      C,
      kT0_H,
      kT0_W,
      kT0_GKernelNCHW.data(),
      OC,
      kT0_KH,
      kT0_KW,
      kT0_Bias.data(),
      kT0_S,
      kT0_S,
      kT0_PY,
      kT0_PX,
      kT0_D,
      kT0_D,
      G,
      output.data());

  log("output:", output);
  log("expected_output:", kT0_GOutputNCHW);
  assert(kT0_GOutputNCHW == output);
}

static void test0_convDW_agpu_IKnhwc() {
  const int64_t G = 3;
  const int64_t C = 3;
  const int64_t OC = 3;

  std::vector<float> output(kT0_N * OC * kT0_OH * kT0_OW);
  agpu::agpu_conv2dDW_buf_IKnhwc(
      kT0_InputNHWC.data(),
      kT0_N,
      C,
      kT0_H,
      kT0_W,
      kT0_GKernelNHWC.data(),
      OC,
      kT0_KH,
      kT0_KW,
      kT0_Bias.data(),
      kT0_S,
      kT0_S,
      kT0_PY,
      kT0_PX,
      kT0_D,
      kT0_D,
      G,
      output.data());

  log("output:", output);
  log("expected_output:", kT0_GOutputNHWC);
  assert(kT0_GOutputNHWC == output);
}

void gbench_main(const std::string& args) {
  ALOGI("pytorch_android_agpu::gbench_main(%s)", args.c_str());
  auto argscv = ArgsCV{args};
  // TODO: --benchmark_filter for some reason does not work
  //  BENCHMARK_CAPTURE(BM_conv, mobilenet_v2, "MobileNet v2")
  //      ->Apply(BM_conv_args_MobileNetV2)
  //      ->Iterations(50)
  //      ->Unit(benchmark::kMicrosecond)
  //      ->ReportAggregatesOnly(true)
  //      ->UseRealTime();

  // BENCHMARK_CAPTURE(BM_conv, base, "base")
  //    ->Apply(BM_conv_args_base)
  //    ->Iterations(10)
  //    ->Repetitions(10)
  //    ->Unit(benchmark::kMicrosecond)
  //    ->ReportAggregatesOnly(true)
  //    ->UseRealTime();

  /*
     static const int kPreburn = 10;
     BENCHMARK_CAPTURE(BM_conv_agpu, base, "a_base")
        ->Apply(BM_conv_agpu_args_base)
        ->Iterations(1)
        ->Repetitions(kPreburn + 20)
        ->ComputeStatistics(
            "_mean*AAA",
            [](const std::vector<double>& v) -> double {
              double sum = std::accumulate(v.begin() + kPreburn, v.end(), 0.);
              return sum / (v.size() - kPreburn);
            })
        ->ComputeStatistics(
            "_std*AAA",
            [](const std::vector<double>& v) -> double {
              std::vector<double> _v{v.begin() + kPreburn, v.end()};
              double sum = std::accumulate(_v.begin(), _v.end(), 0.);
              double avg = sum / _v.size();
              double var = std::accumulate(
                  _v.begin(),
                  _v.end(),
                  0.,
                  [avg](double acc, double x) -> double {
                    double d = x - avg;
                    return acc + d * d;
                  });
              double std = std::sqrt(var / (_v.size() - 1));
              return std;
            })
        ->ComputeStatistics(
            "_p50",
            [](const std::vector<double>& v) -> double {
              std::vector<double> _v{v.begin() + kPreburn, v.end()};
              std::sort(_v.begin(), _v.end());
              return _v[std::round(0.50 * _v.size())];
            })
        ->ComputeStatistics(
            "_p75",
            [](const std::vector<double>& v) -> double {
              std::vector<double> _v{v.begin() + kPreburn, v.end()};
              std::sort(_v.begin(), _v.end());
              int idx = std::round(0.75 * _v.size());
              return _v[idx];
            })
        ->ComputeStatistics(
            "_p90",
            [](const std::vector<double>& v) -> double {
              std::vector<double> _v{v.begin() + kPreburn, v.end()};
              std::sort(_v.begin(), _v.end());
              return _v[std::round(0.9 * _v.size())];
            })
        ->ReportAggregatesOnly(true)
        ->Unit(benchmark::kMicrosecond)
        ->UseManualTime();
  */

  /*
    BENCHMARK_CAPTURE(BM_conv_agpu, base, "conv_once")
        ->Apply(BM_conv_agpu_args_base_once)
        ->Iterations(1)
        ->Repetitions(1)
        ->Unit(benchmark::kMicrosecond)
        ->UseManualTime();


    benchmark::Initialize(&(argscv.c), argscv.v);
    benchmark::RunSpecifiedBenchmarks();
  */

  auto tin = torch::tensor(
      {{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
        {{101, 102, 103}, {104, 105, 106}, {107, 108, 109}},
        {{1001, 1002, 1003}, {1004, 1005, 1006}, {1007, 1008, 1009}}}},
      torch::kFloat);
  auto tw = torch::tensor(
      {
          {{{1, 0}, {0, 0}}, {{0, 1}, {0, 0}}, {{0, 0}, {1, 0}}},
          {{{-1, 0}, {0, 0}}, {{0, -1}, {0, 0}}, {{0, 0}, {-1, 0}}},
      },
      torch::kFloat);

  auto twg = torch::tensor(
      {

          {{{1, 0}, {0, 0}}},

          {{{0, 1}, {0, 0}}},
          {{{0, 0}, {1, 0}}}

      },
      torch::kFloat);
  auto tb = torch::tensor({0, 0, 0}, torch::kFloat);
  int64_t g = 3;
  log("tin.sizes():", tin.sizes());
  log("tin:", tin);
  log("twg.sizes():", twg.sizes());
  log("twg:", twg);
  log("tb:", tb);
  auto tout = at::conv2d(
      tin,
      twg,
      tb,
      c10::IntArrayRef{1}, // stride
      c10::IntArrayRef{0}, // padding
      c10::IntArrayRef{1}, // dilation
      g);
  log("tout:", tout);

  test0_conv_agpu_IKnchw();
  test0_conv_agpu_Inhwc_Knchw();
  test0_conv_agpu_IKnhwc();
  test0_conv_agpu_IKnhwc_KrO4C4HW();
  test0_conv_agpu_IKnhwc_KrO4HWC();
  test0_convDW_agpu_IKnchw();
  test0_convDW_agpu_IKnhwc();
  test0_conv_agpu_IKnchw_SIKOnc4hw_KrO4C4HW();
  test0_conv_agpu_IKnchw_SIKOnc4hw_KrO4HWC();
}

void test_module(torch::jit::script::Module module, const std::string& args) {
  ALOGI("pytorch_android_agpu::test_module(%s)", args.c_str());
  module_ = std::move(module);

  torch::autograd::AutoGradMode no_autograd_guard{false};
  torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
  auto tin = torch::randn({1, 3, 224, 224}, torch::kFloat);
  agpuOff();
  auto toutC = module_.forward({tin}).toTensor();
  at::setAgpuVerbose(false);
  at::setUseAgpuConv(true);
  at::setUseAgpuNorm(true);
  at::setUseAgpuRelu(true);
  at::setUseAgpuAdd(true);
  auto toutT = module_.forward({tin}).toTensor();
  agpuOff();

  assert(almostEqual(toutT, toutC, true));
  ALOGI("ATEST_MODULE PASSED");
}

int stdOutErrToLogcat() {
  static int pfd[2];
  static pthread_t log_thread;
  setvbuf(stdout, 0, _IOLBF, 0);
  setvbuf(stderr, 0, _IOLBF, 0);
  pipe(pfd);
  dup2(pfd[1], 1);
  dup2(pfd[1], 2);
  if (pthread_create(
          &log_thread,
          0,
          [](void*) -> void* {
            ssize_t s;
            char b[1024];
            while ((s = read(pfd[0], b, sizeof(b) - 1)) > 0) {
              if (b[s - 1] == '\n') {
                --s;
              }
              b[s] = 0;
              __android_log_write(ANDROID_LOG_INFO, "cout", b);
            }
            return 0;
          },
          0) == -1) {
    return -1;
  }
  pthread_detach(log_thread);
  return 0;
}

} // namespace pytorch_jni_agpu
