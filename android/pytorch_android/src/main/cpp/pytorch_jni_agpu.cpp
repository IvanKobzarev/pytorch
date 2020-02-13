#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <string>
#include <typeinfo>

#include <ATen/AgpuUtils.h>
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
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
  ALOGI("%s typeid:%s %s", m, typeid(t).name(), os.str().c_str());
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
        std::cout << "(" << i << ","<< j << ")"<< fbuf << std::endl;
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
    //assert(false);
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

static bool almostEqual(const at::Tensor& t, const at::Tensor& expected, bool logOnFalse = false) {
  double rtol = 0.0001;
  double atol = 0.00001;
  bool ret = torch::allclose(t, expected, rtol, atol, true);
  if (logOnFalse && !ret) {
    auto diff = (t - expected).abs();
    auto diff_max = diff.max().item<float>();
    ALOGI("almostEquals abs_diff_max:%12.8f", diff_max);

    double rtoli = 0.1;
    int i = 1;
    while (i < 5) {
      ALOGI("almostEquals allClose(%d rtoli:%12.8f):%d",
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
    int64_t ph,
    int64_t pw,
    int64_t s,
    int64_t d,
    int64_t g,
    int64_t gcin,
    int64_t gcout,
    bool log) {

  auto input = torch::randn({n, gcin, h, w}, torch::kFloat);
  auto weight = torch::randn({gcout, gcin, kh, kw}, torch::kFloat);
  auto bias = torch::randn({gcout}, torch::kFloat);

  agpuOff();
  auto outputC = at::conv2d(
      input,
      weight,
      bias,
      c10::IntArrayRef{s}, // stride
      c10::IntArrayRef{ph, pw}, // padding
      c10::IntArrayRef{d}, // dilation
      g);

  at::setUseAgpuConv(true);
  auto outputT = at::conv2d(
      input,
      weight,
      bias,
      c10::IntArrayRef{s}, // stride
      c10::IntArrayRef{ph, pw}, // padding
      c10::IntArrayRef{d}, // dilation
      g);
  agpuOff();
  assert(almostEqual(outputC, outputT, log));
}

/*
TEST(conv, mn2_0) {
  test_conv_(1, 224, 224, 3, 3, 2, 2, 2, 1, 1, 3, 32, false);
}
*/

//TEST(conv, small) {
//  test_conv_(
//    /* n */ 1,
//    /* h */ 3,
//    /* w */ 3,
//    /* kh */ 3,
//    /* kw */ 3,
//    /* ph */ 0,
//    /* pw */ 0,
//    /* s */ 1,
//    /* d */ 1,
//    /* g */ 1,
//    /* gcin */ 3,
//    /* gcout */ 32,
//    /* log */ false);
//}

TEST(conv, small_padding1) {
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
    /* gcout */ 32,
    /* log */ false);
}
//TEST(conv, small_stride2) {
//  test_conv_(
//    /* n */ 1,
//    /* h */ 3,
//    /* w */ 3,
//    /* kh */ 3,
//    /* kw */ 3,
//    /* ph */ 0,
//    /* pw */ 0,
//    /* s */ 2,
//    /* d */ 1,
//    /* g */ 1,
//    /* gcin */ 3,
//    /* gcout */ 32,
//    /* log */ false);
//}

TEST(add, small) {
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

TEST(threshold, small) {
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

TEST(norm, small) {
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
  auto toutC = at::batch_norm(tin, weight, bias, mean, var, false, 0.1, 0.00001, false);
  at::setUseAgpuNorm(true);
  auto toutT = at::batch_norm(tin, weight, bias, mean, var, false, 0.1, 0.00001, false);
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
  ::testing::InitGoogleTest(&(argscv.c), argscv.v);
  RUN_ALL_TESTS();
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

    const int64_t padding_h = state.range(6);
    const int64_t padding_w = state.range(7);

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
        c10::IntArrayRef{0}, // padding
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
    int64_t ph,
    int64_t pw,
    int64_t s,
    int64_t d,
    int64_t g,
    int64_t gcin,
    int64_t gcout) {
  b->Args({0, n, h, w, kh, kw, ph, pw, s, d, g, gcin, gcout});
  b->Args({1, n, h, w, kh, kw, ph, pw, s, d, g, gcin, gcout});
}

static void BM_conv_args_base(benchmark::internal::Benchmark* b) {
  b->ArgNames({"AGPU",
               "N",
               "H",
               "W",
               "KH",
               "KW",
               "PH",
               "PW",
               "S",
               "D",
               "G",
               "GCin",
               "GCout"});
  /*             N   H    W   KH  KW  PH  PW  S  D    G  GCin  GCout */
  BM_convArgs(b, 1, 224, 224, 3, 3, 2, 2, 2, 1, 1, 3, 32);
  BM_convArgs(b, 1, 112, 112, 3, 3, 2, 2, 2, 1, 96, 1, 1);
  BM_convArgs(b, 1, 56, 56, 1, 1, 0, 0, 1, 1, 1, 144, 24);
  BM_convArgs(b, 1, 28, 28, 3, 3, 2, 2, 2, 1, 192, 1, 1);
  BM_convArgs(b, 1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 384, 96);
  BM_convArgs(b, 1, 7, 7, 3, 3, 2, 2, 1, 1, 960, 1, 1);
}

static void BM_conv_args_MobileNetV2(benchmark::internal::Benchmark* b) {
  b->ArgNames({"AGPU",
               "N",
               "H",
               "W",
               "KH",
               "KW",
               "PH",
               "PW",
               "S",
               "D",
               "G",
               "GCin",
               "GCout"});

  /*             N   H    W   KH  KW  PH  PW  S  D    G  GCin  GCout */
  BM_convArgs(b, 1, 224, 224, 3, 3, 2, 2, 2, 1, 1, 3, 32);
  BM_convArgs(b, 1, 112, 112, 3, 3, 2, 2, 1, 1, 32, 1, 1);
  BM_convArgs(b, 1, 112, 112, 1, 1, 0, 0, 1, 1, 1, 32, 16);
  BM_convArgs(b, 1, 112, 112, 1, 1, 0, 0, 1, 1, 1, 16, 96);
  BM_convArgs(b, 1, 112, 112, 3, 3, 2, 2, 2, 1, 96, 1, 1);
  BM_convArgs(b, 1, 56, 56, 1, 1, 0, 0, 1, 1, 1, 96, 24);
  BM_convArgs(b, 1, 56, 56, 1, 1, 0, 0, 1, 1, 1, 24, 144);
  BM_convArgs(b, 1, 56, 56, 3, 3, 2, 2, 1, 1, 144, 1, 1);
  BM_convArgs(b, 1, 56, 56, 1, 1, 0, 0, 1, 1, 1, 144, 24);
  BM_convArgs(b, 1, 56, 56, 3, 3, 2, 2, 2, 1, 144, 1, 1);
  BM_convArgs(b, 1, 28, 28, 1, 1, 0, 0, 1, 1, 1, 144, 32);
  BM_convArgs(b, 1, 28, 28, 1, 1, 0, 0, 1, 1, 1, 32, 192);
  BM_convArgs(b, 1, 28, 28, 3, 3, 2, 2, 1, 1, 192, 1, 1);
  BM_convArgs(b, 1, 28, 28, 1, 1, 0, 0, 1, 1, 1, 192, 32);
  BM_convArgs(b, 1, 28, 28, 3, 3, 2, 2, 2, 1, 192, 1, 1);
  BM_convArgs(b, 1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 192, 64);
  BM_convArgs(b, 1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 64, 384);
  BM_convArgs(b, 1, 14, 14, 3, 3, 2, 2, 1, 1, 384, 1, 1);
  BM_convArgs(b, 1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 384, 64);
  BM_convArgs(b, 1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 384, 96);
  BM_convArgs(b, 1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 96, 576);
  BM_convArgs(b, 1, 14, 14, 3, 3, 2, 2, 1, 1, 576, 1, 1);
  BM_convArgs(b, 1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 576, 96);
  BM_convArgs(b, 1, 14, 14, 3, 3, 2, 2, 2, 1, 576, 1, 1);
  BM_convArgs(b, 1, 7, 7, 1, 1, 0, 0, 1, 1, 1, 576, 160);
  BM_convArgs(b, 1, 7, 7, 1, 1, 0, 0, 1, 1, 1, 160, 960);
  BM_convArgs(b, 1, 7, 7, 3, 3, 2, 2, 1, 1, 960, 1, 1);
  BM_convArgs(b, 1, 7, 7, 1, 1, 0, 0, 1, 1, 1, 960, 160);
  BM_convArgs(b, 1, 7, 7, 1, 1, 0, 0, 1, 1, 1, 960, 320);
  BM_convArgs(b, 1, 7, 7, 1, 1, 0, 0, 1, 1, 1, 320, 1280);
  BM_convArgs(b, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1280, 1000);
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

  BENCHMARK_CAPTURE(BM_conv, base, "base")
      ->Apply(BM_conv_args_base)
      ->Iterations(10)
      ->Repetitions(10)
      ->Unit(benchmark::kMicrosecond)
      ->ReportAggregatesOnly(true)
      ->UseRealTime();

  benchmark::Initialize(&(argscv.c), argscv.v);
  benchmark::RunSpecifiedBenchmarks();
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
  at::setUseAgpuNorm(false);
  at::setUseAgpuRelu(false);
  at::setUseAgpuAdd(false);
  auto toutT = module_.forward({tin}).toTensor();
  agpuOff();

  assert(almostEqual(toutT, toutC));
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
            char b[256];
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
