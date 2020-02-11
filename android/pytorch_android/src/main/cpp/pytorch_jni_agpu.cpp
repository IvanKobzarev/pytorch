#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <string>

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
  ALOGI("%s %s", m, os.str().c_str());
}

static bool checkRtol(
    const at::Tensor& diff,
    const std::vector<at::Tensor> inputs) {
  double maxValue = 0.0;
  for (auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }
  return diff.abs().max().item<float>() < 2e-6 * maxValue;
}

static bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  return checkRtol(a - b, {a, b});
}

TEST(conv, smoke) {
  std::cout << "*******************************"
            << "ATEST_CONV"
            << "*******************************" << std::endl;
  auto input = torch::tensor( // 1, 3, 3, 3
      {{
          // c_0
          {
              {1, 2, 3},
              {4, 5, 6},
              {7, 8, 9},
          },
          // c_1
          {
              {101, 102, 103},
              {104, 105, 106},
              {107, 108, 109},
          },
          // c_2
          {
              {1001, 1002, 1003},
              {1004, 1005, 1006},
              {1007, 1008, 1009},
          },
      }},
      torch::kFloat);

  auto weight = torch::tensor(
      {
          // 2, 3, 2, 2
          // oc_0 (f_0)
          {{
               // oc_0 c_0
               {1, 0},
               {0, 0},
           },
           {
               // oc_0 c_1
               {0, 1},
               {0, 0},
           },
           {
               // oc_0 c_2
               {0, 0},
               {1, 0},
           }},
          // oc_1 (f_1)
          {{
               // oc_1 c_0
               {-1, 0},
               {0, 0},
           },
           {
               // oc_1 c_1
               {0, -1},
               {0, 0},
           },
           {
               // oc_1 c_2
               {0, 0},
               {-1, 0},
           }},
      },
      torch::kFloat);
  auto bias = torch::tensor({0, 0}, torch::kFloat);
  log("C input sizes:", input.sizes());
  log("C w sizes:", weight.sizes());
  log("C b sizes:", bias.sizes());

  int64_t groups = 1;
  torch::nn::functional::Conv2dFuncOptions o =
      torch::nn::functional::Conv2dFuncOptions().stride(1).padding(0);

  ALOGI("C set useAgpu false");
  at::setUseAgpu(false);
  auto outputC = at::conv2d(
      input,
      weight,
      bias,
      c10::IntArrayRef{1}, // stride
      c10::IntArrayRef{0}, // padding
      c10::IntArrayRef{1}, // dilation
      groups);
  log("C outputC.sizes: ", outputC.sizes());

  ALOGI("C set useAgpu true");
  at::setUseAgpu(true);
  auto outputT = at::conv2d(
      input,
      weight,
      bias,
      c10::IntArrayRef{1}, // stride
      c10::IntArrayRef{0}, // padding
      c10::IntArrayRef{1}, // dilation
      groups);
  log("C outputT.sizes: ", outputT.sizes());

  bool eq = torch::equal(outputC, outputT);
  ALOGI("C outputC eq outputT:%d", eq);
  assert(eq);
  ALOGI("ATEST_CONV PASSED");
}

TEST(add, smoke) {
  std::cout << "*******************************"
            << "ATEST_ADD"
            << "*******************************" << std::endl;
  auto a = torch::tensor( // 1, 2, 2, 3
      {
          {
              {1, 2, 3},
              {4, 5, 6},
          },
          {
              {11, 12, 13},
              {14, 15, 16},
          },
      },
      torch::kFloat);
  auto b = torch::tensor( // 1, 2, 2, 3
      {
          {
              {101, 102, 103},
              {104, 105, 106},
          },
          {
              {111, 112, 113},
              {114, 115, 116},
          },
      },
      torch::kFloat);

  std::cout << "A a:\n" << a << std::endl;
  std::cout << "A b:\n" << b << std::endl;

  ALOGI("A set useAgpu false");
  at::setUseAgpu(false);
  auto outputC = torch::add(a, b);
  log("A outputC.sizes: ", outputC.sizes());

  ALOGI("A set useAgpu true");
  at::setUseAgpu(true);
  auto outputT = torch::add(a, b);
  log("A outputT.sizes: ", outputT.sizes());

  bool eq = torch::equal(outputC, outputT);
  ALOGI("A outputC eq outputT:%d", eq);
  assert(eq);
  ALOGI("ATEST_ADD PASSED");
}

TEST(threshold, smoke) {
  std::cout << "*******************************"
            << "ATEST_THRESHOLD"
            << "*******************************" << std::endl;
  auto input = torch::tensor( // 1, 2, 2, 3
      {
          {
              {1, -2, 3},
              {-4, 5, -6},
          },
          {
              {11, -12, 13},
              {-14, 15, -16},
          },
      },
      torch::kFloat);
  log("T input.sizes():", input.sizes());
  log("T input:", input);
  ALOGI("T set useAgpu false");
  at::setUseAgpu(false);
  auto outputC = at::relu(input); //, 3, 0);
  log("T outputC.sizes: ", outputC.sizes());
  log("T outputC: ", outputC);

  ALOGI("III set useAgpu true");
  at::setUseAgpu(true);
  auto outputT = at::relu(input); //, 3, 0);
  ALOGI("T ===");
  log("T input.sizes():", input.sizes());
  log("T input:", input);
  log("T outputC.sizes: ", outputC.sizes());
  log("T outputC: ", outputC);
  log("T outputT.sizes: ", outputT.sizes());
  log("T outputT: ", outputT);

  bool eq = torch::equal(outputC, outputT);
  ALOGI("T outputC eq outputT:%d", eq);
  assert(eq);
  ALOGI("ATEST_THRESHOLD PASSED");
}

TEST(norm, smoke) {
  std::cout << "*******************************"
            << "ATEST_NORM"
            << "*******************************" << std::endl;
  auto input = torch::tensor( // 1, 2, 2, 3
      {
          {
              {1, -2, 3},
              {-4, 5, -6},
          },
          {
              {11, -12, 13},
              {-14, 15, -16},
          },
      },
      torch::kFloat);
  auto weight = torch::tensor({1, 2}, torch::kFloat);
  auto bias = torch::tensor({3, 4}, torch::kFloat);
  auto mean = torch::tensor({5, 6}, torch::kFloat);
  auto var = torch::tensor({7, 8}, torch::kFloat);

  log("N input.sizes():", input.sizes());
  log("N input:", input);
  ALOGI("N set useAgpu false");
  at::setUseAgpu(false);
  auto outputC = at::batch_norm(
      input, weight, bias, mean, var, false, 0.1, 0.00001, false);
  log("N outputC.sizes: ", outputC.sizes());
  log("N outputC: ", outputC);

  ALOGI("N set useAgpu true");
  at::setUseAgpu(true);
  auto outputT = at::batch_norm(
      input, weight, bias, mean, var, false, 0.1, 0.00001, false);
  at::setUseAgpu(false);
  log("N outputC.sizes: ", outputC.sizes());
  log("N outputC: ", outputC);
  log("N outputT.sizes: ", outputT.sizes());
  log("N outputT: ", outputT);

  bool eq = almostEqual(outputC, outputT);
  ALOGI("N outputC eq outputT:%d", eq);
  assert(eq);
  ALOGI("ATEST_NORM PASSED");
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

    if (useAgpu == 0) {
      at::setUseAgpu(false);
    } else {
      at::setUseAgpu(true);
    }

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
  ALOGI("pytorch_android_agpu::bench(%s)", args.c_str());
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
