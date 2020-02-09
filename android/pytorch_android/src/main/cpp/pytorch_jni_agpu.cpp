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

void gtest(const std::string& args) {
  ALOGI("pytorch_android_agpu::test(%s)", args.c_str());
  auto argscv = ArgsCV{args};
  ::testing::InitGoogleTest(&(argscv.c), argscv.v);
  RUN_ALL_TESTS();
}

void BM_conv(benchmark::State& state) {
  for (auto _ : state) {
    const int64_t useAgpu = state.range(0);
    const int64_t input_n = state.range(1);
    const int64_t input_c = state.range(2);
    const int64_t input_h = state.range(3);
    const int64_t input_w = state.range(4);

    const int64_t kernel_c = state.range(5);
    const int64_t kernel_h = state.range(6);
    const int64_t kernel_w = state.range(7);

    if (useAgpu == 0) {
      at::setUseAgpu(false);
    } else {
      at::setUseAgpu(true);
    }

    auto input =
        torch::randn({input_n, input_c, input_h, input_w}, torch::kFloat);
    auto w =
        torch::randn({kernel_c, input_c, kernel_h, kernel_w}, torch::kFloat);
    auto b = torch::randn({kernel_c}, torch::kFloat);
    int64_t groups = 1;
    torch::nn::functional::Conv2dFuncOptions o =
        torch::nn::functional::Conv2dFuncOptions().stride(1).padding(0);
    auto outputC = at::conv2d(
        input,
        w,
        b,
        c10::IntArrayRef{1}, // stride
        c10::IntArrayRef{0}, // padding
        c10::IntArrayRef{1}, // dilation
        groups);
  }
}

void gbench(const std::string& args) {
  ALOGI("pytorch_android_agpu::bench(%s)", args.c_str());
  auto argscv = ArgsCV{args};

  BENCHMARK(BM_conv)
      ->ArgNames({"AGPU", "N", "C", "H", "W", "KC", "KH", "KW"})
      ->Iterations(10)
      ->Args({0, 1, 3, 3, 3, 3, 2, 2})
      ->Args({1, 1, 3, 3, 3, 3, 2, 2});

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