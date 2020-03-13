#pragma once

namespace pytorch_jni_agpu {
int stdOutErrToLogcat();
void gtest_main(const std::string& args);
void test_module(torch::jit::script::Module module, const std::string& args);
void test0_main(const std::string& args);

void gbench_main(const std::string& args, const std::string& labelPrefix);
void gbench_module(torch::jit::script::Module module, const std::string& args);
} // namespace pytorch_jni_agpu
