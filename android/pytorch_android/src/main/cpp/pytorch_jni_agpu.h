#pragma once

namespace pytorch_jni_agpu {
int stdOutErrToLogcat();
void gtest_main(const std::string& args);
void gbench_main(const std::string& args);
} // namespace pytorch_jni_agpu
