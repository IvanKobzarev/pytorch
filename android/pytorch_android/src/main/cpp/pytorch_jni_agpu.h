#pragma once

namespace pytorch_jni_agpu {
int stdOutErrToLogcat();
void gtest(const std::string& args);
void gbench(const std::string& args);
} // namespace pytorch_jni_agpu
