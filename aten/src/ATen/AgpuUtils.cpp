
namespace at {
thread_local bool kUseAgpu = false;
bool kAgpuVerbose = false;

void setUseAgpu(bool o) {
  kUseAgpu = o;
}

bool getUseAgpu() {
  return kUseAgpu;
}

void setAgpuVerbose(bool agpuVerbose){
  kAgpuVerbose = agpuVerbose;
}

bool isAgpuVerbose() {
  return kAgpuVerbose;
}
} // namespace at
