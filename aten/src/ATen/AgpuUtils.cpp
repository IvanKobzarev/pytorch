
namespace at {
thread_local bool kUseAgpu = false;
void setUseAgpu(bool o) { kUseAgpu = o; }
bool getUseAgpu() { return kUseAgpu; }

bool kUseAgpuConv = false;
void setUseAgpuConv(bool o) { kUseAgpuConv = o; }
bool getUseAgpuConv() { return kUseAgpuConv; }

bool kUseAgpuNorm = false;
void setUseAgpuNorm(bool o) { kUseAgpuNorm = o; }
bool getUseAgpuNorm() { return kUseAgpuNorm; }

bool kUseAgpuRelu = false;
void setUseAgpuRelu(bool o) { kUseAgpuRelu = o; }
bool getUseAgpuRelu() { return kUseAgpuRelu; }

bool kUseAgpuAdd = false;
void setUseAgpuAdd(bool o) { kUseAgpuAdd = o; }
bool getUseAgpuAdd() { return kUseAgpuAdd; }

bool kAgpuVerbose = false;


void setAgpuVerbose(bool agpuVerbose){
  kAgpuVerbose = agpuVerbose;
}

bool isAgpuVerbose() {
  return kAgpuVerbose;
}
} // namespace at
