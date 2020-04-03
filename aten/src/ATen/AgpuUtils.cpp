
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

bool kUseAgpuAddmm = false;
void setUseAgpuAddmm(bool o) { kUseAgpuAddmm = o; }
bool getUseAgpuAddmm() { return kUseAgpuAddmm; }

bool kUseAgpuUpSampleNearest2d = false;
void setUseAgpuUpSampleNearest2d(bool o) { kUseAgpuUpSampleNearest2d = o; }
bool getUseAgpuUpSampleNearest2d() { return kUseAgpuUpSampleNearest2d; }

bool kAgpuVerbose = false;


void setAgpuVerbose(bool agpuVerbose){
  kAgpuVerbose = agpuVerbose;
}

bool isAgpuVerbose() {
  return kAgpuVerbose;
}
} // namespace at
