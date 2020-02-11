
namespace at {

void setUseAgpu(bool o);
bool getUseAgpu();

void setUseAgpuConv(bool o);
bool getUseAgpuConv();

void setUseAgpuNorm(bool o);
bool getUseAgpuNorm();

void setUseAgpuRelu(bool o);
bool getUseAgpuRelu();

void setUseAgpuAdd(bool o);
bool getUseAgpuAdd();


void setAgpuVerbose(bool agpuVerbose);
bool isAgpuVerbose();

struct AgpuGuard {
  AgpuGuard(bool state) : old_state_(getUseAgpu()) {
    setUseAgpu(state);
  }

  ~AgpuGuard() {
    setUseAgpu(old_state_);
  }

  bool old_state_;
};

} // namespace at
