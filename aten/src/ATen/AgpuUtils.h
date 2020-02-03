
namespace at {

void setUseAgpu(bool o);
bool getUseAgpu();

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
