
namespace at {
thread_local bool kUseAgpu = false;

void setUseAgpu(bool o) {
  kUseAgpu = o;
}

bool getUseAgpu() {
  return kUseAgpu;
}

} // namespace at
