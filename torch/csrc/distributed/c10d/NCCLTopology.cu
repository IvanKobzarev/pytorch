#include <torch/library.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

#include <nvml.h>

namespace {
at::Tensor _get_topology(at::Tensor& dummy, std::string group_name) {
  int32_t deviceCount;
  cudaGetDeviceCount(&deviceCount);
  auto n = deviceCount;
  std::cout << "XXX GET_TOPO:" << n << std::endl;

  nvmlInit();

  for (int i = 0; i < n; i++) {
      nvmlDevice_t device;
      nvmlDeviceGetHandleByIndex(i, &device);
      for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; link++) {
        nvmlEnableState_t isActive;
        nvmlReturn_t result = nvmlDeviceGetNvLinkState(device, link, &isActive);
        
        if (result == NVML_SUCCESS && isActive == NVML_FEATURE_ENABLED) {
            unsigned int version;
            result = nvmlDeviceGetNvLinkVersion(device, link, &version);
            if (result == NVML_SUCCESS) {
                std::cout << "GPU " << i << " link:" << link << " NVLink version:" << version << std::endl;
            }
        }

      }

      for (int j = i + 1; j < n; j++) {
            int perfRank = 0;
            cudaError_t err = cudaDeviceGetP2PAttribute(
                &perfRank,
                cudaDevP2PAttrPerformanceRank,
                i,  // source device
                j   // destination device
            );
            
            if (err == cudaSuccess) {
							std::cout << "XXX err:" << err << " cudaSuccess:" << cudaSuccess << " "  << i << "->" << j << " perfRank:" << perfRank << std::endl;
            } else {
              printf(" N/A ");
            }
      }
  }

  auto ret = at::Tensor();
  return ret;
}
}

TORCH_LIBRARY_FRAGMENT(topo, m) {
  m.def("get_topology(Tensor dummy, str group_name) -> Tensor");
};

TORCH_LIBRARY_IMPL(topo, CPU, m) {
  m.impl("get_topology", _get_topology);
};
