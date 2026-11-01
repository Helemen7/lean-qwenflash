// GPU runtime API for the host-only translation unit (nv2_host.cpp). On ROCm the file is built by hipcc after
// torch's hipify has rewritten the cuda* names, so only the right runtime header is needed here.
#pragma once
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime_api.h>
#else
#include <cuda_runtime.h>
#endif
