// GPU runtime API for the host-only translation unit (nv2_host.cpp). CUDA builds use cuda_runtime.h;
// ROCm builds map the few calls used onto HIP here, because torch's hipify can leave .cpp sources untouched.
#pragma once
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime_api.h>
#define cudaEvent_t hipEvent_t
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaSetDevice hipSetDevice
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#else
#include <cuda_runtime.h>
#endif
