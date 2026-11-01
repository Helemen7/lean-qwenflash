// GPU runtime API for the host-only translation unit (nv2_host.cpp).
//   CUDA          : cuda_runtime.h
//   ROCm, clang   : hip/hip_runtime_api.h
//   ROCm, MSVC cl : HIP's headers need clang attributes, so the handful of calls used here are declared by hand
//                   (they resolve against amdhip64.lib, which torch links for ROCm extensions).
// The cuda* names are mapped to hip* so the source works both untouched and after torch's hipify rewrote it.
#pragma once
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#if defined(_MSC_VER) && !defined(__clang__)
#include <stddef.h>
extern "C" {
typedef struct ihipEvent_t* hipEvent_t;
typedef struct ihipStream_t* hipStream_t;
typedef int hipError_t;
enum { hipSuccess = 0 };
enum { hipStreamNonBlocking = 1 };
enum { hipEventDisableTiming = 2 };
enum hipMemcpyKind { hipMemcpyHostToHost = 0, hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2,
                     hipMemcpyDeviceToDevice = 3, hipMemcpyDefault = 4 };
hipError_t hipEventCreateWithFlags(hipEvent_t*, unsigned int);
hipError_t hipEventQuery(hipEvent_t);
hipError_t hipEventRecord(hipEvent_t, hipStream_t);
hipError_t hipEventSynchronize(hipEvent_t);
hipError_t hipMemcpyAsync(void*, const void*, size_t, hipMemcpyKind, hipStream_t);
hipError_t hipSetDevice(int);
hipError_t hipStreamCreateWithFlags(hipStream_t*, unsigned int);
hipError_t hipStreamSynchronize(hipStream_t);
hipError_t hipStreamWaitEvent(hipStream_t, hipEvent_t, unsigned int);
}
#else
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime_api.h>
#endif
#ifndef cudaEvent_t
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
#endif
#else
#include <cuda_runtime.h>
#endif
