// Device-side CUDA/HIP differences used by the nv2 and cpu_avx2 kernels. Included after the torch/CUDA headers
// (hipify rewrites those on ROCm builds; this file only needs the __HIP_PLATFORM_AMD__ switch).
#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
// RDNA/CDNA wall clock: wall_clock64() ticks at a fixed rate (100 MHz on gfx10/11/9). Override with
// -DQW_WALLCLOCK_KHZ=<n> if hipDeviceAttributeWallClockRate reports something else.
#ifndef QW_WALLCLOCK_KHZ
#define QW_WALLCLOCK_KHZ 100000
#endif
__device__ __forceinline__ unsigned long long qw_gtime_ns()
{
    return (unsigned long long) wall_clock64() * (1000000ull / QW_WALLCLOCK_KHZ);
}
#define QW_NANOSLEEP(ns) __builtin_amdgcn_s_sleep(4)
#define QW_LDCV(p) (*(const volatile float*) (p))
#else
__device__ __forceinline__ unsigned long long qw_gtime_ns()
{
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
#define QW_NANOSLEEP(ns) __nanosleep(ns)
#define QW_LDCV(p) __ldcv(p)
#endif
