#pragma once

#ifndef __linux__
    #include <intrin.h>
#endif

bool is_avx2_supported();
bool is_f16c_supported();

#if defined(__linux__) && (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))  // EXL3_AARCH64_STUB
    #define AVX2_TARGET __attribute__((target("avx2")))
    #define AVX2_F16C_TARGET __attribute__((target("avx2,f16c")))
    #define AVX2_TARGET_OPTIONAL __attribute__((target_clones("avx2","default")))
#elif defined(__clang__) && (defined(_M_X64) || defined(__x86_64__))
    // clang on Windows (hipcc): per-function target works, target_clones does not
    #define AVX2_TARGET __attribute__((target("avx2")))
    #define AVX2_F16C_TARGET __attribute__((target("avx2,f16c")))
    #define AVX2_TARGET_OPTIONAL
#else
    #define AVX2_TARGET
    #define AVX2_F16C_TARGET
    #define AVX2_TARGET_OPTIONAL
#endif