// Linux/Windows shims for the host-side engines (nv2, cpu_avx2). Everything the sources need from
// POSIX goes through qw_* so the same code builds with gcc/clang on Linux and MSVC on Windows.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cerrno>
#include <vector>
#include <algorithm>
#include <cstring>
#include <immintrin.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <malloc.h>
#ifdef _MSC_VER
typedef SSIZE_T ssize_t;
#define __restrict__ __restrict
#define __attribute__(x)
#endif

// pread(2) on a CRT fd whose handle was opened with FILE_FLAG_NO_BUFFERING (sector-aligned buffer/offset/length,
// as with O_DIRECT). Explicit offset, so concurrent readers on one handle are safe.
static inline ssize_t qw_pread(int fd, void* buf, size_t len, long long off)
{
    HANDLE h = (HANDLE) _get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
    if (len > (size_t(1) << 30)) len = size_t(1) << 30;
    OVERLAPPED ov; memset(&ov, 0, sizeof(ov));
    ov.Offset = DWORD((unsigned long long) off & 0xFFFFFFFFull);
    ov.OffsetHigh = DWORD((unsigned long long) off >> 32);
    DWORD n = 0;
    if (!ReadFile(h, buf, DWORD(len), &n, &ov))
    {
        if (GetLastError() == ERROR_HANDLE_EOF) return 0;
        errno = EIO; return -1;
    }
    return ssize_t(n);
}
static inline void qw_pin_thread(int cpu)
{
    if (cpu < 0 || cpu >= 64) return;
    SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << cpu);
}
static inline void qw_pin_thread_set(const std::vector<int>& cpus)
{
    DWORD_PTR m = 0;
    for (int c : cpus) if (c >= 0 && c < 64) m |= DWORD_PTR(1) << c;
    if (m) SetThreadAffinityMask(GetCurrentThread(), m);
}
struct QwAffinity { DWORD_PTR mask = 0; };
static inline QwAffinity qw_get_affinity()
{
    QwAffinity a; DWORD_PTR proc = 0, sys = 0;
    GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys);
    a.mask = proc ? proc : sys;
    return a;
}
static inline void qw_set_affinity(const QwAffinity& a) { if (a.mask) SetThreadAffinityMask(GetCurrentThread(), a.mask); }
static inline void* qw_aligned_alloc(size_t al, size_t bytes) { return _aligned_malloc((bytes + al - 1) / al * al, al); }
static inline void* qw_big_alloc(size_t bytes)
{
    // 2 MiB large pages need SeLockMemoryPrivilege; plain aligned pages are the portable path.
    void* p = qw_aligned_alloc(size_t(2) << 20, bytes);
    if (!p) { perror("alloc"); exit(1); }
    return p;
}
// Physical cores first (one logical CPU per core), SMT siblings after.
static inline std::vector<int> qw_cpu_order()
{
    std::vector<int> first, sib;
    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> info(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (len && GetLogicalProcessorInformation(info.data(), &len))
        for (auto& i : info)
        {
            if (i.Relationship != RelationProcessorCore) continue;
            bool f = true;
            for (int b = 0; b < 64; ++b) if (i.ProcessorMask & (ULONG_PTR(1) << b)) { (f ? first : sib).push_back(b); f = false; }
        }
    std::sort(first.begin(), first.end());
    first.insert(first.end(), sib.begin(), sib.end());
    return first;
}
static inline long long qw_realtime_ns()
{
    FILETIME ft; GetSystemTimePreciseAsFileTime(&ft);
    unsigned long long t = ((unsigned long long) ft.dwHighDateTime << 32) | ft.dwLowDateTime;   // 100 ns since 1601
    return (long long) ((t - 116444736000000000ull) * 100ull);
}
#else
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

static inline ssize_t qw_pread(int fd, void* buf, size_t len, long long off)
{
    for (;;)
    {
        ssize_t r = pread(fd, buf, len, off_t(off));
        if (r < 0 && errno == EINTR) continue;
        return r;
    }
}
static inline void qw_pin_thread(int cpu)
{
    if (cpu < 0) return;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}
static inline void qw_pin_thread_set(const std::vector<int>& cpus)
{
    if (cpus.empty()) return;
    cpu_set_t s; CPU_ZERO(&s); for (int c : cpus) CPU_SET(c, &s); pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}
struct QwAffinity { cpu_set_t set; };
static inline QwAffinity qw_get_affinity() { QwAffinity a; sched_getaffinity(0, sizeof(a.set), &a.set); return a; }
static inline void qw_set_affinity(const QwAffinity& a) { sched_setaffinity(0, sizeof(a.set), &a.set); }
static inline void* qw_aligned_alloc(size_t al, size_t bytes)
{
    void* p = nullptr;
    return posix_memalign(&p, al, (bytes + al - 1) / al * al) ? nullptr : p;
}
static inline void* qw_big_alloc(size_t bytes)
{
    const size_t al = size_t(2) << 20;
    void* p = qw_aligned_alloc(al, bytes);
    if (!p) { perror("alloc"); exit(1); }
    madvise(p, bytes, MADV_HUGEPAGE);
    return p;
}
#include <fstream>
#include <string>
#include <thread>
static inline std::vector<int> qw_cpu_order()
{
    std::vector<int> first, sib; std::vector<std::pair<int, int>> seen;
    const int ncpu = int(std::thread::hardware_concurrency());
    for (int c = 0; c < ncpu; ++c)
    {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/core_id"); int core = -1; f >> core;
        std::ifstream f2("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/physical_package_id"); int pk = 0; f2 >> pk;
        auto key = std::make_pair(pk, core);
        if (std::find(seen.begin(), seen.end(), key) == seen.end()) { seen.push_back(key); first.push_back(c); } else sib.push_back(c);
    }
    first.insert(first.end(), sib.begin(), sib.end());
    return first;
}
static inline long long qw_realtime_ns() { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec; }
#endif

// Buffered read of nb bytes at offset off (loader/test path; the hot path uses qw_pread on O_DIRECT fds).
static inline bool qw_read_file_at(const char* path, void* dst, size_t nb, long long off)
{
    FILE* f = fopen(path, "rb");
    if (!f) return false;
#ifdef _WIN32
    const int sk = _fseeki64(f, off, SEEK_SET);
#else
    const int sk = fseeko(f, off_t(off), SEEK_SET);
#endif
    const bool ok = sk == 0 && fread(dst, 1, nb, f) == nb;
    fclose(f);
    return ok;
}
