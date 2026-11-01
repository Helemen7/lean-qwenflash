"""Linux/Windows platform shims: anonymous host memory, O_DIRECT-style reads, CPU affinity, RAM accounting,
extension build flags. Linux paths are unchanged; Windows replaces cgroups, mmap flags, O_DIRECT and
sched_*affinity with their Win32 equivalents."""
import ctypes, mmap, os, sys

IS_WIN = os.name == "nt"
GiB = 1024 ** 3


# ---------------------------------------------------------------------------------------------------- memory
def anon_alloc(nbytes):
    """Page-aligned anonymous host memory usable with cudaHostRegister and torch.frombuffer.
    Returns a buffer object; ctypes.addressof(ctypes.c_char.from_buffer(buf)) is its address."""
    if not IS_WIN:
        return mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    p = k32.VirtualAlloc(None, nbytes, 0x3000, 0x04)   # MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE (zeroed)
    if not p:
        raise MemoryError(f"VirtualAlloc({nbytes}) failed: {ctypes.get_last_error()}")
    return (ctypes.c_char * nbytes).from_address(p)


def total_ram_bytes():
    if IS_WIN:
        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_uint32), ("load", ctypes.c_uint32), ("total", ctypes.c_uint64),
                        ("avail", ctypes.c_uint64), ("tpf", ctypes.c_uint64), ("apf", ctypes.c_uint64),
                        ("tv", ctypes.c_uint64), ("av", ctypes.c_uint64), ("ext", ctypes.c_uint64)]
        m = MS(); m.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return int(m.total), int(m.avail)
    with open("/proc/meminfo") as f:
        d = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in f}
    return d["MemTotal"], d.get("MemAvailable", d["MemFree"])


def process_ram_bytes():
    """Resident bytes of this process (Windows stand-in for memory.current of the cgroup)."""
    if IS_WIN:
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint32), ("pf", ctypes.c_uint32), ("peak", ctypes.c_size_t),
                        ("ws", ctypes.c_size_t), ("qpp", ctypes.c_size_t), ("qp", ctypes.c_size_t),
                        ("qnpp", ctypes.c_size_t), ("qnp", ctypes.c_size_t), ("pagefile", ctypes.c_size_t),
                        ("peakpagefile", ctypes.c_size_t)]
        k32 = ctypes.WinDLL("kernel32"); ps = ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        ps.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return int(c.ws)
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def cgroup_mem():
    """Same keys as nv_tier.cgroup_mem (GiB): current / peak / max / anon / file / host_avail."""
    out = {}
    if IS_WIN:
        total, avail = total_ram_bytes()
        out["current"] = process_ram_bytes() / GiB
        out["max"] = float(os.environ.get("QWEN_RAM_LIMIT_BYTES", total)) / GiB
        out["host_avail"] = avail / GiB
        return {k: round(v, 3) for k, v in out.items()}
    try:
        cg = os.environ.get("QWEN_MEMORY_CGROUP", "/sys/fs/cgroup")
        out["current"] = int(open(f"{cg}/memory.current").read()) / GiB
        if os.path.exists(f"{cg}/memory.peak"):
            out["peak"] = int(open(f"{cg}/memory.peak").read()) / GiB
        mx = open(f"{cg}/memory.max").read().strip()
        out["max"] = int(mx) / GiB if mx != "max" else None
        for line in open(f"{cg}/memory.stat"):
            k, v = line.split()
            if k in ("anon", "file", "kernel", "shmem", "unevictable", "pagetables", "slab"):
                out[k] = int(v) / GiB
    except OSError:
        pass
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            out["host_avail"] = int(line.split()[1]) / 1048576
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


def drop_file_cache(path):
    """Ask the OS to drop cached pages of a file (posix_fadvise DONTNEED); no-op on Windows."""
    if IS_WIN or not hasattr(os, "posix_fadvise"):
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------------------------------- files
def open_direct(path):
    """Read-only fd for page-cache-bypassing reads (offset/length/buffer sector-aligned). The C++ readers
    call qw_pread on it (pread on Linux, ReadFile + OVERLAPPED offset on a CRT fd on Windows)."""
    if not IS_WIN:
        return os.open(path, os.O_RDONLY | os.O_DIRECT)
    import msvcrt
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = ctypes.c_void_p
    k32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    # GENERIC_READ, share read|write, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS
    h = k32.CreateFileW(os.fspath(path), 0x80000000, 3, None, 3, 0x20000000 | 0x10000000, None)
    if h in (None, ctypes.c_void_p(-1).value):
        raise OSError(f"CreateFileW({path}) failed: {ctypes.get_last_error()}")
    return msvcrt.open_osfhandle(h, os.O_RDONLY)


# ---------------------------------------------------------------------------------------------------- cpus
def get_affinity():
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    n = os.cpu_count() or 1
    if IS_WIN:
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        proc, sysm = ctypes.c_size_t(), ctypes.c_size_t()
        if k32.GetProcessAffinityMask(ctypes.c_void_p(k32.GetCurrentProcess()), ctypes.byref(proc), ctypes.byref(sysm)):
            return [i for i in range(64) if proc.value >> i & 1]
    return list(range(n))


def set_affinity(cpus):
    cpus = set(cpus)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)
    elif IS_WIN:
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetProcessAffinityMask(ctypes.c_void_p(k32.GetCurrentProcess()),
                                   ctypes.c_size_t(sum(1 << c for c in cpus if c < 64)))


def physical_core_order(cpus, only_first=False):
    """cpus ordered physical-cores-first (one logical CPU per core), SMT siblings after."""
    cpus = list(cpus)
    if IS_WIN:
        try:
            class SLPI(ctypes.Structure):
                _fields_ = [("mask", ctypes.c_size_t), ("rel", ctypes.c_int), ("u", ctypes.c_ubyte * 16)]
            k32 = ctypes.WinDLL("kernel32"); n = ctypes.c_uint32(0)
            k32.GetLogicalProcessorInformation(None, ctypes.byref(n))
            arr = (SLPI * (n.value // ctypes.sizeof(SLPI)))()
            if not k32.GetLogicalProcessorInformation(arr, ctypes.byref(n)):
                raise OSError
            first, sib = [], []
            for e in arr:
                if e.rel != 0:   # RelationProcessorCore
                    continue
                bits = [b for b in range(64) if e.mask >> b & 1 and b in cpus]
                first += bits[:1]; sib += bits[1:]
            return sorted(first) if only_first else sorted(first) + sorted(sib)
        except Exception:
            return cpus
    seen, first, sib = set(), [], []
    for c in cpus:
        try:
            core = open(f"/sys/devices/system/cpu/cpu{c}/topology/core_id").read().strip()
            pkg = open(f"/sys/devices/system/cpu/cpu{c}/topology/physical_package_id").read().strip()
        except OSError:
            core, pkg = str(c), "0"
        (sib if (pkg, core) in seen else first).append(c)
        seen.add((pkg, core))
    return first if only_first else first + sib


# ---------------------------------------------------------------------------------------------------- build
def cxx_flags(avx2=True):
    if IS_WIN:
        return ["/O2", "/std:c++17", "/EHsc", "/Zc:__cplusplus", "/permissive-"] + (["/arch:AVX2"] if avx2 else [])
    return ["-O3", "-std=c++17"] + (["-mavx2", "-mfma", "-mf16c", "-mtune=znver3"] if avx2 else [])


def ld_flags():
    return [] if IS_WIN else ["-lpthread"]


def cuda_flags(*extra):
    # nvcc on Windows: pass /Zc:__cplusplus to the MSVC host compiler (hipcc takes no -Xcompiler)
    import torch
    return ["-O3", *extra] + (["-Xcompiler", "/Zc:__cplusplus"] if IS_WIN and not torch.version.hip else [])
