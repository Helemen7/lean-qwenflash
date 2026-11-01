"""Windows bring-up check: environment, compilers, platform shims, then the three native builds.
Run from the repo root inside the venv:  python windows_check.py   (writes windows_report.txt)"""
import ctypes, os, platform, shutil, subprocess, sys, tempfile, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'qwen38-flash-offload' / 'qwen38'))
LOG = open(ROOT / 'windows_report.txt', 'w', encoding='utf-8')
FAILS = []

def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

def step(name):
    def deco(fn):
        say(f'\n=== {name}')
        t = time.time()
        try:
            r = fn(); say(f'OK   {name} ({time.time()-t:.1f}s)', '' if r is None else r)
        except BaseException:
            FAILS.append(name); say(f'FAIL {name}\n' + traceback.format_exc())
        return fn
    return deco

@step('system')
def _():
    say(platform.platform(), '| python', sys.version.split()[0], '|', platform.machine())
    for exe in ('cl', 'nvcc', 'hipcc', 'clang', 'nvidia-smi', 'git'):
        say(f'{exe:11}', shutil.which(exe) or 'NOT FOUND')
    import plat
    total, avail = plat.total_ram_bytes()
    say(f'RAM total {total/2**30:.1f} GiB, available {avail/2**30:.1f} GiB, logical CPUs {os.cpu_count()}')
    say('free disk here', f'{shutil.disk_usage(ROOT).free/2**30:.0f} GiB (need ~85 GiB for model + store)')

@step('torch + GPU')
def _():
    import torch
    say('torch', torch.__version__, '| cuda', torch.version.cuda, '| hip', torch.version.hip)
    for v in ('CUDA_HOME', 'ROCM_HOME', 'ROCM_PATH', 'HIP_PATH', 'PYTORCH_ROCM_ARCH'): say(f'  {v}={os.environ.get(v)}')
    try:
        import triton; say('triton', triton.__version__)
    except Exception as e: say('triton import failed:', e)
    assert torch.cuda.is_available(), 'torch.cuda.is_available() is False (wrong torch build or driver?)'
    p = torch.cuda.get_device_properties(0)
    say('GPU', p.name, f'{p.total_memory/2**30:.1f} GiB', 'cc', (p.major, p.minor), getattr(p, 'gcnArchName', ''))

@step('plat.py shims')
def _():
    import plat, torch
    say('anon_alloc + cudaHostRegister(Mapped)')
    buf = plat.anon_alloc(8 << 20)
    a = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    assert a % 4096 == 0
    r = torch.cuda.cudart().cudaHostRegister(a, 8 << 20, 2)
    assert int(r) == 0, f'cudaHostRegister failed: {r}'
    torch.cuda.cudart().cudaHostUnregister(a)
    say('open_direct (unbuffered) aligned read')
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'x.bin'; f.write_bytes(bytes(range(256)) * 4096 * 4)   # 4 MiB
        fd = plat.open_direct(f)
        if plat.IS_WIN:
            # same call as qw_pread: ReadFile on the CRT fd's handle with an explicit offset
            k32 = ctypes.WinDLL('kernel32', use_last_error=True)
            class OV(ctypes.Structure):
                _fields_ = [('i', ctypes.c_size_t), ('ih', ctypes.c_size_t), ('off', ctypes.c_uint32), ('offh', ctypes.c_uint32), ('ev', ctypes.c_void_p)]
            k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(OV)]
            import msvcrt
            got = ctypes.c_uint32(0); ov = OV(); ov.off = 1 << 20
            ok = k32.ReadFile(msvcrt.get_osfhandle(fd), a, 1 << 20, ctypes.byref(got), ctypes.byref(ov))
            assert ok, f'ReadFile failed, GetLastError={ctypes.get_last_error()}'
            n = got.value
        else:
            n = os.readv(fd, [memoryview(buf)[:1 << 20]])
        os.close(fd)
        assert n == 1 << 20, f'aligned read returned {n}'
    say('affinity', plat.get_affinity()[:16], '| cores-first', plat.physical_core_order(plat.get_affinity(), True)[:16])
    say('cgroup_mem', plat.cgroup_mem())
    from memory_guard import DeviceMemory
    d = DeviceMemory(); say('device mem used', d.sample() / 2**20, 'MiB', '(NVML)' if d.lib else '(torch fallback)'); d.running = False

@step('exllamav3 imports')
def _():
    import exllamav3; say(exllamav3.__file__)
    from exllamav3.ext import exllamav3_ext as xe; say('ext loaded')

def build(name, fn):
    @step(f'build {name}')
    def _():
        t = fn(); say(getattr(t, '__file__', t))

import nv_tier, expert_cache, nv2
build('nv (nv_tier)', nv_tier._ext)
build('ec (expert_cache)', expert_cache._ext)
build('nv2 (host + device)', nv2.ext)

say('\n=== summary')
say('ALL OK' if not FAILS else 'FAILED: ' + ', '.join(FAILS))
say('Send back windows_report.txt (and the build output above if a build failed).')
sys.exit(1 if FAILS else 0)
