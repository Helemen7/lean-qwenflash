"""Compile offload extensions for the current GPU and runtime."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'qwen38-flash-offload/qwen38'))
import torch,exllamav3
print('RUNTIME',torch.__version__,torch.version.cuda,exllamav3.__file__,flush=True)
print('GPU',torch.cuda.get_device_name(),torch.cuda.get_device_capability(),flush=True)
import os
if torch.version.hip: os.environ.setdefault('PYTORCH_ROCM_ARCH',torch.cuda.get_device_properties(0).gcnArchName.split(':')[0])
else:
 major,minor=torch.cuda.get_device_capability(); os.environ.setdefault('TORCH_CUDA_ARCH_LIST',f'{major}.{minor}')
import nv_tier,expert_cache,nv2
for name,fn in [('nv',nv_tier._ext),('ec',expert_cache._ext),('nv2',nv2.ext)]:
 start=time.monotonic();native=fn()
 print('COMPILED',name,native.__file__,time.monotonic()-start,flush=True)
print('LOCAL_EXTENSIONS_READY',flush=True)
