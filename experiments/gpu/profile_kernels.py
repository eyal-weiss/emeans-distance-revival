"""CUDA time per kernel (torch.profiler) of one fit on SPACEV 1M (k = 16,384, 1 epoch), library vs ours,
for e-means and o-means: python experiments/gpu/profile_kernels.py"""
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from torch.profiler import profile, ProfilerActivity
from faiss.contrib.o_means import Omeans
from faiss.contrib.e_means import Emeans
from distance_revival.omeans import OmeansDistanceRevival
from distance_revival.torch_emeans import DistanceAwareRevivalTorchEmeans
DATA = os.environ.get("DISTANCE_REVIVAL_DATA", os.path.join(HERE, "..", "..", "data"))
x = torch.from_numpy(np.array(np.load(os.path.join(DATA, "spacev1m", "train.npy"), mmap_mode="r"), dtype=np.float32)).cuda()
d, k = x.shape[1], 16384
makers = {
    "omeans lib": lambda: Omeans(d, k, nepochs=1, seed=3, capture_assignments=False),
    "omeans ours": lambda: OmeansDistanceRevival(d, k, nepochs=1, seed=3, capture_assignments=False),
    "emeans lib": lambda: Emeans(d, k, nepochs=1, seed=3, capture_assignments=False, unused_threshold=0.1),
    "emeans ours": lambda: DistanceAwareRevivalTorchEmeans(d, k, nepochs=1, seed=3, capture_assignments=False),
}
for name, mk in makers.items():
    mk().train(x); torch.cuda.synchronize()          # warm-up
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        mk().train(x); torch.cuda.synchronize()
    ev = [e for e in prof.key_averages() if e.device_type.name == "CUDA" or getattr(e, "self_device_time_total", 0) > 0]
    tot = sum(e.self_device_time_total for e in prof.key_averages()) / 1000
    print(f"\n== {name}: total CUDA kernel time {tot:.1f} ms")
    for e in sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)[:9]:
        if e.self_device_time_total > 0:
            print(f"   {e.self_device_time_total/1000:7.2f} ms  x{e.count:<5d} {e.key[:90]}")
