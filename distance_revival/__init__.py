"""e-means with distance-aware revival. See revival.py (CPU) and, for GPUs, torch_emeans.py
(FAISS's PyTorch e-means) and omeans.py (o-means). The GPU classes are imported on first use, so
the CPU class needs neither PyTorch nor Triton."""
from .revival import DistanceAwareRevivalEmeans, RATIO_BINS

__all__ = ["DistanceAwareRevivalEmeans", "RATIO_BINS", "DistanceAwareRevivalTorchEmeans",
           "OmeansDistanceRevival"]


def __getattr__(name):
    if name == "DistanceAwareRevivalTorchEmeans":
        from .torch_emeans import DistanceAwareRevivalTorchEmeans
        return DistanceAwareRevivalTorchEmeans
    if name == "OmeansDistanceRevival":
        from .omeans import OmeansDistanceRevival
        return OmeansDistanceRevival
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
