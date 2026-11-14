"""Choose v3 Turbo GPU settings that suit the installed NVIDIA card.

Cards older than Volta (compute capability < 7.0, e.g. GTX 10-series) run half precision
far slower than single precision and have no bfloat16, so they get float32.
"""
from typing import Any, Dict


def cuda_card() -> Dict[str, Any]:
    """Name, compute capability and memory of the first CUDA device, or {} without one."""
    try:
        import torch
    except Exception:
        return {}
    if not torch.cuda.is_available():
        return {}
    props = torch.cuda.get_device_properties(0)
    return {"name": props.name, "capability": (props.major, props.minor), "memory_gb": props.total_memory / 2**30}


def turbo_gpu_kwargs() -> Dict[str, Any]:
    """Keyword arguments for ``Vieneu("v3turbo", ...)`` on this machine's GPU; {} means use the CPU."""
    card = cuda_card()
    if not card:
        return {}
    kwargs: Dict[str, Any] = {"device": "cuda", "backend": "pytorch"}
    if card["capability"] < (7, 0):
        kwargs["dtype"] = "float32"
    if card["memory_gb"] < 8:
        kwargs["max_batch_size"] = 8
        kwargs["max_streams"] = 4
    return kwargs
