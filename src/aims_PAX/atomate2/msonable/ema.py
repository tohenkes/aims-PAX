"""
This module contains the custom serializable version of torch_ema.ExponentialMovingAverage
"""
from pathlib import Path

import torch
from torch_ema import ExponentialMovingAverage
from .serialization import register, register_override

def _ema_as_dict(instance) -> dict:
    path = Path.cwd() / "ema_shadow_params.pt"
    torch.save(instance.shadow_params, path)
    return {
        "shadow_params_path": str(path),
        "decay": instance.decay,
        "num_updates": instance.num_updates,  # None means use_num_updates=False
    }

def _ema_from_dict(torch_cls, d: dict):
    shadow_params = torch.load(d["shadow_params_path"], weights_only=True)
    ema = torch_cls.__new__(torch_cls)
    ema.shadow_params = shadow_params
    ema.decay = d["decay"]
    ema.num_updates = d["num_updates"]
    ema.collected_params = None
    return ema


# as EMA has positional arguments at initialization time, we need
# serialization functions override for EMA
register_override(ExponentialMovingAverage, _ema_as_dict, _ema_from_dict)
MSONableEMA = register(ExponentialMovingAverage, stateless=True)