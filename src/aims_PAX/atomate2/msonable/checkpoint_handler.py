"""
This module contains an MSONable version of mace CheckpointHandler.
"""
import importlib

from mace.tools import CheckpointIO, CheckpointHandler
from monty.serialization import MontyDecoder

from .serialization import register, register_override, wrap


def _checkpoint_io_as_dict(instance) -> dict:
    return {
        "directory": str(instance.directory),
        "tag":       instance.tag,
        "swa_start": instance.swa_start,  # from vars() — adjust if absent
    }

def _checkpoint_io_from_dict(torch_cls, d: dict):
    return torch_cls(directory=d["directory"], tag=d["tag"], swa_start=d.get("swa_start"))


def _checkpoint_handler_as_dict(instance) -> dict:
    return {
        # Serialize the nested CheckpointIO via its own MSONable wrapper
        "io": wrap(instance.io).as_dict(),
        # CheckpointBuilder is stateless with no args — just store the class path
        "builder_class": f"{instance.builder.__class__.__module__}.{instance.builder.__class__.__qualname__}",
    }

def _checkpoint_handler_from_dict(torch_cls, d: dict):
    handler = torch_cls.__new__(torch_cls)
    handler.io = MontyDecoder().process_decoded(d["io"])
    # Reconstruct builder from its fully qualified class name
    module_name, cls_name = d["builder_class"].rsplit(".", 1)
    builder_cls = getattr(importlib.import_module(module_name), cls_name)
    handler.builder = builder_cls()
    return handler


# as Checkpoint Handler has positional arguments at initialization time, we need
# serialization functions override for it
register_override(CheckpointIO,      _checkpoint_io_as_dict,      _checkpoint_io_from_dict)
register_override(CheckpointHandler, _checkpoint_handler_as_dict, _checkpoint_handler_from_dict)

MSONableCheckpointIO      = register(CheckpointIO,      stateless=True)
MSONableCheckpointHandler = register(CheckpointHandler, stateless=True)