"""
This module contains an MSONable version of the Model.
"""
import inspect
import sys
import torch
import torch.serialization
from pathlib import Path
from typing import Type

from monty.json import MSONable

# e3nn <=0.5.x stores constants.pt with slice objects;
# PyTorch >=2.6 requires explicit allowance for weights_only=True
torch.serialization.add_safe_globals([slice])



class MSONableModel(MSONable):

    @classmethod
    def from_parent(cls, instance):
        instance.__class__ = cls
        return instance


# Tensor serialization and deserialization
def _serialize_value(v):
    """Convert non-JSON-serializable types to serializable ones."""
    if isinstance(v, torch.Tensor):
        return {"@tensor": v.detach().cpu().tolist()}
    return v


def _deserialize_value(v):
    """Reconstruct types from their serialized form."""
    if isinstance(v, dict) and "@tensor" in v:
        return torch.tensor(v["@tensor"])
    return v


def _msonable_torch_stateful(torch_cls: Type) -> Type[MSONableModel]:
    """
    Factory that creates an MSONable subclass for stateful torch class (models / optimizers / etc).
    Saves/loads via torch.save / torch.load.
    """

    class MSONableTorch(torch_cls, MSONableModel):

        _torch_cls = torch_cls  # capture in closure-safe way

        def as_dict(self) -> dict:
            path = Path.cwd() / f"{self._torch_cls.__name__}.pt"
            torch.save(self, path)
            return {
                "@module": self.__class__.__module__,
                "@class": self.__class__.__name__,
                "checkpoint": str(path),
                "torch_cls_module": self._torch_cls.__module__,
                "torch_cls_name": self._torch_cls.__qualname__,
            }

        @classmethod
        def from_dict(cls, d: dict):
            return cls.from_parent(
                torch.load(d["checkpoint"], weights_only=False)
            )

    MSONableTorch.__name__ = f"MSONable{torch_cls.__name__}"
    MSONableTorch.__qualname__ = f"MSONable{torch_cls.__qualname__}"

    return MSONableTorch


def _msonable_torch_stateless(torch_cls: Type) -> Type[MSONableModel]:
    """
    Factory for stateless torch classes (loss functions, activations, etc.)
    Serializes constructor arguments instead of a checkpoint file.
    """

    class MSONableTorch(torch_cls, MSONableModel):

        _torch_cls = torch_cls

        def as_dict(self) -> dict:
            # Introspect __init__ params and pull their values from the instance
            sig = inspect.signature(self._torch_cls.__init__)
            params = {
                name: _serialize_value(getattr(self, name, param.default))
                for name, param in sig.parameters.items()
                if name != "self" and param.default is not inspect.Parameter.empty
            }
            return {
                "@module": self.__class__.__module__,
                "@class": self.__class__.__name__,
                "torch_cls_module": self._torch_cls.__module__,
                "torch_cls_name": self._torch_cls.__qualname__,
                **params,
            }

        @classmethod
        def from_dict(cls, d: dict):
            excluded = {"@module", "@class", "torch_cls_module", "torch_cls_name"}
            kwargs = {k: _deserialize_value(v) for k, v in d.items() if k not in excluded}
            return cls.from_parent(cls._torch_cls(**kwargs))

    MSONableTorch.__name__ = f"MSONable{torch_cls.__name__}"
    MSONableTorch.__qualname__ = f"MSONable{torch_cls.__qualname__}"

    return MSONableTorch


def _is_stateless(instance) -> bool:
    """
    Stateless = no learned parameters and no optimizer state.
    Covers loss functions, activations, regularizers, etc.
    """
    if isinstance(instance, torch.nn.Module):
        return len(list(instance.parameters())) == 0
    if isinstance(instance, torch.optim.Optimizer):
        return False
    if isinstance(instance, torch.optim.lr_scheduler.LRScheduler):
        return False
    return True  # fallback: treat unknown torch objects as stateless


# Cache of MSONableTorch classes (to create them once in lazy manner)
_REGISTRY: dict[Type, Type[MSONableModel]] = {}
_REVERSE_REGISTRY: dict[Type[MSONableModel], Type] = {}  # MSONable cls -> original torch cls


def register(torch_cls: Type, stateless: bool = False) -> Type[MSONableModel]:
    """Explicitly register a class and return the MSONable variant."""

    # Add the class to the registry if not exists
    if torch_cls not in _REGISTRY:
        factory = _msonable_torch_stateless if stateless else _msonable_torch_stateful

        msonable_cls = factory(torch_cls)

        # Make the class findable by pickle via attribute lookup, as dynamically created classes are not.
        # For this, update sys.modules with the correct module and set the class __module__ attribute.
        # These two have to agree to work.
        target_module = __name__
        msonable_cls.__module__ = target_module
        setattr(sys.modules[target_module], msonable_cls.__name__, msonable_cls)
        # Cache the class
        _REGISTRY[torch_cls] = msonable_cls
        _REVERSE_REGISTRY[msonable_cls] = torch_cls

    return _REGISTRY[torch_cls]


def wrap(instance: torch.nn.Module) -> MSONableModel:
    """Wrap a torch object into an MSONable version."""
    torch_cls = type(instance)
    msonable_cls = register(torch_cls, _is_stateless(instance))
    return msonable_cls.from_parent(instance)
