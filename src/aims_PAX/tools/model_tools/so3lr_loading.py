"""One loader for every SO3LR source.

A SO3LR model can come from a bundled keyword (``v1``, ``tauri-s-2606``,
...), from a ``<stem>.pt`` state dict with a sibling
``<stem>_settings.yaml``, or from a pickled model file. The bundled
SO3LR-tauri models carry 16 theory-level energy heads; aims-PAX trains on
a single DFT level, so this module is the only place that deals with
theory levels: whatever it returns has exactly one.
"""

import inspect
import logging
from pathlib import Path
from typing import Any, Literal, Optional

import torch
from so3krates_torch.calculator.so3 import (
    _PRETRAINED_SO3LR_MODELS,
    SO3LRCalculator,
    TorchkratesCalculator,
    # Private upstream helpers; imported only here so that a change in
    # so3krates-torch has to be fixed in one place.
    _build_so3lr_from_settings_and_state_dict,
    _shared_theory_levels,
    load_pretrained_so3lr,
    settings_path_for_checkpoint,
)
from so3krates_torch.modules.models import SO3LR, MultiHeadSO3LR
from so3krates_torch.tools.multihead_utils import (
    DEFAULT_THEORY_LEVEL,
    select_theory_level,
)

from aims_PAX.settings.model import SO3LRArchitectureSettings

BUNDLED_SO3LR_MODELS = frozenset(_PRETRAINED_SO3LR_MODELS)

# Settings field name -> model attribute, where they differ.
_SETTINGS_TO_ATTRIBUTE = {
    "cutoff_fn": "cutoff_fn_name",
    "activation_fn": "activation_fn_name",
    "energy_activation_fn": "energy_activation_fn_name",
}

# Fields of SO3LRArchitectureSettings that are not model attributes.
_AIMS_PAX_ONLY_FIELDS = frozenset(
    {"model_choice", "use_multihead_model", "num_multihead_heads"}
)


def _num_theory_levels(model: torch.nn.Module) -> int:
    return getattr(model.atomic_energy_output_block, "num_theory_levels", 1)


def _to_single_theory_level(
    model: SO3LR, theory_level: Optional[int]
) -> SO3LR:
    """Collapse a multi-theory-level model to one level.

    ``theory_level=None`` keeps ``DEFAULT_THEORY_LEVEL`` for a
    multi-level model; any ``theory_level`` on a single-level model is an
    error.
    """
    num_levels = _num_theory_levels(model)
    if num_levels == 1:
        if theory_level is not None:
            raise ValueError(
                f"theory_level={theory_level} was given, but the model has "
                "a single theory level; theory_level does not apply."
            )
        return model
    if theory_level is None:
        theory_level = DEFAULT_THEORY_LEVEL
        logging.info(
            f"Model has {num_levels} theory levels; using the default "
            f"theory_level={theory_level} (set theory_level to choose "
            "another)."
        )
    shared = _shared_theory_levels(model.atomic_energy_output_block)
    if theory_level in shared:
        trained = [lvl for lvl in range(num_levels) if lvl not in shared]
        logging.warning(
            f"theory_level={theory_level} has parameters identical to "
            "those of other theory levels, so it was most likely never "
            f"trained. The trained theory levels are {trained}."
        )
    return select_theory_level(model, theory_level)


def _resolve_spec(spec: str) -> tuple[str, Optional[Path]]:
    """Classify ``spec`` as ``bundled``, ``settings`` or ``pickle``."""
    if spec in BUNDLED_SO3LR_MODELS:
        return "bundled", None
    path = Path(spec)
    if not path.is_file():
        raise FileNotFoundError(
            f"SO3LR model {spec!r} is neither a bundled model "
            f"({sorted(BUNDLED_SO3LR_MODELS)}) nor an existing file."
        )
    if (
        path.suffix == ".pt"
        and path.with_name(path.stem + "_settings.yaml").exists()
    ):
        return "settings", path
    return "pickle", path


def load_so3lr(
    spec: str,
    device: str,
    dtype: Literal["float32", "float64"],
    theory_level: Optional[int] = None,
    **architecture_overrides: Any,
) -> SO3LR:
    """Load a SO3LR model with a single theory level.

    Args:
        spec: A bundled keyword (see ``BUNDLED_SO3LR_MODELS``), a ``.pt``
            state dict with a sibling ``<stem>_settings.yaml``, or a
            pickled model file (``.model``, or ``.pt`` without settings).
        device: Device the model is moved to.
        dtype: Dtype of the returned model's parameters.
        theory_level: Theory level kept from a multi-level model; only
            valid for those. ``None`` keeps ``DEFAULT_THEORY_LEVEL``.
        **architecture_overrides: Passed to the ``SO3LR`` constructor
            (e.g. ``use_pme=True``). Not allowed for a pickled model.

    ``torch.get_default_dtype()`` is the same before and after the call.
    A pickled ``MultiHeadSO3LR`` is returned as it is.
    """
    kind, path = _resolve_spec(spec)
    if kind == "pickle" and architecture_overrides:
        raise ValueError(
            "architecture overrides "
            f"({sorted(architecture_overrides)}) cannot be applied to a "
            f"pickled model ({spec!r}); they only apply to bundled models "
            "and '<stem>.pt' + '<stem>_settings.yaml' pairs."
        )

    # the So3krates constructor sets the global default dtype
    previous_dtype = torch.get_default_dtype()
    try:
        if kind == "pickle":
            model = torch.load(path, map_location=device, weights_only=False)
            if not isinstance(model, torch.nn.Module):
                raise ValueError(
                    f"{spec!r} does not contain a pickled model (found "
                    f"{type(model).__name__}). A state dict needs a "
                    "sibling '<stem>_settings.yaml'."
                )
            if isinstance(model, MultiHeadSO3LR):
                if theory_level is not None:
                    raise ValueError(
                        "theory_level does not apply to a MultiHeadSO3LR."
                    )
                return model
            model.to(getattr(torch, dtype))
            model.dtype = getattr(torch, dtype)
        elif kind == "bundled":
            if spec.startswith("tauri-"):
                logging.info(
                    f"The weights of {spec!r} are licensed under "
                    "CC BY-NC 4.0 (non-commercial use only)."
                )
            model = load_pretrained_so3lr(
                spec, device=device, dtype=dtype, **architecture_overrides
            )
        else:
            model = _build_so3lr_from_settings_and_state_dict(
                settings_path_for_checkpoint(path),
                path,
                device=device,
                dtype=dtype,
                **architecture_overrides,
            )
        return _to_single_theory_level(model, theory_level)
    finally:
        torch.set_default_dtype(previous_dtype)


def _resolve_long_range_value(
    model: SO3LR, name: str, value: Optional[float]
) -> float:
    """``value``, else the model's stored ``name``, else the default of
    ``SO3LRCalculator`` (read from its signature)."""
    if value is not None:
        return value
    stored = getattr(model, name, None)
    if stored is not None:
        return stored
    default = inspect.signature(SO3LRCalculator.__init__).parameters[name]
    logging.info(
        f"The model stores no {name}; using the SO3LRCalculator default "
        f"{name}={default.default}."
    )
    return default.default


def make_so3lr_calculator(
    spec: str,
    *,
    device: str,
    dtype: str,
    theory_level: Optional[int] = None,
    compute_stress: bool = False,
    r_max_lr: Optional[float] = None,
    dispersion_energy_cutoff_lr_damping: Optional[float] = None,
    key_specification=None,
    **architecture_overrides: Any,
) -> TorchkratesCalculator:
    """ASE calculator around ``load_so3lr(spec, ...)``.

    ``r_max_lr`` and ``dispersion_energy_cutoff_lr_damping`` are each
    resolved as: the argument if not ``None``, else the value stored in
    the model, else the default of ``SO3LRCalculator`` (v1 stores
    neither). The calculator builds the long-range neighbor list itself,
    so it always needs an ``r_max_lr``.
    """
    model = load_so3lr(
        spec,
        device,
        dtype,
        theory_level=theory_level,
        **architecture_overrides,
    )
    r_max_lr = _resolve_long_range_value(model, "r_max_lr", r_max_lr)
    damping = _resolve_long_range_value(
        model,
        "dispersion_energy_cutoff_lr_damping",
        dispersion_energy_cutoff_lr_damping,
    )
    model.r_max_lr = r_max_lr
    return TorchkratesCalculator(
        models=[model],
        device=device,
        default_dtype=dtype,
        compute_stress=compute_stress,
        r_max_lr=r_max_lr,
        dispersion_energy_cutoff_lr_damping=damping,
        key_specification=key_specification,
    )


def so3lr_architecture_dict(model: SO3LR) -> dict[str, Any]:
    """Architecture of ``model`` as ``SO3LRArchitectureSettings`` values.

    Contains every settings field the model stores as an attribute
    (``pme_*`` only exist when PME is on), keyed by settings field name.
    The aims-PAX-only fields (``model_choice``, ``use_multihead_model``,
    ``num_multihead_heads``) are not included. ``final_layer_bias`` is not
    stored as an attribute; it is read from the energy head.
    """
    architecture = {}
    for field in SO3LRArchitectureSettings.model_fields:
        if field in _AIMS_PAX_ONLY_FIELDS:
            continue
        attribute = _SETTINGS_TO_ATTRIBUTE.get(field, field)
        if hasattr(model, attribute):
            architecture[field] = getattr(model, attribute)
    if "final_layer_bias" not in architecture:
        final_layer = model.atomic_energy_output_block.final_layer
        architecture["final_layer_bias"] = final_layer.bias is not None
    return architecture
