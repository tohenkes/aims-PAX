"""MBD-NL a0/C6 ratio labels: placeholder until FHI-aims can provide them.

The SO3LR-tauri models have an a0 (polarizability) ratio head and a C6
ratio head, trained on PBE0+MBD-NL values: the in-system VV value divided
by the free-atom VV value (the free-atom tables are so3krates-torch's
``alphas`` / ``C6_coef``). Stock FHI-aims does not print these per-atom
values; QCML/QCell used a customised FHI-aims build. Until aims-PAX can
obtain them, the ratio losses are not available and the heads of a
pretrained tauri model are frozen, so that training on energies and
forces does not move them away from their pretrained values.

v1-type models (``legacy_dispersion_bool=True``) are not affected: their
a0 head predicts Hirshfeld volume ratios, which stock FHI-aims provides,
and they have no C6 head.
"""

import logging

import torch

from aims_PAX.settings.model import TrainingSettings

MBD_NL_RATIO_LABELS_AVAILABLE = False
"""Whether aims-PAX can label structures with MBD-NL a0/C6 ratios."""

_C6_HEAD = "c6_ratios_output_block"
_A0_HEAD = "hirshfeld_output_block"
_HEAD_FREEZE_SETTING = {
    _C6_HEAD: "freeze_c6_ratios",
    _A0_HEAD: "freeze_hirshfeld",
}
_UNAVAILABLE = (
    "MBD-NL a0/C6 ratio labels need per-atom output that stock FHI-aims "
    "does not print"
)


def has_mbd_nl_a0_head(model: torch.nn.Module) -> bool:
    """Whether ``model``'s a0 head predicts MBD-NL (VV) polarizability
    ratios (SO3LR-tauri) rather than Hirshfeld volume ratios (v1)."""
    return getattr(model, "dispersion_energy_bool", False) and not getattr(
        model, "legacy_dispersion_bool", True
    )


def unlabelled_ratio_heads(model: torch.nn.Module) -> list[str]:
    """Names of ``model``'s ratio heads that aims-PAX cannot label."""
    if MBD_NL_RATIO_LABELS_AVAILABLE:
        return []
    heads = []
    if getattr(model, "c6_ratios_bool", False) and hasattr(model, _C6_HEAD):
        heads.append(_C6_HEAD)
    if has_mbd_nl_a0_head(model) and hasattr(model, _A0_HEAD):
        heads.append(_A0_HEAD)
    return heads


def frozen_ratio_heads(model: torch.nn.Module) -> list[str]:
    """Unlabelled ratio heads of ``model`` whose parameters are all
    frozen already (e.g. by an earlier initial-dataset run)."""
    return [
        head
        for head in unlabelled_ratio_heads(model)
        if not any(p.requires_grad for p in getattr(model, head).parameters())
    ]


def freeze_unlabelled_ratio_heads(
    model: torch.nn.Module, training: TrainingSettings
) -> list[str]:
    """Freeze the ratio heads aims-PAX cannot label and return their
    names. Call it after fine-tuning settings are applied, which may
    unfreeze every parameter. Warns when the user asked for a head to be
    trainable."""
    heads = unlabelled_ratio_heads(model)
    for head in heads:
        for param in getattr(model, head).parameters():
            param.requires_grad = False
        setting = _HEAD_FREEZE_SETTING[head]
        if setting in training.model_fields_set and not getattr(
            training, setting
        ):
            logging.warning(
                f"TRAINING.{setting}=false is ignored: {_UNAVAILABLE}, so "
                f"{head} stays frozen at its pretrained values."
            )
    if heads:
        logging.info(
            f"Frozen pretrained ratio heads ({', '.join(heads)}): "
            f"{_UNAVAILABLE}."
        )
    return heads


def warn_ratio_loss_weights(training: TrainingSettings) -> None:
    """Warn that the MBD-NL ratio losses are not available yet."""
    if MBD_NL_RATIO_LABELS_AVAILABLE:
        return
    for key in ("c6_ratios_weight", "a0_ratios_weight"):
        if getattr(training, key) > 0:
            logging.warning(
                f"TRAINING.{key} is ignored: {_UNAVAILABLE} (QCML/QCell "
                "used a customised FHI-aims build). This loss is not "
                "available yet."
            )


def warn_hirshfeld_loss_on_mbd_nl_head(
    model: torch.nn.Module, training: TrainingSettings
) -> None:
    """Warn when a Hirshfeld loss would train an MBD-NL a0 head."""
    if (
        has_mbd_nl_a0_head(model)
        and "hirshfeld" in training.loss.lower()
        and training.hirshfeld_weight > 0
    ):
        logging.warning(
            f"TRAINING.loss={training.loss!r} uses Hirshfeld volume ratios "
            "from FHI-aims as labels, but this model's a0 head predicts "
            "MBD-NL (VV) polarizability ratios. The head stays frozen; the "
            "loss term would still push the shared representation towards "
            "a different quantity. Use a loss without Hirshfeld ratios."
        )
