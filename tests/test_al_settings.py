"""
Tests for ALSettings Pydantic model (active learning config).
"""

import sys

import pytest
from pydantic import ValidationError

from aims_PAX.settings.project import ALSettings

# ---------------------------------------------------------------------------
# Baseline dict – always use dict-merge to override keys
# ---------------------------------------------------------------------------

VALID_BASE = dict(
    num_trajectories=4,
    desired_acc=0.1,
)


def make_base(tmp_path, **overrides):
    """Return a kwargs dict with a real species_dir and any extra overrides."""
    return {**VALID_BASE, "species_dir": tmp_path, **overrides}


# ===========================================================================
# §1 — Missing required field
# ===========================================================================


def test_missing_num_trajectories(tmp_path):
    """Omitting the required field raises ValidationError."""
    with pytest.raises(ValidationError):
        ALSettings(
            desired_acc=0.1,
            species_dir=tmp_path,
        )


# ===========================================================================
# §2 — gt=0 field constraints
# ===========================================================================


def test_num_trajectories_zero(tmp_path):
    """num_trajectories=0 violates gt=0."""
    with pytest.raises(ValidationError, match="num_trajectories"):
        ALSettings(**make_base(tmp_path, num_trajectories=0))


def test_valid_ratio_zero(tmp_path):
    """valid_ratio=0.0 violates gt=0."""
    with pytest.raises(ValidationError, match="valid_ratio"):
        ALSettings(**make_base(tmp_path, valid_ratio=0.0))


def test_ensemble_size_zero(tmp_path):
    """ensemble_size=0 violates gt=0."""
    with pytest.raises(ValidationError, match="ensemble_size"):
        ALSettings(**make_base(tmp_path, ensemble_size=0))


def test_convergence_patience_zero(tmp_path):
    """convergence_patience=0 violates gt=0."""
    with pytest.raises(ValidationError, match="convergence_patience"):
        ALSettings(**make_base(tmp_path, convergence_patience=0))


def test_max_convergence_epochs_zero(tmp_path):
    """max_convergence_epochs=0 violates gt=0."""
    with pytest.raises(ValidationError, match="max_convergence_epochs"):
        ALSettings(**make_base(tmp_path, max_convergence_epochs=0))


# ===========================================================================
# §3 — validate_species_dir
# ===========================================================================


def test_species_dir_required_by_default():
    """No species_dir and use_teacher_reference=False raises ValidationError."""
    with pytest.raises(ValidationError, match="species_dir"):
        ALSettings(
            num_trajectories=4,
            desired_acc=0.1,
        )


def test_species_dir_not_required_with_teacher():
    """species_dir=None is allowed when use_teacher_reference=True."""
    s = ALSettings(
        num_trajectories=4,
        desired_acc=0.1,
        species_dir=None,
        use_teacher_reference=True,
        teacher_reference_settings={"model_type": "mace-mp"},
    )
    assert s.species_dir is None


def test_species_dir_valid_path(tmp_path):
    """A valid directory path for species_dir passes validation."""
    s = ALSettings(**make_base(tmp_path))
    assert s.species_dir == tmp_path


def test_species_dir_file_raises(tmp_path):
    """A path pointing to a file (not a directory) is rejected."""
    file_path = tmp_path / "notadir.txt"
    file_path.touch()
    with pytest.raises(ValidationError, match="species_dir"):
        ALSettings(
            num_trajectories=4,
            desired_acc=0.1,
            species_dir=file_path,
        )


# ===========================================================================
# §4 — check_at_least_one_required
# ===========================================================================


def test_no_stopping_criterion(tmp_path):
    """All stopping criteria at defaults → ValidationError."""
    with pytest.raises(
        ValidationError, match="stopping criterion"
    ):
        ALSettings(
            **make_base(
                tmp_path,
                desired_acc=0.0,
                max_MD_steps=sys.maxsize,
                max_train_set_size=sys.maxsize,
            )
        )


def test_desired_acc_satisfies(tmp_path):
    """desired_acc > 0 alone satisfies the stopping criterion."""
    s = ALSettings(
        **make_base(
            tmp_path,
            desired_acc=0.01,
            max_MD_steps=sys.maxsize,
            max_train_set_size=sys.maxsize,
        )
    )
    assert s.desired_acc == 0.01


def test_max_MD_steps_satisfies(tmp_path):
    """max_MD_steps < sys.maxsize alone satisfies the stopping criterion."""
    s = ALSettings(
        **make_base(
            tmp_path,
            desired_acc=0.0,
            max_MD_steps=1000,
        )
    )
    assert s.max_MD_steps == 1000


def test_max_train_set_size_satisfies(tmp_path):
    """max_train_set_size < sys.maxsize alone satisfies the stopping
    criterion."""
    s = ALSettings(
        **make_base(
            tmp_path,
            desired_acc=0.0,
            max_train_set_size=100,
        )
    )
    assert s.max_train_set_size == 100


def test_all_three_satisfy(tmp_path):
    """All three stopping criteria set → no error."""
    s = ALSettings(
        **make_base(
            tmp_path,
            desired_acc=0.01,
            max_MD_steps=1000,
            max_train_set_size=100,
        )
    )
    assert s.desired_acc == 0.01
    assert s.max_MD_steps == 1000
    assert s.max_train_set_size == 100


# ===========================================================================
# §5 — validate_teacher_reference
# ===========================================================================


def test_teacher_requires_model_type_key():
    """use_teacher_reference=True without model_type key raises
    ValidationError."""
    with pytest.raises(ValidationError, match="model_type"):
        ALSettings(
            num_trajectories=4,
            desired_acc=0.1,
            use_teacher_reference=True,
            teacher_reference_settings={},
        )


def test_teacher_valid():
    """use_teacher_reference=True with correct settings passes validation."""
    s = ALSettings(
        num_trajectories=4,
        desired_acc=0.1,
        use_teacher_reference=True,
        teacher_reference_settings={"model_type": "mace-mp"},
    )
    assert s.use_teacher_reference is True


def test_teacher_disables_analysis():
    """analysis=True is silently set to False when use_teacher_reference=True."""
    s = ALSettings(
        num_trajectories=4,
        desired_acc=0.1,
        use_teacher_reference=True,
        teacher_reference_settings={"model_type": "mace-mp"},
        analysis=True,
    )
    assert s.analysis is False


# ===========================================================================
# §6 — uncertainty_type acceptance (positive-only, plain str field)
# ===========================================================================


def test_uncertainty_type_max_atomic_sd(tmp_path):
    """uncertainty_type='max_atomic_sd' is accepted."""
    s = ALSettings(**make_base(tmp_path, uncertainty_type="max_atomic_sd"))
    assert s.uncertainty_type == "max_atomic_sd"


def test_uncertainty_type_mean_atomic_sd(tmp_path):
    """uncertainty_type='mean_atomic_sd' is accepted."""
    s = ALSettings(**make_base(tmp_path, uncertainty_type="mean_atomic_sd"))
    assert s.uncertainty_type == "mean_atomic_sd"


def test_uncertainty_type_ensemble_sd(tmp_path):
    """uncertainty_type='ensemble_sd' is accepted."""
    s = ALSettings(**make_base(tmp_path, uncertainty_type="ensemble_sd"))
    assert s.uncertainty_type == "ensemble_sd"


# ===========================================================================
# §7 — save_trajectories defaults
# ===========================================================================


def test_save_trajectories_defaults(tmp_path):
    """save_trajectories defaults to True and interval defaults to 5."""
    s = ALSettings(**make_base(tmp_path))
    assert s.save_trajectories is True
    assert s.save_trajectories_interval == 5


# ===========================================================================
# §8 — from_file()
# ===========================================================================


def test_from_yaml_valid(tmp_path):
    """Load a valid YAML file via from_file() returns a correct
    ALSettings instance."""
    yaml_file = tmp_path / "al_settings.yaml"
    yaml_file.write_text(
        "num_trajectories: 4\n"
        "desired_acc: 0.1\n"
        f"species_dir: {tmp_path}\n"
    )
    s = ALSettings.from_file(yaml_file)
    assert isinstance(s, ALSettings)
    assert s.num_trajectories == 4
    assert s.desired_acc == 0.1


def test_from_yaml_missing_file(tmp_path):
    """Passing a non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ALSettings.from_file(tmp_path / "nonexistent.yaml")


def test_from_yaml_invalid_data(tmp_path):
    """A YAML with num_trajectories=0 raises ValidationError."""
    yaml_file = tmp_path / "bad_settings.yaml"
    yaml_file.write_text(
        "num_trajectories: 0\n"
        "desired_acc: 0.1\n"
        f"species_dir: {tmp_path}\n"
    )
    with pytest.raises(ValidationError):
        ALSettings.from_file(yaml_file)


# ===========================================================================
# §9 — Defaults sanity check
# ===========================================================================


def test_defaults(tmp_path):
    """All default values are as documented."""
    s = ALSettings(**make_base(tmp_path))
    assert s.analysis is False
    assert s.ensemble_size == 4
    assert s.valid_ratio == 0.1
    assert s.valid_skip == 1
    assert s.convergence_patience == 50
    assert s.max_convergence_epochs == 500
    assert s.converge_al is True
    assert s.converge_best is True
    assert s.margin == 0.002
    assert s.replay_strategy == "full_dataset"
    assert s.uncertainty_type == "max_atomic_sd"
    assert s.save_trajectories is True
    assert s.save_trajectories_interval == 5
    assert s.update_md_checkpoints is True
    assert s.max_MD_steps == sys.maxsize
    assert s.max_train_set_size == sys.maxsize


# ===========================================================================
# §10 — replay_strategy / train_subset_size
# ===========================================================================


def test_random_subset_without_subset_size(tmp_path):
    """replay_strategy='random_subset' without train_subset_size raises."""
    with pytest.raises(ValidationError, match="train_subset_size"):
        ALSettings(**make_base(tmp_path, replay_strategy="random_subset"))


def test_random_subset_with_subset_size(tmp_path):
    """replay_strategy='random_subset' with train_subset_size passes."""
    s = ALSettings(
        **make_base(
            tmp_path, replay_strategy="random_subset", train_subset_size=50
        )
    )
    assert s.replay_strategy == "random_subset"
    assert s.train_subset_size == 50
