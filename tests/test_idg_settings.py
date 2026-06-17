"""
Tests for IDGSettings Pydantic model (initial dataset generation config).
"""

import sys

import pytest
from pydantic import ValidationError

from aims_PAX.settings.project import (
    FMSettings,
    IDGSettings,
    MaceFMSettings,
    So3lrFMSettings,
)

# ---------------------------------------------------------------------------
# Baseline dict – always use dict-merge to override keys
# ---------------------------------------------------------------------------

VALID_BASE = dict(
    n_points_per_sampling_step_idg=10,
    max_initial_epochs=50,
)


def make_base(tmp_path, **overrides):
    """Return a kwargs dict with a real species_dir and any extra overrides."""
    return {**VALID_BASE, "species_dir": tmp_path, **overrides}


# ===========================================================================
# §2.1 — Missing required field
# ===========================================================================


def test_missing_n_points(tmp_path):
    """Omitting the required field raises ValidationError."""
    with pytest.raises(ValidationError):
        IDGSettings(
            max_initial_epochs=50,
            species_dir=tmp_path,
        )


# ===========================================================================
# §2.2 — gt=0 field constraints
# ===========================================================================


def test_n_points_zero(tmp_path):
    """n_points_per_sampling_step_idg=0 violates gt=0."""
    with pytest.raises(
        ValidationError, match="n_points_per_sampling_step_idg"
    ):
        IDGSettings(**make_base(tmp_path, n_points_per_sampling_step_idg=0))


def test_ensemble_size_zero(tmp_path):
    """ensemble_size=0 violates gt=0."""
    with pytest.raises(ValidationError, match="ensemble_size"):
        IDGSettings(**make_base(tmp_path, ensemble_size=0))


def test_intermediate_epochs_zero(tmp_path):
    """intermediate_epochs_idg=0 violates gt=0."""
    with pytest.raises(ValidationError, match="intermediate_epochs_idg"):
        IDGSettings(**make_base(tmp_path, intermediate_epochs_idg=0))


def test_max_initial_epochs_zero(tmp_path):
    """max_initial_epochs=0 violates gt=0."""
    with pytest.raises(ValidationError, match="max_initial_epochs"):
        IDGSettings(**make_base(tmp_path, max_initial_epochs=0))


def test_max_initial_set_size_zero(tmp_path):
    """max_initial_set_size=0 violates gt=0."""
    with pytest.raises(ValidationError, match="max_initial_set_size"):
        IDGSettings(**make_base(tmp_path, max_initial_set_size=0))


def test_progress_dft_update_zero(tmp_path):
    """progress_dft_update=0 violates gt=0."""
    with pytest.raises(ValidationError, match="progress_dft_update"):
        IDGSettings(**make_base(tmp_path, progress_dft_update=0))


def test_skip_step_initial_zero(tmp_path):
    """skip_step_initial=0 violates gt=0."""
    with pytest.raises(ValidationError, match="skip_step_initial"):
        IDGSettings(**make_base(tmp_path, skip_step_initial=0))


def test_valid_ratio_zero(tmp_path):
    """valid_ratio=0.0 violates gt=0."""
    with pytest.raises(ValidationError, match="valid_ratio"):
        IDGSettings(**make_base(tmp_path, valid_ratio=0.0))


# ===========================================================================
# §2.3 — validate_species_dir
# ===========================================================================


def test_species_dir_required_by_default():
    """No species_dir and use_teacher_reference=False raises
    ValidationError."""
    with pytest.raises(ValidationError, match="species_dir is required"):
        IDGSettings(
            n_points_per_sampling_step_idg=10,
            max_initial_epochs=50,
        )


def test_species_dir_not_required_with_teacher():
    """species_dir=None is allowed when use_teacher_reference=True."""
    s = IDGSettings(
        n_points_per_sampling_step_idg=10,
        max_initial_epochs=50,
        species_dir=None,
        use_teacher_reference=True,
        initial_sampling="foundational",
        teacher_reference_settings={"model_type": "mace-mp"},
    )
    assert s.species_dir is None


def test_species_dir_valid_path(tmp_path):
    """A valid directory path for species_dir passes validation."""
    s = IDGSettings(**make_base(tmp_path))
    assert s.species_dir == tmp_path


def test_species_dir_file_raises(tmp_path):
    """A path pointing to a file (not a directory) is rejected."""
    file_path = tmp_path / "notadir.txt"
    file_path.touch()
    with pytest.raises(ValidationError, match="species_dir"):
        IDGSettings(
            n_points_per_sampling_step_idg=10,
            max_initial_epochs=50,
            species_dir=file_path,
        )


# ===========================================================================
# §2.4 — check_at_least_one_required
# ===========================================================================


def test_no_stopping_criterion(tmp_path):
    """All stopping criteria at defaults → ValidationError."""
    with pytest.raises(
        ValidationError, match="at least one stopping criterion"
    ):
        IDGSettings(
            **make_base(
                tmp_path,
                max_initial_epochs=sys.maxsize,
                desired_acc=0.0,
                max_initial_set_size=sys.maxsize,
            )
        )


def test_desired_acc_satisfies(tmp_path):
    """desired_acc > 0 alone satisfies the stopping criterion."""
    s = IDGSettings(
        **make_base(
            tmp_path,
            max_initial_epochs=sys.maxsize,
            desired_acc=0.01,
            max_initial_set_size=sys.maxsize,
        )
    )
    assert s.desired_acc == 0.01


def test_max_epochs_satisfies(tmp_path):
    """max_initial_epochs < sys.maxsize alone satisfies the stopping
    criterion."""
    s = IDGSettings(
        **make_base(
            tmp_path,
            max_initial_epochs=100,
            desired_acc=0.0,
            max_initial_set_size=sys.maxsize,
        )
    )
    assert s.max_initial_epochs == 100


def test_max_set_size_satisfies(tmp_path):
    """max_initial_set_size < sys.maxsize alone satisfies stopping
    criterion."""
    s = IDGSettings(
        **make_base(
            tmp_path,
            max_initial_epochs=sys.maxsize,
            desired_acc=0.0,
            max_initial_set_size=200,
        )
    )
    assert s.max_initial_set_size == 200


def test_all_three_satisfy(tmp_path):
    """All three stopping criteria set → no error."""
    s = IDGSettings(
        **make_base(
            tmp_path,
            max_initial_epochs=100,
            desired_acc=0.01,
            max_initial_set_size=200,
        )
    )
    assert s.desired_acc == 0.01
    assert s.max_initial_epochs == 100
    assert s.max_initial_set_size == 200


# ===========================================================================
# §2.5 — validate_teacher_reference
# ===========================================================================


def test_teacher_requires_foundational_sampling():
    """use_teacher_reference=True with initial_sampling='aimd' raises
    ValidationError."""
    with pytest.raises(
        ValidationError, match="initial_sampling: foundational"
    ):
        IDGSettings(
            n_points_per_sampling_step_idg=10,
            max_initial_epochs=50,
            use_teacher_reference=True,
            initial_sampling="aimd",
            teacher_reference_settings={"model_type": "mace-mp"},
        )


def test_teacher_requires_model_type_key():
    """use_teacher_reference=True without model_type key raises
    ValidationError."""
    with pytest.raises(ValidationError, match="model_type"):
        IDGSettings(
            n_points_per_sampling_step_idg=10,
            max_initial_epochs=50,
            use_teacher_reference=True,
            initial_sampling="foundational",
            teacher_reference_settings={},
        )


def test_teacher_valid():
    """use_teacher_reference=True with correct settings passes validation."""
    s = IDGSettings(
        n_points_per_sampling_step_idg=10,
        max_initial_epochs=50,
        use_teacher_reference=True,
        initial_sampling="foundational",
        teacher_reference_settings={"model_type": "mace-mp"},
    )
    assert s.use_teacher_reference is True


# ===========================================================================
# §2.6 — validate_nested_settings
# ===========================================================================


def test_so3lr_settings_cast(tmp_path):
    """foundational_model='so3lr' casts foundational_model_settings
    to So3lrFMSettings."""
    s = IDGSettings(
        **make_base(
            tmp_path,
            foundational_model="so3lr",
            foundational_model_settings={},
        )
    )
    assert isinstance(s.foundational_model_settings, So3lrFMSettings)


def test_mace_mp_with_custom_model(tmp_path):
    """Custom mace_model value is preserved after casting."""
    s = IDGSettings(
        **make_base(
            tmp_path,
            foundational_model="mace-mp",
            foundational_model_settings={"mace_model": "large"},
        )
    )
    assert isinstance(s.foundational_model_settings, MaceFMSettings)
    assert s.foundational_model_settings.mace_model == "large"


def test_so3lr_with_dispersion(tmp_path):
    """So3lrFMSettings fields are set correctly via dict."""
    s = IDGSettings(
        **make_base(
            tmp_path,
            foundational_model="so3lr",
            foundational_model_settings={
                "dispersion": True,
                "r_max_lr": 6.0,
            },
        )
    )
    assert isinstance(s.foundational_model_settings, So3lrFMSettings)
    assert s.foundational_model_settings.dispersion is True
    assert s.foundational_model_settings.r_max_lr == 6.0


def test_mace_extra_key_ignored(tmp_path):
    """Unknown keys in foundational_model_settings are silently ignored
    (Pydantic v2)."""
    s = IDGSettings(
        **make_base(
            tmp_path,
            foundational_model="mace-mp",
            foundational_model_settings={"nonexistent_key": 42},
        )
    )
    assert isinstance(s.foundational_model_settings, MaceFMSettings)


def test_mace_invalid_nested_field(tmp_path):
    """dispersion_cutoff=0.0 violates gt=0 in nested FMSettings
    → ValidationError."""
    with pytest.raises(ValidationError, match="greater than 0"):
        IDGSettings(
            **make_base(
                tmp_path,
                foundational_model="mace-mp",
                foundational_model_settings={"dispersion_cutoff": 0.0},
            )
        )


# ===========================================================================
# §2.7 — to_lower field_validator
# ===========================================================================


def test_foundational_model_uppercase(tmp_path):
    """'MACE-MP' is normalised to 'mace-mp'."""
    s = IDGSettings(**make_base(tmp_path, foundational_model="MACE-MP"))
    assert s.foundational_model == "mace-mp"


def test_foundational_model_mixed_case(tmp_path):
    """'So3lr' is normalised to 'so3lr'."""
    s = IDGSettings(**make_base(tmp_path, foundational_model="So3lr"))
    assert s.foundational_model == "so3lr"


# ===========================================================================
# §2.8 — Literal rejection + to_lower pipeline
# ===========================================================================


def test_invalid_foundational_model(tmp_path):
    """'mace' is not a valid Literal value → ValidationError."""
    with pytest.raises(ValidationError, match="foundational_model"):
        IDGSettings(**make_base(tmp_path, foundational_model="mace"))


def test_invalid_initial_sampling(tmp_path):
    """'random' is not a valid initial_sampling value → ValidationError."""
    with pytest.raises(ValidationError, match="initial_sampling"):
        IDGSettings(**make_base(tmp_path, initial_sampling="random"))


def test_invalid_model_uppercase(tmp_path):
    """'MACE' lowercases to 'mace', which still fails the Literal
    → ValidationError."""
    with pytest.raises(ValidationError, match="foundational_model"):
        IDGSettings(**make_base(tmp_path, foundational_model="MACE"))


def test_aimd_sampling_valid(tmp_path):
    """initial_sampling='aimd' with use_teacher_reference=False is valid."""
    s = IDGSettings(**make_base(tmp_path, initial_sampling="aimd"))
    assert s.initial_sampling == "aimd"


# ===========================================================================
# §2.9 — Nested model standalone
# ===========================================================================


def test_fmsettings_defaults():
    """FMSettings has the expected default values."""
    s = FMSettings()
    assert s.dispersion is False
    assert s.dispersion_xc == "pbe"
    assert s.dispersion_cutoff == 12.0
    assert s.damping == "bj"


def test_fmsettings_cutoff_zero():
    """FMSettings(dispersion_cutoff=0.0) violates gt=0 → ValidationError."""
    with pytest.raises(ValidationError, match="dispersion_cutoff"):
        FMSettings(dispersion_cutoff=0.0)


def test_mace_defaults():
    """MaceFMSettings has the expected default mace_model."""
    s = MaceFMSettings()
    assert s.mace_model == "small"


def test_so3lr_defaults():
    """So3lrFMSettings has correct None defaults for optional fields."""
    s = So3lrFMSettings()
    assert s.r_max_lr is None
    assert s.dispersion_lr_damping is None


# ===========================================================================
# §2.10 — from_file
# ===========================================================================


def test_from_yaml_valid(tmp_path):
    """Load a valid YAML file via from_file() returns a correct
    IDGSettings instance."""
    yaml_file = tmp_path / "idg_settings.yaml"
    yaml_file.write_text(
        "n_points_per_sampling_step_idg: 10\n"
        "max_initial_epochs: 50\n"
        f"species_dir: {tmp_path}\n"
    )
    s = IDGSettings.from_file(yaml_file)
    assert isinstance(s, IDGSettings)
    assert s.n_points_per_sampling_step_idg == 10
    assert s.max_initial_epochs == 50


def test_from_yaml_missing_file(tmp_path):
    """Passing a non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        IDGSettings.from_file(tmp_path / "nonexistent.yaml")


def test_from_yaml_invalid_data(tmp_path):
    """A YAML with n_points_per_sampling_step_idg=0 raises ValidationError."""
    yaml_file = tmp_path / "bad_settings.yaml"
    yaml_file.write_text(
        "n_points_per_sampling_step_idg: 0\n"
        "max_initial_epochs: 50\n"
        f"species_dir: {tmp_path}\n"
    )
    with pytest.raises(ValidationError):
        IDGSettings.from_file(yaml_file)


# ===========================================================================
# §2.11 — Defaults sanity check
# ===========================================================================


def test_defaults(tmp_path):
    """All default values are as documented."""
    s = IDGSettings(
        n_points_per_sampling_step_idg=5,
        max_initial_epochs=10,
        species_dir=tmp_path,
    )
    assert s.analysis is False
    assert s.desired_acc == 0.0
    assert s.desired_acc_scale_idg == 10.0
    assert s.distinct_model_sets is True
    assert s.ensemble_size == 4
    assert s.foundational_model == "mace-mp"
    assert isinstance(s.foundational_model_settings, MaceFMSettings)
    assert s.intermediate_epochs_idg == 5
    assert s.scheduler_initial is True
    assert s.skip_step_initial == 25
    assert s.valid_ratio == 0.1
    assert s.valid_skip == 1
    assert s.converge_initial is False
    assert s.convergence_patience == 50
    assert s.margin == 0.002
    assert s.max_convergence_epochs == 500
    assert s.initial_sampling == "foundational"
    assert s.use_teacher_reference is False
    assert s.max_initial_set_size == sys.maxsize
    assert s.progress_dft_update == 10
    assert s.teacher_reference_settings == {}


def test_max_initial_epochs_default(tmp_path):
    """max_initial_epochs defaults to sys.maxsize when not provided."""
    s = IDGSettings(
        n_points_per_sampling_step_idg=10,
        desired_acc=0.01,
        species_dir=tmp_path,
    )
    assert s.max_initial_epochs == sys.maxsize
