"""
Tests for `aims_PAX.tools.utilities.input_utils`:
  - read_input_files
  - read_geometry

All tests are fast (no @pytest.mark.slow).
"""

from pathlib import Path

import ase
import pytest
import yaml

from aims_PAX.settings import AimsPAXSettings, ModelSettings
from aims_PAX.tools.utilities.input_utils import (
    read_geometry,
    read_input_files,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patched_aimspax_yaml(
    tmp_path: Path,
    data_dir: Path,
    source_yaml: Path,
) -> Path:
    """
    Write a copy of source_yaml into tmp_path with:
      - species_dir replaced by absolute paths
      - path_to_control / path_to_geometry set to real test-data files
      - output_dir set to tmp_path (so resolve_all_dirs can create dirs)
    """
    with open(source_yaml) as f:
        data = yaml.safe_load(f)

    species_dir = str(data_dir / "species_defaults" / "light")
    if "INITIAL_DATASET_GENERATION" in data and data[
        "INITIAL_DATASET_GENERATION"
    ] is not None:
        data["INITIAL_DATASET_GENERATION"]["species_dir"] = species_dir
    if "ACTIVE_LEARNING" in data and data["ACTIVE_LEARNING"] is not None:
        data["ACTIVE_LEARNING"]["species_dir"] = species_dir

    # Point to real files so FilePath validators pass
    data.setdefault("MISC", {})
    data["MISC"]["path_to_control"] = str(
        data_dir / "control_files" / "periodic.in"
    )
    data["MISC"]["path_to_geometry"] = str(
        data_dir / "structures" / "Si.in"
    )
    data["MISC"]["output_dir"] = str(tmp_path)

    patched = tmp_path / "aimsPAX_patched.yaml"
    with open(patched, "w") as f:
        yaml.dump(data, f)
    return patched


# ---------------------------------------------------------------------------
# read_input_files
# ---------------------------------------------------------------------------


class TestReadInputFiles:
    """Tests for read_input_files()."""

    def test_read_input_files_returns_4_tuple(
        self, tmp_path, data_dir
    ):
        """
        read_input_files returns a 4-tuple:
        (ModelSettings, AimsPAXSettings, path_to_control, path_to_geometry)
        """
        source_yaml = (
            data_dir / "project_settings" / "aimsPAX.yaml"
        )
        model_yaml = data_dir / "project_settings" / "model.yaml"
        patched = _patched_aimspax_yaml(tmp_path, data_dir, source_yaml)

        result = read_input_files(
            path_to_model_settings=str(model_yaml),
            path_to_aimsPAX_settings=str(patched),
        )

        assert len(result) == 4, "Expected a 4-tuple"
        model_settings, aimspax_settings, path_to_control, path_to_geometry = (
            result
        )

        assert isinstance(model_settings, ModelSettings)
        assert isinstance(aimspax_settings, AimsPAXSettings)
        # all_teacher is False in the test YAML → path_to_control is not None
        assert path_to_control is not None
        assert path_to_geometry is not None

    @pytest.mark.parametrize(
        "procedure", ["full", "al", "initial-ds"]
    )
    def test_read_input_files_procedure_variants(
        self, tmp_path, data_dir, procedure
    ):
        """
        read_input_files runs without error for all supported procedure values.
        """
        source_yaml = (
            data_dir / "project_settings" / "aimsPAX.yaml"
        )
        model_yaml = data_dir / "project_settings" / "model.yaml"
        # Each parametrize invocation gets its own sub-directory to avoid
        # directory creation conflicts across parallel test runs.
        out_dir = tmp_path / procedure
        out_dir.mkdir()
        patched = _patched_aimspax_yaml(out_dir, data_dir, source_yaml)

        result = read_input_files(
            path_to_model_settings=str(model_yaml),
            path_to_aimsPAX_settings=str(patched),
            procedure=procedure,
        )

        model_s, pax_s, ctrl, geom = result
        assert isinstance(model_s, ModelSettings)
        assert isinstance(pax_s, AimsPAXSettings)
        assert geom is not None


# ---------------------------------------------------------------------------
# read_geometry
# ---------------------------------------------------------------------------


class TestReadGeometry:
    """Tests for read_geometry()."""

    # ------------------------------------------------------------------
    # Form A / Form D – single file (str path and Path object)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "source_factory",
        [
            lambda p: str(p),  # Form A: str path
            lambda p: p,  # Form D: Path object
        ],
        ids=["str", "Path"],
    )
    def test_single_file_returns_si(self, source_factory, data_dir):
        si_path = data_dir / "structures" / "Si.in"
        result = read_geometry(source_factory(si_path))

        assert isinstance(result, dict)
        assert set(result.keys()) == {0}
        assert isinstance(result[0], ase.Atoms)
        assert "Si" in result[0].get_chemical_symbols()

    # ------------------------------------------------------------------
    # Form B – directory
    # ------------------------------------------------------------------

    def test_directory_returns_two_entries(self, data_dir):
        structures_dir = str(data_dir / "structures")
        result = read_geometry(structures_dir)

        assert isinstance(result, dict)
        assert len(result) == 2
        assert all(isinstance(v, ase.Atoms) for v in result.values())

    def test_directory_keys_are_0_and_1(self, data_dir):
        structures_dir = str(data_dir / "structures")
        result = read_geometry(structures_dir)

        assert set(result.keys()) == {0, 1}

    # ------------------------------------------------------------------
    # Form C – dict of paths
    # ------------------------------------------------------------------

    def test_dict_of_paths_geometry(self, data_dir):
        si_path = str(data_dir / "structures" / "Si.in")
        asp_path = str(data_dir / "structures" / "aspirin.in")
        result = read_geometry({0: si_path, 1: asp_path})

        assert set(result.keys()) == {0, 1}
        assert "Si" in result[0].get_chemical_symbols()
        assert "C" in result[1].get_chemical_symbols()
        assert "H" in result[1].get_chemical_symbols()

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_integer_source_raises_type_error(self):
        with pytest.raises(TypeError):
            read_geometry(42)  # type: ignore[arg-type]

    def test_empty_directory_raises_value_error(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError):
            read_geometry(str(empty_dir))
