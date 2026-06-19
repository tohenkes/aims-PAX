"""
Packaging and CLI smoke test — verify installed package is usable.

- Import aims_PAX succeeds
- Each console entry point launches --help with exit code 0 and
  non-empty output containing "usage" or the command name
- Atomate2 entry points guarded with importorskip and skipped if
  the entry point is not installed
"""

import shutil
import subprocess

import pytest


def test_import_aims_pax():
    """Verify aims_PAX package imports cleanly."""
    import aims_PAX  # noqa: F401


@pytest.mark.parametrize(
    "entry_point",
    [
        "aims-PAX",
        "aims-PAX-initial-ds",
        "aims-PAX-al",
        "aims-PAX-recalc",
    ],
)
def test_core_entry_point_help(entry_point):
    """Core entry points respond to --help with exit code 0."""
    result = subprocess.run(
        [entry_point, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{entry_point} --help failed with exit code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    output = result.stdout.lower()
    assert (
        "usage" in output or entry_point in output
    ), f"{entry_point} --help produced no usage info: {result.stdout}"


@pytest.mark.parametrize(
    "entry_point",
    [
        "aims-PAX-idg-v2",
        "aims-PAX-idg-split",
    ],
)
def test_atomate2_entry_point_help(entry_point):
    """Atomate2 entry points (optional) respond to --help if available."""
    pytest.importorskip("atomate2")
    # Skip if the entry point is not installed (atomate2 extra not installed)
    if shutil.which(entry_point) is None:
        pytest.skip(f"{entry_point} not installed")
    result = subprocess.run(
        [entry_point, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{entry_point} --help failed with exit code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    output = result.stdout.lower()
    assert (
        "usage" in output or entry_point in output
    ), f"{entry_point} --help produced no usage info: {result.stdout}"
