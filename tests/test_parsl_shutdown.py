"""
shutdown_parsl_on_exit: a failing run must shut PARSL down and exit
instead of leaving non-daemon PARSL threads alive (which made the process
hang and left Slurm blocks running until their idle timeout).
"""

import logging
import subprocess
import sys
import textwrap
from unittest.mock import MagicMock, patch

import parsl
import pytest
from parsl.config import Config
from parsl.errors import NoDataFlowKernelError
from parsl.executors import ThreadPoolExecutor as ParslThreadPoolExecutor

from aims_PAX.tools.utilities.parsl_utils import shutdown_parsl_on_exit


@pytest.fixture
def local_dfk(tmp_path):
    parsl.load(
        Config(
            executors=[ParslThreadPoolExecutor(label="local", max_threads=1)],
            run_dir=str(tmp_path / "rd"),
            initialize_logging=False,
            retries=0,
        )
    )
    yield
    try:
        parsl.dfk().cleanup()
    except Exception:
        pass
    parsl.clear()


def test_cleans_up_loaded_dfk_and_reraises(local_dfk):
    with pytest.raises(RuntimeError, match="boom"):
        with shutdown_parsl_on_exit():
            raise RuntimeError("boom")
    with pytest.raises(NoDataFlowKernelError):
        parsl.dfk()


def test_reraises_without_dfk():
    with pytest.raises(ValueError, match="x"):
        with shutdown_parsl_on_exit():
            raise ValueError("x")


def test_cleans_up_loaded_dfk_on_normal_exit(local_dfk):
    with shutdown_parsl_on_exit():
        pass
    with pytest.raises(NoDataFlowKernelError):
        parsl.dfk()


def test_normal_exit_without_dfk_is_a_noop(caplog):
    with caplog.at_level(logging.ERROR):
        with shutdown_parsl_on_exit():
            pass
    assert not caplog.records


def test_cleanup_failure_does_not_mask_original_error():
    dfk = MagicMock()
    dfk.cleanup.side_effect = OSError("cleanup failed")
    with patch("parsl.dfk", return_value=dfk), patch("parsl.clear"):
        with pytest.raises(RuntimeError, match="boom"):
            with shutdown_parsl_on_exit():
                raise RuntimeError("boom")
    dfk.cleanup.assert_called_once()


def test_dfk_is_cleared_even_if_cleanup_fails():
    dfk = MagicMock()
    dfk.cleanup.side_effect = OSError("cleanup failed")
    with patch("parsl.dfk", return_value=dfk), patch("parsl.clear") as clear:
        with pytest.raises(RuntimeError, match="boom"):
            with shutdown_parsl_on_exit():
                raise RuntimeError("boom")
    clear.assert_called_once()


def test_keyboard_interrupt_cleans_up_and_propagates(caplog):
    with (
        patch("parsl.dfk") as mock_dfk,
        patch("parsl.clear"),
        caplog.at_level(logging.ERROR),
    ):
        with pytest.raises(KeyboardInterrupt):
            with shutdown_parsl_on_exit():
                raise KeyboardInterrupt
    mock_dfk.return_value.cleanup.assert_called_once()
    assert not [r for r in caplog.records if r.exc_info]


def test_logs_traceback(caplog):
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError):
            with shutdown_parsl_on_exit():
                raise RuntimeError("boom")
    records = [r for r in caplog.records if r.exc_info]
    assert len(records) == 1
    assert records[0].exc_info[0] is RuntimeError


_WQ_SCRIPT = textwrap.dedent("""
    import sys
    import parsl
    from parsl.config import Config
    from parsl.executors import WorkQueueExecutor
    from aims_PAX.tools.utilities.parsl_utils import shutdown_parsl_on_exit

    tmp = sys.argv[1]
    parsl.load(
        Config(
            executors=[
                WorkQueueExecutor(
                    label="wq",
                    address="127.0.0.1",
                    port=0,
                    shared_fs=True,
                    function_dir=tmp + "/fd",
                    autocategory=False,
                )
            ],
            run_dir=tmp + "/rd",
            initialize_logging=False,
        )
    )
    with shutdown_parsl_on_exit():
        if sys.argv[2] == "raise":
            raise FileNotFoundError("geometry.in")
    """)


def _run_wq_script(tmp_path, mode):
    return subprocess.run(
        [sys.executable, "-c", _WQ_SCRIPT, str(tmp_path), mode],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )


@pytest.mark.slow
def test_process_exits_after_error_with_workqueue_loaded(tmp_path):
    result = _run_wq_script(tmp_path, "raise")
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr


@pytest.mark.slow
def test_process_exits_after_normal_finish_with_workqueue_loaded(tmp_path):
    result = _run_wq_script(tmp_path, "ok")
    assert result.returncode == 0, result.stderr
