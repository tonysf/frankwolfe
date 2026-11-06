import os
from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY_ROOT / "scripts" / "run_qpt_jax_gpu.sh"
SETUP = REPOSITORY_ROOT / "scripts" / "setup_qpt_jax_gpu.sh"


def _make_fake_python(tmp_path):
    fake_python = tmp_path / "fake python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\0' \"$@\" > \"$QPT_TEST_ARGUMENTS\"\n"
        "printf '%s' \"${JAX_ENABLE_X64-}\" > \"$QPT_TEST_X64\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return fake_python


def _run_launcher(tmp_path, *arguments, x64=None):
    fake_python = _make_fake_python(tmp_path)
    captured_arguments = tmp_path / "arguments"
    captured_x64 = tmp_path / "x64"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_BIN": str(fake_python),
            "QPT_TEST_ARGUMENTS": str(captured_arguments),
            "QPT_TEST_X64": str(captured_x64),
        }
    )
    if x64 is None:
        environment.pop("JAX_ENABLE_X64", None)
    else:
        environment["JAX_ENABLE_X64"] = x64

    completed = subprocess.run(
        [str(LAUNCHER), *map(str, arguments)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    invoked_arguments = None
    if captured_arguments.exists():
        invoked_arguments = [
            item.decode()
            for item in captured_arguments.read_bytes().split(b"\0")
            if item
        ]
    invoked_x64 = (
        captured_x64.read_text(encoding="utf-8")
        if captured_x64.exists()
        else None
    )
    return completed, invoked_arguments, invoked_x64


def _make_fake_setup_python(fake_bin):
    fake_python = fake_bin / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "count_file=\"$QPT_SETUP_LOG/count\"\n"
        "count=0\n"
        "if [[ -f \"$count_file\" ]]; then\n"
        "    IFS= read -r count < \"$count_file\"\n"
        "fi\n"
        "count=$((count + 1))\n"
        "printf '%s\\n' \"$count\" > \"$count_file\"\n"
        "printf '%s\\0' \"$@\" > \"$QPT_SETUP_LOG/invocation.$count\"\n"
        "if [[ \"${1-}\" == '-c' ]]; then\n"
        "    case \"${2-}\" in\n"
        "        *platform.python_version*)\n"
        "            printf '%s\\n' \"$QPT_FAKE_PYTHON_VERSION\"\n"
        "            exit 0\n"
        "            ;;\n"
        "        *sys.version_info*)\n"
        "            exit \"$QPT_FAKE_VERSION_CHECK_EXIT\"\n"
        "            ;;\n"
        "    esac\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return fake_python


def _make_fake_nvidia_smi(fake_bin):
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$QPT_FAKE_DRIVER_VERSION\"\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)


def _read_setup_invocations(log_directory):
    count_file = log_directory / "count"
    if not count_file.exists():
        return []
    count = int(count_file.read_text(encoding="utf-8"))
    invocations = []
    for index in range(1, count + 1):
        raw_arguments = (log_directory / f"invocation.{index}").read_bytes()
        invocations.append(
            [item.decode() for item in raw_arguments.split(b"\0") if item]
        )
    return invocations


def _run_setup(
    tmp_path,
    *,
    driver_version="580.82.07",
    cuda_variant="auto",
    python_version="3.12.8",
    python_supported=True,
    preflight_arguments=(),
):
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    setup_log = tmp_path / "setup-log"
    setup_log.mkdir()
    fake_python = _make_fake_setup_python(fake_bin)
    _make_fake_nvidia_smi(fake_bin)

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": str(fake_bin) + os.pathsep + environment.get("PATH", ""),
            "PYTHON_BIN": str(fake_python),
            "JAX_CUDA_VARIANT": cuda_variant,
            "QPT_SETUP_LOG": str(setup_log),
            "QPT_FAKE_DRIVER_VERSION": driver_version,
            "QPT_FAKE_PYTHON_VERSION": python_version,
            "QPT_FAKE_VERSION_CHECK_EXIT": (
                "0" if python_supported else "1"
            ),
        }
    )
    completed = subprocess.run(
        [str(SETUP), *map(str, preflight_arguments)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed, _read_setup_invocations(setup_log)


def _find_pip_install(invocations):
    return next(
        (
            arguments
            for arguments in invocations
            if arguments[:3] == ["-m", "pip", "install"]
        ),
        None,
    )


def _find_preflight(invocations):
    return next(
        (
            arguments
            for arguments in invocations
            if arguments
            and Path(arguments[0]).name
            == "check_qpt_jax_environment.py"
        ),
        None,
    )


def test_launcher_forwards_schedules_and_enforces_gpu_configuration(tmp_path):
    data_path = tmp_path / "QPT data with spaces.h5"
    data_path.touch()

    completed, arguments, x64 = _run_launcher(
        tmp_path,
        data_path,
        "--steps",
        "17",
        "--rho-scale",
        "2.5",
        "--smoothing-exponent",
        "0.4",
        "--step-offset",
        "3",
        # Attempts to weaken the launch settings are superseded by the final
        # enforced arguments below.
        "--device",
        "cpu",
        "--precision",
        "32",
        "--execution-mode",
        "step",
    )

    assert completed.returncode == 0, completed.stderr
    assert arguments == [
        "-m",
        "paper.experiments.quantum_process_tomography_jax",
        "--h5",
        str(data_path),
        "--steps",
        "17",
        "--rho-scale",
        "2.5",
        "--smoothing-exponent",
        "0.4",
        "--step-offset",
        "3",
        "--device",
        "cpu",
        "--precision",
        "32",
        "--execution-mode",
        "step",
        "--device",
        "gpu",
        "--precision",
        "64",
        "--execution-mode",
        "scan",
    ]
    assert x64 == "1"


def test_launcher_preserves_explicit_x64_environment_setting(tmp_path):
    data_path = tmp_path / "data.h5"
    data_path.touch()

    completed, _, x64 = _run_launcher(tmp_path, data_path, x64="0")

    assert completed.returncode == 0, completed.stderr
    assert x64 == "0"


def test_launcher_requires_an_hdf5_path(tmp_path):
    completed = subprocess.run(
        [str(LAUNCHER)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 64
    assert "Usage:" in completed.stderr


def test_launcher_rejects_a_missing_data_file_before_starting_python(
    tmp_path,
):
    completed, arguments, x64 = _run_launcher(
        tmp_path, tmp_path / "missing.h5"
    )

    assert completed.returncode == 66
    assert "does not exist" in completed.stderr
    assert arguments is None
    assert x64 is None


def test_setup_auto_selects_cuda13_and_forwards_gpu_preflight(tmp_path):
    data_path = tmp_path / "QPT input with spaces.h5"
    completed, invocations = _run_setup(
        tmp_path,
        driver_version="580.82.07",
        preflight_arguments=("--h5", data_path),
    )

    assert completed.returncode == 0, completed.stderr
    assert "JAX wheel:  jax[cuda13]" in completed.stdout
    pip_install = _find_pip_install(invocations)
    assert pip_install is not None
    assert f"{REPOSITORY_ROOT}[qpt-jax]" in pip_install
    assert "jax[cuda13]" in pip_install
    preflight = _find_preflight(invocations)
    assert preflight is not None
    assert preflight[1:3] == ["--h5", str(data_path)]
    assert preflight[-2:] == ["--device", "gpu"]


def test_setup_auto_selects_cuda12_for_a_579_driver(tmp_path):
    completed, invocations = _run_setup(
        tmp_path, driver_version="579.99.01"
    )

    assert completed.returncode == 0, completed.stderr
    assert "JAX wheel:  jax[cuda12]" in completed.stdout
    pip_install = _find_pip_install(invocations)
    assert pip_install is not None
    assert "jax[cuda12]" in pip_install
    assert "jax[cuda13]" not in pip_install


def test_setup_rejects_a_pre_525_driver_before_pip(tmp_path):
    completed, invocations = _run_setup(
        tmp_path, driver_version="524.99.01"
    )

    assert completed.returncode != 0
    assert "too old" in completed.stderr
    assert _find_pip_install(invocations) is None
    assert _find_preflight(invocations) is None


def test_setup_rejects_explicit_cuda13_with_an_old_driver_before_pip(
    tmp_path,
):
    completed, invocations = _run_setup(
        tmp_path,
        driver_version="579.99.01",
        cuda_variant="cuda13",
    )

    assert completed.returncode != 0
    assert "cuda13 requires NVIDIA driver 580+" in completed.stderr
    assert _find_pip_install(invocations) is None
    assert _find_preflight(invocations) is None


def test_setup_rejects_python_older_than_3_12_before_pip(tmp_path):
    completed, invocations = _run_setup(
        tmp_path,
        python_version="3.11.9",
        python_supported=False,
    )

    assert completed.returncode != 0
    assert "require Python 3.12 or newer" in completed.stderr
    assert "Python 3.11.9" in completed.stderr
    assert _find_pip_install(invocations) is None
    assert _find_preflight(invocations) is None
