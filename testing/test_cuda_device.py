"""
Unit tests for GPU / CPU device selection across the workflow scripts.

These cover the behaviour of the ``cuda_device`` handling:

  * ``helpers.io.resolve_cuda_device`` / ``check_cuda_setup`` (the fallback logic),
  * ``helpers.io.load_model_calculator`` resolving the device it is handed,
  * the ``cuda_device`` field of every YAML-driven workflow script (``helpers.inputs``),
  * a smoke test that every module in ``react_ml_test_public.scripts`` still imports.

Everything here passes without a GPU: the CUDA introspection of ``torch`` is patched
where a particular device layout needs to be simulated, and the few tests that need a
real GPU, the MACE stack or the model files skip themselves when those are missing.

Run from the repository root (the YAML inputs use repo-root-relative paths):

    pytest testing/test_cuda_device.py
"""

import importlib
import os
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import pytest

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers import io as rmlt_io
from react_ml_test_public.helpers.io import (
    check_cuda_setup,
    load_model_calculator,
    resolve_cuda_device,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUTS_DIR = REPO_ROOT / "inputs"
GO_MACE_MODEL = REPO_ROOT / "models" / "GO-MACE-23.model"

# YAML-driven scripts that build MLIP calculators: script -> (resolver, sample input YAML)
YAML_RESOLVERS = {
    "test_set_comparison": (
        rmlt_inputs.resolve_test_set_comparison_yaml,
        "test/CPD_MVK_DA_FT42_test.yaml",
    ),
    "reaction_barrier_comparison": (
        rmlt_inputs.resolve_reaction_barrier_comparison_yaml,
        "irc/CPD_MVK_DA_IRC_barrier.yaml",
    ),
    "pretrain_bias_analysis": (
        rmlt_inputs.resolve_pretrain_bias_analysis_yaml,
        "pretrain_bias/CPD_MVK_DA_pretrain_bias.yaml",
    ),
    "mlip_md": (rmlt_inputs.resolve_mlip_md_yaml, "md/CPD_MVK_DA_dynamics_vac.yaml"),
    "mlip_ts_opt": (rmlt_inputs.resolve_mlip_ts_opt_yaml, "ts/CPD_MVK_DA_ts_opt_vac.yaml"),
}

# input directory -> resolver, for the shipped YAML files of each workflow
INPUT_DIR_RESOLVERS = {
    "test": rmlt_inputs.resolve_test_set_comparison_yaml,
    "irc": rmlt_inputs.resolve_reaction_barrier_comparison_yaml,
    "pretrain_bias": rmlt_inputs.resolve_pretrain_bias_analysis_yaml,
    "md": rmlt_inputs.resolve_mlip_md_yaml,
    "ts": rmlt_inputs.resolve_mlip_ts_opt_yaml,
}


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    """Resolve the repo-root-relative paths used in the YAML inputs."""
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture
def fake_cuda(monkeypatch):
    """Patch torch's CUDA introspection so tests need no particular GPU layout.

    Returns a factory: ``fake_cuda(available=True, count=2)`` simulates a machine with
    two GPUs, ``fake_cuda(available=False, count=0)`` a CPU-only machine. The factory
    returns the list of device indices that were set as the current device.
    """
    torch = pytest.importorskip("torch")

    def _fake(available: bool = True, count: int = 1):
        set_device_calls = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
        monkeypatch.setattr(torch.cuda, "set_device", set_device_calls.append)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"Fake GPU {index}")
        return set_device_calls

    return _fake


class TestResolveCudaDevice:
    """resolve_cuda_device() must always hand back a device that exists."""

    def test_cpu_request_is_kept(self):
        for request in ("cpu", "CPU", "Cpu", " cpu "):
            assert resolve_cuda_device(request) == "cpu"

    def test_blank_and_none_requests_mean_the_default_gpu(self, fake_cuda):
        fake_cuda(available=True, count=1)
        default = resolve_cuda_device("cuda:0")
        for request in (None, "", "   ", "none", "None"):
            assert resolve_cuda_device(request) == default

    def test_no_cuda_falls_back_to_cpu(self, fake_cuda):
        fake_cuda(available=False, count=0)
        for request in ("cuda", "cuda:0", "cuda:3"):
            assert resolve_cuda_device(request) == "cpu"

    def test_missing_gpu_index_falls_back_to_cpu(self, fake_cuda):
        fake_cuda(available=True, count=1)  # single-GPU machine
        assert resolve_cuda_device("cuda:0") == "cuda:0"
        assert resolve_cuda_device("cuda:1") == "cpu"
        assert resolve_cuda_device("cuda:99") == "cpu"

    def test_available_gpu_resolves_and_becomes_current_device(self, fake_cuda):
        set_device_calls = fake_cuda(available=True, count=4)
        assert resolve_cuda_device("cuda:2") == "cuda:2"
        assert set_device_calls == [2]  # the resolved index is made current
        assert resolve_cuda_device("cuda") == "cuda:0"  # bare 'cuda' means device 0
        assert set_device_calls == [2, 0]

    def test_unrecognised_requests_fall_back_to_cpu(self, fake_cuda):
        fake_cuda(available=True, count=1)
        for request in ("mps", "banana", "cuda:x", "cuda:", "gpu"):
            assert resolve_cuda_device(request) == "cpu"

    def test_unpatched_default_request_is_usable(self):
        """On this machine, 'cuda:0' must resolve to 'cuda:0' or 'cpu' - never fail."""
        assert resolve_cuda_device("cuda:0") in ("cuda:0", "cpu")

    def test_unpatched_second_gpu_request_matches_this_machine(self):
        torch = pytest.importorskip("torch")
        has_two_gpus = torch.cuda.is_available() and torch.cuda.device_count() > 1
        assert resolve_cuda_device("cuda:1") == ("cuda:1" if has_two_gpus else "cpu")


class TestCheckCudaSetup:
    """check_cuda_setup() logs and returns the device that will actually be used."""

    def test_returns_resolved_device_not_a_bool(self, monkeypatch):
        monkeypatch.setattr(rmlt_io, "resolve_cuda_device", lambda device="cuda:0": "cpu")
        assert check_cuda_setup("cuda:1") == "cpu"

    def test_moves_test_tensor_onto_resolved_device(self):
        # the tensor move happens inside, so a broken device here would raise
        assert check_cuda_setup("cuda:0") in ("cuda:0", "cpu")


class TestLoadModelCalculatorDevice:
    """load_model_calculator() resolves its device argument itself."""

    def test_missing_gpu_with_missing_model_returns_none(self, fake_cuda):
        fake_cuda(available=False, count=0)
        assert load_model_calculator("no-such-model", device="cuda:0") is None

    def test_out_of_range_device_with_missing_model_returns_none(self, fake_cuda):
        fake_cuda(available=True, count=1)
        assert load_model_calculator("no-such-model", device="cuda:7") is None

    @pytest.mark.skipif(not GO_MACE_MODEL.is_file(), reason="models/GO-MACE-23.model not present")
    def test_cpu_inference_for_cpu_and_missing_gpu_requests(self):
        """An explicit 'cpu' and a nonexistent 'cuda:99' both run on CPU, identically."""
        pytest.importorskip("mace")
        molecule = pytest.importorskip("ase.build").molecule
        atoms = molecule("CH4")

        energies = []
        for device in ("cpu", "cuda:99"):
            calc = load_model_calculator("GO-MACE-23", device=device, dtype="float32")
            if calc is None:
                pytest.skip("GO-MACE-23 could not be loaded in this environment")
            atoms.calc = calc
            energies.append(atoms.get_potential_energy())
            assert str(calc.device) == "cpu"

        assert energies[0] == pytest.approx(energies[1])


class TestJitLoadOnDevice:
    """MACE/e3nn models must load on CPU even where CUDA is not visible.

    e3nn restores its compiled TorchScript submodules with a bare
    ``torch.jit.load(buffer)``, which reproduces the device the model was saved on and so
    raises "No CUDA GPUs are available" on a machine without that GPU - so
    ``load_model_calculator`` injects the resolved device into those calls.
    """

    def test_bare_jit_load_calls_get_the_device(self, monkeypatch):
        torch = pytest.importorskip("torch")
        recorded = []

        def fake_jit_load(f, map_location=None, *args, **kwargs):
            recorded.append((f, map_location))
            return "sentinel"

        monkeypatch.setattr(torch.jit, "load", fake_jit_load)
        with rmlt_io._jit_load_on_device("cpu"):
            assert torch.jit.load("buffer") == "sentinel"

        assert recorded == [("buffer", "cpu")]

    def test_explicit_map_location_is_not_overridden(self, monkeypatch):
        torch = pytest.importorskip("torch")
        recorded = []

        def fake_jit_load(f, map_location=None, *args, **kwargs):
            recorded.append(map_location)
            return None

        monkeypatch.setattr(torch.jit, "load", fake_jit_load)
        with rmlt_io._jit_load_on_device("cpu"):
            torch.jit.load("buffer", map_location="cuda:0")

        assert recorded == ["cuda:0"]

    def test_original_is_restored(self):
        torch = pytest.importorskip("torch")
        original = torch.jit.load
        with rmlt_io._jit_load_on_device("cpu"):
            assert torch.jit.load is not original
        assert torch.jit.load is original

    def test_original_is_restored_after_an_exception(self):
        torch = pytest.importorskip("torch")
        original = torch.jit.load
        with pytest.raises(RuntimeError):
            with rmlt_io._jit_load_on_device("cpu"):
                raise RuntimeError("boom")
        assert torch.jit.load is original

    @pytest.mark.skipif(not GO_MACE_MODEL.is_file(), reason="models/GO-MACE-23.model not present")
    def test_mace_loading_injects_the_resolved_device(self, monkeypatch):
        pytest.importorskip("mace")
        torch = pytest.importorskip("torch")
        original_jit_load = torch.jit.load
        recorded = []

        def spy(f, map_location=None, *args, **kwargs):
            recorded.append(map_location)
            return original_jit_load(f, map_location=map_location, *args, **kwargs)

        monkeypatch.setattr(torch.jit, "load", spy)
        calc = load_model_calculator("GO-MACE-23", device="cpu", dtype="float32")

        assert calc is not None
        # the bundled MACE models do serialise e3nn codegen submodules, and every such
        # load must have been given the resolved device
        assert recorded, "no torch.jit.load call was made while loading GO-MACE-23"
        assert set(recorded) == {"cpu"}

    @pytest.mark.skipif(not GO_MACE_MODEL.is_file(), reason="models/GO-MACE-23.model not present")
    def test_mace_cpu_inference_where_cuda_is_invisible(self):
        """Regression test: this failed with 'No CUDA GPUs are available' before the fix.

        Needs its own process because CUDA_VISIBLE_DEVICES is only read when torch is
        first imported.
        """
        pytest.importorskip("mace")
        code = (
            "import torch;"
            "from ase.build import molecule;"
            "from react_ml_test_public.helpers.io import load_model_calculator;"
            "assert not torch.cuda.is_available(), 'CUDA unexpectedly visible';"
            "calc = load_model_calculator('GO-MACE-23', device='cpu', dtype='float32');"
            "assert calc is not None, 'GO-MACE-23 failed to load on CPU';"
            "atoms = molecule('CH4'); atoms.calc = calc;"
            "print(calc.device, atoms.get_potential_energy())"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, result.stderr[-2000:]

        device, energy = result.stdout.split()[-2:]
        assert device == "cpu"
        # loose sanity band around the CH4 energy of this model (-221.27 eV)
        assert -222.0 < float(energy) < -220.0


class TestYamlCudaDevice:
    """Every YAML-driven script reads a cuda_device field, defaulting to 'cuda:0'."""

    @pytest.mark.parametrize("script", sorted(YAML_RESOLVERS))
    def test_default_is_cuda_0_when_field_is_absent(self, script, tmp_path):
        resolver, sample = YAML_RESOLVERS[script]
        text = (INPUTS_DIR / sample).read_text()
        without_field = re.sub(r"^cuda_device:.*\n", "", text, flags=re.M)
        assert "cuda_device:" not in without_field

        fpath = tmp_path / "no_cuda_device.yaml"
        fpath.write_text(without_field)
        assert resolver(str(fpath))["cuda_device"] == "cuda:0"

    @pytest.mark.parametrize("script", sorted(YAML_RESOLVERS))
    def test_default_is_cuda_0_in_shipped_yaml(self, script):
        resolver, sample = YAML_RESOLVERS[script]
        assert resolver(str(INPUTS_DIR / sample))["cuda_device"] == "cuda:0"

    @pytest.mark.parametrize("device_request", ["cpu", "cuda", "cuda:1"])
    @pytest.mark.parametrize("script", sorted(YAML_RESOLVERS))
    def test_requested_device_is_read_from_yaml(self, script, device_request, tmp_path):
        """'cpu' (and any other device) can be specified in the YAML."""
        resolver, sample = YAML_RESOLVERS[script]
        text = (INPUTS_DIR / sample).read_text()
        assert re.search(r"^cuda_device:", text, flags=re.M)

        fpath = tmp_path / "cuda_device.yaml"
        fpath.write_text(re.sub(r"^cuda_device:.*$", f'cuda_device: "{device_request}"', text, flags=re.M))
        assert resolver(str(fpath))["cuda_device"] == device_request

    @pytest.mark.parametrize("directory", sorted(INPUT_DIR_RESOLVERS))
    def test_all_shipped_yamls_resolve_with_a_device(self, directory):
        resolver = INPUT_DIR_RESOLVERS[directory]
        fpaths = sorted((INPUTS_DIR / directory).glob("*.yaml"))
        assert fpaths, f"no input YAML files found in inputs/{directory}"

        for fpath in fpaths:
            assert resolver(str(fpath))["cuda_device"] == "cuda:0"

    def test_trajectory_analysis_takes_no_device(self):
        """run_trajectory_analysis.py does no MLIP inference, so it has no device field."""
        fpaths = sorted((INPUTS_DIR / "trj_analysis").glob("*.yaml"))
        assert fpaths

        for fpath in fpaths:
            assert "cuda_device:" not in fpath.read_text()
            assert "cuda_device" not in rmlt_inputs.resolve_trajectory_analysis_yaml(str(fpath))


# pre-existing import failures in this repository (unrelated to device handling)
KNOWN_BROKEN_IMPORTS = {
    "download_datasets": "optional 'load_atoms' package is not installed",
    "run_mace_mhft": "react_ml_test_public.helpers.training_justine is missing from this repo",
    "save_orca_out_to_xyz": "this mlptrain version has no mlptrain.utils.orca_output_to_npz",
}

_scripts_pkg = importlib.import_module("react_ml_test_public.scripts")
SCRIPT_MODULES = sorted(module.name for module in pkgutil.iter_modules(_scripts_pkg.__path__))


class TestScriptModulesImport:
    """Import smoke test: every workflow module must at least import cleanly."""

    @pytest.mark.parametrize("module_name", SCRIPT_MODULES)
    def test_script_module_imports(self, module_name, request):
        if module_name in KNOWN_BROKEN_IMPORTS:
            # not strict: if the missing dependency is added the test just reports XPASS
            request.node.add_marker(
                pytest.mark.xfail(reason=KNOWN_BROKEN_IMPORTS[module_name], strict=False)
            )
        # importing must not touch a GPU, raise NameError on module constants, etc.
        importlib.import_module(f"react_ml_test_public.scripts.{module_name}")
