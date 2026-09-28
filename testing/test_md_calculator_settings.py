"""
Unit tests for the calculator settings run_mlip_md.py passes down to its models.

An input YAML's ``enable_cueq`` has to reach every calculator an MD run builds - both
through the dynamics helper that owns the run and through ``run_md_eval``, which forwards
it on. It was silently dropped on the way, so these tests pin the whole chain: a YAML that
asks for the cuEquivariance speed-up used to run without it.

These use a stub loader in place of a MACE model, so nothing here needs a GPU, the
cuEquivariance stack or any model files.

Run from the repository root:

    pytest testing/test_md_calculator_settings.py
"""

import ast
import inspect
import logging
from pathlib import Path

import pytest

from react_ml_test_public.helpers import dynamics as rmlt_dynamics
from react_ml_test_public.helpers.io import load_model_calculator
from react_ml_test_public.helpers.settings import MDSettings

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "react_ml_test_public"

# the helper every MD run goes through, and the setting that was dropped on the way to it
HELPER_NAME = "get_trajectory_with_ase"
FORWARDED_SETTINGS = ("enable_cueq", "model_dir")

# a short run: 5 steps of 0.5 fs
MD_CONFIG = MDSettings(fs=5, temp=300, dt=0.5, interval=1, md_method="ase")


@pytest.fixture
def loader_kwargs(monkeypatch):
    """A stub model loader that records the settings it was handed."""
    from ase.calculators.emt import EMT

    recorded = {}

    def fake_loader(model_name, **kwargs):
        recorded.update(kwargs)
        return EMT()

    monkeypatch.setattr(rmlt_dynamics, "load_model_calculator", fake_loader)
    return recorded


class TestGetTrajectoryWithAseForwardsEnableCueq:

    @pytest.mark.parametrize("enable_cueq", [True, False])
    def test_enable_cueq_reaches_the_loader(self, tmp_path, loader_kwargs, enable_cueq):
        from ase.build import molecule

        rmlt_dynamics.get_trajectory_with_ase(molecule("H2O"),
                                              "any-model-name",
                                              MD_CONFIG,
                                              "enable_cueq_forwarding_test",
                                              out_dir=str(tmp_path),
                                              cuda_device="cpu",
                                              enable_cueq=enable_cueq)

        assert loader_kwargs["enable_cueq"] is enable_cueq

    def test_it_defaults_to_off(self, tmp_path, loader_kwargs):
        from ase.build import molecule

        rmlt_dynamics.get_trajectory_with_ase(molecule("H2O"), "any-model-name",
                                              MD_CONFIG, "enable_cueq_default_test",
                                              out_dir=str(tmp_path), cuda_device="cpu")

        assert loader_kwargs["enable_cueq"] is False


class TestRunMdEvalForwardsEnableCueq:

    @pytest.mark.parametrize("enable_cueq", [True, False])
    def test_enable_cueq_reaches_the_loader(self, tmp_path, loader_kwargs, enable_cueq):
        """The YAML's enable_cueq gets past run_md_eval to the calculator it builds."""
        from ase.build import molecule

        from react_ml_test_public.helpers.dynamics import run_md_eval

        traj_dir = tmp_path / "trj"
        traj_dir.mkdir()

        run_md_eval([molecule("H2O")],
                    MD_CONFIG,
                    md_runs_per_config=1,
                    model_name="any-model-name",
                    formatted_model_name="model",
                    system_name="system",
                    traj_dir=str(traj_dir),
                    cuda_device="cpu",
                    enable_cueq=enable_cueq)

        assert loader_kwargs["enable_cueq"] is enable_cueq


class TestEveryCallSiteForwardsTheSettings:
    """The scripts' own calls to the helper must pass these on.

    A script that forgets one drops the setting silently - the YAML asks for it, the run
    ignores it, and nothing fails - which is how ``enable_cueq`` was lost here. The call
    sites live in ``__main__`` blocks, so they cannot be run in a test; this reads them.
    """

    @staticmethod
    def _keyword_arguments_of_helper_calls(source_fpath):
        """{call node: [keyword names]} for every call to the helper in a source file."""
        tree = ast.parse(source_fpath.read_text(), filename=str(source_fpath))

        calls = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == HELPER_NAME:
                calls[node] = [keyword.arg for keyword in node.keywords]
        return calls

    def test_helper_calls_pass_every_setting_explicitly(self):
        call_sites = {}

        for source_fpath in sorted(PACKAGE_ROOT.rglob("*.py")):
            calls = self._keyword_arguments_of_helper_calls(source_fpath)
            for node, keywords in calls.items():
                call_sites[(source_fpath, node.lineno)] = keywords

        # a renamed helper would otherwise make this pass by finding nothing to check
        assert call_sites, f"no calls to {HELPER_NAME}() found under {PACKAGE_ROOT}"

        for (source_fpath, lineno), keywords in call_sites.items():
            missing = [setting for setting in FORWARDED_SETTINGS if setting not in keywords]
            assert not missing, (f"{source_fpath.relative_to(REPO_ROOT)}:{lineno} calls "
                                 f"{HELPER_NAME}() without {missing}")

    def test_the_helper_accepts_them(self):
        parameters = inspect.signature(rmlt_dynamics.get_trajectory_with_ase).parameters
        assert all(setting in parameters for setting in FORWARDED_SETTINGS)


class TestCueqFailuresAreExplained:
    """Asking for the cuEquivariance acceleration and not getting it is a failure.

    It is deliberately not a best-effort setting - a run that silently continued
    unaccelerated would compare timings from two different calculators. The failure is
    therefore kept, but the warning has to say what cueq needs, since the error raised from
    inside cuequivariance (a version mismatch, in the ``mf_mat_react`` env) does not.
    """

    MODEL_NAME = "MACE-OFF23_medium"

    # what cuequivariance 0.1.0 raises for the mace-torch 0.3.16 that wants it
    CUEQ_ERROR = TypeError("Linear.__init__() got an unexpected keyword argument 'method'")

    @staticmethod
    def _stub_mace_calculator(monkeypatch, cueq_error=None, other_error=None):
        """Stub MACE's calculator, recording whether each build asked for cueq.

        ``cueq_error`` is raised by a build that asks for cueq (the acceleration being
        unusable); ``other_error`` by every build, i.e. a failure that is not cueq's fault.
        """
        pytest.importorskip("mace")
        import mace.calculators

        built = []

        class StubCalculator:
            def __init__(self, enable_cueq=False, **kwargs):
                built.append(enable_cueq)
                if enable_cueq and cueq_error is not None:
                    raise cueq_error
                if other_error is not None:
                    raise other_error
                self.enable_cueq = enable_cueq

        monkeypatch.setattr(mace.calculators, "MACECalculator", StubCalculator)
        return built

    def test_a_cueq_failure_does_not_load_the_model(self, monkeypatch):
        """No silent demotion to the standard calculator: the run fails, as asked."""
        built = self._stub_mace_calculator(monkeypatch, cueq_error=self.CUEQ_ERROR)

        assert load_model_calculator(self.MODEL_NAME, device="cpu", enable_cueq=True) is None
        assert built == [True]   # no second attempt without cueq

    def test_the_warning_names_cueq_and_the_underlying_error(self, monkeypatch, caplog):
        self._stub_mace_calculator(monkeypatch, cueq_error=self.CUEQ_ERROR)

        with caplog.at_level(logging.WARNING):
            load_model_calculator(self.MODEL_NAME, device="cpu", enable_cueq=True)

        warnings = [record.message for record in caplog.records]
        assert any("could not be loaded with cuEquivariance acceleration" in message
                   for message in warnings)
        assert any(str(self.CUEQ_ERROR) in message for message in warnings)

    def test_the_warning_says_what_cueq_needs(self, monkeypatch, caplog):
        """The version skew that causes this is invisible in the error it raises."""
        self._stub_mace_calculator(monkeypatch, cueq_error=self.CUEQ_ERROR)

        with caplog.at_level(logging.WARNING):
            load_model_calculator(self.MODEL_NAME, device="cpu", enable_cueq=True)

        cueq_warning = [record.message for record in caplog.records
                        if "cuEquivariance" in record.message]
        assert cueq_warning, "no cueq warning was logged"

        message = cueq_warning[0]
        assert "cueq extras" in message          # what to install
        assert "CUDA GPU" in message             # why a cpu-only machine can never work
        assert "enable_cueq: false" in message   # how to opt out

    def test_it_names_the_versions_on_both_sides(self, monkeypatch, caplog):
        """mace-torch's requirement and the installed package, when both can be read."""
        pytest.importorskip("mace")
        self._stub_mace_calculator(monkeypatch, cueq_error=self.CUEQ_ERROR)

        with caplog.at_level(logging.WARNING):
            load_model_calculator(self.MODEL_NAME, device="cpu", enable_cueq=True)

        message = [record.message for record in caplog.records
                   if "cuEquivariance" in record.message][0]
        assert "mace-torch asks for cuequivariance-torch" in message
        assert "installed cuequivariance-torch is" in message

    # the cluster env of 2026-09-24: core 8 minors behind the torch layer that calls it
    MIXED_SET = {"cuequivariance": "0.1.0",
                 "cuequivariance-torch": "0.12.0",
                 "cuequivariance-ops-torch-cu12": "0.1.0"}

    def _cueq_warning(self, monkeypatch, caplog, installed):
        """The cueq warning, with the reported package set stubbed."""
        from react_ml_test_public.helpers import io as rmlt_io

        pytest.importorskip("mace")
        monkeypatch.setattr(rmlt_io, "_cueq_versions", lambda: installed)
        self._stub_mace_calculator(monkeypatch, cueq_error=self.CUEQ_ERROR)

        with caplog.at_level(logging.WARNING):
            load_model_calculator(self.MODEL_NAME, device="cpu", enable_cueq=True)

        return [record.message for record in caplog.records
                if "cuEquivariance" in record.message][0]

    def test_a_mixed_cuequivariance_set_is_called_out(self, monkeypatch, caplog):
        """A set at mixed versions fails as an AttributeError naming a class the core
        never had, so the hint has to say the versions are the problem."""
        message = self._cueq_warning(monkeypatch, caplog, self.MIXED_SET)

        assert "must all be one version" in message
        for name, dist_version in self.MIXED_SET.items():
            assert f"{name} {dist_version}" in message

    def test_a_matched_set_is_not_called_mixed(self, monkeypatch, caplog):
        """An installed set that is in step gets the generic advice, not a false alarm."""
        matched = {name: "0.12.0" for name in self.MIXED_SET}

        message = self._cueq_warning(monkeypatch, caplog, matched)

        assert "must all be one version" not in message
        assert "cueq extras" in message

    def test_cueq_is_used_when_it_loads(self, monkeypatch, caplog):
        built = self._stub_mace_calculator(monkeypatch)

        with caplog.at_level(logging.WARNING):
            calc = load_model_calculator(self.MODEL_NAME, device="cpu", enable_cueq=True)

        assert calc.enable_cueq is True
        assert built == [True]
        assert not [record for record in caplog.records if "cuEquivariance" in record.message]

    def test_a_failure_without_cueq_asked_for_is_reported_as_before(self, monkeypatch, caplog):
        """A failure that is not cueq's fault keeps the original message."""
        built = self._stub_mace_calculator(monkeypatch, other_error=ValueError("corrupt model"))

        with caplog.at_level(logging.WARNING):
            calc = load_model_calculator(self.MODEL_NAME, device="cpu", enable_cueq=False)

        assert calc is None
        assert built == [False]   # no cueq attempt, so no retry either
        warnings = [record.message for record in caplog.records]
        assert any("unable to be loaded" in message for message in warnings)
        assert not any("cuEquivariance" in message for message in warnings)
