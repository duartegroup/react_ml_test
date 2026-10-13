"""
Unit tests for loading FairChem models through ``load_model_calculator``.

FairChem cannot be installed alongside MACE, so it lives in its own ``fair_chem`` env and the
loader imports it lazily. These cover:

  * fairchem models being recognised by name (``settings.FAIRCHEM_MODELS``), with the MACE /
    ANI names left alone by that,
  * the calculator FairChem is asked for: hub model name, device, task and inference settings,
  * the failure when fairchem is not installed - i.e. a fairchem model run from a MACE env -
    pointing at the ``fair_chem`` env rather than only reporting a missing module,
  * the MD path forwarding the FairChem settings down to the loader.

``fairchem.core`` is faked in ``sys.modules`` rather than imported, so these run in any env,
including the MACE ones where fairchem is deliberately absent. Only the test that compares the
model list against fairchem's own registry needs fairchem itself, and skips without it.

Run from the repository root:

    pytest testing/test_fairchem_loader.py
"""

import logging
import sys
import types

import pytest

from react_ml_test_public.helpers import dynamics as rmlt_dynamics
from react_ml_test_public.helpers import settings
from react_ml_test_public.helpers.io import load_model_calculator
from react_ml_test_public.helpers.settings import MDSettings

# a fairchem model name, straight out of the registry
FAIRCHEM_MODEL = "uma-s-1p2p1"

# a short run: 5 fs at a 0.5 fs time step, i.e. 10 steps
MD_CONFIG = MDSettings(fs=5, temp=300, dt=0.5, interval=1, md_method="ase")


@pytest.fixture
def fairchem(monkeypatch):
    """A fake ``fairchem.core`` that records how it is asked for a calculator."""

    class FakeCalculator:
        def __init__(self, predict_unit, task_name=None, seed=None):
            self.predict_unit = predict_unit
            self.task_name = task_name

    fake_mlip = types.ModuleType("fairchem.core.pretrained_mlip")

    def fake_get_predict_unit(model_name, **kwargs):
        fake_mlip.called_with = {"model_name": model_name, **kwargs}
        return "predict-unit"

    fake_mlip.get_predict_unit = fake_get_predict_unit
    fake_mlip.called_with = None

    fake_core = types.ModuleType("fairchem.core")
    fake_core.pretrained_mlip = fake_mlip
    fake_core.FAIRChemCalculator = FakeCalculator

    monkeypatch.setitem(sys.modules, "fairchem.core", fake_core)
    return fake_mlip


class TestFairchemModelsAreRecognised:

    @pytest.mark.parametrize("model_name", ["uma-s-1p2p1", "uma-m-1p1",
                                            "esen-md-direct-all-omol",
                                            "allscaip-md-direct-all-omol"])
    def test_registry_names_are_loaded_with_fairchem(self, model_name, fairchem):
        load_model_calculator(model_name, device="cpu")

        assert fairchem.called_with["model_name"] == model_name

    @pytest.mark.parametrize("model_name", ["GO-MACE-23", "MACE-OFF23_medium", "ani-1xnr"])
    def test_the_other_models_are_not_claimed_by_fairchem(self, model_name, fairchem):
        """Recognition is by name, so it must not swallow the MACE / ANI models."""
        load_model_calculator(model_name, device="cpu")

        assert fairchem.called_with is None

    def test_the_registry_matches_what_fairchem_serves(self):
        """A name listed here that fairchem does not serve would only fail at load time."""
        fairchem_core = pytest.importorskip("fairchem.core")

        served = set(fairchem_core.pretrained_mlip.available_models)

        assert not set(settings.FAIRCHEM_MODELS) - served


class TestTheCalculatorFairchemIsAskedFor:

    def test_a_fairchem_calculator_is_built(self, fairchem):
        calc = load_model_calculator(FAIRCHEM_MODEL, device="cpu")

        assert calc.predict_unit == "predict-unit"

    def test_the_device_index_is_dropped(self, fairchem):
        """FairChem takes 'cuda' / 'cpu' / None, not a torch 'cuda:N'."""
        load_model_calculator(FAIRCHEM_MODEL, device="cuda:0")

        # the device is resolved first, so this is 'cuda' on a machine with a usable GPU and
        # 'cpu' on one without - either way with no index on it
        assert fairchem.called_with["device"] in ("cuda", "cpu")

    def test_the_task_and_inference_settings_are_forwarded(self, fairchem):
        calc = load_model_calculator(FAIRCHEM_MODEL, device="cpu", fair_chem_task="omol",
                                     fair_chem_setting="turbo")

        assert calc.task_name == "omol"
        assert fairchem.called_with["inference_settings"] == "turbo"

    def test_the_inference_settings_default_to_fairchems_own(self, fairchem):
        load_model_calculator(FAIRCHEM_MODEL, device="cpu")

        assert fairchem.called_with["inference_settings"] == "default"


class TestWithoutFairchemInstalled:
    """The likely mistake: a fairchem model run from one of the MACE envs."""

    @pytest.fixture(autouse=True)
    def _no_fairchem(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "fairchem.core", None)   # importing it now fails

    def test_the_model_is_not_loaded(self):
        assert load_model_calculator(FAIRCHEM_MODEL, device="cpu") is None

    def test_the_warning_points_at_the_fair_chem_env(self, caplog):
        with caplog.at_level(logging.WARNING):
            load_model_calculator(FAIRCHEM_MODEL, device="cpu")

        warnings = [record.message for record in caplog.records]
        assert any("is a FairChem model" in message for message in warnings)
        assert any("fair_chem env" in message for message in warnings)


class TestTheMdPathForwardsTheFairchemSettings:
    """run_mlip_md's path down to the loader, with a cheap stand-in for the model."""

    @pytest.fixture
    def reported_to_loader(self, monkeypatch):
        from ase.calculators.emt import EMT

        recorded = {}

        def fake_loader(model_name, **kwargs):
            recorded.update(kwargs, model_name=model_name)
            return EMT()

        monkeypatch.setattr(rmlt_dynamics, "load_model_calculator", fake_loader)
        return recorded

    def test_get_trajectory_with_ase_forwards_them(self, tmp_path, reported_to_loader):
        from ase.build import molecule

        rmlt_dynamics.get_trajectory_with_ase(molecule("H2O"), FAIRCHEM_MODEL, MD_CONFIG,
                                              "fairchem_forwarding_test",
                                              out_dir=str(tmp_path), cuda_device="cpu",
                                              fair_chem_task="omol",
                                              fair_chem_setting="turbo")

        assert reported_to_loader["fair_chem_task"] == "omol"
        assert reported_to_loader["fair_chem_setting"] == "turbo"

    def test_run_md_eval_forwards_them(self, tmp_path, reported_to_loader):
        from ase.build import molecule

        from react_ml_test_public.helpers.dynamics import run_md_eval

        traj_dir = tmp_path / "trj"
        traj_dir.mkdir()

        run_md_eval([molecule("H2O")], MD_CONFIG, md_runs_per_config=1,
                    model_name=FAIRCHEM_MODEL, formatted_model_name="model",
                    system_name="system", traj_dir=str(traj_dir), cuda_device="cpu",
                    fair_chem_task="omol", fair_chem_setting="turbo")

        assert reported_to_loader["fair_chem_task"] == "omol"
        assert reported_to_loader["fair_chem_setting"] == "turbo"
        assert reported_to_loader["model_name"] == FAIRCHEM_MODEL


class TestTheYamlFieldsAreResolved:
    """The fair_chem_task / fair_chem_setting fields of a run_mlip_md input YAML."""

    @staticmethod
    def _resolve(tmp_path, body):
        from react_ml_test_public.helpers import inputs as rmlt_inputs

        yaml_fpath = tmp_path / "dynamics.yaml"
        yaml_fpath.write_text("""
system_name: test_system
model_names:
  - uma-s-1p2p1
init_configs_fpath: null
""" + body)
        return rmlt_inputs.resolve_mlip_md_yaml(str(yaml_fpath))

    def test_they_default_to_none(self, tmp_path):
        resolved = self._resolve(tmp_path, "")

        assert resolved["fair_chem_task"] is None
        assert resolved["fair_chem_setting"] is None

    def test_they_are_read_as_strings(self, tmp_path):
        resolved = self._resolve(tmp_path, 'fair_chem_task: "omol"\nfair_chem_setting: "turbo"\n')

        assert resolved["fair_chem_task"] == "omol"
        assert resolved["fair_chem_setting"] == "turbo"
