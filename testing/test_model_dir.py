"""
Unit tests for the ``model_dir`` YAML override.

``data_dir`` and ``out_dir`` let an input YAML read and write somewhere other than the
repository it is run from; ``model_dir`` does the same for models, so a repository that
keeps its own ``models/`` directory beside its own ``inputs/`` can name the models it
ships with instead of the ones in the default ``MODEL_DIR``.

These cover:

  * ``settings.resolve_model_dir`` (the fallback logic),
  * the ``model_dir`` field of every YAML-driven workflow script (``helpers.inputs``),
  * ``io.load_model_calculator`` searching the ``model_dir`` it is handed - including an
    explicit-but-wrong directory failing loudly instead of falling back to ``MODEL_DIR``,
  * the helpers forwarding ``model_dir`` on to the loader.

Only the tests that load a real MACE model need the ML stack; they skip themselves when
``models/GO-MACE-23.model`` or mace is absent. The forwarding test runs a short real MD
with the cheap ASE EMT calculator, so it needs neither.

Run from the repository root (the sample YAML inputs use repo-root-relative paths):

    pytest testing/test_model_dir.py
"""

import inspect
from pathlib import Path

import pytest

from react_ml_test_public.helpers import dynamics as rmlt_dynamics
from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers import model_eval as rmlt_model_eval
from react_ml_test_public.helpers import ts_opt as rmlt_ts_opt
from react_ml_test_public.helpers.settings import MODEL_DIR, MDSettings, resolve_model_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
GO_MACE_MODEL = REPO_ROOT / "models" / "GO-MACE-23.model"

# the YAML-driven scripts that build MLIP calculators: script -> (resolver, sample input YAML)
YAML_RESOLVERS = {
    "test_set_comparison": (
        rmlt_inputs.resolve_test_set_comparison_yaml,
        "inputs/test/CPD_MVK_DA_FT42_test.yaml",
    ),
    "reaction_barrier_comparison": (
        rmlt_inputs.resolve_reaction_barrier_comparison_yaml,
        "inputs/irc/CPD_MVK_DA_IRC_barrier.yaml",
    ),
    "pretrain_bias_analysis": (
        rmlt_inputs.resolve_pretrain_bias_analysis_yaml,
        "inputs/pretrain_bias/CPD_MVK_DA_pretrain_bias.yaml",
    ),
    "mlip_md": (rmlt_inputs.resolve_mlip_md_yaml, "inputs/md/CPD_MVK_DA_dynamics_vac.yaml"),
    "mlip_ts_opt": (rmlt_inputs.resolve_mlip_ts_opt_yaml, "inputs/ts/CPD_MVK_DA_ts_opt_vac.yaml"),
}


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    """Resolve the repo-root-relative paths used in the sample YAML inputs."""
    monkeypatch.chdir(REPO_ROOT)


class TestResolveModelDir:
    """resolve_model_dir() defaults to MODEL_DIR and otherwise keeps what it is given."""

    def test_none_falls_back_to_model_dir(self):
        assert resolve_model_dir(None) == MODEL_DIR

    def test_explicit_directory_is_kept(self):
        assert resolve_model_dir("somewhere/other/models") == "somewhere/other/models"

    def test_absolute_directory_is_kept(self):
        assert resolve_model_dir("/tmp/other_repo/models") == "/tmp/other_repo/models"

    def test_path_is_stringified(self):
        assert resolve_model_dir(Path("/tmp/other_repo/models")) == "/tmp/other_repo/models"


class TestYamlModelDir:
    """Every YAML-driven script reads a model_dir, defaulting to MODEL_DIR."""

    @pytest.mark.parametrize("script", sorted(YAML_RESOLVERS))
    def test_default_is_model_dir_when_field_is_absent(self, script):
        resolver, sample = YAML_RESOLVERS[script]
        assert "model_dir:" not in (REPO_ROOT / sample).read_text()

        assert resolver(sample)["model_dir"] == MODEL_DIR

    @pytest.mark.parametrize("script", sorted(YAML_RESOLVERS))
    def test_override_is_read_from_the_yaml(self, script, tmp_path):
        resolver, sample = YAML_RESOLVERS[script]
        override = str(tmp_path / "other_repo" / "models")

        # a field added at the top level, so the rest of the shipped input is untouched
        fpath = tmp_path / f"{script}.yaml"
        fpath.write_text(f"model_dir: {override}\n" + (REPO_ROOT / sample).read_text())

        assert resolver(str(fpath))["model_dir"] == override


@pytest.mark.skipif(not GO_MACE_MODEL.is_file(), reason="models/GO-MACE-23.model not present")
class TestLoadModelCalculatorModelDir:
    """load_model_calculator() searches the model_dir it is handed, not just MODEL_DIR."""

    def test_model_is_found_in_a_nested_override_directory(self, tmp_path):
        """The recursive search happens under the override, as it does under MODEL_DIR."""
        pytest.importorskip("mace")
        from react_ml_test_public.helpers.io import load_model_calculator

        nested = tmp_path / "models" / "fine_tuned"
        nested.mkdir(parents=True)
        (nested / "GO-MACE-23.model").symlink_to(GO_MACE_MODEL)

        calc = load_model_calculator("GO-MACE-23", device="cpu", dtype="float32",
                                     model_dir=str(tmp_path / "models"))

        assert calc is not None
        assert str(calc.device) == "cpu"

    def test_explicit_but_wrong_directory_fails_loudly(self, tmp_path):
        """An explicit model_dir must not silently fall back to MODEL_DIR.

        Silently loading a different repository's model would be worse than not loading
        one at all: it would produce results attributed to the wrong model.
        """
        pytest.importorskip("mace")
        from react_ml_test_public.helpers.io import load_model_calculator

        assert load_model_calculator("GO-MACE-23", device="cpu",
                                     model_dir=str(tmp_path / "empty")) is None

    def test_the_default_model_dir_still_loads(self):
        pytest.importorskip("mace")
        from react_ml_test_public.helpers.io import load_model_calculator

        assert load_model_calculator("GO-MACE-23", device="cpu", dtype="float32") is not None


class TestHelpersForwardModelDir:
    """model_dir reaches the loader through the helper that owns the run."""

    def test_get_trajectory_with_ase_forwards_model_dir(self, monkeypatch, tmp_path):
        """A short real MD with the cheap ASE EMT calculator, so this needs no model file."""
        from ase.build import molecule
        from ase.calculators.emt import EMT

        loaded_with = {}

        def fake_loader(model_name, **kwargs):
            loaded_with.update(kwargs)
            return EMT()

        monkeypatch.setattr(rmlt_dynamics, "load_model_calculator", fake_loader)

        trajectory = rmlt_dynamics.get_trajectory_with_ase(
            molecule("H2O"),
            "any-model-name",
            MDSettings(fs=5, temp=300, dt=0.5, interval=1, md_method="ase"),
            "model_dir_forwarding_test",
            out_dir=str(tmp_path),
            cuda_device="cpu",
            model_dir="/some/other/repo/models",
        )

        assert loaded_with["model_dir"] == "/some/other/repo/models"
        assert len(trajectory) > 0

    @pytest.mark.parametrize("func", [
        rmlt_ts_opt.mlip_dimer_method,
        rmlt_model_eval.get_model_predictions,
        rmlt_model_eval.get_test_and_pred_data,
        rmlt_model_eval.reaction_barrier_comparison,
    ])
    def test_helpers_that_load_models_accept_a_model_dir(self, func):
        """A dropped or renamed parameter would silently break the override."""
        assert "model_dir" in inspect.signature(func).parameters
