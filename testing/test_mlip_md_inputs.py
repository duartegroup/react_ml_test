"""
Unit tests for the run_mlip_md.py input YAML resolution.

These cover the ``data_dir`` / ``out_dir`` fields and the optional ``init_configs_fpath``
starting-config file: a relative ``init_configs_fpath`` is taken as relative to ``data_dir``,
trajectories are written under ``<out_dir>/trj`` and the MD benchmark CSVs under
``<out_dir>/csv``.

Nothing here needs a GPU, the MACE stack or any model files.

Run from the repository root (the YAML inputs use repo-root-relative paths):

    pytest testing/test_mlip_md_inputs.py
"""

from pathlib import Path

import pytest

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.settings import (
    CSV_DIR,
    DATA_DIR,
    DEFAULT_CHECKPOINT_EVERY_PS,
    OUT_DIR,
    TRAJ_DIR,
    resolve_csv_dir,
    resolve_traj_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# one H2 config, as a stand-in for a solvated starting config
CONFIG = """2
Properties=species:S:1:pos:R:3
H 0.0 0.0 0.0
H 0.0 0.0 0.7
"""

# the minimum an MD input YAML needs: a system name and at least one model
BASE_BODY = """
system_name: test_system
model_names:
  - MACE-OFF23_medium
"""


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    """Resolve the repo-root-relative paths used in the YAML inputs."""
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture
def config_fpath(tmp_path):
    """A two-config .xyz file to point init_configs_fpath at."""
    fpath = tmp_path / "config.xyz"
    fpath.write_text(CONFIG + CONFIG)
    return fpath


def _write_yaml(tmp_path, body):
    fpath = tmp_path / "dynamics.yaml"
    fpath.write_text(body)
    return str(fpath)


class TestResolveTrajDir:

    def test_defaults_come_from_settings(self):
        assert resolve_traj_dir() == TRAJ_DIR

    def test_override(self):
        assert resolve_traj_dir("somewhere/else") == "somewhere/else/trj"


class TestResolveCsvDir:

    def test_defaults_come_from_settings(self):
        assert resolve_csv_dir() == CSV_DIR

    def test_override(self):
        assert resolve_csv_dir("somewhere/else") == "somewhere/else/csv"


class TestMlipMdYaml:

    def test_dirs_default_to_settings_and_no_init_configs(self, tmp_path):
        yaml_fpath = _write_yaml(tmp_path, BASE_BODY)
        resolved = rmlt_inputs.resolve_mlip_md_yaml(yaml_fpath)

        assert resolved["data_dir"] == DATA_DIR
        assert resolved["out_dir"] == OUT_DIR
        assert resolved["traj_dir"] == TRAJ_DIR
        assert resolved["csv_dir"] == CSV_DIR
        # unset: the script falls back to each model's optimised TS in out/opt
        assert resolved["init_configs_fpath"] is None

    def test_data_out_dirs_and_init_configs_fpath(self, tmp_path):
        yaml_fpath = _write_yaml(tmp_path, f"""
data_dir: "{tmp_path}/data"
out_dir: "{tmp_path}/out"
init_configs_fpath: "pbc/config.xyz"
""" + BASE_BODY)

        # the config lives under <data_dir>/pbc, not in the cwd, so it is found via data_dir
        (tmp_path / "data" / "pbc").mkdir(parents=True)
        (tmp_path / "data" / "pbc" / "config.xyz").write_text(CONFIG)

        resolved = rmlt_inputs.resolve_mlip_md_yaml(yaml_fpath)

        assert resolved["data_dir"] == f"{tmp_path}/data"
        assert resolved["out_dir"] == f"{tmp_path}/out"
        assert resolved["traj_dir"] == f"{tmp_path}/out/trj"
        assert resolved["csv_dir"] == f"{tmp_path}/out/csv"
        assert resolved["init_configs_fpath"] == f"{tmp_path}/data/pbc/config.xyz"

    def test_init_configs_fpath_templates_from_data_dir(self, tmp_path):
        yaml_fpath = _write_yaml(tmp_path, f"""
data_dir: "{tmp_path}/data"
init_configs_fpath: "${{data_dir}}/pbc/config.xyz"
""" + BASE_BODY)
        (tmp_path / "data" / "pbc").mkdir(parents=True)
        (tmp_path / "data" / "pbc" / "config.xyz").write_text(CONFIG)

        resolved = rmlt_inputs.resolve_mlip_md_yaml(yaml_fpath)

        assert resolved["init_configs_fpath"] == f"{tmp_path}/data/pbc/config.xyz"

    def test_absolute_init_configs_fpath_is_left_alone(self, tmp_path, config_fpath):
        yaml_fpath = _write_yaml(tmp_path, f"""
out_dir: "{tmp_path}/out"
init_configs_fpath: "{config_fpath}"
""" + BASE_BODY)

        resolved = rmlt_inputs.resolve_mlip_md_yaml(yaml_fpath)

        assert resolved["init_configs_fpath"] == str(config_fpath)
        assert resolved["traj_dir"] == f"{tmp_path}/out/trj"

    def test_missing_init_configs_fpath_is_reported(self, tmp_path):
        yaml_fpath = _write_yaml(tmp_path, f"""
data_dir: "{tmp_path}/data"
init_configs_fpath: "pbc/nope.xyz"
""" + BASE_BODY)

        with pytest.raises(ValueError, match="init_configs_fpath not found"):
            rmlt_inputs.resolve_mlip_md_yaml(yaml_fpath)


class TestMlipMdCheckpointFields:
    """The three MD checkpoint fields, and their defaults."""

    def test_they_default_to_off_at_ten_ps(self, tmp_path):
        yaml_fpath = _write_yaml(tmp_path, BASE_BODY)
        resolved = rmlt_inputs.resolve_mlip_md_yaml(yaml_fpath)

        assert resolved["save_md_checkpoints"] is False
        assert resolved["load_md_checkpoint"] is False
        assert resolved["checkpoint_every"] == DEFAULT_CHECKPOINT_EVERY_PS == 10

    def test_the_checkpoint_dir_follows_out_dir(self, tmp_path):
        yaml_fpath = _write_yaml(tmp_path, BASE_BODY)
        assert rmlt_inputs.resolve_mlip_md_yaml(yaml_fpath)["md_checkpoint_dir"] == (
            f"{OUT_DIR}/md_checkpoints")

        yaml_fpath = _write_yaml(tmp_path, f'out_dir: "{tmp_path}/out"\n' + BASE_BODY)
        assert rmlt_inputs.resolve_mlip_md_yaml(yaml_fpath)["md_checkpoint_dir"] == (
            f"{tmp_path}/out/md_checkpoints")

    def test_saving_and_resuming_every_run_is_a_switch(self, tmp_path):
        yaml_fpath = _write_yaml(tmp_path, """
save_md_checkpoints: true
load_md_checkpoint: true
checkpoint_every: 5
""" + BASE_BODY)
        resolved = rmlt_inputs.resolve_mlip_md_yaml(yaml_fpath)

        assert resolved["save_md_checkpoints"] is True
        assert resolved["load_md_checkpoint"] is True
        assert resolved["checkpoint_every"] == 5

    def test_a_checkpoint_path_is_kept_as_one(self, tmp_path):
        """load_md_checkpoint resumes a single run from a file when it is not a switch."""
        yaml_fpath = _write_yaml(tmp_path, f"""
load_md_checkpoint: "{tmp_path}/cp/run_checkpoint.yaml"
""" + BASE_BODY)
        resolved = rmlt_inputs.resolve_mlip_md_yaml(yaml_fpath)

        assert resolved["load_md_checkpoint"] == f"{tmp_path}/cp/run_checkpoint.yaml"

    def test_a_quoted_switch_is_still_a_switch(self, tmp_path):
        yaml_fpath = _write_yaml(tmp_path, 'load_md_checkpoint: "true"\n' + BASE_BODY)

        assert rmlt_inputs.resolve_mlip_md_yaml(yaml_fpath)["load_md_checkpoint"] is True

    def test_a_checkpoint_path_may_reference_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHECKPOINT_DIR", f"{tmp_path}/cp")
        yaml_fpath = _write_yaml(tmp_path, """
load_md_checkpoint: "$CHECKPOINT_DIR/run_checkpoint.yaml"
""" + BASE_BODY)

        assert rmlt_inputs.resolve_mlip_md_yaml(yaml_fpath)["load_md_checkpoint"] == (
            f"{tmp_path}/cp/run_checkpoint.yaml")

    @pytest.mark.parametrize("value", ["0", "-5", "ten"])
    def test_an_interval_that_is_not_positive_number_of_ps_is_reported(self, tmp_path, value):
        yaml_fpath = _write_yaml(tmp_path, f"checkpoint_every: {value}\n" + BASE_BODY)

        with pytest.raises(ValueError, match="checkpoint_every"):
            rmlt_inputs.resolve_mlip_md_yaml(yaml_fpath)
