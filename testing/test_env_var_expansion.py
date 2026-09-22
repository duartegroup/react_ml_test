"""
Unit tests for ``$VAR`` / ``${VAR}`` environment expansion in YAML path fields.

A YAML input can name a location that is only known at run time - most usefully the shared
``$DATA/models`` on the ARC clusters, which has no fixed absolute path::

    model_dir: "$DATA/models"

These cover:

  * ``settings.expand_env_vars`` (both reference forms, ``$$`` escaping, unset variables),
  * the ``resolve_*`` helpers expanding the values they are given,
  * the three directory fields (``data_dir`` / ``out_dir`` / ``model_dir``) being expanded
    while the config is loaded, so an unset variable is reported as a config error,
  * the regression that matters most: the ``${...}`` templating a YAML does against its own
    keys (``init_configs_fpath: '${data_dir}/pbc/x.xyz'``) still works, and is resolved
    *before* environment expansion so a YAML key is never mistaken for an unset variable.

Run from the repository root (the sample YAML inputs use repo-root-relative paths):

    pytest testing/test_env_var_expansion.py
"""

from pathlib import Path

import pytest
import yaml

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.settings import (
    expand_env_vars,
    resolve_data_path,
    resolve_model_dir,
    resolve_out_root,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MD_YAML = "inputs/md/CPD_MVK_DA_dynamics_vac.yaml"


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    """Resolve the repo-root-relative paths used in the sample YAML inputs."""
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture
def models_env(monkeypatch, tmp_path):
    """A set $MY_MODELS pointing at a real directory, for the expansion tests."""
    models_dir = tmp_path / "shared" / "models"
    models_dir.mkdir(parents=True)
    monkeypatch.setenv("MY_MODELS", str(models_dir))
    return models_dir


class TestExpandEnvVars:
    """expand_env_vars() handles both reference forms, and fails loudly on an unset name."""

    def test_bare_reference_is_expanded(self, models_env):
        assert expand_env_vars("$MY_MODELS") == str(models_env)

    def test_braced_reference_is_expanded(self, models_env):
        assert expand_env_vars("${MY_MODELS}") == str(models_env)

    def test_reference_inside_a_path_is_expanded(self, models_env):
        assert expand_env_vars("$MY_MODELS/fine_tuned") == str(models_env / "fine_tuned")

    def test_relative_path_without_a_reference_is_unchanged(self):
        assert expand_env_vars("models") == "models"
        assert expand_env_vars("data/reaction_data") == "data/reaction_data"

    def test_absolute_path_without_a_reference_is_unchanged(self):
        assert expand_env_vars("/abs/models") == "/abs/models"

    def test_double_dollar_gives_a_literal_dollar(self, models_env):
        assert expand_env_vars("$$MY_MODELS") == "$MY_MODELS"

    def test_unset_variable_is_an_error_naming_it(self, monkeypatch):
        monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
        with pytest.raises(ValueError, match="'NOT_SET_ANYWHERE'"):
            expand_env_vars("$NOT_SET_ANYWHERE/models")

    def test_a_literal_dollar_in_a_path_is_an_error_suggesting_the_escape(self):
        with pytest.raises(ValueError, match=r"\$\$"):
            expand_env_vars("/a$b/c")


class TestResolversExpandEnvVars:
    """The resolve_* helpers expand the values a YAML hands them."""

    def test_model_dir(self, models_env):
        assert resolve_model_dir("$MY_MODELS") == str(models_env)

    def test_model_dir_default_is_untouched(self):
        from react_ml_test_public.helpers.settings import MODEL_DIR
        assert resolve_model_dir(None) == MODEL_DIR

    def test_out_dir(self, models_env):
        assert resolve_out_root("$MY_MODELS/out") == str(models_env / "out")

    def test_data_path_absolute_reference(self, models_env, tmp_path):
        data_file = models_env / "x.xyz"
        data_file.write_text("1\n\nH 0 0 0\n")
        assert resolve_data_path("$MY_MODELS/x.xyz") == str(data_file)

    def test_data_path_relative_resolved_against_an_expanded_data_dir(self, models_env):
        (models_env / "test.xyz").write_text("1\n\nH 0 0 0\n")
        assert resolve_data_path("test.xyz", "$MY_MODELS") == str(models_env / "test.xyz")


class TestConfigLoadingExpandsEnvVars:
    """The directory fields are expanded while the config loads, with a config error if unset."""

    @staticmethod
    def _write_yaml(tmp_path, **overrides):
        config = yaml.safe_load((REPO_ROOT / MD_YAML).read_text())
        config.update(overrides)
        fpath = tmp_path / "input.yaml"
        fpath.write_text(yaml.safe_dump(config))
        return str(fpath)

    def test_model_dir_is_expanded(self, models_env, tmp_path):
        fpath = self._write_yaml(tmp_path, model_dir="$MY_MODELS")
        assert rmlt_inputs.resolve_mlip_md_yaml(fpath)["model_dir"] == str(models_env)

    def test_data_and_out_dirs_are_expanded(self, models_env, tmp_path):
        fpath = self._write_yaml(tmp_path, data_dir="$MY_MODELS/data", out_dir="$MY_MODELS/out")
        resolved = rmlt_inputs.resolve_mlip_md_yaml(fpath)

        assert resolved["data_dir"] == str(models_env / "data")
        assert resolved["out_dir"] == str(models_env / "out")
        assert resolved["traj_dir"] == str(models_env / "out" / "trj")

    @pytest.mark.parametrize("field", ["model_dir", "data_dir", "out_dir"])
    def test_unset_variable_is_reported_as_a_config_error(self, field, tmp_path, monkeypatch):
        """A literal '$DATA/models' directory must never be searched instead."""
        monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
        fpath = self._write_yaml(tmp_path, **{field: "$NOT_SET_ANYWHERE/models"})

        with pytest.raises(ValueError, match="'NOT_SET_ANYWHERE'"):
            rmlt_inputs.resolve_mlip_md_yaml(fpath)

    def test_the_shipped_inputs_need_no_environment_variables(self):
        """Every shipped YAML must keep resolving on a machine where $DATA is unset."""
        resolved = rmlt_inputs.resolve_mlip_md_yaml(MD_YAML)

        assert "$" not in resolved["model_dir"]
        assert "$" not in resolved["data_dir"]


class TestYamlSelfTemplatingStillWorks:
    """The ${...} templating against a YAML's own keys must be unaffected.

    It runs first, so a ``${data_dir}`` reference is already resolved by the time
    environment expansion sees the string - otherwise it would be reported as an unset
    environment variable called 'data_dir' and every existing input would break.
    """

    @staticmethod
    def _stage(tmp_path, data_dir_value, filename="config.xyz"):
        """A YAML whose init_configs_fpath templates from its own data_dir, plus that file."""
        config = yaml.safe_load((REPO_ROOT / MD_YAML).read_text())
        config["data_dir"] = data_dir_value
        config["init_configs_fpath"] = "${data_dir}/pbc/" + filename
        fpath = tmp_path / "input.yaml"
        fpath.write_text(yaml.safe_dump(config))
        return str(fpath)

    @staticmethod
    def _make_starting_config(data_dir, filename="config.xyz"):
        pbc = Path(data_dir) / "pbc"
        pbc.mkdir(parents=True, exist_ok=True)
        (pbc / filename).write_text("1\n\nH 0 0 0\n")
        return Path(data_dir) / "pbc" / filename

    def test_data_dir_reference_is_substituted_not_treated_as_an_env_var(self, tmp_path):
        data_dir = tmp_path / "my_data"
        expected = self._make_starting_config(data_dir)
        fpath = self._stage(tmp_path, str(data_dir))

        resolved = rmlt_inputs.resolve_mlip_md_yaml(fpath)

        assert resolved["data_dir"] == str(data_dir)
        # substituted from the YAML's own data_dir, then resolved against it
        assert resolved["init_configs_fpath"] == str(expected)
        assert "${" not in resolved["init_configs_fpath"]

    def test_self_templating_and_env_expansion_compose(self, models_env, tmp_path):
        """A YAML key can hold an environment reference that another field templates from."""
        data_dir = models_env / "data"
        expected = self._make_starting_config(data_dir)
        fpath = self._stage(tmp_path, "$MY_MODELS/data")

        resolved = rmlt_inputs.resolve_mlip_md_yaml(fpath)

        assert resolved["data_dir"] == str(data_dir)
        # ${data_dir} expands to the raw '$MY_MODELS/data', which is then env-expanded
        assert resolved["init_configs_fpath"] == str(expected)
