"""
Unit tests for reading reaction-profile trajectories and their YAML inputs.

These cover the handling added for the reaction-profile files written in Hartree
(``REF_energy=<float>``, ``energy_units="Hartree"``) and the ``data_dir`` / ``out_dir`` fields
of the reaction-barrier input YAML.

Nothing here needs a GPU, the MACE stack or any model files.

Run from the repository root (the YAML inputs use repo-root-relative paths):

    pytest testing/test_reaction_profile_inputs.py
"""

from pathlib import Path

import pytest

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.io import flexible_ase_read, hartree_props_parser
from react_ml_test_public.helpers.settings import (
    DATA_DIR,
    HARTREE_TO_EV,
    OUT_DIR,
    resolve_data_path,
    resolve_out_dirs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# one H2 config labelled with a Hartree energy, as in the reaction-profile files
CONFIG = """2
Properties=species:S:1:pos:R:3 REF_energy=-1.5 energy_units="Hartree"
H 0.0 0.0 0.0
H 0.0 0.0 0.7
"""


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    """Resolve the repo-root-relative paths used in the YAML inputs."""
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture
def profile_fpath(tmp_path):
    """A 2-config profile of Hartree-labelled configs."""
    fpath = tmp_path / "h2_profile.xyz"
    fpath.write_text(CONFIG + CONFIG)
    return str(fpath)


class TestHartreePropsParser:

    def test_converts_hartree_to_ev(self):
        props = hartree_props_parser('Properties=species:S:1:pos:R:3 REF_energy=-460.9729')
        assert props["energy"] == pytest.approx(-460.9729 * HARTREE_TO_EV)

    def test_raises_when_the_label_is_absent(self):
        with pytest.raises(ValueError, match="No REF_energy"):
            hartree_props_parser('Properties=species:S:1:pos:R:3 species="pro"')

    def test_reads_a_trajectory_and_overrides_the_ase_energy(self, profile_fpath):
        configs = flexible_ase_read(profile_fpath, hartree_props_parser)
        assert len(configs) == 2
        assert configs[0].info["energy"] == pytest.approx(-1.5 * HARTREE_TO_EV)

    def test_a_config_without_the_label_raises_on_read(self, tmp_path):
        fpath = tmp_path / "incomplete.xyz"
        fpath.write_text(CONFIG + '2\nProperties=species:S:1:pos:R:3 species="pro"\n'
                                  "H 0.0 0.0 0.0\nH 0.0 0.0 1.4\n")
        with pytest.raises(ValueError, match="No REF_energy"):
            flexible_ase_read(str(fpath), hartree_props_parser)


class TestResolveDirs:

    def test_defaults_come_from_settings(self):
        out_dir, plot_dir, csv_dir = resolve_out_dirs()
        assert (out_dir, plot_dir, csv_dir) == (OUT_DIR, f"{OUT_DIR}/plt", f"{OUT_DIR}/csv")

    def test_override(self):
        assert resolve_out_dirs("somewhere/else") == ("somewhere/else", "somewhere/else/plt",
                                                      "somewhere/else/csv")

    def test_path_resolving_from_the_cwd_is_left_alone(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "profile.xyz").write_text("")
        assert resolve_data_path("profile.xyz", "data") == "profile.xyz"

    def test_relative_path_is_taken_as_relative_to_the_data_dir(self, tmp_path):
        (tmp_path / "data" / "test").mkdir(parents=True)
        fpath = tmp_path / "data" / "test" / "profile.xyz"
        fpath.write_text("")
        assert resolve_data_path("test/profile.xyz", str(tmp_path / "data")) == str(fpath)

    def test_unknown_path_is_returned_as_written(self, tmp_path):
        assert resolve_data_path("nope/missing.xyz", str(tmp_path)) == "nope/missing.xyz"

    def test_absolute_path_is_left_alone(self, tmp_path):
        fpath = str(tmp_path / "profile.xyz")
        assert resolve_data_path(fpath, "data") == fpath


class TestReactionBarrierYaml:

    def _write_yaml(self, tmp_path, body):
        fpath = tmp_path / "profile.yaml"
        fpath.write_text(body)
        return str(fpath)

    def test_data_out_dirs_and_parser(self, tmp_path):
        (tmp_path / "data" / "cluster").mkdir(parents=True)
        (tmp_path / "data" / "cluster" / "profile.xyz").write_text(CONFIG)

        yaml_fpath = self._write_yaml(tmp_path, f"""
data_dir: "{tmp_path}/data"
out_dir: "{tmp_path}/out"
model_names:
  - MACE-OFF23_medium
save_name: test_profile
num_base_models: 1
num_base_model_comps: 1
test_trj_fpath: "cluster/profile.xyz"
custom_xyz_parser: "hartree_props_parser"
""")
        resolved = rmlt_inputs.resolve_reaction_barrier_comparison_yaml(yaml_fpath)

        assert resolved["data_dir"] == f"{tmp_path}/data"
        assert resolved["out_dir"] == f"{tmp_path}/out"
        assert resolved["plot_dir"] == f"{tmp_path}/out/plt"
        assert resolved["csv_dir"] == f"{tmp_path}/out/csv"
        assert resolved["test_trj_fpath"] == f"{tmp_path}/data/cluster/profile.xyz"
        assert resolved["custom_xyz_parser"] is hartree_props_parser

    def test_dirs_default_to_settings(self, tmp_path):
        (tmp_path / "profile.xyz").write_text(CONFIG)
        yaml_fpath = self._write_yaml(tmp_path, f"""
model_names:
  - MACE-OFF23_medium
save_name: test_profile
num_base_models: 1
num_base_model_comps: 1
test_trj_fpath: "{tmp_path}/profile.xyz"
""")
        resolved = rmlt_inputs.resolve_reaction_barrier_comparison_yaml(yaml_fpath)

        assert resolved["data_dir"] == DATA_DIR
        assert resolved["out_dir"] == OUT_DIR
        assert resolved["plot_dir"] == f"{OUT_DIR}/plt"

    def test_unknown_parser_name_is_rejected(self, tmp_path):
        (tmp_path / "profile.xyz").write_text(CONFIG)
        yaml_fpath = self._write_yaml(tmp_path, f"""
model_names:
  - MACE-OFF23_medium
save_name: test_profile
num_base_models: 1
num_base_model_comps: 1
test_trj_fpath: "{tmp_path}/profile.xyz"
custom_xyz_parser: "not_a_parser"
""")
        with pytest.raises(ValueError, match="Unknown custom_xyz_parser"):
            rmlt_inputs.resolve_reaction_barrier_comparison_yaml(yaml_fpath)
