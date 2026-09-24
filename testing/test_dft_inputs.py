"""
Unit tests for the save_dft_inputs_from_xyz.py input YAML resolution and the ORCA
``%basis`` block it writes into the generated input files.

These cover the ``method`` dispatch (ORCA / VASP), the per-method required fields, the
``input_fpath`` / ``out_dir`` path handling and the optional ``basis`` field.

Nothing here needs a DFT code, the MACE stack or any model files.

Run from the repository root:

    pytest testing/test_dft_inputs.py
"""

from pathlib import Path

import pytest

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.io import (
    format_orca_basis_block,
    insert_orca_input_block,
)
from react_ml_test_public.helpers.settings import DFT_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]

# a single H2 config, as a stand-in for the .xyz being converted
CONFIG = """2
Properties=species:S:1:pos:R:3
H 0.0 0.0 0.0
H 0.0 0.0 0.7
"""

# the minimum an ORCA input YAML needs
BASE_ORCA_BODY = """
method: "orca"
charge: 0
mult: 1
"""


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    """Resolve the repo-root-relative paths used in the YAML inputs."""
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture
def config_fpath(tmp_path):
    """A two-config .xyz file to point input_fpath at."""
    fpath = tmp_path / "config.xyz"
    fpath.write_text(CONFIG + CONFIG)
    return fpath


def _write_yaml(tmp_path, body):
    fpath = tmp_path / "dft.yaml"
    fpath.write_text(body)
    return str(fpath)


class TestSaveDftInputsYaml:

    def test_orca_defaults(self, tmp_path, config_fpath):
        yaml_fpath = _write_yaml(tmp_path, f"""
input_fpath: "{config_fpath}"
""" + BASE_ORCA_BODY)
        resolved = rmlt_inputs.resolve_save_dft_inputs_yaml(yaml_fpath)

        assert resolved["input_fpath"] == str(config_fpath)
        assert resolved["method"] == "orca"
        assert resolved["charge"] == 0
        assert resolved["mult"] == 1
        # unset: the output root defaults to settings and the config values are left alone
        assert resolved["out_dir"] == DFT_DIR
        assert resolved["n_cores"] is None
        assert resolved["orca_keywords"] is None
        assert resolved["max_core"] is None
        assert resolved["extra_name_tag"] is None
        # VASP-only settings are not resolved for ORCA
        assert resolved["box_size"] is None
        assert resolved["incar_settings"] is None

    def test_orca_settings_and_out_dir(self, tmp_path, config_fpath):
        yaml_fpath = _write_yaml(tmp_path, f"""
input_fpath: "{config_fpath}"
out_dir: "somewhere/else"
n_cores: 16
orca_keywords: ["WB97M-D3BJ", "def2-TZVPPD", "TightSCF", "EnGrad"]
max_core: 4000
extra_name_tag: "wB97M_TZVPPD"
""" + BASE_ORCA_BODY)
        resolved = rmlt_inputs.resolve_save_dft_inputs_yaml(yaml_fpath)

        assert resolved["out_dir"] == "somewhere/else"
        assert resolved["n_cores"] == 16
        assert resolved["orca_keywords"] == ["WB97M-D3BJ", "def2-TZVPPD", "TightSCF", "EnGrad"]
        assert resolved["max_core"] == 4000.0
        assert resolved["extra_name_tag"] == "wB97M_TZVPPD"

    def test_vasp_settings(self, tmp_path, config_fpath):
        yaml_fpath = _write_yaml(tmp_path, f"""
input_fpath: "{config_fpath}"
method: "vasp"
box_size: 50.0
incar_settings:
  ENCUT: 550
  EDIFF: "1E-5"
  SIGMA: 0.1
  ISPIN: 1
""")
        resolved = rmlt_inputs.resolve_save_dft_inputs_yaml(yaml_fpath)

        assert resolved["method"] == "vasp"
        assert resolved["box_size"] == 50.0
        assert resolved["incar_settings"] == {"ENCUT": 550, "EDIFF": "1E-5", "SIGMA": 0.1, "ISPIN": 1}
        # VASP inputs carry no charge / multiplicity, so neither is required or resolved
        assert resolved["charge"] is None
        assert resolved["mult"] is None

    def test_vasp_incar_settings_are_optional(self, tmp_path, config_fpath):
        yaml_fpath = _write_yaml(tmp_path, f"""
input_fpath: "{config_fpath}"
method: "vasp"
box_size: 40.0
""")
        resolved = rmlt_inputs.resolve_save_dft_inputs_yaml(yaml_fpath)

        assert resolved["incar_settings"] == {}

    def test_input_fpath_relative_to_data_dir(self, tmp_path):
        yaml_fpath = _write_yaml(tmp_path, f"""
data_dir: "{tmp_path}/data"
input_fpath: "test/config.xyz"
""" + BASE_ORCA_BODY)

        # the config lives under <data_dir>/test, not in the cwd, so it is found via data_dir
        (tmp_path / "data" / "test").mkdir(parents=True)
        (tmp_path / "data" / "test" / "config.xyz").write_text(CONFIG)

        resolved = rmlt_inputs.resolve_save_dft_inputs_yaml(yaml_fpath)

        assert resolved["input_fpath"] == f"{tmp_path}/data/test/config.xyz"

    def test_missing_input_fpath_is_reported(self, tmp_path):
        yaml_fpath = _write_yaml(tmp_path, f"""
data_dir: "{tmp_path}/data"
input_fpath: "test/nope.xyz"
""" + BASE_ORCA_BODY)

        with pytest.raises(ValueError, match="input_fpath not found"):
            rmlt_inputs.resolve_save_dft_inputs_yaml(yaml_fpath)

    def test_missing_method_is_reported(self, tmp_path, config_fpath):
        yaml_fpath = _write_yaml(tmp_path, f"""
input_fpath: "{config_fpath}"
charge: 0
mult: 1
""")

        with pytest.raises(ValueError, match="Missing required field 'method'"):
            rmlt_inputs.resolve_save_dft_inputs_yaml(yaml_fpath)

    def test_unknown_method_is_reported(self, tmp_path, config_fpath):
        yaml_fpath = _write_yaml(tmp_path, f"""
input_fpath: "{config_fpath}"
method: "gaussian"
""")

        with pytest.raises(ValueError, match="Unrecognised method"):
            rmlt_inputs.resolve_save_dft_inputs_yaml(yaml_fpath)

    def test_missing_charge_is_reported_for_orca(self, tmp_path, config_fpath):
        yaml_fpath = _write_yaml(tmp_path, f"""
input_fpath: "{config_fpath}"
method: "orca"
mult: 1
""")

        with pytest.raises(ValueError, match="Missing required field 'charge'"):
            rmlt_inputs.resolve_save_dft_inputs_yaml(yaml_fpath)

    def test_missing_box_size_is_reported_for_vasp(self, tmp_path, config_fpath):
        yaml_fpath = _write_yaml(tmp_path, f"""
input_fpath: "{config_fpath}"
method: "vasp"
""")

        with pytest.raises(ValueError, match="Missing required field 'box_size'"):
            rmlt_inputs.resolve_save_dft_inputs_yaml(yaml_fpath)

    def test_orca_keywords_must_be_a_list(self, tmp_path, config_fpath):
        yaml_fpath = _write_yaml(tmp_path, f"""
input_fpath: "{config_fpath}"
orca_keywords: "WB97M-D3BJ def2-TZVPPD"
""" + BASE_ORCA_BODY)

        with pytest.raises(ValueError, match="'orca_keywords' must be a list"):
            rmlt_inputs.resolve_save_dft_inputs_yaml(yaml_fpath)

    def test_incar_settings_must_be_a_mapping(self, tmp_path, config_fpath):
        yaml_fpath = _write_yaml(tmp_path, f"""
input_fpath: "{config_fpath}"
method: "vasp"
box_size: 50.0
incar_settings: ["ENCUT", 550]
""")

        with pytest.raises(ValueError, match="'incar_settings' must be a mapping"):
            rmlt_inputs.resolve_save_dft_inputs_yaml(yaml_fpath)

    def test_non_yaml_input_is_reported(self, tmp_path):
        with pytest.raises(ValueError, match="must be a YAML file"):
            rmlt_inputs.resolve_save_dft_inputs_yaml(str(tmp_path / "dft.txt"))

    def test_basis_defaults_to_none(self, tmp_path, config_fpath):
        yaml_fpath = _write_yaml(tmp_path, f"""
input_fpath: "{config_fpath}"
""" + BASE_ORCA_BODY)
        resolved = rmlt_inputs.resolve_save_dft_inputs_yaml(yaml_fpath)

        # unset: no %basis block is written into the generated input
        assert resolved["basis"] is None


class TestSaveDftInputsBasis:
    """The optional ``basis`` field, written into each input as an ORCA %basis block."""

    def _basis_of(self, tmp_path, config_fpath, basis_yaml):
        yaml_fpath = _write_yaml(tmp_path, f"""
input_fpath: "{config_fpath}"
{basis_yaml}
""" + BASE_ORCA_BODY)
        return rmlt_inputs.resolve_save_dft_inputs_yaml(yaml_fpath)["basis"]

    def test_block_key_and_value(self, tmp_path, config_fpath):
        # the spelling used in the mf_mat_react GO_cluster_test_profiles.yaml
        basis = self._basis_of(tmp_path, config_fpath, 'basis: AuxJ "aug-cc-pVTZ/JK"')

        assert basis == [("AuxJ", "aug-cc-pVTZ/JK")]

    def test_value_quoted_as_a_whole(self, tmp_path, config_fpath):
        basis = self._basis_of(tmp_path, config_fpath, 'basis: "AuxJ aug-cc-pVTZ/JK"')

        assert basis == [("AuxJ", "aug-cc-pVTZ/JK")]

    def test_unquoted_value(self, tmp_path, config_fpath):
        basis = self._basis_of(tmp_path, config_fpath, "basis: AuxJ aug-cc-pVTZ/JK")

        assert basis == [("AuxJ", "aug-cc-pVTZ/JK")]

    def test_bare_basis_set_name_is_the_orbital_basis(self, tmp_path, config_fpath):
        basis = self._basis_of(tmp_path, config_fpath, "basis: def2-TZVPPD")

        assert basis == [("Basis", "def2-TZVPPD")]

    def test_key_case_is_normalised(self, tmp_path, config_fpath):
        basis = self._basis_of(tmp_path, config_fpath, 'basis: auxjk "aug-cc-pVTZ/JK"')

        assert basis == [("AuxJK", "aug-cc-pVTZ/JK")]

    def test_mapping(self, tmp_path, config_fpath):
        basis = self._basis_of(tmp_path, config_fpath, """
basis:
  Basis: def2-TZVPPD
  AuxJ: "aug-cc-pVTZ/JK"
  DecontractAuxJ: false
""")

        assert basis == [("Basis", "def2-TZVPPD"),
                         ("AuxJ", "aug-cc-pVTZ/JK"),
                         ("DecontractAuxJ", "False")]

    def test_list(self, tmp_path, config_fpath):
        basis = self._basis_of(tmp_path, config_fpath, """
basis:
  - 'Basis "def2-TZVPPD"'
  - 'AuxJ "aug-cc-pVTZ/JK"'
""")

        assert basis == [("Basis", "def2-TZVPPD"), ("AuxJ", "aug-cc-pVTZ/JK")]

    def test_basis_is_reported_for_vasp(self, tmp_path, config_fpath):
        yaml_fpath = _write_yaml(tmp_path, f"""
input_fpath: "{config_fpath}"
method: "vasp"
box_size: 40.0
basis: AuxJ "aug-cc-pVTZ/JK"
""")

        with pytest.raises(ValueError, match="'basis' is only used by the ORCA method"):
            rmlt_inputs.resolve_save_dft_inputs_yaml(yaml_fpath)

    def test_empty_value_is_reported(self, tmp_path, config_fpath):
        with pytest.raises(ValueError, match="Missing value for the 'basis' block key"):
            self._basis_of(tmp_path, config_fpath, 'basis: AuxJ ""')

    def test_unknown_entry_type_is_reported(self, tmp_path, config_fpath):
        with pytest.raises(ValueError, match="'basis' entries must be strings or mappings"):
            self._basis_of(tmp_path, config_fpath, "basis: [1, 2]")


class TestOrcaBasisBlock:

    def test_block_lines(self):
        block = format_orca_basis_block([("Basis", "def2-TZVPPD"),
                                         ("AuxJ", "aug-cc-pVTZ/JK")])

        assert block == ('%basis\n'
                         '  Basis "def2-TZVPPD"\n'
                         '  AuxJ "aug-cc-pVTZ/JK"\n'
                         'end\n')

    def test_booleans_are_left_unquoted(self):
        # ORCA takes the Decontract* switches as bare booleans, not quoted strings
        block = format_orca_basis_block([("DecontractAuxJ", False),
                                         ("DecontractBas", "false")])

        assert block == '%basis\n  DecontractAuxJ false\n  DecontractBas false\nend\n'

    def test_block_is_inserted_after_the_simple_input_line(self, tmp_path):
        # the shape of the file autode's ORCA wrapper generates: the ! line, blocks, coords
        inp_fpath = tmp_path / "config_0_orca.inp"
        inp_fpath.write_text("! WB97M-D3BJ def2-TZVPPD TightSCF EnGrad\n"
                             "%scf \nmaxiter 250 \nend\n"
                             "*xyz 0 1\nH 0.0 0.0 0.0\n*\n")

        insert_orca_input_block(str(inp_fpath),
                                format_orca_basis_block([("AuxJ", "aug-cc-pVTZ/JK")]))

        assert inp_fpath.read_text() == (
            "! WB97M-D3BJ def2-TZVPPD TightSCF EnGrad\n"
            "%basis\n"
            '  AuxJ "aug-cc-pVTZ/JK"\n'
            "end\n"
            "%scf \nmaxiter 250 \nend\n"
            "*xyz 0 1\nH 0.0 0.0 0.0\n*\n")
