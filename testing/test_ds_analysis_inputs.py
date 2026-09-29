"""
Unit tests for the run_ds_analysis.py input YAML (see resolve_ds_analysis_yaml).

These cover the dataset list / label / parser handling, the filter_config_type field that
replaces the hard-coded config-type filter the script used to carry, the energy / force
toggles and the data_dir / out_dir fields.

Nothing here needs a GPU, the MACE stack, a model or the real datasets (the datasets the
shipped YAMLs name are only checked for existence).

Run from the repository root (the YAML inputs use repo-root-relative paths):

    pytest testing/test_ds_analysis_inputs.py
"""

from pathlib import Path

import pytest

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.io import orca_props_parser
from react_ml_test_public.helpers.settings import (
    DATA_DIR,
    MEV_PER_ATOM,
    OUT_DIR,
    KCAL_PER_MOL,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_YAMLS = [
    "inputs/dataset/guajadial_DA_train_subsets.yaml",
    "inputs/dataset/CPD_MVK_DA_train_subsets.yaml",
    "inputs/dataset/anthracene_DA_train_subsets.yaml",
]

# the config types the datasets of these systems are labelled with
CONFIG_TYPES = ["vacuum", "solvated"]


# one H2 config, as a small stand-in for a dataset
CONFIG = """2
Properties=species:S:1:pos:R:3:REF_forces:R:3 config_type={config_type} REF_energy=-1.5
H 0.0 0.0 0.0 0.01 0.0 0.0
H 0.0 0.0 0.7 0.0 0.01 0.0
"""


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    """Resolve the repo-root-relative paths used in the YAML inputs."""
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture
def ds_fpath(tmp_path):
    """A 2-config dataset path (in tmp_path, so the resolver's file check passes)."""
    fpath = tmp_path / "ds.xyz"
    fpath.write_text(CONFIG.format(config_type="vacuum") * 2)
    return str(fpath)


@pytest.fixture
def ts_fpath(tmp_path):
    """A single-config TS reference path."""
    fpath = tmp_path / "ts.xyz"
    fpath.write_text(CONFIG.format(config_type="vacuum"))
    return str(fpath)


def write_yaml(tmp_path, ds_fpath, ts_fpath, **fields):
    """Write an input YAML holding the fields needed for a minimal, valid config."""
    import yaml

    config = {
        "input_ds_fps": [ds_fpath],
        "ts_ref_fpath": ts_fpath,
        "react_coords": [[0, 1], [1, 0]],
        "save_name": "test_ds",
    }
    config.update({key: value for key, value in fields.items() if value is not None})
    for key, value in fields.items():
        if value is None and key in config:
            del config[key]
    fpath = tmp_path / "input.yaml"
    fpath.write_text(yaml.safe_dump(config))
    return str(fpath)


class TestShippedYamls:
    """The YAMLs in inputs/dataset/ (one per system) must resolve as shipped."""

    @pytest.mark.parametrize("yaml_fpath", DATASET_YAMLS)
    def test_resolves(self, yaml_fpath):
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(str(REPO_ROOT / yaml_fpath))
        assert Path(resolved["ts_ref_fpath"]).is_file()
        assert resolved["dataset_labels"] == ["Train", "Valid", "Test"]
        assert len(resolved["input_ds_fps"]) == 3
        assert resolved["energy_units"] == MEV_PER_ATOM
        assert resolved["include_energies"] and resolved["include_forces"]
        # no config-type filter / no custom parsers unless the YAML asks for them, and the
        # SOAP-descriptor calculation is opt-in
        assert resolved["filter_config_type"] is None
        assert resolved["custom_xyz_parsers"] is None
        assert resolved["calc_soap"] is False

    @pytest.mark.parametrize("yaml_fpath", DATASET_YAMLS)
    def test_datasets_exist(self, yaml_fpath):
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(str(REPO_ROOT / yaml_fpath))
        for dataset_fpath in resolved["input_ds_fps"]:
            assert Path(dataset_fpath).is_file(), dataset_fpath

    @pytest.mark.parametrize("yaml_fpath", DATASET_YAMLS)
    def test_config_type_filter_is_documented(self, yaml_fpath):
        """Every dataset YAML shows the vacuum / solvated filter field."""
        text = (REPO_ROOT / yaml_fpath).read_text()
        assert "filter_config_type" in text


class TestFilterConfigType:
    """filter_config_type is None (no filtering), a config type to keep, or an error."""

    def test_defaults_to_none(self, tmp_path, ds_fpath, ts_fpath):
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, ds_fpath, ts_fpath))
        assert resolved["filter_config_type"] is None

    @pytest.mark.parametrize("config_type", CONFIG_TYPES)
    def test_accepts_a_config_type(self, tmp_path, ds_fpath, ts_fpath, config_type):
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, ds_fpath, ts_fpath, filter_config_type=config_type))
        assert resolved["filter_config_type"] == config_type

    def test_rejects_an_unknown_config_type(self, tmp_path, ds_fpath, ts_fpath):
        with pytest.raises(ValueError, match="Unrecognised filter_config_type"):
            rmlt_inputs.resolve_ds_analysis_yaml(
                write_yaml(tmp_path, ds_fpath, ts_fpath, filter_config_type="water"))

    def test_selects_the_configs_of_that_type(self, ds_fpath):
        """The filter the script applies keeps only the named config_type."""
        from ase.io import read

        fpath = Path(ds_fpath).parent / "mixed.xyz"
        fpath.write_text(CONFIG.format(config_type="vacuum")
                         + CONFIG.format(config_type="solvated"))
        configs = read(str(fpath), ':')

        for config_type in CONFIG_TYPES:
            kept = [config for config in configs
                    if config.info.get("config_type") == config_type]
            assert len(kept) == 1
            assert kept[0].info["config_type"] == config_type


class TestDatasets:
    """The dataset list, its legend labels and its per-dataset parsers."""

    def test_data_dir_resolves_relative_dataset_paths(self, tmp_path, ts_fpath):
        (tmp_path / "ds.xyz").write_text(CONFIG.format(config_type="vacuum"))
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, "ds.xyz", ts_fpath, data_dir=str(tmp_path)))
        assert resolved["input_ds_fps"] == [str(tmp_path / "ds.xyz")]
        assert resolved["data_dir"] == str(tmp_path)

    def test_labels_default_to_the_file_names(self, tmp_path, ds_fpath, ts_fpath):
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, ds_fpath, ts_fpath))
        assert resolved["dataset_labels"] == ["ds"]

    def test_labels_must_match_the_datasets(self, tmp_path, ds_fpath, ts_fpath):
        with pytest.raises(ValueError, match="one label per entry of 'input_ds_fps'"):
            rmlt_inputs.resolve_ds_analysis_yaml(
                write_yaml(tmp_path, ds_fpath, ts_fpath, dataset_labels=["A", "B"]))

    def test_missing_dataset_is_reported(self, tmp_path, ts_fpath):
        with pytest.raises(ValueError, match="Dataset file\\(s\\) not found"):
            rmlt_inputs.resolve_ds_analysis_yaml(
                write_yaml(tmp_path, "nope.xyz", ts_fpath))

    def test_empty_dataset_list_is_reported(self, tmp_path, ts_fpath):
        with pytest.raises(ValueError, match="'input_ds_fps' must be a non-empty list"):
            rmlt_inputs.resolve_ds_analysis_yaml(
                write_yaml(tmp_path, "nope.xyz", ts_fpath, input_ds_fps=[]))

    def test_custom_parsers_default_to_none(self, tmp_path, ds_fpath, ts_fpath):
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, ds_fpath, ts_fpath, custom_xyz_parsers=[None]))
        assert resolved["custom_xyz_parsers"] == [None]

    def test_custom_parsers_map_names_onto_parsers(self, tmp_path, ds_fpath, ts_fpath):
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, ds_fpath, ts_fpath,
                       custom_xyz_parsers=["orca_props_parser"]))
        assert resolved["custom_xyz_parsers"] == [orca_props_parser]

    def test_custom_parsers_must_match_the_datasets(self, tmp_path, ds_fpath, ts_fpath):
        with pytest.raises(ValueError, match="one entry per entry of 'input_ds_fps'"):
            rmlt_inputs.resolve_ds_analysis_yaml(
                write_yaml(tmp_path, ds_fpath, ts_fpath,
                           custom_xyz_parsers=["autode_props_parser", None]))

    def test_unknown_custom_parser_is_reported(self, tmp_path, ds_fpath, ts_fpath):
        with pytest.raises(ValueError, match="Unknown custom_xyz_parsers entry"):
            rmlt_inputs.resolve_ds_analysis_yaml(
                write_yaml(tmp_path, ds_fpath, ts_fpath, custom_xyz_parsers=["nope"]))


class TestCalculationSettings:

    def test_toggles_default_to_true(self, tmp_path, ds_fpath, ts_fpath):
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, ds_fpath, ts_fpath))
        assert resolved["include_energies"] is True
        assert resolved["include_forces"] is True

    def test_toggles_are_read(self, tmp_path, ds_fpath, ts_fpath):
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, ds_fpath, ts_fpath,
                       include_energies=False, include_forces=False))
        assert resolved["include_energies"] is False
        assert resolved["include_forces"] is False

    def test_energy_units_default_to_mev_per_atom(self, tmp_path, ds_fpath, ts_fpath):
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, ds_fpath, ts_fpath))
        assert resolved["energy_units"] == MEV_PER_ATOM

    @pytest.mark.parametrize("energy_units", ["kcal / mol", "kcal/mol", "KCAL/MOL"])
    def test_energy_units_are_spelling_insensitive(self, tmp_path, ds_fpath, ts_fpath,
                                                   energy_units):
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, ds_fpath, ts_fpath, energy_units=energy_units))
        assert resolved["energy_units"] == KCAL_PER_MOL

    def test_unsupported_energy_units_are_reported(self, tmp_path, ds_fpath, ts_fpath):
        """The script only converts to meV / atom and kcal / mol, so eV is an error."""
        with pytest.raises(ValueError, match="Unrecognised energy_units"):
            rmlt_inputs.resolve_ds_analysis_yaml(
                write_yaml(tmp_path, ds_fpath, ts_fpath, energy_units="eV"))

    def test_calc_soap_defaults_to_false(self, tmp_path, ds_fpath, ts_fpath):
        """The SOAP PCA plots are opt-in, so a YAML that does not ask for them gets none."""
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, ds_fpath, ts_fpath))
        assert resolved["calc_soap"] is False

    def test_calc_soap_can_be_switched_on(self, tmp_path, ds_fpath, ts_fpath):
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, ds_fpath, ts_fpath, calc_soap=True))
        assert resolved["calc_soap"] is True

    def test_calc_soap_accepts_a_quoted_boolean(self, tmp_path, ds_fpath, ts_fpath):
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, ds_fpath, ts_fpath, calc_soap="true"))
        assert resolved["calc_soap"] is True

    def test_test_elements_default_to_every_element(self, tmp_path, ds_fpath, ts_fpath):
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, ds_fpath, ts_fpath))
        assert resolved["test_elements"] is None

    def test_test_elements_are_read_as_a_set(self, tmp_path, ds_fpath, ts_fpath):
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, ds_fpath, ts_fpath, test_elements=["C", "H", "O"]))
        assert resolved["test_elements"] == {"C", "H", "O"}


class TestReactionCoordinate:

    def test_react_coords_are_pairs_of_indices(self, tmp_path, ds_fpath, ts_fpath):
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, ds_fpath, ts_fpath, react_coords=[[1, 12], [6, 11]]))
        assert resolved["react_coords"] == [(1, 12), (6, 11)]

    def test_a_non_pair_is_reported(self, tmp_path, ds_fpath, ts_fpath):
        with pytest.raises(ValueError, match="'react_coords' must be a list of pairs"):
            rmlt_inputs.resolve_ds_analysis_yaml(
                write_yaml(tmp_path, ds_fpath, ts_fpath, react_coords=[[1, 12, 13]]))

    def test_a_missing_ts_is_reported(self, tmp_path, ds_fpath):
        with pytest.raises(ValueError, match="ts_ref_fpath not found"):
            rmlt_inputs.resolve_ds_analysis_yaml(
                write_yaml(tmp_path, ds_fpath, "nope.xyz"))

    def test_a_ts_is_optional(self, tmp_path, ds_fpath, ts_fpath):
        """Without a TS the reaction-coordinate distances are still measured."""
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, ds_fpath, ts_fpath, ts_ref_fpath=None))
        assert resolved["ts_ref_fpath"] is None
        assert resolved["react_coords"] == [(0, 1), (1, 0)]


class TestRequiredFields:

    @pytest.mark.parametrize("field", ["react_coords", "save_name"])
    def test_missing_field_is_reported(self, tmp_path, ds_fpath, ts_fpath, field):
        with pytest.raises(ValueError, match=f"Missing required field '{field}'"):
            rmlt_inputs.resolve_ds_analysis_yaml(
                write_yaml(tmp_path, ds_fpath, ts_fpath, **{field: None}))

    def test_missing_dataset_list_is_reported(self, tmp_path, ds_fpath, ts_fpath):
        with pytest.raises(ValueError, match="'input_ds_fps' must be a non-empty list"):
            rmlt_inputs.resolve_ds_analysis_yaml(
                write_yaml(tmp_path, ds_fpath, ts_fpath, input_ds_fps=None))


class TestOutputDirs:

    def test_defaults_come_from_settings(self, tmp_path, ds_fpath, ts_fpath):
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, ds_fpath, ts_fpath))
        assert resolved["out_dir"] == OUT_DIR
        assert resolved["plot_dir"] == f"{OUT_DIR}/plt"
        assert resolved["data_dir"] == DATA_DIR

    def test_out_dir_moves_the_plots(self, tmp_path, ds_fpath, ts_fpath):
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, ds_fpath, ts_fpath, out_dir=str(tmp_path / "elsewhere")))
        assert resolved["plot_dir"] == f"{tmp_path / 'elsewhere'}/plt"

    def test_plot_limits_are_parsed(self, tmp_path, ds_fpath, ts_fpath):
        resolved = rmlt_inputs.resolve_ds_analysis_yaml(
            write_yaml(tmp_path, ds_fpath, ts_fpath,
                       plot_2d_xlim=[1.0, 7.5], plot_2d_ylim=[2.0, 8.0]))
        assert resolved["plot_2d_xlim"] == (1.0, 7.5)
        assert resolved["plot_2d_ylim"] == (2.0, 8.0)
