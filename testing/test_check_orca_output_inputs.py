"""
Unit tests for the check_orca_output.py input YAML resolution, and the check it drives.

These cover:

  * the two fields the YAML takes (``work_dir`` / ``input_name``), both required, and the
    ``$VAR`` expansion of the directory,
  * the example YAML the repository ships, so it keeps naming the outputs it was written for,
  * the script running from end to end on a directory of ORCA outputs, reporting each run by
    whether it terminated normally.

Nothing here needs an ORCA installation, a GPU or any model files: the outputs it reads are
written by the tests themselves.

Run from the repository root:

    pytest testing/test_check_orca_output_inputs.py
"""

import subprocess
import sys
from pathlib import Path

import pytest

from react_ml_test_public.helpers import inputs as rmlt_inputs

REPO_ROOT = Path(__file__).resolve().parents[1]

# the example YAML the repository ships, and the run it was written for (the settings the
# script hardcoded before it took one)
EXAMPLE_YAML = "inputs/dft/check_orca_output_guajadial_eg.yaml"
EXAMPLE_INPUT_NAME = "guajadial_DA_test_wB97M_TZVPPD"

# a minimal ORCA output that terminated, and one that did not
SUCCESS_OUT = """\
Number of atoms: 3
ORCA TERMINATED NORMALLY
"""
FAILURE_OUT = """\
Number of atoms: 4
SCF NOT CONVERGED
"""


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    """Resolve the repo-root-relative paths used in the YAML inputs."""
    monkeypatch.chdir(REPO_ROOT)


def _write_yaml(tmp_path, body):
    fpath = tmp_path / "orca_check.yaml"
    fpath.write_text(body)
    return str(fpath)


class TestCheckOrcaOutputYaml:
    """The fields the YAML takes, and what it does with them."""

    def test_it_takes_a_work_dir_and_an_input_name(self, tmp_path):
        yaml_fpath = _write_yaml(tmp_path, """
work_dir: "out/dft/sp"
input_name: "CPD_MVK_DA_test_wB97M_TZVPPD"
""")
        resolved = rmlt_inputs.resolve_check_orca_output_yaml(yaml_fpath)

        assert resolved["work_dir"] == "out/dft/sp"
        assert resolved["input_name"] == "CPD_MVK_DA_test_wB97M_TZVPPD"

    def test_the_shipped_example_holds_the_settings_it_replaced(self):
        """The example carries what the script hardcoded: out/dft/sp and that run's name.

        The directory is spelled relative to the repository root, so it holds locally; on the
        cluster, where that directory is not cwd-relative, the same field is edited to
        "$DATA/out/dft/sp" (see the example's own notes).
        """
        resolved = rmlt_inputs.resolve_check_orca_output_yaml(str(REPO_ROOT / EXAMPLE_YAML))

        assert resolved["work_dir"] == "out/dft/sp"
        assert resolved["input_name"] == EXAMPLE_INPUT_NAME

    def test_missing_work_dir_is_reported(self, tmp_path):
        yaml_fpath = _write_yaml(tmp_path, 'input_name: "CPD_MVK_DA_test"\n')

        with pytest.raises(ValueError, match="Missing required field 'work_dir'"):
            rmlt_inputs.resolve_check_orca_output_yaml(yaml_fpath)

    def test_missing_input_name_is_reported(self, tmp_path):
        yaml_fpath = _write_yaml(tmp_path, 'work_dir: "out/dft/sp"\n')

        with pytest.raises(ValueError, match="Missing required field 'input_name'"):
            rmlt_inputs.resolve_check_orca_output_yaml(yaml_fpath)

    def test_work_dir_may_reference_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ORCA_OUT_ROOT", "/somewhere/else")
        yaml_fpath = _write_yaml(tmp_path, """
work_dir: "$ORCA_OUT_ROOT/out/dft/sp"
input_name: "CPD_MVK_DA_test"
""")
        resolved = rmlt_inputs.resolve_check_orca_output_yaml(yaml_fpath)

        assert resolved["work_dir"] == "/somewhere/else/out/dft/sp"

    def test_an_unset_variable_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
        yaml_fpath = _write_yaml(tmp_path, """
work_dir: "$NOT_SET_ANYWHERE/out/dft/sp"
input_name: "CPD_MVK_DA_test"
""")

        with pytest.raises(ValueError, match="NOT_SET_ANYWHERE"):
            rmlt_inputs.resolve_check_orca_output_yaml(yaml_fpath)

    def test_a_directory_that_does_not_exist_is_left_to_the_check(self, tmp_path):
        """A directory holding no outputs is a set of zero runs, not a config error."""
        yaml_fpath = _write_yaml(tmp_path, """
work_dir: "not/a/real/directory"
input_name: "CPD_MVK_DA_test"
""")
        resolved = rmlt_inputs.resolve_check_orca_output_yaml(yaml_fpath)

        assert resolved["work_dir"] == "not/a/real/directory"

    def test_a_file_that_is_not_yaml_is_reported(self, tmp_path):
        fpath = tmp_path / "orca_check.txt"
        fpath.write_text('work_dir: "out/dft/sp"\ninput_name: "CPD_MVK_DA_test"\n')

        with pytest.raises(ValueError, match="must be a YAML file"):
            rmlt_inputs.resolve_check_orca_output_yaml(str(fpath))


class TestTheScriptRuns:
    """The script goes from its YAML to the report on a directory of outputs."""

    def test_it_reports_every_run_under_the_directory(self, tmp_path):
        work_dir = tmp_path / "dft" / "sp"
        work_dir.mkdir(parents=True)
        (work_dir / "conf_my_set_0.out").write_text(SUCCESS_OUT)
        (work_dir / "conf_my_set_1.out").write_text(FAILURE_OUT)
        yaml_fpath = _write_yaml(tmp_path, f"""
work_dir: "{work_dir}"
input_name: "my_set"
""")

        result = subprocess.run([sys.executable, "-m",
                                 "react_ml_test_public.scripts.check_orca_output", yaml_fpath],
                                cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)

        assert result.returncode == 0, f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        # the two outputs are found, and told apart by whether they terminated
        assert "Total Runs:  2" in result.stdout
        assert "Success:  1" in result.stdout
        assert "Success Jobs:  [1]" in result.stdout
        assert "Success Num Atoms:  {1: 3}" in result.stdout
        assert "Fail:  1" in result.stdout
        assert "Fail Jobs:  [2]" in result.stdout
        assert "Fail Num Atoms:  {2: 4}" in result.stdout

    def test_it_reports_no_runs_for_a_directory_without_outputs(self, tmp_path):
        work_dir = tmp_path / "empty"
        work_dir.mkdir()
        yaml_fpath = _write_yaml(tmp_path, f"""
work_dir: "{work_dir}"
input_name: "my_set"
""")

        result = subprocess.run([sys.executable, "-m",
                                 "react_ml_test_public.scripts.check_orca_output", yaml_fpath],
                                cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)

        assert result.returncode == 0, f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        assert "Total Runs:  0" in result.stdout
        assert "Success:  0" in result.stdout
        assert "Fail:  0" in result.stdout

    def test_the_input_yaml_is_the_only_argument_it_needs(self):
        result = subprocess.run([sys.executable, "-m",
                                 "react_ml_test_public.scripts.check_orca_output", "--help"],
                                cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)

        assert result.returncode == 0, result.stderr[-2000:]
        assert "usage" in result.stdout.lower()
        assert "input_yaml" in result.stdout
