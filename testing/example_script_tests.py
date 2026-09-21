"""
Example-script tests: run the worked examples listed in README.md, end to end.

Each test runs a documented example command - ``rmlt <script> <input.yaml>`` - as a
subprocess from the repository root, and checks the outputs the README says it writes
(plots under ``out/plt/``, CSVs under ``out/csv/``, optimised TSs under ``out/opt/``,
trajectories under ``out/trj/``).

These are integration tests: they need the MLIP stack, the trained models in ``models/``
and the reaction data in ``data/``, which the README copies over from the ORA repository.
A test skips itself when what it needs is missing - the models it would load (looked up
recursively in ``models/``, as ``load_model_calculator`` does), the data it would read,
or the output of an earlier step of a chained example (TS optimisation -> dynamics ->
trajectory analysis). The scripts skip work whose outputs already exist, so an example
with cached ``.npz`` results and saved trajectories runs quickly.

Every test carries the ``examples`` marker, and the long-running ones (the MD runs and
the solvent variants of the TS optimisations) also carry ``slow``:

    pytest -m examples                     # run the examples
    pytest -m "examples and not slow"      # skip the MD / solvent runs

Run from the repository root (the examples use repo-root-relative paths):

    pytest testing/example_script_tests.py
"""

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.io import get_name_from_path
from react_ml_test_public.helpers.settings import CSV_DIR, OPT_DIR, PLOT_DIR, PS, TRAJ_DIR

pytestmark = pytest.mark.examples

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data"

# a cached-results analysis: minutes
ANALYSIS_TIMEOUT = 3600
# a dimer TS search (+ frequencies), or one MD run per model: hours
DYNAMICS_TIMEOUT = 6 * ANALYSIS_TIMEOUT

# --- README example 1: test-set energy / force errors ---------------------------------
TEST_SET_EXAMPLE = "run_test_set_comparison"
TEST_SET_YAMLS = {
    "inputs/test/CPD_MVK_DA_FT42_test.yaml": "CPD_MVK_DA",
    "inputs/test/anthracene_DA_FT46_test.yaml": "anthracene_DA",
    "inputs/test/guajadial_DA_FT50_test.yaml": "guajadial_DA",
}

# --- README example 2: IRC reaction barriers ------------------------------------------
IRC_EXAMPLE = "run_reaction_barrier_comparison"
IRC_YAMLS = {
    "inputs/irc/CPD_MVK_DA_IRC_barrier.yaml": "CPD_MVK_DA",
    "inputs/irc/anthracene_DA_IRC_barrier.yaml": "anthracene_DA",
    "inputs/irc/anthracene_1_4_DA_IRC_barrier.yaml": "anthracene_DA",
    "inputs/irc/guajadial_DA_IRC_barrier.yaml": "guajadial_DA",
}

# --- README example 3: TS optimisation -> dynamics -> trajectory analysis --------------
# The three steps of the example are one chain per system: each step needs the previous
# step's outputs, so the dynamics / analysis tests skip until those exist.
TS_OPT_EXAMPLE = "run_mlip_ts_opt"
TS_OPT_YAMLS = {
    "inputs/ts/CPD_MVK_DA_ts_opt_vac.yaml": "CPD_MVK_DA",
    "inputs/ts/anthracene_DA_ts_opt_vac.yaml": "anthracene_DA",
    "inputs/ts/guajadial_DA_ts_opt_vac.yaml": "guajadial_DA",
    "inputs/ts/CPD_MVK_DA_ts_opt_in_solvent.yaml": "CPD_MVK_DA",
    "inputs/ts/anthracene_DA_ts_opt_in_solvent.yaml": "anthracene_DA",
    "inputs/ts/guajadial_DA_ts_opt_in_solvent.yaml": "guajadial_DA",
}

MD_EXAMPLE = "run_mlip_md"
MD_YAMLS = {
    "inputs/md/CPD_MVK_DA_dynamics_vac.yaml": "CPD_MVK_DA",
    "inputs/md/anthracene_DA_dynamics_vac.yaml": "anthracene_DA",
    "inputs/md/guajadial_DA_dynamics_vac.yaml": "guajadial_DA",
    "inputs/md/CPD_MVK_DA_dynamics_in_solvent.yaml": "CPD_MVK_DA",
    "inputs/md/anthracene_DA_dynamics_in_solvent.yaml": "anthracene_DA",
    "inputs/md/guajadial_DA_dynamics_in_solvent.yaml": "guajadial_DA",
}

TRJ_ANALYSIS_EXAMPLE = "run_trajectory_analysis"
TRJ_ANALYSIS_YAMLS = {
    "inputs/trj_analysis/CPD_MVK_DA_trj_analysis_vac.yaml": "CPD_MVK_DA",
    "inputs/trj_analysis/anthracene_DA_trj_analysis_vac.yaml": "anthracene_DA",
    "inputs/trj_analysis/guajadial_DA_trj_analysis_vac.yaml": "guajadial_DA",
    "inputs/trj_analysis/CPD_MVK_DA_trj_analysis_in_solvent.yaml": "CPD_MVK_DA",
    "inputs/trj_analysis/anthracene_DA_trj_analysis_in_solvent.yaml": "anthracene_DA",
    "inputs/trj_analysis/guajadial_DA_trj_analysis_in_solvent.yaml": "guajadial_DA",
}

# --- README example 4: pretraining-bias analysis ---------------------------------------
PRETRAIN_BIAS_EXAMPLE = "run_pretrain_bias_analysis"
PRETRAIN_BIAS_YAMLS = {
    "inputs/pretrain_bias/CPD_MVK_DA_pretrain_bias.yaml": "CPD_MVK_DA",
    "inputs/pretrain_bias/anthracene_DA_pretrain_bias.yaml": "anthracene_DA",
    "inputs/pretrain_bias/guajadial_DA_pretrain_bias.yaml": "guajadial_DA",
}

EXAMPLE_SCRIPTS = sorted({TEST_SET_EXAMPLE, IRC_EXAMPLE, TS_OPT_EXAMPLE, MD_EXAMPLE,
                          TRJ_ANALYSIS_EXAMPLE, PRETRAIN_BIAS_EXAMPLE})


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    """The examples use repo-root-relative paths."""
    monkeypatch.chdir(REPO_ROOT)


def _params(yamls: dict) -> list:
    """Parametrize over a YAML table, marking the solvent variants as slow."""
    return [pytest.param(yaml_rel, marks=pytest.mark.slow) if "in_solvent" in yaml_rel else yaml_rel
            for yaml_rel in sorted(yamls)]


def _run(script: str, yaml_rel: str, timeout: int = ANALYSIS_TIMEOUT) -> subprocess.CompletedProcess:
    """Run one example from the repository root, requiring it to exit cleanly.

    Prefers the documented ``rmlt <script>`` command, falling back to the module form
    (``python -m react_ml_test_public.scripts.<script>``) when ``rmlt`` is not on PATH.
    """
    rmlt = shutil.which("rmlt")
    if rmlt is not None:
        command = [rmlt, script.replace("_", "-"), yaml_rel]
    else:
        command = [sys.executable, "-m", f"react_ml_test_public.scripts.{script}", yaml_rel]

    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, (f"{' '.join(command)} exited with {result.returncode}:\n"
                                    f"{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
    return result


def _resolved(yaml_rel: str, resolver) -> dict:
    """Resolve an input YAML with the resolver its script uses."""
    return resolver(str(REPO_ROOT / yaml_rel))


def _require_mlip_stack() -> None:
    """The examples build MLIP calculators, which needs the mlptrain-mace environment."""
    try:
        importlib.import_module("mace.calculators")
    except ImportError as exc:
        pytest.skip(f"the MACE stack is not importable here ({exc}) - see the README setup")


def _model_fpath(model_name: str):
    """Path of a model in models/, searched recursively exactly as load_model_calculator does."""
    matches = sorted(MODELS_DIR.glob(f"**/{model_name}.model"))
    return matches[0] if matches else None


def _require_mlip_models(resolved: dict) -> None:
    """Check the MLIP stack, and that every model the example would load is present.

    ``model_names`` is enough for the test-set / IRC / TS-opt / MD examples; the
    pretraining-bias analysis also loads the fine-tuned and fully fine-tuned models it
    derives from the fine-tuning naming components.
    """
    _require_mlip_stack()

    model_names = list(resolved.get("model_names") or [])
    for key in ("ft_model_names", "full_ft_model_names", "rel_comp_model_names"):
        model_names += [name for name in (resolved.get(key) or []) if name]

    missing = [name for name in dict.fromkeys(model_names) if _model_fpath(name) is None]
    if missing:
        pytest.skip(f"model(s) not in models/: {', '.join(missing)} - "
                    f"copy the ORA models over (see the README examples note)")


def _require_data(system_dir: str) -> None:
    """The reaction data of a system (README: copy `data/` from the ORA repository)."""
    data_dir = DATA_DIR / "reaction_data" / system_dir
    if not data_dir.is_dir():
        pytest.skip(f"{data_dir.relative_to(REPO_ROOT)} is not present - "
                    f"copy the ORA data over (see the README examples note)")


def _require_paths(*fpaths) -> None:
    """Check files an example reads, given as repo-root-relative paths."""
    missing = [fpath for fpath in fpaths if fpath and not (REPO_ROOT / fpath).exists()]
    if missing:
        pytest.skip(f"missing input file(s): {', '.join(missing)} - "
                    f"copy the ORA data over (see the README examples note)")


def _opt_ts_stem(resolved: dict) -> str:
    """Stem of a model's optimised-TS file name, as run_mlip_md.py looks it up in OPT_DIR."""
    if resolved["include_solvent"]:
        return f"{resolved['system_name']}_in_{resolved['solvent_name']}"
    return f"{resolved['system_name']}_vac"


def _traj_stem(resolved: dict) -> str:
    """Stem of the MD trajectory names, as run_mlip_md.py builds them in TRAJ_DIR."""
    if resolved["include_solvent"]:
        return f"{resolved['system_name']}_{resolved['solvent_name']}_opt_TS"
    return f"{resolved['system_name']}_vac_opt_TS"


def _require_opt_ts_outputs(resolved: dict) -> None:
    """MD starts from each model's optimised TS, so the TS-optimisation step comes first."""
    prefix = _opt_ts_stem(resolved)
    missing = [model_name for model_name in resolved["model_names"]
               if not (REPO_ROOT / OPT_DIR / f"{prefix}_{model_name}_opt_TS.xyz").is_file()]
    if missing:
        pytest.skip(f"no optimised TS for {', '.join(missing)} in {OPT_DIR} - "
                    f"run the run-mlip-ts-opt example for this system first")


def _require_trajectories(resolved: dict) -> None:
    """The trajectory analysis reads MD trajectories, so the dynamics step comes first."""
    traj_dir = REPO_ROOT / resolved["traj_dir"]
    stubs = [f"{resolved['traj_type']}_{model_name}_{resolved['run_name']}"
             for model_name in resolved["model_names"]]
    if not any(list(traj_dir.glob(f"{stub}*")) for stub in stubs):
        pytest.skip(f"no MD trajectories matching {stubs[0]}* in {resolved['traj_dir']} - "
                    f"run the run-mlip-md example for this system first")


class TestExampleScripts:
    """The `rmlt` command itself: listing scripts, and how an example is invoked."""

    def test_rmlt_lists_every_example_script(self):
        """README: `rmlt list` - "List all available scripts with:"."""
        rmlt = shutil.which("rmlt")
        if rmlt is None:
            pytest.skip("the `rmlt` command is not on PATH - install the package (see the README setup)")

        result = subprocess.run([rmlt, "list"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stderr[-2000:]
        for script in EXAMPLE_SCRIPTS:
            assert script.replace("_", "-") in result.stdout

    @pytest.mark.parametrize("script", EXAMPLE_SCRIPTS)
    def test_scripts_take_the_input_yaml_as_an_argument(self, script):
        """README: scripts also run as `python -m react_ml_test_public.scripts.<script>`."""
        result = subprocess.run([sys.executable, "-m", f"react_ml_test_public.scripts.{script}", "--help"],
                                cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stderr[-2000:]
        assert "usage" in result.stdout.lower()


class TestExample1TestSetComparison:
    """README example 1: `rmlt run-test-set-comparison inputs/test/<system>_test.yaml`.

    Figures land in ``out/plt/bar_plots/`` and the RMSE / R2 table in ``out/csv/``.
    """

    @pytest.mark.parametrize("yaml_rel", sorted(TEST_SET_YAMLS))
    def test_runs_and_writes_the_plots_and_csv(self, yaml_rel):
        _require_data(TEST_SET_YAMLS[yaml_rel])
        resolved = _resolved(yaml_rel, rmlt_inputs.resolve_test_set_comparison_yaml)
        _require_mlip_models(resolved)

        _run(TEST_SET_EXAMPLE, yaml_rel)

        csv_fpath = REPO_ROOT / CSV_DIR / f"test_set_comparison_{resolved['test_name']}_{resolved['save_name']}.csv"
        assert csv_fpath.is_file(), f"no CSV written to {csv_fpath.relative_to(REPO_ROOT)}"

        plots = list((REPO_ROOT / PLOT_DIR / "bar_plots").glob(
            f"{resolved['save_name']}_{resolved['test_name']}*.png"))
        assert plots, f"no bar plots written to {PLOT_DIR}/bar_plots"


class TestExample2ReactionBarrier:
    """README example 2: `rmlt run-reaction-barrier-comparison inputs/irc/<system>_IRC_barrier.yaml`.

    Figures land in ``out/plt/irc/`` and the per-model table in ``out/csv/``.
    """

    @pytest.mark.parametrize("yaml_rel", sorted(IRC_YAMLS))
    def test_runs_and_writes_the_reaction_profile_and_csv(self, yaml_rel):
        _require_data(IRC_YAMLS[yaml_rel])
        resolved = _resolved(yaml_rel, rmlt_inputs.resolve_reaction_barrier_comparison_yaml)
        _require_paths(resolved["test_trj_fpath"])
        _require_mlip_models(resolved)

        _run(IRC_EXAMPLE, yaml_rel)

        # both the plot and CSV are named after the trajectory and the run's save name
        trj_name = get_name_from_path(resolved["test_trj_fpath"])
        plot_dir = REPO_ROOT / resolved["plot_dir"] / "irc"
        for plot in ("reaction_plot.png", "energy_errors.png"):
            plot_fpath = plot_dir / f"{trj_name}_{resolved['save_name']}_{plot}"
            assert plot_fpath.is_file(), f"no plot written to {plot_fpath.relative_to(REPO_ROOT)}"

        csv_fpath = REPO_ROOT / resolved["csv_dir"] / f"{trj_name}_{resolved['save_name']}.csv"
        assert csv_fpath.is_file(), f"no CSV written to {csv_fpath.relative_to(REPO_ROOT)}"


class TestExample3TsOptimisation:
    """README example 3.1: `rmlt run-mlip-ts-opt inputs/ts/<system>_ts_opt_{vac,in_solvent}.yaml`.

    "Output optimised TSs, optimisation trajectories, logs and vibrational modes will be
    stored in ``out/opt/``."
    """

    @pytest.mark.parametrize("yaml_rel", _params(TS_OPT_YAMLS))
    def test_runs_and_writes_the_optimised_ts_and_frequencies(self, yaml_rel):
        _require_data(TS_OPT_YAMLS[yaml_rel])
        resolved = _resolved(yaml_rel, rmlt_inputs.resolve_mlip_ts_opt_yaml)
        _require_paths(resolved["input_config_fp"], resolved["ref_config_fp"])
        if resolved["include_solvent"]:
            _require_paths(resolved["solvent_xyz_fpath"])
        _require_mlip_models(resolved)

        _run(TS_OPT_EXAMPLE, yaml_rel, timeout=DYNAMICS_TIMEOUT)

        out_dir = REPO_ROOT / resolved["out_dir"]
        # every run optimises the TS in vacuum first; the solvent variants then equilibrate
        # and optimise the solvated shell with that solvent held fixed
        prefixes = [f"{resolved['save_name']}_vac"]
        if resolved["include_solvent"]:
            prefixes.append(f"{resolved['save_name']}_in_{resolved['solvent_name']}")

        for model_name in resolved["model_names"]:
            for suffix in ("opt_TS.xyz", "dimer_traj.xyz", "dimer_opt_log.txt"):
                for prefix in prefixes:
                    out_fpath = out_dir / f"{prefix}_{model_name}_{suffix}"
                    assert out_fpath.is_file(), f"no output written to {out_fpath.relative_to(REPO_ROOT)}"

            # frequencies confirm the optimised TS (vacuum only)
            for suffix in ("vib.txt", "vib_mode_0.xyz"):
                out_fpath = out_dir / f"{resolved['save_name']}_vac_{model_name}_{suffix}"
                assert out_fpath.is_file(), f"no output written to {out_fpath.relative_to(REPO_ROOT)}"


class TestExample3Dynamics:
    """README example 3.2: `rmlt run-mlip-md inputs/md/<system>_dynamics_{vac,in_solvent}.yaml`.

    "This will take the output optimised TS in vacuum from 1. and run 10 1ps dynamics
    runs, with the trajectories output to ``out/trj/``."
    """

    pytestmark = pytest.mark.slow

    @pytest.mark.parametrize("yaml_rel", sorted(MD_YAMLS))
    def test_runs_and_writes_the_trajectories(self, yaml_rel):
        _require_data(MD_YAMLS[yaml_rel])
        resolved = _resolved(yaml_rel, rmlt_inputs.resolve_mlip_md_yaml)
        _require_mlip_models(resolved)
        _require_opt_ts_outputs(resolved)

        _run(MD_EXAMPLE, yaml_rel, timeout=DYNAMICS_TIMEOUT)

        # the trajectories are named after the formatted system / model labels, which the
        # script derives itself, so check the set rather than reconstructing each label
        traj_dir = REPO_ROOT / TRAJ_DIR
        trajectories = sorted(traj_dir.glob(
            f"{_traj_stem(resolved)}_*_{resolved['temp']}K_{resolved['fs'] / PS}ps_config_*_run_*_traj.xyz"))
        assert trajectories, f"no trajectories written to {TRAJ_DIR}"
        # a saved trajectory is not re-run, so at least one per model is what can be required
        assert len(trajectories) >= len(resolved["model_names"])


class TestExample3TrajectoryAnalysis:
    """README example 3.3: `rmlt run-trajectory-analysis inputs/trj_analysis/<system>_trj_analysis_*.yaml`.

    "Which will produce 2D trajectory plots like that in Figure 3d) in
    ``out/plt/trj_atom_dist/``, and print relevant RS / PS formation ratios."
    """

    @pytest.mark.parametrize("yaml_rel", sorted(TRJ_ANALYSIS_YAMLS))
    def test_runs_and_writes_the_trajectory_plots(self, yaml_rel):
        _require_data(TRJ_ANALYSIS_YAMLS[yaml_rel])
        resolved = _resolved(yaml_rel, rmlt_inputs.resolve_trajectory_analysis_yaml)
        _require_paths(resolved["ts_ref_fp"])
        _require_trajectories(resolved)

        result = _run(TRJ_ANALYSIS_EXAMPLE, yaml_rel)

        # the reaction-coordinate plot and its formed-product time-gap inset
        plot_dir = REPO_ROOT / PLOT_DIR / "trj_atom_dist"
        for plot in ("atom_distance_plot.png", "inset_time_plot.png"):
            plot_fpath = plot_dir / f"{resolved['save_name']}_{plot}"
            assert plot_fpath.is_file(), f"no plot written to {plot_fpath.relative_to(REPO_ROOT)}"

        # README: the RS / PS formation percentages are printed to the terminal
        assert "Product Formation Percentages" in result.stdout


class TestExample4PretrainBiasAnalysis:
    """README example 4: `rmlt run-pretrain-bias-analysis inputs/pretrain_bias/<system>_pretrain_bias.yaml`.

    "This will produce plots in ``out/plt/descriptors`` of the overall correlation scores,
    linear coefficients, and predicted error magnitudes across all models for each
    breakdown of the data."
    """

    @pytest.mark.parametrize("yaml_rel", sorted(PRETRAIN_BIAS_YAMLS))
    def test_runs_and_writes_the_descriptor_plots(self, yaml_rel):
        _require_data(PRETRAIN_BIAS_YAMLS[yaml_rel])
        resolved = _resolved(yaml_rel, rmlt_inputs.resolve_pretrain_bias_analysis_yaml)
        _require_paths(resolved["test_fpath"], *resolved["input_lot_test_fpaths"])
        _require_mlip_models(resolved)

        _run(PRETRAIN_BIAS_EXAMPLE, yaml_rel)

        plot_dir = REPO_ROOT / PLOT_DIR / "descriptors"
        test_name = resolved["test_name"]
        for plot in (f"{test_name}_model_lot_maes.png", f"{test_name}_model_abs_sims.png"):
            plot_fpath = plot_dir / plot
            assert plot_fpath.is_file(), f"no plot written to {plot_fpath.relative_to(REPO_ROOT)}"

        # Pearson's r / beta / beta*u across the subsets, for the descriptor-similarity and
        # the lower-level-of-theory error analysis: 3 metrics x 2 analyses
        analyses = list(plot_dir.glob(f"{test_name}_*_{resolved['subsets_name']}_*model_pred_analysis.png"))
        assert len(analyses) == 6, (
            f"expected 3 correlation plots per analysis in {PLOT_DIR}/descriptors, found {len(analyses)}")
