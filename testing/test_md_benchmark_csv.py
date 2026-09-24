"""
Unit tests for the MD benchmark CSV that run_mlip_md.py writes, one per MD run.

These cover:

  * the CSV layout (``dynamics.save_md_run_csv``): the device / calculator, the wall time
    of every step, and the summary values,
  * ``io.compute_device_label`` (the 'CPU' / GPU-name reporting),
  * ``dynamics.get_trajectory_with_ase`` recording one time per step taken, and
  * ``run_mlip_md.run_md_eval`` writing the CSV to ``<csv_dir>/md`` for a run.

The MD runs here use the cheap ASE EMT calculator instead of a MACE model, so nothing
needs a GPU, the MACE stack or any model files.

Run from the repository root:

    pytest testing/test_md_benchmark_csv.py
"""

import csv

import pytest

from react_ml_test_public.helpers import dynamics as rmlt_dynamics
from react_ml_test_public.helpers.io import compute_device_label
from react_ml_test_public.helpers.settings import MDSettings

# a short run: 5 steps of 0.5 fs
MD_CONFIG = MDSettings(fs=5, temp=300, dt=0.5, interval=1, md_method="ase")


def _read_pairs(csv_fpath):
    """Read a benchmark CSV into its rows, as lists of strings."""
    with open(csv_fpath, newline="") as csv_file:
        return list(csv.reader(csv_file))


def _summary(csv_fpath):
    """Read a benchmark CSV's summary values into a {name: value} dict."""
    rows = _read_pairs(csv_fpath)
    return {row[0]: row[1] for row in rows if len(row) == 2 and row[0] not in ("step",)}


@pytest.fixture
def emt_md(monkeypatch):
    """Run MD with the ASE EMT calculator, whichever model name is asked for."""
    from ase.calculators.emt import EMT

    monkeypatch.setattr(rmlt_dynamics, "load_model_calculator",
                        lambda model_name, **kwargs: EMT())


class TestSaveMdRunCsv:
    """The layout of a run's CSV, and the values it reports."""

    def test_layout(self, tmp_path):
        csv_fpath = tmp_path / "run.csv"
        rmlt_dynamics.save_md_run_csv(str(csv_fpath),
                                      step_times=[0.0154, 0.0112, 0.0110],
                                      dt=0.5,
                                      run_time_s=12.4,
                                      device="NVIDIA H100 80GB HBM3",
                                      calculator="ASE")

        assert _read_pairs(csv_fpath) == [
            ["device", "NVIDIA H100 80GB HBM3"],
            ["calculator", "ASE"],
            ["step", "step_time_s"],
            ["0", "0.015400"],
            ["1", "0.011200"],
            ["2", "0.011000"],
            ["mean_step_time_s", "0.012533"],
            ["total_steps", "3"],
            ["total_simulation_time_ps", "0.0015"],
            ["total_run_time_s", "12.400"],
        ]

    def test_simulated_time_comes_from_the_steps_actually_taken(self, tmp_path):
        """1000 steps of 0.5 fs is 0.5 ps - not the fs/PS figure the trajectory name uses."""
        csv_fpath = tmp_path / "run.csv"
        rmlt_dynamics.save_md_run_csv(str(csv_fpath),
                                      step_times=[0.01] * 1000,
                                      dt=0.5,
                                      run_time_s=10.0,
                                      device="CPU")

        summary = _summary(csv_fpath)
        assert summary["total_steps"] == "1000"
        assert summary["total_simulation_time_ps"] == "0.5000"

    def test_no_steps_is_reported_as_zero_not_an_error(self, tmp_path):
        csv_fpath = tmp_path / "run.csv"
        rmlt_dynamics.save_md_run_csv(str(csv_fpath),
                                      step_times=[],
                                      dt=0.5,
                                      run_time_s=1.0,
                                      device="CPU")

        summary = _summary(csv_fpath)
        assert summary["mean_step_time_s"] == "0.000000"
        assert summary["total_steps"] == "0"
        assert summary["total_simulation_time_ps"] == "0.0000"

    def test_values_are_parseable_and_the_step_list_is_contiguous(self, tmp_path):
        csv_fpath = tmp_path / "run.csv"
        step_times = [0.001 * step for step in range(1, 11)]
        rmlt_dynamics.save_md_run_csv(str(csv_fpath), step_times, dt=1.0,
                                      run_time_s=0.055, device="CPU")

        rows = _read_pairs(csv_fpath)
        header = rows.index(["step", "step_time_s"])
        steps = rows[header + 1:header + 1 + len(step_times)]

        assert [int(row[0]) for row in steps] == list(range(len(step_times)))
        assert [float(row[1]) for row in steps] == pytest.approx(step_times)


class TestComputeDeviceLabel:
    """The device is reported by name, so runs on different hardware are comparable."""

    def test_cpu(self):
        assert compute_device_label("cpu") == "CPU"

    def test_other_devices_are_passed_through(self):
        assert compute_device_label("mps") == "mps"

    def test_cuda_reports_the_gpu_model_name(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device on this machine")

        label = compute_device_label("cuda:0")

        assert label and label != "cuda:0"
        assert label == torch.cuda.get_device_name(0)


class TestStepTimesAreRecorded:
    """One time per step taken, from the ASE observer attached by the helper."""

    def test_one_time_per_step(self, tmp_path, emt_md):
        from ase.build import molecule

        step_times = []
        rmlt_dynamics.get_trajectory_with_ase(molecule("H2O"),
                                              "any-model-name",
                                              MD_CONFIG,
                                              "step_time_test",
                                              out_dir=str(tmp_path),
                                              cuda_device="cpu",
                                              step_times=step_times)

        # 5 steps, so 5 times - the observer's pre-first-step call is not one of them
        assert len(step_times) == MD_CONFIG.fs
        assert all(step_time > 0 for step_time in step_times)

    def test_a_reused_trajectory_records_nothing(self, tmp_path, emt_md):
        from ase.build import molecule

        # first run writes the trajectory...
        rmlt_dynamics.get_trajectory_with_ase(molecule("H2O"), "any-model-name",
                                              MD_CONFIG, "reused_test",
                                              out_dir=str(tmp_path), cuda_device="cpu")

        # ...so the second takes no steps and has nothing to report
        step_times = []
        rmlt_dynamics.get_trajectory_with_ase(molecule("H2O"), "any-model-name",
                                              MD_CONFIG, "reused_test",
                                              out_dir=str(tmp_path), cuda_device="cpu",
                                              step_times=step_times)

        assert step_times == []

    def test_no_step_times_list_is_optional(self, tmp_path, emt_md):
        """The helper's existing callers pass no list and still work."""
        from ase.build import molecule

        trajectory = rmlt_dynamics.get_trajectory_with_ase(
            molecule("H2O"), "any-model-name", MD_CONFIG, "optional_test",
            out_dir=str(tmp_path), cuda_device="cpu")

        assert len(trajectory) > 0


class TestRunMdEvalWritesCsv:
    """run_md_eval writes one CSV per run, under <csv_dir>/md."""

    def test_csv_is_written_next_to_the_trajectory(self, tmp_path, emt_md):
        from ase.build import molecule

        from react_ml_test_public.scripts.run_mlip_md import run_md_eval

        traj_dir = tmp_path / "trj"
        csv_dir = tmp_path / "csv"
        (csv_dir / "md").mkdir(parents=True)
        traj_dir.mkdir()

        run_md_eval([molecule("H2O")],
                    MD_CONFIG,
                    md_runs_per_config=1,
                    model_name="any-model-name",
                    formatted_model_name="model",
                    system_name="system",
                    traj_dir=str(traj_dir),
                    csv_dir=str(csv_dir),
                    cuda_device="cpu",
                    device_label="CPU")

        csv_fpath = csv_dir / "md" / f"system_model_{MD_CONFIG.temp}K_0.005ps_config_0_run_0.csv"
        assert csv_fpath.is_file()

        summary = _summary(csv_fpath)
        assert summary["device"] == "CPU"
        assert summary["calculator"] == "ASE"
        assert summary["total_steps"] == str(MD_CONFIG.fs)

    def test_a_skipped_run_writes_no_csv(self, tmp_path, emt_md):
        """A trajectory that already exists is reused, so there is nothing to benchmark."""
        from ase.build import molecule

        from react_ml_test_public.scripts.run_mlip_md import run_md_eval

        traj_dir = tmp_path / "trj"
        csv_dir = tmp_path / "csv"
        (csv_dir / "md").mkdir(parents=True)
        traj_dir.mkdir()

        kwargs = dict(md_runs_per_config=1,
                      model_name="any-model-name",
                      formatted_model_name="model",
                      system_name="system",
                      traj_dir=str(traj_dir),
                      csv_dir=str(csv_dir),
                      cuda_device="cpu",
                      device_label="CPU")
        run_md_eval([molecule("H2O")], MD_CONFIG, **kwargs)
        run_md_eval([molecule("H2O")], MD_CONFIG, **kwargs)   # second: traj already there

        assert len(list((csv_dir / "md").glob("*.csv"))) == 1
