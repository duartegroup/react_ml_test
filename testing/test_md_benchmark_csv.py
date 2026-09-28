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
from react_ml_test_public.helpers.settings import PS, MDSettings

# a short run: 5 steps of 0.5 fs
MD_CONFIG = MDSettings(fs=5, temp=300, dt=0.5, interval=1, md_method="ase")


def _md_config(fs=MD_CONFIG.fs, dt=MD_CONFIG.dt):
    """The settings of an MD run, for the CSV's ``md_config`` argument."""
    return MDSettings(fs=fs, temp=MD_CONFIG.temp, dt=dt, interval=MD_CONFIG.interval,
                      md_method="ase")


def _dying_calculator(calls):
    """An EMT calculator that raises after ``calls`` calculations, as a killed job stops.

    Counting calculator calls rather than steps is close enough: each MD step calculates once.
    """
    from ase.calculators.calculator import all_changes
    from ase.calculators.emt import EMT

    class _DyingEMT(EMT):
        def __init__(self):
            super().__init__()
            self.calls = calls

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            self.calls -= 1
            if self.calls < 0:
                raise RuntimeError("killed mid-run")
            return super().calculate(atoms, properties, system_changes)

    return _DyingEMT()


def _dying_loader(calls):
    """A model loader giving a calculator that dies after ``calls`` calculations."""
    return lambda model_name, **kwargs: _dying_calculator(calls)


def _finishing_loader():
    """A model loader giving a calculator that runs to the end, as the EMT fixture does."""
    from ase.calculators.emt import EMT

    return lambda model_name, **kwargs: EMT()


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
                                      md_config=_md_config(),
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
            # 3 steps of 0.5 fs, i.e. what the run simulated - not the fs/PS figure its
            # trajectory name carries
            ["total_simulation_time_ps", "0.0015"],
            ["total_run_time_s", "12.400"],
        ]

    def test_simulated_time_comes_from_the_steps_actually_taken(self, tmp_path):
        """1000 steps of 0.5 fs is 0.5 ps - not the fs/PS figure the trajectory name uses."""
        csv_fpath = tmp_path / "run.csv"
        rmlt_dynamics.save_md_run_csv(str(csv_fpath),
                                      step_times=[0.01] * 1000,
                                      md_config=_md_config(fs=1000),
                                      run_time_s=10.0,
                                      device="CPU")

        summary = _summary(csv_fpath)
        assert summary["total_steps"] == "1000"
        assert summary["total_simulation_time_ps"] == "0.5000"

    def test_no_steps_is_reported_as_zero_not_an_error(self, tmp_path):
        csv_fpath = tmp_path / "run.csv"
        rmlt_dynamics.save_md_run_csv(str(csv_fpath),
                                      step_times=[],
                                      md_config=MD_CONFIG,
                                      run_time_s=1.0,
                                      device="CPU")

        summary = _summary(csv_fpath)
        assert summary["mean_step_time_s"] == "0.000000"
        assert summary["total_steps"] == "0"
        assert summary["total_simulation_time_ps"] == "0.0000"

    def test_the_simulated_time_follows_the_steps_and_the_time_step(self, tmp_path):
        """The row is steps * dt, so a changed time step changes it and a changed fs does not."""
        def simulated_time_ps(step_count, md_config):
            csv_fpath = tmp_path / f"run_{step_count}_{md_config.dt}.csv"
            rmlt_dynamics.save_md_run_csv(str(csv_fpath),
                                          step_times=[0.01] * step_count,
                                          md_config=md_config,
                                          run_time_s=1.0,
                                          device="CPU")
            return _summary(csv_fpath)["total_simulation_time_ps"]

        assert simulated_time_ps(10, _md_config(dt=0.5)) == "0.0050"
        assert simulated_time_ps(10, _md_config(dt=1.0)) == "0.0100"
        # fs is not part of the calculation: 10 steps at 0.5 fs is 5 fs whatever the run's
        # target was, which is what makes the row the time actually simulated
        assert simulated_time_ps(10, _md_config(fs=10, dt=0.5)) == (
            simulated_time_ps(10, _md_config(fs=999, dt=0.5)))

    def test_values_are_parseable_and_the_step_list_is_contiguous(self, tmp_path):
        csv_fpath = tmp_path / "run.csv"
        step_times = [0.001 * step for step in range(1, 11)]
        rmlt_dynamics.save_md_run_csv(str(csv_fpath), step_times,
                                      md_config=_md_config(fs=len(step_times), dt=1.0),
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
    """run_md_eval writes one CSV per run, under <md_csv_dir>."""

    def test_csv_is_written_next_to_the_trajectory(self, tmp_path, emt_md):
        from ase.build import molecule

        from react_ml_test_public.helpers.dynamics import run_md_eval

        traj_dir = tmp_path / "trj"
        md_csv_dir = tmp_path / "csv" / "md"
        md_csv_dir.mkdir(parents=True)
        traj_dir.mkdir()

        run_md_eval([molecule("H2O")],
                    MD_CONFIG,
                    md_runs_per_config=1,
                    model_name="any-model-name",
                    formatted_model_name="model",
                    system_name="system",
                    traj_dir=str(traj_dir),
                    md_csv_dir=str(md_csv_dir),
                    cuda_device="cpu",
                    device_label="CPU")

        csv_fpath = md_csv_dir / f"system_model_{MD_CONFIG.temp}K_0.005ps_config_0_run_0.csv"
        assert csv_fpath.is_file()

        summary = _summary(csv_fpath)
        assert summary["device"] == "CPU"
        assert summary["calculator"] == "ASE"
        assert summary["total_steps"] == str(MD_CONFIG.fs)
        # the steps the run took (5) at the time step it took them with (0.5 fs), i.e. one
        # time step's worth per step - 0.0030 here would mean a step was counted twice
        assert summary["total_simulation_time_ps"] == f"{MD_CONFIG.fs * MD_CONFIG.dt / PS:.4f}"
        assert summary["total_simulation_time_ps"] == "0.0025"

    def test_a_skipped_run_writes_no_csv(self, tmp_path, emt_md):
        """A trajectory that already exists is reused, so there is nothing to benchmark."""
        from ase.build import molecule

        from react_ml_test_public.helpers.dynamics import run_md_eval

        traj_dir = tmp_path / "trj"
        md_csv_dir = tmp_path / "csv" / "md"
        md_csv_dir.mkdir(parents=True)
        traj_dir.mkdir()

        kwargs = dict(md_runs_per_config=1,
                      model_name="any-model-name",
                      formatted_model_name="model",
                      system_name="system",
                      traj_dir=str(traj_dir),
                      md_csv_dir=str(md_csv_dir),
                      cuda_device="cpu",
                      device_label="CPU")
        run_md_eval([molecule("H2O")], MD_CONFIG, **kwargs)
        run_md_eval([molecule("H2O")], MD_CONFIG, **kwargs)   # second: traj already there

        assert len(list(md_csv_dir.glob("*.csv"))) == 1


class TestEachStretchKeepsItsOwnCsv:
    """The CSVs of a run that is resumed do not overwrite each other (see _md_csv_fpath)."""

    TRAJ_NAME = f"system_model_{MD_CONFIG.temp}K_0.005ps_config_0_run_0"

    @classmethod
    def _csv_fpaths(cls, tmp_path):
        """The run's CSVs, by name, in the order the stretches wrote them."""
        md_csv_dir = tmp_path / "csv" / "md"
        return sorted(md_csv_dir.glob("*.csv"))

    def _run_stretch(self, tmp_path, load_md_checkpoint=False):
        """Run one stretch of the run, checkpointing and benchmarking it."""
        from ase.build import molecule

        from react_ml_test_public.helpers.dynamics import run_md_eval

        traj_dir = tmp_path / "trj"
        md_csv_dir = tmp_path / "csv" / "md"
        traj_dir.mkdir(exist_ok=True)
        md_csv_dir.mkdir(parents=True, exist_ok=True)

        return run_md_eval([molecule("H2O")],
                           MD_CONFIG,
                           md_runs_per_config=1,
                           model_name="any-model-name",
                           formatted_model_name="model",
                           system_name="system",
                           traj_dir=str(traj_dir),
                           md_csv_dir=str(md_csv_dir),
                           cuda_device="cpu",
                           device_label="CPU",
                           save_md_checkpoints=True,
                           md_checkpoint_dir=str(tmp_path),
                           # a one-step cadence, so a stretch that dies has written a CSV
                           checkpoint_every=0.0005,
                           load_md_checkpoint=load_md_checkpoint)

    def _kill_a_stretch(self, tmp_path, monkeypatch, load_md_checkpoint):
        """Run one stretch that dies part way through, as a job that hit its time limit does."""
        monkeypatch.setattr(rmlt_dynamics, "load_model_calculator", _dying_loader(calls=3))

        with pytest.raises(RuntimeError, match="killed mid-run"):
            self._run_stretch(tmp_path, load_md_checkpoint=load_md_checkpoint)

    def test_a_first_stretch_writes_the_runs_own_name(self, tmp_path, emt_md):
        self._run_stretch(tmp_path)

        assert [f.name for f in self._csv_fpaths(tmp_path)] == [f"{self.TRAJ_NAME}.csv"]

    def test_each_resume_writes_its_own_numbered_csv(self, tmp_path, monkeypatch, emt_md):
        """A run killed twice and then carried to the end keeps all three stretches' CSVs."""
        self._kill_a_stretch(tmp_path, monkeypatch, load_md_checkpoint=False)
        self._kill_a_stretch(tmp_path, monkeypatch, load_md_checkpoint=True)
        monkeypatch.setattr(rmlt_dynamics, "load_model_calculator", _finishing_loader())
        self._run_stretch(tmp_path, load_md_checkpoint=True)

        assert [f.name for f in self._csv_fpaths(tmp_path)] == [
            f"{self.TRAJ_NAME}.csv",
            f"{self.TRAJ_NAME}_contd_1.csv",
            f"{self.TRAJ_NAME}_contd_2.csv",
        ]

    def test_each_csv_holds_the_steps_of_its_own_stretch(self, tmp_path, monkeypatch, emt_md):
        """Every stretch accounts for the steps it took, and they add up to the run."""
        self._kill_a_stretch(tmp_path, monkeypatch, load_md_checkpoint=False)
        self._kill_a_stretch(tmp_path, monkeypatch, load_md_checkpoint=True)
        monkeypatch.setattr(rmlt_dynamics, "load_model_calculator", _finishing_loader())
        self._run_stretch(tmp_path, load_md_checkpoint=True)

        steps = [int(_summary(fpath)["total_steps"]) for fpath in self._csv_fpaths(tmp_path)]
        assert all(step > 0 for step in steps)
        assert sum(steps) == MD_CONFIG.fs

    def test_the_original_csv_is_left_as_it_was(self, tmp_path, monkeypatch, emt_md):
        """A resumed stretch does not touch the CSV of the stretch before it."""
        self._kill_a_stretch(tmp_path, monkeypatch, load_md_checkpoint=False)
        original = self._csv_fpaths(tmp_path)[0]
        before = original.read_text()

        monkeypatch.setattr(rmlt_dynamics, "load_model_calculator", _finishing_loader())
        self._run_stretch(tmp_path, load_md_checkpoint=True)

        assert original.read_text() == before

    def test_a_stretch_that_is_killed_keeps_what_it_measured(self, tmp_path, monkeypatch, emt_md):
        """The CSV is refreshed during the run, so a killed stretch does not lose its timings.

        Without that, the only CSV a run could leave is the one written after its last step,
        and a job that hit its time limit - the whole reason for checkpointing - would leave
        no benchmark of the steps it did run.
        """
        self._kill_a_stretch(tmp_path, monkeypatch, load_md_checkpoint=False)

        csv_fpaths = self._csv_fpaths(tmp_path)
        assert [f.name for f in csv_fpaths] == [f"{self.TRAJ_NAME}.csv"]
        # some of the run, but not all of it: the steps before the kill, not the five it aimed at
        measured = int(_summary(csv_fpaths[0])["total_steps"])
        assert 0 < measured < MD_CONFIG.fs
