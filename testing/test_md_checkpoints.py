"""
Unit tests for the MD checkpoints that get_trajectory_with_ase can save and resume from.

These cover:

  * the checkpoint interval conversion (``dynamics.checkpoint_interval_steps``: ps -> steps),
  * the checkpoint file itself (``dynamics.write_md_checkpoint`` / ``read_md_checkpoint``):
    what it records, and that what it records comes back,
  * resuming a run from a checkpoint (``dynamics.get_trajectory_with_ase``): that a resumed run
    carries the run that wrote the checkpoint on - the same trajectory, frame for frame -
    rather than starting a similar one,
  * the runs of ``run_mlip_md.run_md_eval``: a checkpoint per run, and a resume continuing one
    that already has a trajectory.

The MD runs here use the cheap ASE EMT calculator instead of a MACE model, so nothing needs a
GPU, the MACE stack or any model files.

Run from the repository root:

    pytest testing/test_md_checkpoints.py
"""

import csv
import os

import numpy as np
import pytest
import yaml
from ase.io import read

from react_ml_test_public.helpers import dynamics as rmlt_dynamics
from react_ml_test_public.helpers.settings import (
    DEFAULT_CHECKPOINT_EVERY_PS,
    PS,
    MDSettings,
)

# short runs, so the tests stay quick: 5 steps at 0.5 fs, a frame and a timing per step
MD_CONFIG = MDSettings(fs=5, temp=300, dt=0.5, interval=1, md_method="ase")

CHECKPOINT_NAME = "md_checkpoint_test_checkpoint.yaml"

# the seed a checkpointed run is created from, pinned so two runs of it are comparable
FIXED_SEED = 20260925


@pytest.fixture
def emt_md(monkeypatch):
    """Run MD with the ASE EMT calculator, whichever model name is asked for."""
    from ase.calculators.emt import EMT

    monkeypatch.setattr(rmlt_dynamics, "load_model_calculator",
                        lambda model_name, **kwargs: EMT())


@pytest.fixture
def fixed_seed(monkeypatch):
    """Pin the seed a checkpointed run is created from, so runs are comparable bit for bit.

    The helper seeds a checkpointed run from OS entropy (see ``get_trajectory_with_ase``);
    pinning it lets a run and a checkpointed-then-resumed run start from the very same random
    stream, which is what makes them comparable.
    """
    real_seed_sequence = np.random.SeedSequence

    def pinned_seed_sequence(*args, **kwargs):
        return real_seed_sequence(FIXED_SEED)

    monkeypatch.setattr(rmlt_dynamics.np.random, "SeedSequence", pinned_seed_sequence)


def _run(out_dir, fs=MD_CONFIG.fs, dt=MD_CONFIG.dt, output_name="md_checkpoint_test", **kwargs):
    """Run one MD trajectory with EMT, checkpointing to (and reading from) out_dir."""
    from ase.build import molecule

    # the trajectory and checkpoint directories are made by the script, not the helper
    os.makedirs(out_dir, exist_ok=True)

    md_config = MDSettings(fs=fs, temp=MD_CONFIG.temp, dt=dt,
                           interval=MD_CONFIG.interval, md_method="ase")
    return rmlt_dynamics.get_trajectory_with_ase(molecule("H2O"),
                                                 "any-model-name",
                                                 md_config,
                                                 output_name,
                                                 out_dir=str(out_dir),
                                                 cuda_device="cpu",
                                                 md_checkpoint_dir=str(out_dir),
                                                 **kwargs)


def _csv_summary(csv_fpath):
    """Read a benchmark CSV's summary values into a {name: value} dict."""
    with open(csv_fpath, newline="") as csv_file:
        rows = list(csv.reader(csv_file))
    return {row[0]: row[1] for row in rows if len(row) == 2 and not row[0].isdigit()}


class TestCheckpointIntervalSteps:
    """checkpoint_every is given in ps, while ASE's observers count steps."""

    def test_ps_are_converted_at_the_time_step_used(self):
        # one step advances the simulation by dt fs, so 10 ps at 0.5 fs/step is 20000 steps
        assert rmlt_dynamics.checkpoint_interval_steps(10, 0.5) == 20000
        assert rmlt_dynamics.checkpoint_interval_steps(10, 1.0) == 10000

    def test_the_default_is_ten_ps(self):
        assert DEFAULT_CHECKPOINT_EVERY_PS == 10
        assert (rmlt_dynamics.checkpoint_interval_steps(DEFAULT_CHECKPOINT_EVERY_PS, 0.5)
                == 20000)

    def test_an_interval_shorter_than_a_step_is_still_a_step(self):
        assert rmlt_dynamics.checkpoint_interval_steps(0.0001, 1.0) == 1

    @pytest.mark.parametrize("checkpoint_every", [0, -1])
    def test_a_non_positive_interval_is_rejected(self, checkpoint_every):
        with pytest.raises(ValueError, match="checkpoint_every"):
            rmlt_dynamics.checkpoint_interval_steps(checkpoint_every, 0.5)


class TestCheckpointFile:
    """What a checkpoint records, and that reading it back gives the same thing."""

    def test_nothing_is_written_unless_asked_for(self, tmp_path, emt_md):
        _run(tmp_path)

        assert list(tmp_path.glob("*_checkpoint.yaml")) == []

    def test_it_is_written_under_the_checkpoint_dir(self, tmp_path, emt_md):
        _run(tmp_path, save_md_checkpoints=True)

        assert (tmp_path / CHECKPOINT_NAME).is_file()

    def test_it_records_the_state_a_run_is_continued_from(self, tmp_path, emt_md):
        _run(tmp_path, save_md_checkpoints=True)
        checkpoint = yaml.safe_load((tmp_path / CHECKPOINT_NAME).read_text())

        # the run is over, so the checkpoint is its final state
        assert checkpoint["step"] == MD_CONFIG.fs
        assert checkpoint["total_steps"] == MD_CONFIG.fs
        assert checkpoint["timestep_fs"] == pytest.approx(MD_CONFIG.dt)
        assert checkpoint["temperature_K"] == pytest.approx(MD_CONFIG.temp)
        assert checkpoint["friction"] == pytest.approx(rmlt_dynamics.MD_FRICTION)
        assert checkpoint["model_name"] == "any-model-name"
        assert checkpoint["output_name"] == "md_checkpoint_test"

        # the properties calculated for that frame
        assert isinstance(checkpoint["energy_eV"], float)
        assert np.shape(checkpoint["forces_eV_per_A"]) == (3, 3)

        # and the RNG the run used and a resume continues
        assert isinstance(checkpoint["rng_seed"], int)
        assert checkpoint["rng_state"]["bit_generator"] == "PCG64"
        assert isinstance(checkpoint["rng_state"]["state"]["state"], int)

    def test_the_frame_is_the_final_configuration_with_its_momenta(self, tmp_path, emt_md):
        trajectory = _run(tmp_path, save_md_checkpoints=True)
        checkpoint = rmlt_dynamics.read_md_checkpoint(str(tmp_path / CHECKPOINT_NAME))

        frame = checkpoint["atoms"]
        final = trajectory[-1]
        assert frame.get_chemical_symbols() == final.get_chemical_symbols()
        # a trajectory frame is rounded to the 8 decimal places extxyz writes, the
        # checkpoint's own copy of the configuration is not
        assert np.allclose(frame.get_positions(), final.get_positions(), atol=1e-8)
        assert np.allclose(frame.get_momenta(), final.get_momenta(), atol=1e-8)

        assert np.array_equal(frame.get_positions(), np.array(checkpoint["positions"]))
        assert np.array_equal(frame.get_momenta(), np.array(checkpoint["momenta"]))

    def test_the_frame_is_an_xyz_frame(self, tmp_path, emt_md):
        """The frame is a real XYZ frame, so it can be looked at on its own."""
        from ase.io import read as ase_read
        import io

        _run(tmp_path, save_md_checkpoints=True)
        checkpoint = yaml.safe_load((tmp_path / CHECKPOINT_NAME).read_text())

        frame = ase_read(io.StringIO(checkpoint["frame"]), format="extxyz")
        assert len(frame) == 3
        assert "momenta" in frame.arrays

    def test_the_momenta_a_run_stopped_at_are_not_zero(self, tmp_path, emt_md):
        """Momenta are half the integration state, and an XYZ file need not carry them."""
        _run(tmp_path, save_md_checkpoints=True)
        checkpoint = rmlt_dynamics.read_md_checkpoint(str(tmp_path / CHECKPOINT_NAME))

        assert not np.allclose(checkpoint["momenta"], 0.0)
        assert np.array_equal(checkpoint["atoms"].get_momenta(),
                              np.array(checkpoint["momenta"]))

    def test_a_missing_checkpoint_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="checkpoint"):
            rmlt_dynamics.read_md_checkpoint(str(tmp_path / "not_there.yaml"))

    def test_a_yaml_that_is_not_a_checkpoint_is_rejected(self, tmp_path):
        fpath = tmp_path / "other.yaml"
        fpath.write_text("system_name: not_a_checkpoint\n")

        with pytest.raises(ValueError, match="not an MD checkpoint"):
            rmlt_dynamics.read_md_checkpoint(str(fpath))

    def test_an_older_layout_is_rejected_rather_than_guessed_at(self, tmp_path, emt_md):
        _run(tmp_path, save_md_checkpoints=True)
        fpath = tmp_path / CHECKPOINT_NAME
        checkpoint = yaml.safe_load(fpath.read_text())
        checkpoint["format_version"] = rmlt_dynamics.MD_CHECKPOINT_FORMAT_VERSION - 1
        fpath.write_text(yaml.safe_dump(checkpoint))

        with pytest.raises(ValueError, match="version"):
            rmlt_dynamics.read_md_checkpoint(str(fpath))


class TestResumingARun:
    """A resumed run is the same run carried on, not a similar one started again."""

    @staticmethod
    def _checkpointed_run(tmp_path, fs=MD_CONFIG.fs):
        """A checkpointed run of ``fs`` steps, and the checkpoint file it left."""
        trajectory = _run(tmp_path, fs=fs, save_md_checkpoints=True)
        return trajectory, str(tmp_path / CHECKPOINT_NAME)

    def test_it_continues_the_trajectory_the_checkpoint_came_from(self, tmp_path, emt_md):
        part, checkpoint_fp = self._checkpointed_run(tmp_path)

        # the same run, asked to go further
        resumed = _run(tmp_path, fs=8, load_md_checkpoint=checkpoint_fp)

        # the frames up to the checkpoint are still there, and the new ones follow them
        assert len(resumed) > len(part)
        for earlier, later in zip(part, resumed):
            assert np.array_equal(earlier.get_positions(), later.get_positions())

        # and the checkpoint now records the state the run reached, not the one it was given
        assert yaml.safe_load(open(checkpoint_fp))["step"] == 8

    def test_the_target_counts_the_whole_run(self, tmp_path, emt_md):
        """fs is the run's total, so a resume from step 5 of 8 takes 3 steps, not 8."""
        self._checkpointed_run(tmp_path)
        checkpoint_fp = str(tmp_path / CHECKPOINT_NAME)

        step_times = []
        resumed = _run(tmp_path, fs=8, load_md_checkpoint=checkpoint_fp, step_times=step_times)

        assert len(step_times) == 3
        # a 5-step run writes frames 0-5, and the resume continues them to frame 8
        assert len(resumed) == MD_CONFIG.fs + 1 + 3

    def test_a_resume_reproduces_an_uninterrupted_run(self, tmp_path, emt_md, fixed_seed):
        """The strongest check that there is no state outside the checkpoint.

        The same seed and the same starting configuration are used both for a run that goes
        straight to 8 steps and for a run that stops at 5 and is carried on. If the resume
        dropped the momenta, the thermostat, the step count or the RNG state - anything the
        checkpoint holds - the two trajectories would part company, and MD amplifies even a
        1e-8 difference. Both runs are checkpointed, since that is what gives them the seeded
        generator the seed pins (see save_md_checkpoints).
        """
        whole = _run(tmp_path / "whole", fs=8, save_md_checkpoints=True)

        self._checkpointed_run(tmp_path / "interrupted")
        checkpoint_fp = str(tmp_path / "interrupted" / CHECKPOINT_NAME)
        resumed = _run(tmp_path / "interrupted", fs=8, load_md_checkpoint=checkpoint_fp)

        assert len(whole) == len(resumed)
        for uninterrupted_frame, resumed_frame in zip(whole, resumed):
            assert np.array_equal(uninterrupted_frame.get_positions(),
                                  resumed_frame.get_positions())
            assert np.array_equal(uninterrupted_frame.get_momenta(),
                                  resumed_frame.get_momenta())

    def test_two_resumes_of_one_checkpoint_agree(self, tmp_path, emt_md):
        """Nothing but the checkpoint decides a resumed run - not even the global RNG.

        Each resume loads its own copy of the same checkpoint, since resuming keeps the
        checkpoint it came from up to date.
        """
        self._checkpointed_run(tmp_path)
        original = (tmp_path / CHECKPOINT_NAME).read_text()

        trajectories = []
        for name in ("first", "second"):
            checkpoint_fp = tmp_path / f"{name}.yaml"
            checkpoint_fp.write_text(original)
            trajectories.append(_run(tmp_path, fs=8, load_md_checkpoint=str(checkpoint_fp),
                                     output_name=name))

        assert np.array_equal(trajectories[0][-1].get_positions(),
                              trajectories[1][-1].get_positions())

    def test_a_checkpoint_at_the_target_runs_no_further_steps(self, tmp_path, emt_md):
        part, checkpoint_fp = self._checkpointed_run(tmp_path)

        step_times = []
        trajectory = _run(tmp_path, load_md_checkpoint=checkpoint_fp, step_times=step_times)

        assert step_times == []
        assert len(trajectory) == len(part)

    def test_a_resume_keeps_checkpointing_itself(self, tmp_path, emt_md):
        """A resume that is itself cut short has to be resumable, so it writes too."""
        _, checkpoint_fp = self._checkpointed_run(tmp_path)
        _run(tmp_path, fs=8, load_md_checkpoint=checkpoint_fp)

        assert yaml.safe_load(open(checkpoint_fp))["step"] == 8

    def test_a_missing_checkpoint_is_not_a_fresh_start(self, tmp_path, emt_md):
        with pytest.raises(FileNotFoundError, match="checkpoint"):
            _run(tmp_path, load_md_checkpoint=str(tmp_path / "not_there.yaml"))

    def test_it_starts_the_trajectory_when_there_is_none_left(self, tmp_path, emt_md):
        """Resuming from a checkpoint whose trajectory is gone still gives a whole one."""
        self._checkpointed_run(tmp_path)
        checkpoint_fp = str(tmp_path / CHECKPOINT_NAME)
        os.remove(tmp_path / "md_checkpoint_test_traj.xyz")

        resumed = _run(tmp_path, fs=8, load_md_checkpoint=checkpoint_fp)

        # the frame resumed from, plus the 3 steps to the new target
        assert len(resumed) == 1 + 3

    def test_the_csv_counts_a_resumed_run_at_the_checkpoints_time_step(self, tmp_path, emt_md):
        """The CSV's simulated time is steps * dt, with the dt the steps were taken with.

        A resume integrates at the checkpoint's time step rather than the input's (see
        get_trajectory_with_ase), so the benchmark of that stretch has to be timed at it too.
        """
        self._checkpointed_run(tmp_path)                             # 5 steps at dt 0.5
        checkpoint_fp = str(tmp_path / CHECKPOINT_NAME)

        csv_fpath = tmp_path / "resumed.csv"
        step_times = []
        _run(tmp_path, fs=8, dt=1.0, load_md_checkpoint=checkpoint_fp,
             step_times=step_times, csv_fpath=str(csv_fpath), csv_device="CPU")

        summary = _csv_summary(csv_fpath)
        assert summary["total_steps"] == str(len(step_times)) == "3"
        # 3 steps of the checkpoint's 0.5 fs, not of the 1.0 fs the input asks for
        assert summary["total_simulation_time_ps"] == f"{3 * MD_CONFIG.dt / PS:.4f}"

    def test_a_stretch_with_no_steps_to_take_writes_no_csv(self, tmp_path, emt_md):
        """There is nothing to benchmark, so no empty CSV is left behind."""
        self._checkpointed_run(tmp_path)

        csv_fpath = tmp_path / "nothing_to_do.csv"
        _run(tmp_path, load_md_checkpoint=str(tmp_path / CHECKPOINT_NAME), step_times=[],
             csv_fpath=str(csv_fpath), csv_device="CPU")

        assert not os.path.exists(csv_fpath)


class TestTheScriptsRuns:
    """run_md_eval checkpoints each of its runs, and resumes them."""

    TRAJ_NAME = "system_model_300K_0.005ps_config_0_run_0"

    @classmethod
    def _checkpoint_fp(cls, tmp_path):
        return tmp_path / f"{cls.TRAJ_NAME}_checkpoint.yaml"

    @classmethod
    def _trajectory_fp(cls, tmp_path):
        return tmp_path / "trj" / f"{cls.TRAJ_NAME}_traj.xyz"

    @staticmethod
    def _run_md_eval(tmp_path, **kwargs):
        from ase.build import molecule

        from react_ml_test_public.helpers.dynamics import run_md_eval

        traj_dir = tmp_path / "trj"
        traj_dir.mkdir(exist_ok=True)

        return run_md_eval([molecule("H2O")],
                           MD_CONFIG,
                           md_runs_per_config=1,
                           model_name="any-model-name",
                           formatted_model_name="model",
                           system_name="system",
                           traj_dir=str(traj_dir),
                           cuda_device="cpu",
                           md_checkpoint_dir=str(tmp_path),
                           **kwargs)

    def test_a_run_checkpoints_itself(self, tmp_path, emt_md):
        self._run_md_eval(tmp_path, save_md_checkpoints=True)

        assert self._checkpoint_fp(tmp_path).is_file()
        assert yaml.safe_load(self._checkpoint_fp(tmp_path).read_text())["step"] == MD_CONFIG.fs

    def test_a_resume_is_not_skipped_for_having_a_trajectory(self, tmp_path, emt_md):
        """A resume continues a trajectory that is already there, so it must not be reused."""
        self._run_md_eval(tmp_path, save_md_checkpoints=True)
        frames = len(read(str(self._trajectory_fp(tmp_path)), index=":"))

        trajectories = self._run_md_eval(tmp_path, load_md_checkpoint=True,
                                         save_md_checkpoints=True)

        # a skipped run returns nothing, so this is a run that resumed (and, its checkpoint
        # being at the run's target, had nothing left to take)
        assert trajectories
        assert len(read(str(self._trajectory_fp(tmp_path)), index=":")) == frames

    def test_a_resume_carries_the_run_on_from_its_checkpoint(self, tmp_path, emt_md):
        """With the trajectory gone, only the checkpoint can produce the resumed frames."""
        self._run_md_eval(tmp_path, save_md_checkpoints=True)
        os.remove(self._trajectory_fp(tmp_path))

        self._run_md_eval(tmp_path, load_md_checkpoint=True, save_md_checkpoints=True)

        # the frame resumed from and nothing else: a fresh start would have run all 5 steps
        assert len(read(str(self._trajectory_fp(tmp_path)), index=":")) == 1

    def test_a_resume_with_no_checkpoint_yet_starts_the_run(self, tmp_path, emt_md):
        """True resumes what there is to resume, and a first run has nothing to resume."""
        trajectories = self._run_md_eval(tmp_path, load_md_checkpoint=True,
                                         save_md_checkpoints=True)

        assert trajectories
        assert self._checkpoint_fp(tmp_path).is_file()

    def test_a_path_that_is_not_a_checkpoint_is_reported(self, tmp_path, emt_md):
        """An explicitly named checkpoint is used as given, so a wrong one is an error."""
        with pytest.raises(FileNotFoundError, match="checkpoint"):
            self._run_md_eval(tmp_path, load_md_checkpoint=str(tmp_path / "not_there.yaml"))
