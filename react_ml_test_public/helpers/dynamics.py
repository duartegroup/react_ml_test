"""
Helpers for running dynamics with MLIPs.
"""
import csv
import io
import os
from time import time

import ase
import numpy as np
import yaml
from ase import units
from ase.io import read
from ase.md.langevin import Langevin

from react_ml_test_public.helpers.io import compute_device_label, load_model_calculator
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.settings import (
    DEFAULT_CHECKPOINT_EVERY_PS,
    MD_CHECKPOINT_DIR,
    PS,
    TRAJ_DIR,
    MDSettings,
)

# the Langevin friction every MD run uses, taken from mlptrain's standard value. Runs
# started from a checkpoint continue with the friction stored in it (see
# write_md_checkpoint), so this only sets a fresh run's thermostat.
MD_FRICTION = 0.02

# the MD engine the runs are driven with, i.e. get_trajectory_with_ase, which is ASE. Only ASE
# is wired up, so this is also what the benchmark CSVs record as the calculator.
MD_CALCULATOR = 'ASE'

# layout version of the checkpoint YAML that write_md_checkpoint writes and
# read_md_checkpoint understands. Bump it when a key changes meaning, so that an old
# checkpoint is rejected instead of silently resumed from the wrong state.
MD_CHECKPOINT_FORMAT_VERSION = 1


def checkpoint_interval_steps(checkpoint_every: float, dt: float) -> int:
    """
    Convert a checkpoint interval in ps into a whole number of MD steps.

    One ASE step advances the simulation by ``dt`` fs and there are ``PS`` fs in a ps, so
    ``checkpoint_every`` ps is ``checkpoint_every * PS / dt`` steps - e.g. the default 10 ps
    at a 0.5 fs time step is 20000 steps.

    :param checkpoint_every: interval between checkpoint writes, in ps.
    :param dt: MD integration time step, in fs.
    :return: the interval in MD steps, at least one.
    :raises ValueError: if ``checkpoint_every`` is not positive.
    """
    if checkpoint_every <= 0:
        raise ValueError(f"checkpoint_every must be a positive number of ps, "
                         f"got {checkpoint_every!r}")

    # never 0: an interval shorter than one step would otherwise check point only at the end
    # of the run, which is not what asking for regular checkpoints means
    return max(1, round(checkpoint_every * PS / dt))


def md_checkpoint_fpath(output_name: str, md_checkpoint_dir: str = None) -> str:
    """
    Path of an MD run's checkpoint file: ``<md_checkpoint_dir>/<output_name>_checkpoint.yaml``.

    One file per run, overwritten in place on every write (see write_md_checkpoint).

    :param output_name: name of the MD run, as passed to get_trajectory_with_ase.
    :param md_checkpoint_dir: directory to write checkpoints to, or None for
                              ``MD_CHECKPOINT_DIR`` (i.e. ``out/md_checkpoints``).
    :return: the checkpoint file path.
    """
    md_checkpoint_dir = MD_CHECKPOINT_DIR if md_checkpoint_dir is None else md_checkpoint_dir
    return f"{md_checkpoint_dir}/{output_name}_checkpoint.yaml"


def write_md_checkpoint(checkpoint_fpath: str,
                        dynamics: Langevin,
                        total_steps: int,
                        output_name: str = None,
                        model_name: str = None,
                        rng_seed: int = None) -> str:
    """
    Write the state of an ASE Langevin dynamics run to a YAML checkpoint.

    The checkpoint holds what is needed to carry the run on rather than start it again:

      * ``timestep_fs``, ``temperature_K`` and ``friction`` - the integration time step and
        the thermostat state,
      * ``step`` - how many steps the run has taken, and ``total_steps`` - how many it targets,
      * ``rng_seed`` / ``rng_state`` - the seed the run's random number generator was created
        from, and the state of its bit generator, which is what makes a resumed run continue
        the same random stream rather than a new one,
      * ``frame`` - the current configuration as an extxyz frame, i.e. the positions, the cell
        and periodic boundary conditions, the momenta the propagator integrates, and the
        energy / forces calculated at those positions,
      * ``positions`` / ``momenta`` - that same geometry as exact numbers, which is what a
        resume restores: the frame is rounded to 8 decimal places as text, and continuing
        from rounded coordinates would be continuing a different trajectory,
      * ``energy_eV`` and ``forces_eV_per_A`` - the calculated properties on their own, so they
        can be read without parsing the frame. A resumed run recalculates them rather than
        reusing them (its calculator is a new object), so they are recorded for inspection.

    The file is written whole (not appended to), so one file tracks one run's latest state.

    :param checkpoint_fpath: path of the YAML file to write (its directory is created).
    :param dynamics: the Langevin object to checkpoint.
    :param total_steps: number of steps the whole run targets (``MDSettings.fs``).
    :param output_name: name of the MD run, recorded for provenance.
    :param model_name: model the run uses, recorded for provenance.
    :param rng_seed: seed ``dynamics.rng`` was created from, recorded for provenance.
    :return: the path written.
    """
    atoms = dynamics.atoms

    # an extxyz frame carries positions, cell, pbc and momenta (the calculated energy /
    # forces ride along too, since the atoms hold them), so a resume restores the whole
    # integration state rather than the geometry alone
    frame = io.StringIO()
    atoms.write(frame, format='extxyz')

    checkpoint = {
        'format_version': MD_CHECKPOINT_FORMAT_VERSION,
        'output_name': output_name,
        'model_name': model_name,
        'step': int(dynamics.nsteps),
        'total_steps': int(total_steps),
        # ASE keeps the time step in ASE time units and the temperature in eV, so both are
        # converted back to the fs / K they were set in
        'timestep_fs': dynamics.dt / units.fs,
        'temperature_K': dynamics.temp / units.kB,
        'friction': dynamics.fr,
        'rng_seed': rng_seed,
        'rng_state': _rng_state(dynamics.rng),
        'energy_eV': float(atoms.get_potential_energy()),
        'forces_eV_per_A': atoms.get_forces().tolist(),
        # the exact integration state. The frame below holds the same configuration, but as
        # text it is rounded to 8 decimal places (ASE's extxyz format), and continuing a
        # chaotic trajectory from rounded coordinates is a different trajectory
        'positions': atoms.get_positions().tolist(),
        'momenta': atoms.get_momenta().tolist(),
        'frame': frame.getvalue(),
    }

    checkpoint_dir = os.path.dirname(checkpoint_fpath)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    with open(checkpoint_fpath, 'w') as checkpoint_file:
        # never flow-style: a frame's forces are a list of lists, and reading them back as
        # YAML mappings of coordinates would be a different object entirely
        yaml.safe_dump(checkpoint, checkpoint_file, sort_keys=False,
                       default_flow_style=False)

    logger.info(f"Saved MD checkpoint at step {checkpoint['step']} to {checkpoint_fpath}")

    return checkpoint_fpath


def read_md_checkpoint(checkpoint_fpath: str) -> dict:
    """
    Read an MD checkpoint written by ``write_md_checkpoint``.

    The returned dict is the checkpoint's YAML with the ``frame`` parsed into an ``ase.Atoms``
    object under the ``'atoms'`` key - i.e. the configuration the run stopped at, momenta
    included, at the full precision the checkpoint recorded rather than the frame's 8 decimal
    places.

    :param checkpoint_fpath: path of the checkpoint YAML.
    :return: the checkpoint as a dict, holding the keys ``write_md_checkpoint`` documents.
    :raises FileNotFoundError: if there is no such file.
    :raises ValueError: if the file is not a checkpoint that can be resumed from.
    """
    if not os.path.isfile(checkpoint_fpath):
        raise FileNotFoundError(
            f"MD checkpoint not found: {checkpoint_fpath!r} - a run is only resumable from a "
            f"checkpoint saved with save_md_checkpoints = True (see write_md_checkpoint)."
        )

    with open(checkpoint_fpath, 'r') as checkpoint_file:
        checkpoint = yaml.safe_load(checkpoint_file)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"{checkpoint_fpath!r} is not an MD checkpoint (expected a YAML "
                         f"mapping, found {type(checkpoint).__name__}).")

    # the keys first, so that a YAML that simply is not a checkpoint is reported as one,
    # rather than as a checkpoint of an unrecognised version
    missing = [key for key in ('step', 'timestep_fs', 'temperature_K', 'friction', 'frame',
                               'positions', 'momenta')
               if checkpoint.get(key) is None]
    if missing:
        raise ValueError(f"{checkpoint_fpath!r} is not an MD checkpoint: it has no {missing}.")

    version = checkpoint.get('format_version')
    if version != MD_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"{checkpoint_fpath!r} is a version {version!r} MD checkpoint, but this code "
            f"writes and reads version {MD_CHECKPOINT_FORMAT_VERSION} - re-run the "
            f"trajectory rather than resume it."
        )

    atoms = read(io.StringIO(checkpoint['frame']), format='extxyz')
    # the frame carries the cell / pbc / symbols (written at full precision), but its
    # coordinates are rounded, so the exact ones take over
    atoms.set_positions(checkpoint['positions'])
    atoms.set_momenta(checkpoint['momenta'])
    checkpoint['atoms'] = atoms

    return checkpoint


def _rng_state(rng) -> dict:
    """
    The YAML-safe state of a run's random number generator, i.e. of its bit generator.

    Only a ``numpy.random.Generator`` - what a checkpointed run is seeded with - has a state
    that can be restored exactly. ASE's default is the legacy global ``numpy.random``, whose
    state a checkpoint cannot own; that case is warned about and recorded as no state at all,
    so a resume restarts the random stream instead of restoring a wrong one.

    :param rng: the ``Langevin.rng`` object to record.
    :return: the bit generator's own state dict (which names its type, e.g. ``PCG64``), or
             None if the generator has no usable state.
    """
    bit_generator = getattr(rng, 'bit_generator', None)
    if bit_generator is None:
        logger.warning(f"Cannot save the state of {rng!r} (not a numpy Generator), so a "
                       f"resumed run will not continue the same random stream.")
        return None

    return bit_generator.state


def _rng_from_state(rng_state: dict):
    """
    Rebuild the random number generator a checkpoint recorded.

    :param rng_state: the ``rng_state`` value of a checkpoint, i.e. what ``_rng_state`` wrote.
    :return: a ``numpy.random.Generator`` continuing the checkpointed run's stream, or None if
             the checkpoint recorded no state (Langevin then uses ``numpy.random``).
    """
    if not rng_state:
        return None

    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state

    return rng



def get_trajectory_with_ase(init_config: ase.Atoms,
                            model_name: str,
                            md_config: MDSettings,
                            output_name: str,
                            out_dir: str = TRAJ_DIR,
                            constraint: str = None, # IndexedConstraint = None,
                            cuda_device: str = "cuda:0",  # falls back to 'cpu' if unavailable
                            override_cache: bool = False,
                            model_dir: str = None,
                            enable_cueq: bool = False,
                            step_times: list = None,
                            fair_chem_task: str = None,
                            fair_chem_setting: str = None,
                            save_md_checkpoints: bool = False,
                            load_md_checkpoint: str = None,
                            checkpoint_every: float = DEFAULT_CHECKPOINT_EVERY_PS,
                            md_checkpoint_dir: str = None,
                            csv_fpath: str = None,
                            csv_device: str = None,
                            csv_calculator: str = 'ASE'):
    """

    :param init_config_fp:
    :param model_fp:
    :param output_name:
    :param enable_cueq: use the cuEquivariance MACE speed-up for the calculator (needs a CUDA
                        GPU and mace-torch's cueq extras).
    :param fair_chem_task: FairChem task to run a fairchem model as, e.g. 'omol' for molecules
                           (see load_model_calculator).
    :param fair_chem_setting: FairChem inference settings, e.g. 'turbo' (see
                              load_model_calculator).
    :param step_times: optional list to append the wall time (s) of every MD step to, for
                       benchmarking (see save_md_run_csv). A run that returns an existing
                       trajectory without taking any steps leaves it empty.
    :param save_md_checkpoints: write this run's state to a YAML checkpoint under
                               ``md_checkpoint_dir``, every ``checkpoint_every`` ps and once
                               more at the end of the run, so an interrupted run can be
                               resumed. Checkpoints are overwritten in place, one file per
                               run (see md_checkpoint_fpath / write_md_checkpoint).
    :param load_md_checkpoint: path of a checkpoint to carry a previous run on from, instead
                               of starting this one from ``init_config`` - the checkpoint's
                               frame, thermostat and RNG state are used, and the run takes
                               the steps still missing from ``md_config.fs`` (which counts
                               the steps of the whole run, not of this stretch). The existing
                               trajectory is continued rather than overwritten, the checkpoint
                               is kept up to date in place (whether or not
                               ``save_md_checkpoints`` is set, so a resume is itself
                               resumable), and a missing/unreadable checkpoint is an error
                               rather than a fresh start. ``init_config`` and ``constraint``
                               still come from this call: the checkpoint does not record
                               constraints.
    :param checkpoint_every: interval between checkpoint writes, in ps of simulated time
                             (default 10 ps). One step advances the simulation by
                             ``md_config.dt`` fs, so this is converted to steps by
                             ``checkpoint_interval_steps``. Used whenever the run
                             checkpoints itself, i.e. with ``save_md_checkpoints`` or a
                             resume, and as the refresh interval of the benchmark CSV.
    :param md_checkpoint_dir: directory to write checkpoints to, or None for
                              ``MD_CHECKPOINT_DIR`` (i.e. ``out/md_checkpoints``).
    :param csv_fpath: optional path of a benchmark CSV to write this stretch's step wall
                      times to (see save_md_run_csv), refreshed every ``checkpoint_every`` ps
                      and once more at the end of the stretch - so a stretch that is killed
                      keeps what it measured instead of losing it with the process. It is
                      written from ``step_times``, so that list has to be given too, and a
                      stretch that takes no steps writes no CSV. The caller names the file
                      (see _md_csv_fpath, which gives each resumed stretch its own).
    :param csv_device: compute device to record in that CSV, e.g. 'CPU' (see
                       compute_device_label).
    :param csv_calculator: MD calculator/engine to record in that CSV.
    :return:
    """
    traj_fp = f"{out_dir}/{output_name}_traj.xyz"
    checkpoint_fp = md_checkpoint_fpath(output_name, md_checkpoint_dir)

    # the benchmark CSV reports the wall time of the run as a whole, loading the model
    # included, so its clock starts here rather than at its first write
    started = time()

    # a resumed run continues the trajectory a checkpoint came from, so the cache check (and
    # the re-write it can trigger) does not apply to it
    resuming = load_md_checkpoint is not None
    if resuming and override_cache:
        logger.warning(f"Resuming from {load_md_checkpoint} and asked to overwrite cached "
                       f"outputs: keeping {traj_fp}, which holds the frames up to the "
                       f"checkpoint.")

    # re-write trajectory file if it doesn't exist already
    if os.path.exists(traj_fp) and not resuming:
        if override_cache:
            logger.info(f"Overwriting {traj_fp}...")
            os.remove(traj_fp)
        else:
            logger.info(f"Found {traj_fp}, returning existing trj...")
            return read(traj_fp, index=":")

    # unpack MDConfig
    fs = md_config.fs
    temp = md_config.temp
    dt = md_config.dt
    interval = md_config.interval

    # the thermostat a fresh run starts from; a resumed one continues the checkpoint's, since
    # a run carried on at another temperature or friction is not the run that was checkpointed
    friction = MD_FRICTION
    steps_done = 0
    rng = None
    rng_seed = None

    if resuming:
        checkpoint = read_md_checkpoint(load_md_checkpoint)
        # the frame the run stopped at replaces the caller's starting config
        init_config = checkpoint['atoms']
        steps_done = int(checkpoint['step'])
        dt = float(checkpoint['timestep_fs'])
        temp = float(checkpoint['temperature_K'])
        friction = float(checkpoint['friction'])
        rng = _rng_from_state(checkpoint.get('rng_state'))
        rng_seed = checkpoint.get('rng_seed')
        # a resumed run keeps the checkpoint it came from up to date, so a resume that is
        # itself interrupted is resumable from where it got to
        checkpoint_fp = load_md_checkpoint

        # these are the values the resumed stretch runs with; report the YAML disagreeing
        # rather than mixing the two silently
        if dt != md_config.dt:
            logger.warning(f"Resuming {output_name} with the checkpoint's time step "
                           f"{dt} fs, not the {md_config.dt} fs the input asks for.")
        if temp != md_config.temp:
            logger.warning(f"Resuming {output_name} at the checkpoint's {temp} K, not the "
                           f"{md_config.temp} K the input asks for.")
        if checkpoint.get('total_steps') is not None and int(checkpoint['total_steps']) != fs:
            logger.warning(f"Resuming {output_name} from step {steps_done} of "
                           f"{checkpoint['total_steps']} to a new target of {fs} steps.")
        if checkpoint.get('model_name') not in (None, model_name):
            logger.warning(f"Resuming {output_name} from a checkpoint of "
                           f"{checkpoint['model_name']!r} with {model_name!r}.")

    elif save_md_checkpoints:
        # seed the run explicitly, so the checkpoint's seed reproduces it and its saved state
        # carries the random stream on to the resumed stretch
        rng_seed = int(np.random.SeedSequence().entropy)
        rng = np.random.default_rng(rng_seed)

    # load calculator (model_dir None -> MODEL_DIR, see resolve_model_dir; the last two
    # settings are FairChem's, and ignored by the other loaders)
    # the settings the run really integrates with: a resume takes the checkpoint's time step,
    # and the benchmark CSV's simulated time is the steps at the time step they were taken
    # with, so it has to be given those rather than the ones the caller asked for
    run_md_config = md_config._replace(dt=dt)

    calc = load_model_calculator(model_name, device=cuda_device, dtype='float32',
                                 model_dir=model_dir, enable_cueq=enable_cueq,
                                 fair_chem_task=fair_chem_task,
                                 fair_chem_setting=fair_chem_setting)

    # make copy of init config
    atoms = init_config if resuming else init_config.copy()
    if constraint is not None:
        atoms.set_constraint(constraint)

    # 1. attach the calculator to the Atoms object
    atoms.set_calculator(calc)

    # 2. create the dynamics object (rng None -> numpy.random, ASE's own default)
    dynamics = Langevin(
        atoms,
        timestep=dt * units.fs,
        temperature_K=temp,
        friction=friction, #/ units.fs # currently using mlptrain standard, ,
        rng=rng,
    )

    if resuming:
        # continue the run's own step count, so the interval-based observers below keep
        # firing on its schedule rather than from zero again
        dynamics.nsteps = steps_done

    # 3. time every step, for the benchmark CSV (attached before the writer below, so a
    # step's time is the step itself plus the trajectory write of the previous one)
    if step_times is not None:
        last_call = time()
        stepping = False

        def _record_step_time():
            nonlocal last_call, stepping
            call_time = time()
            # ASE calls its observers once before the first step (writing the initial
            # frame), so the first call only sets the baseline - every call after it
            # closes one step
            if stepping:
                step_times.append(call_time - last_call)
            stepping = True
            last_call = call_time

        dynamics.attach(_record_step_time, interval=1)

        if resuming:
            # a resumed run's nsteps is not 0, so ASE skips that pre-step call and the first
            # resumed step would go untimed - make it here
            _record_step_time()

    # 4. attach a callback to periodically write the trajectory to a file
    dynamics.attach(
        lambda: atoms.write(traj_fp, format='extxyz', append=True),
        interval=interval,
    )

    # 5. attach a callback to periodically checkpoint the run, and, like the trajectory, one
    # more write of the final state once the run is over (below) - a run that is killed
    # between two intervals is then only ever this many steps out of date. A resumed run
    # writes too, whatever save_md_checkpoints says, since it was started from a checkpoint.
    if save_md_checkpoints or resuming:
        dynamics.attach(
            lambda: write_md_checkpoint(checkpoint_fp, dynamics, total_steps=fs,
                                        output_name=output_name, model_name=model_name,
                                        rng_seed=rng_seed),
            interval=checkpoint_interval_steps(checkpoint_every, dt),
        )

    def _write_step_csv():
        """Write the benchmark CSV of the steps taken so far, if there are any to write."""
        if step_times:
            save_md_run_csv(csv_fpath, step_times, run_md_config, time() - started,
                            device=csv_device, calculator=csv_calculator)

    # 6. refresh that CSV on the same cadence as the checkpoint, so a stretch of run that is
    # killed keeps the timings it managed (the end of this function writes the final ones).
    # Nothing is written before the first step, or by a stretch that takes none.
    if csv_fpath is not None and step_times is not None:
        dynamics.attach(_write_step_csv,
                        interval=checkpoint_interval_steps(checkpoint_every, dt))

    # the frames a checkpoint came from live in the trajectory file; if that file is gone,
    # start it at the frame being resumed from so the trajectory is not missing its start
    if resuming and not os.path.exists(traj_fp):
        logger.info(f"No {traj_fp} to continue, starting it from the checkpoint's frame...")
        atoms.write(traj_fp, format='extxyz', append=True)

    remaining_steps = fs - steps_done
    if remaining_steps <= 0:
        logger.info(f"Checkpoint is already at step {steps_done} of the {fs} steps "
                    f"{output_name} targets, so there is nothing left to run.")
    else:
        if resuming:
            logger.info(f"Continuing dynamics from step {steps_done}, "
                        f"{remaining_steps} steps to go...")
        else:
            logger.info("Running dynamics trajectory...")
        dynamics.run(steps=remaining_steps)

    # the observers only fire on their interval, so a run that ends between two of them has
    # not had its final state written - this is that write
    if save_md_checkpoints or resuming:
        write_md_checkpoint(checkpoint_fp, dynamics, total_steps=fs,
                            output_name=output_name, model_name=model_name,
                            rng_seed=rng_seed)
    if csv_fpath is not None and step_times is not None:
        _write_step_csv()

    return read(traj_fp, index=":")


def save_md_run_csv(csv_fpath: str,
                    step_times: list,
                    md_config: MDSettings,
                    run_time_s: float,
                    device: str,
                    calculator: str = "ASE"):
    """
    Save the benchmark results of one MD run to a CSV.

    The file holds the run's identity (device, calculator) first, then the wall time of
    every MD step in order, then the summary values::

        device,NVIDIA H100 80GB HBM3
        calculator,ASE
        step,step_time_s
        0,0.015400
        ...
        999,0.011000
        mean_step_time_s,0.011300
        total_steps,1000
        total_simulation_time_ps,0.5000
        total_run_time_s,12.400

    Note that the simulated time is the steps benchmarked here at the time step they were
    taken with, i.e. what the run really simulated - not the ``fs / PS`` figure its trajectory
    name carries, since ASE runs ``fs`` steps rather than ``fs / dt`` of them: the run above
    takes 1000 steps at a 0.5 fs time step, which simulates 0.5 ps, while its name says 1.0 ps.

    The rows describe the stretch of run this CSV covers, so the steps of a run picked up from
    a checkpoint are the steps of that stretch only. The CSVs of a run that was killed and
    resumed therefore add up to its whole simulated time.

    :param csv_fpath: path of the CSV to write (its directory must exist).
    :param step_times: wall time (s) of every step taken, in order.
    :param md_config: MDSettings object used for the MD run. Its time step is what the
                      simulated time is calculated with, so it has to be the time step the
                      run actually integrated at - for a resumed run the checkpoint's, not
                      necessarily the input's (see get_trajectory_with_ase).
    :param run_time_s: overall wall time (s) of the run, including loading the model.
    :param device: compute device the run used, e.g. 'CPU' (see compute_device_label).
    :param calculator: MD calculator/engine that was used.
    """
    total_steps = len(step_times)
    mean_step_time = sum(step_times) / total_steps if total_steps else 0.0

    with open(csv_fpath, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(['device', device])
        writer.writerow(['calculator', calculator])

        writer.writerow(['step', 'step_time_s'])
        for step, step_time in enumerate(step_times):
            writer.writerow([step, f"{step_time:.6f}"])

        writer.writerow(['mean_step_time_s', f"{mean_step_time:.6f}"])
        writer.writerow(['total_steps', total_steps])
        # the steps above at the time step they were taken with, i.e. what the run really
        # simulated (steps * dt / PS), not the fs / PS figure the trajectory name carries
        writer.writerow(['total_simulation_time_ps', f"{total_steps * md_config.dt / PS:.4f}"])
        writer.writerow(['total_run_time_s', f"{run_time_s:.3f}"])

    logger.info(f"Saved MD benchmark CSV to {csv_fpath}")


# ---------------------------------------------------------------------------
# Running a whole set of trajectories: the model-level loop that run_mlip_md.py's CLI drives
# ---------------------------------------------------------------------------

def _checkpoint_to_resume(checkpoint_fp: str, load_md_checkpoint, traj_name: str):
    """Path of the checkpoint a run resumes from, or None to start it fresh.

    ``load_md_checkpoint`` is either the input YAML's switch - True to resume every run from
    its own checkpoint at ``checkpoint_fp``, where a run that has no checkpoint yet starts
    fresh and says so, since a batch's runs need not all have got far enough to write one -
    or the path of a single checkpoint, which is used as given so that a wrong path is
    reported rather than quietly starting the run over.

    :param checkpoint_fp: the run's own checkpoint path (see md_checkpoint_fpath).
    :param load_md_checkpoint: the resolved ``load_md_checkpoint`` input value.
    :param traj_name: the run's name, for logging.
    :return: the checkpoint to resume from, or None.
    """
    if load_md_checkpoint is True:
        if os.path.exists(checkpoint_fp):
            return checkpoint_fp
        logger.info(f"No checkpoint for {traj_name} at {checkpoint_fp}, starting it fresh...")
        return None

    return load_md_checkpoint or None


def _md_csv_fpath(md_csv_dir: str, traj_name: str, resumed: bool) -> str:
    """Path of the benchmark CSV of one stretch of a run.

    A run that has never been resumed writes ``<traj_name>.csv``, as it always has. Each
    stretch after that writes its own numbered file, ``<traj_name>_contd_<n>.csv``, so that
    picking a run up from a checkpoint does not overwrite the benchmark of the stretch before
    it - resuming a run twice leaves ``eg.csv``, ``eg_contd_1.csv`` and ``eg_contd_2.csv``.
    The number is one past the continuation CSVs already in the directory, which is also what
    makes a stretch that is killed before writing anything take the number it would have had.

    The stub is the run's own name, the one its checkpoint is named after
    (``<traj_name>_checkpoint.yaml``, see md_checkpoint_fpath), so a run's stretches, its
    trajectory and its checkpoint all line up by name.

    :param md_csv_dir: directory the CSVs are written to.
    :param traj_name: the run's name.
    :param resumed: whether this stretch carries a previous one on.
    :return: the path to write this stretch's CSV to.
    """
    if not resumed:
        return f"{md_csv_dir}/{traj_name}.csv"

    return f"{md_csv_dir}/{traj_name}_contd_{_continuation_number(md_csv_dir, traj_name)}.csv"


def _continuation_number(md_csv_dir: str, traj_name: str) -> int:
    """The number the next continuation CSV of a run takes, i.e. one past the ones on disk.

    A directory that does not exist yet holds no CSVs, so the next one is the first - writing
    it is what reports a directory that is really missing (see save_md_run_csv).

    :param md_csv_dir: directory the CSVs are written to.
    :param traj_name: the run's name.
    :return: the number to put in the next ``<traj_name>_contd_<n>.csv``.
    """
    prefix, suffix = f"{traj_name}_contd_", ".csv"
    # read the directory rather than glob it: a traj_name carries model names, which can hold
    # characters glob would take as a pattern
    existing = [] if not os.path.isdir(md_csv_dir) else [
        name for name in os.listdir(md_csv_dir)
        if name.startswith(prefix) and name.endswith(suffix)]

    return len(existing) + 1


def run_md_eval(init_configs: list,
                md_config: MDSettings,
                md_runs_per_config: int,
                model_name: str,
                formatted_model_name: str,
                system_name: str,
                traj_dir: str = TRAJ_DIR,
                md_csv_dir: str = None,
                model_dir: str = None,
                constraint: str = None,# IndexedConstraint = None,
                enable_cueq: bool = False,
                override_saved: bool = False,
                cuda_device: str = "cuda:0",
                device_label: str = None,
                fair_chem_task: str = None,
                fair_chem_setting: str = None,
                save_md_checkpoints: bool = False,
                load_md_checkpoint=False,
                checkpoint_every: float = DEFAULT_CHECKPOINT_EVERY_PS,
                md_checkpoint_dir: str = None):

    """Run every MD trajectory for one model, and write a benchmark CSV for each of them.

    With ``md_csv_dir`` set, one CSV per run is saved under ``<md_csv_dir>`` (see
    ``save_md_run_csv``): the device the run used, the calculator, the wall time of every
    step, the mean step time, and the step count / simulated time / overall run time. Each
    stretch of a run writes its own - the run's own name, then ``_contd_1``, ``_contd_2`` and
    so on for the stretches a resume carries on (see ``_md_csv_fpath``) - and the helper
    refreshes it every ``checkpoint_every`` ps rather than only at the end, so a stretch that
    is killed keeps what it measured. The simulated time is those steps at the time step they
    were taken with, so a resumed run's CSV counts it at the checkpoint's time step rather
    than the input's. ``device_label`` names that device in the CSV and defaults to the name
    of the resolved ``cuda_device`` (see ``compute_device_label``). ``enable_cueq``,
    ``model_dir`` and the FairChem pair ``fair_chem_task`` / ``fair_chem_setting`` are
    forwarded to every run's calculator.

    With ``save_md_checkpoints`` set, every run also checkpoints itself, every
    ``checkpoint_every`` ps of simulated time, under ``md_checkpoint_dir``
    (``MD_CHECKPOINT_DIR``, i.e. ``out/md_checkpoints``, when None). ``load_md_checkpoint``
    then carries runs on from a checkpoint instead of starting them over: True resumes each
    run from its own checkpoint, and a path resumes a single run from that file (see
    ``_checkpoint_to_resume``).
    """

    # the CSV records the device by name (e.g. 'NVIDIA H100 80GB HBM3'), once for the run
    if md_csv_dir is not None and device_label is None:
        device_label = compute_device_label(cuda_device)

    trajectories = []
    for i in range(len(init_configs)):

        init_config = init_configs[i].copy()

        for j in range(md_runs_per_config):
            logger.info(f"Running MD trajectory for init config {i + 1}, run number {j}...")

            start_time = time()
            traj_name = f'{system_name}_{formatted_model_name}_{md_config.temp}K_{md_config.fs / PS}ps_config_{i}_run_{j}'

            # the checkpoint this run resumes from, if any
            resume_fp = _checkpoint_to_resume(md_checkpoint_fpath(traj_name,
                                                                  md_checkpoint_dir),
                                              load_md_checkpoint,
                                              traj_name)

            out_trj_name = f"{traj_dir}/{traj_name}_traj.xyz"
            # a resumed run continues its own trajectory rather than returning it as it
            # stands, so reusing a saved trajectory only applies to a run starting fresh
            if resume_fp is None and os.path.exists(out_trj_name) and not override_saved:
                logger.info(f"{out_trj_name} already exists, skipping...")
                continue

            step_times = [] if md_csv_dir is not None else None
            # the helper writes this CSV, every checkpoint_every ps and at the end of the
            # stretch, so a killed stretch keeps its timings (see get_trajectory_with_ase)
            csv_fpath = None if md_csv_dir is None else _md_csv_fpath(md_csv_dir, traj_name,
                                                                     resume_fp is not None)

            trajectory = get_trajectory_with_ase(init_config,
                                                 model_name,
                                                 md_config,
                                                 traj_name,
                                                 out_dir=traj_dir,
                                                 constraint=constraint,
                                                 cuda_device=cuda_device,
                                                 override_cache=override_saved,
                                                 model_dir=model_dir,
                                                 enable_cueq=enable_cueq,
                                                 step_times=step_times,
                                                 fair_chem_task=fair_chem_task,
                                                 fair_chem_setting=fair_chem_setting,
                                                 save_md_checkpoints=save_md_checkpoints,
                                                 load_md_checkpoint=resume_fp,
                                                 checkpoint_every=checkpoint_every,
                                                 md_checkpoint_dir=md_checkpoint_dir,
                                                 csv_fpath=csv_fpath,
                                                 csv_device=device_label,
                                                 csv_calculator=MD_CALCULATOR)
            end_time = time()
            md_time = end_time - start_time
            # the steps the run took: the benchmarked ones where the CSV is on, otherwise the
            # run's target, which a resume completes
            steps_taken = len(step_times) if step_times else md_config.fs
            logger.info(f"Ran MD trajectory '{traj_name}' in {md_time:.4f}s, with an average step time of {float(md_time) / steps_taken}...")

            trajectories.append(trajectory)

    return trajectories


def get_trajectory_with_mlptrain(init_config,
                                 mlp_model: object,
                                 md_config: MDSettings,
                                 output_name: str):
    """

    :param system:
    :return:
    """
    import mlptrain as mlt

    # unpack MDConfig
    fs = md_config.fs
    temp = md_config.temp
    dt = md_config.dt
    interval = md_config.interval
    md_method = md_config.md_method

    if md_method == 'ase':

        # run ase dynamics
        trajectory = mlt.md.run_mlp_md(
            configuration=init_config,
            mlp=mlp_model,
            fs=fs,
            temp=temp,
            dt=dt,
            interval=interval,
        )

    elif md_method == 'openmm':

        # run openmm dynamics
        trajectory = mlt.md_openmm.run_mlp_md_openmm(
            configuration=init_config,
            mlp=mlp_model,
            fs=fs,
            temp=temp,
            dt=dt,
            interval=interval,
        )

    else:
        logger.info("Please select one md_method from ('ase', 'openmm')...")
        return

    # save and return trajectory
    trajectory.save(filename=f'{TRAJ_DIR}/{output_name}_traj.xyz')
    return trajectory
