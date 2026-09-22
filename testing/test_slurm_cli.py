"""
Unit tests for `rmlt slurm` and the generic slurm script it submits.

These cover:

  * the argument splitting that decides what goes to sbatch and what goes to the job
    (``_cli._split_slurm_args`` / ``_cli._run_slurm``),
  * the sbatch command that gets built, and the errors reported before anything is
    submitted,
  * the slurm script itself (``slurm/eg.sh``): its syntax, its usage guard, that it runs
    from the submitting directory, and that it forwards its arguments to ``rmlt`` intact.

Nothing here submits a job, needs a GPU, or needs ``sbatch``: the command is inspected
with ``--dry-run`` or a monkeypatched ``subprocess.run``, and ``slurm/eg.sh`` is run for
real against a stub ``rmlt`` (its environment setup is guarded so that works off-cluster).

Run from the repository root:

    pytest testing/test_slurm_cli.py
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from react_ml_test_public import _cli

REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_SCRIPT = REPO_ROOT / "slurm" / "eg.sh"

MD_YAML = "inputs/md/CPD_MVK_DA_dynamics_vac.yaml"


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    """Resolve the repo-root-relative paths (slurm/eg.sh, inputs/...) used here."""
    monkeypatch.chdir(REPO_ROOT)


class _SubmitSpy:
    """Stands in for ``subprocess.run``: records the command instead of submitting it."""

    def __init__(self, returncode: int = 0):
        self.commands: list[list[str]] = []
        self.returncode = returncode

    def __call__(self, command, *args, **kwargs):
        self.commands.append(list(command))
        return subprocess.CompletedProcess(command, self.returncode)


@pytest.fixture
def submit_spy(monkeypatch):
    """Capture the sbatch command, with `sbatch` pretended to be on PATH."""
    spy = _SubmitSpy()
    monkeypatch.setattr(_cli.subprocess, "run", spy)
    monkeypatch.setattr(_cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    return spy


class TestSplitSlurmArgs:
    """Everything before the workflow script goes to sbatch; it and the rest go to the job."""

    def test_sbatch_options_come_before_the_script(self):
        options, sbatch_options, workflow = _cli._split_slurm_args(
            ["--time=12:00:00", "--partition=medium", "run-mlip-md", MD_YAML]
        )
        assert sbatch_options == ["--time=12:00:00", "--partition=medium"]
        assert workflow == ["run-mlip-md", MD_YAML]
        assert options["script"] is None

    def test_space_separated_option_values_are_not_taken_as_the_script(self):
        """sbatch accepts `--time X` as well as `--time=X`, so the value is not the script."""
        _, sbatch_options, workflow = _cli._split_slurm_args(
            ["--time", "12:00:00", "--partition", "medium", "run-mlip-md", MD_YAML]
        )
        assert sbatch_options == ["--time", "12:00:00", "--partition", "medium"]
        assert workflow == ["run-mlip-md", MD_YAML]

    @pytest.mark.parametrize("token", ["run-mlip-md", "run_mlip_md"])
    def test_script_name_accepts_dashes_and_underscores(self, token):
        _, _, workflow = _cli._split_slurm_args([token, MD_YAML])
        assert workflow == [token, MD_YAML]

    def test_leading_shell_script_is_taken_as_the_slurm_script(self):
        options, sbatch_options, workflow = _cli._split_slurm_args(["eg.sh", "run-mlip-md", MD_YAML])
        assert options["script"] == "eg.sh"
        assert sbatch_options == []
        assert workflow == ["run-mlip-md", MD_YAML]

    @pytest.mark.parametrize("form", [["--script", "other.sh"], ["--script=other.sh"]])
    def test_script_option_is_read_in_both_forms(self, form):
        options, _, _ = _cli._split_slurm_args([*form, "run-mlip-md", MD_YAML])
        assert options["script"] == "other.sh"

    def test_script_option_without_a_value_is_an_error(self):
        with pytest.raises(_cli._SlurmUsageError, match="--script needs the path"):
            _cli._split_slurm_args(["--script"])

    def test_trailing_arguments_stay_with_the_workflow(self):
        """Anything after the script + YAML reaches the workflow script in the job."""
        _, _, workflow = _cli._split_slurm_args(["run-mlip-md", MD_YAML, "--foo", "bar"])
        assert workflow == ["run-mlip-md", MD_YAML, "--foo", "bar"]

    def test_help_before_the_script_is_ours(self):
        options, _, workflow = _cli._split_slurm_args(["--help"])
        assert options["help"] is True
        assert workflow == []

    def test_help_after_the_script_is_the_workflows(self):
        """`--help` past the script name must reach the workflow, not print our own."""
        options, _, workflow = _cli._split_slurm_args(["run-mlip-md", MD_YAML, "--help"])
        assert options["help"] is False
        assert workflow == ["run-mlip-md", MD_YAML, "--help"]

    def test_mistyped_script_name_is_reported_as_unknown(self):
        """The likeliest mistake gets the same message style as `rmlt <typo>`."""
        with pytest.raises(_cli._SlurmUsageError, match="unknown script 'run-mlip-mdd'"):
            _cli._split_slurm_args(["run-mlip-mdd", MD_YAML])

    @pytest.mark.parametrize("args", [["--time=1:00:00"], ["--partition", "medium"]])
    def test_missing_script_name_is_reported_as_such(self, args):
        with pytest.raises(_cli._SlurmUsageError, match="no workflow script"):
            _cli._split_slurm_args(args)


class TestSlurmCommand:
    """The command handed to sbatch, checked through --dry-run and a spy."""

    def test_dry_run_prints_the_command_and_nothing_is_submitted(self, capsys, monkeypatch):
        monkeypatch.setattr(_cli.shutil, "which", lambda name: None)  # no sbatch anywhere
        assert _cli.main(["slurm", "--dry-run", "run-mlip-md", MD_YAML]) == 0

        out = capsys.readouterr().out.splitlines()
        assert out[0] == f"sbatch slurm/eg.sh run-mlip-md {MD_YAML}"

    def test_dry_run_reports_the_resources_the_script_requests(self, capsys):
        assert _cli.main(["slurm", "-n", "run-mlip-md", MD_YAML]) == 0

        out = capsys.readouterr().out
        assert "#   #SBATCH --partition=devel" in out
        assert "#   #SBATCH --time=00:10:00" in out
        # the script's commented-out constraints are not active directives
        assert "#   # --constraint" not in out

    def test_overrides_are_passed_to_sbatch_before_the_script(self, submit_spy):
        assert _cli.main(["slurm", "--time=12:00:00", "--gres=gpu:1", "run-mlip-md", MD_YAML]) == 0

        assert submit_spy.commands == [
            ["sbatch", "--time=12:00:00", "--gres=gpu:1", "slurm/eg.sh", "run-mlip-md", MD_YAML]
        ]

    def test_bare_script_name_resolves_into_slurm_dir(self, capsys):
        assert _cli.main(["slurm", "-n", "eg.sh", "run-mlip-md", MD_YAML]) == 0

        assert capsys.readouterr().out.splitlines()[0] == f"sbatch slurm/eg.sh run-mlip-md {MD_YAML}"

    def test_script_path_is_quoted_when_it_needs_to_be(self, submit_spy, capsys):
        """The echoed command should be copy-pasteable, so paths go through shlex.join."""
        _cli.main(["slurm", "run-mlip-md", "inputs/md/a file.yaml"])

        printed = capsys.readouterr().out.splitlines()[0]
        assert printed.endswith("run-mlip-md 'inputs/md/a file.yaml'")
        assert submit_spy.commands[0][-1] == "inputs/md/a file.yaml"

    def test_submit_exit_code_is_propagated(self, submit_spy):
        submit_spy.returncode = 1
        assert _cli.main(["slurm", "run-mlip-md", MD_YAML]) == 1

    def test_list_documents_the_subcommand(self, capsys):
        assert _cli.main(["list"]) == 0
        assert "rmlt slurm" in capsys.readouterr().out


class TestSlurmErrors:
    """Failures must be reported before a job is queued, not inside it."""

    def test_unknown_script_exits_2(self, capsys):
        assert _cli.main(["slurm", "run-mlip-mdd", MD_YAML]) == 2

        err = capsys.readouterr().err
        assert "unknown script 'run-mlip-mdd'" in err
        assert "rmlt list" in err

    def test_missing_input_file_exits_2(self, capsys):
        assert _cli.main(["slurm", "run-mlip-md"]) == 2
        assert "no input file given for 'run-mlip-md'" in capsys.readouterr().err

    def test_missing_slurm_script_exits_2(self, capsys):
        assert _cli.main(["slurm", "--script", "nosuch.sh", "run-mlip-md", MD_YAML]) == 2

        err = capsys.readouterr().err
        assert "slurm script not found: nosuch.sh" in err
        assert "repository root" in err

    def test_missing_sbatch_exits_2_with_a_cluster_hint(self, capsys, monkeypatch):
        monkeypatch.setattr(_cli.shutil, "which", lambda name: None)
        assert _cli.main(["slurm", "run-mlip-md", MD_YAML]) == 2

        err = capsys.readouterr().err
        assert "not on PATH" in err
        assert "login node" in err

    def test_bare_slurm_prints_help(self, capsys):
        assert _cli.main(["slurm"]) == 0
        assert "Usage: rmlt slurm" in capsys.readouterr().out


class TestSbatchDirectives:
    """Reading the script's own '#SBATCH' lines for display."""

    def test_active_directives_are_listed_in_order(self):
        directives = _cli._sbatch_directives(str(SLURM_SCRIPT))
        assert directives[0].startswith("#SBATCH")
        assert "#SBATCH --partition=devel" in directives
        assert len(directives) == len({*directives})  # no duplicates

    def test_commented_out_directives_are_not_reported(self):
        """`# --constraint=...` lines are notes, not active directives."""
        assert not [d for d in _cli._sbatch_directives(str(SLURM_SCRIPT))
                    if "--constraint" in d]

    def test_unreadable_script_yields_nothing(self):
        assert _cli._sbatch_directives("does/not/exist.sh") == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
class TestSlurmScript:
    """slurm/eg.sh run for real, against a stub `rmlt`."""

    @pytest.fixture
    def stub_rmlt(self, tmp_path):
        """A fake `rmlt` on PATH that records its arguments and working directory."""
        bindir = tmp_path / "bin"
        bindir.mkdir()
        args_record, cwd_record = tmp_path / "args", tmp_path / "cwd"
        stub = bindir / "rmlt"
        stub.write_text(
            "#!/bin/bash\n"
            f'printf "%s\\n" "$@" > "{args_record}"\n'
            f'pwd > "{cwd_record}"\n'
        )
        stub.chmod(0o755)
        return bindir, args_record, cwd_record

    @staticmethod
    def _run(args, bindir, cwd=REPO_ROOT, submit_dir=None):
        """Run slurm/eg.sh with the stub rmlt first on PATH and no cluster environment."""
        env = {key: value for key, value in os.environ.items() if key != "DATA"}
        env.pop("SLURM_JOB_ID", None)
        env["PATH"] = f"{bindir}{os.pathsep}{env['PATH']}"
        if submit_dir is not None:
            env["SLURM_SUBMIT_DIR"] = str(submit_dir)
        return subprocess.run(
            ["bash", str(SLURM_SCRIPT), *args],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=300,
        )

    def test_script_is_valid_bash(self):
        assert subprocess.run(["bash", "-n", str(SLURM_SCRIPT)]).returncode == 0

    @pytest.mark.parametrize("args", [[], ["run-mlip-md"]])
    def test_too_few_arguments_is_refused(self, args, stub_rmlt):
        """A malformed submission fails at job start with a usage message, not inside rmlt."""
        bindir, args_record, _ = stub_rmlt
        result = self._run(args, bindir)

        assert result.returncode == 2
        assert "usage:" in result.stderr
        assert not args_record.exists(), "rmlt was called despite the missing input file"

    def test_arguments_reach_rmlt_unchanged(self, stub_rmlt):
        bindir, args_record, _ = stub_rmlt
        result = self._run(["run-mlip-md", MD_YAML], bindir)

        assert result.returncode == 0, result.stderr
        assert args_record.read_text().splitlines() == ["run-mlip-md", MD_YAML]

    def test_extra_arguments_are_forwarded(self, stub_rmlt):
        bindir, args_record, _ = stub_rmlt
        result = self._run(["run-mlip-md", MD_YAML, "--foo", "bar"], bindir)

        assert result.returncode == 0, result.stderr
        assert args_record.read_text().splitlines() == ["run-mlip-md", MD_YAML, "--foo", "bar"]

    def test_job_runs_from_the_submitting_directory(self, stub_rmlt, tmp_path):
        """The YAML path is resolved relative to the job's working directory."""
        bindir, _, cwd_record = stub_rmlt
        submit_dir = tmp_path / "submitted-from"
        submit_dir.mkdir()

        result = self._run(["run-mlip-md", MD_YAML], bindir, submit_dir=submit_dir)

        assert result.returncode == 0, result.stderr
        assert Path(cwd_record.read_text().strip()) == submit_dir

    def test_no_cluster_environment_is_tolerated(self, stub_rmlt):
        """No `module`, no $DATA, no scontrol, no SLURM_* - still runs (used by these tests)."""
        bindir, _, _ = stub_rmlt
        result = self._run(["run-mlip-md", MD_YAML], bindir)

        assert result.returncode == 0, result.stderr
        assert "Running on" in result.stdout
