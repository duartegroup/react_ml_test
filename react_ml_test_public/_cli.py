"""
The ``rmlt`` command: run react_ml_test_public workflow scripts.

Installing this package (``pip install -e .``) provides a single ``rmlt``
executable, which dispatches to the workflow modules in
``react_ml_test_public.scripts``:

    rmlt list                          # list available scripts
    rmlt run-test-set-comparison       # run one (dashes == underscores)
    rmlt run-test-set-comparison ...   # anything after the name is passed
                                       # through to the script
    rmlt slurm run-test-set-comparison inputs/test/x.yaml
                                       # submit one to SLURM with sbatch

The scripts read ``data/``, ``models/``, ``inputs/`` and write to ``out/``
relative to the current working directory, so run them from the repository
root (or any directory containing those folders).
"""

from __future__ import annotations

import ast
import importlib
import os
import runpy
import shlex
import shutil
import subprocess
import sys

_SCRIPTS_PKG = "react_ml_test_public.scripts"

# The generic submission script `rmlt slurm` hands to sbatch. Its '#SBATCH' lines hold
# the resources for the job, so it is meant to be edited by hand
# (see slurm/slurm_script.sh).
_DEFAULT_SLURM_SCRIPT = os.path.join("slurm", "slurm_script.sh")

_SLURM_USAGE = (
    "Usage: rmlt slurm [sbatch options] [<slurm-script.sh>] <script> <input.yaml> [args...]"
)


class _SlurmUsageError(Exception):
    """A `rmlt slurm` command line that cannot be acted on."""


def _scripts_dir() -> str:
    return os.path.dirname(importlib.import_module(_SCRIPTS_PKG).__file__)


def _script_names() -> list[str]:
    """Sorted module names of the runnable scripts (no __init__/private modules)."""
    names = []
    for entry in sorted(os.listdir(_scripts_dir())):
        if entry.startswith("_") or not entry.endswith(".py"):
            continue
        names.append(entry[:-3])
    return names


def _doc_summary(name: str) -> str:
    """First line of a script's module docstring, read without importing it."""
    try:
        with open(os.path.join(_scripts_dir(), name + ".py"), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        doc = ast.get_docstring(tree)
    except (OSError, SyntaxError, ValueError):
        return ""
    return doc.strip().splitlines()[0].strip() if doc else ""


def _token(name: str) -> str:
    return name.replace("_", "-")


def _module_name(token: str) -> str | None:
    """Map a command token (dashes or underscores) back to a module name."""
    token = token.strip()
    if token.endswith(".py"):
        token = token[:-3]
    key = token.replace("-", "_")
    return key if key in _script_names() else None


def _print_list(stream=None) -> None:
    # resolve sys.stdout at call time rather than as a default, so the output can be
    # redirected or captured (a `stream=sys.stdout` default binds the import-time object)
    stream = sys.stdout if stream is None else stream

    print("Usage: rmlt <script> [args...]", file=stream)
    print("       rmlt slurm [sbatch options] <script> <input.yaml>   # submit to SLURM", file=stream)
    print("       rmlt list", file=stream)
    print(file=stream)
    print("Available scripts:", file=stream)
    names = _script_names()
    tokens = [_token(name) for name in names]
    width = max(len(token) for token in tokens) if tokens else 0
    for name, token in zip(names, tokens):
        summary = _doc_summary(name)
        print(f"  {token:<{width}}  {summary}" if summary else f"  {token}", file=stream)


def _print_slurm_help(stream=None) -> None:
    stream = sys.stdout if stream is None else stream

    print(_SLURM_USAGE, file=stream)
    print(file=stream)
    print("Submit a workflow script to SLURM with sbatch. The job runs on the compute node",
          file=stream)
    print("and is equivalent to `sbatch <slurm-script.sh> <script> <input.yaml>`.", file=stream)
    print(file=stream)
    print("Options:", file=stream)
    print(f"  --script PATH   slurm script to submit (default: {_DEFAULT_SLURM_SCRIPT}). A leading",
          file=stream)
    print("                  argument ending in .sh is also taken as the script path.",
          file=stream)
    print("  -n, --dry-run   print the sbatch command and the resources the slurm script",
          file=stream)
    print("                  requests, without submitting anything.", file=stream)
    print("  -h, --help      show this message.", file=stream)
    print(file=stream)
    print("Any other option is passed to sbatch unchanged, and so overrides the matching",
          file=stream)
    print("'#SBATCH' line in the slurm script for this submission, e.g.:", file=stream)
    print("  rmlt slurm --time=12:00:00 --partition=medium --gres=gpu:1 run-mlip-md \\",
          file=stream)
    print("      inputs/md/CPD_MVK_DA_dynamics_vac.yaml", file=stream)
    print(file=stream)
    print("The resources themselves live in the slurm script's '#SBATCH' header, because",
          file=stream)
    print("SLURM reads directives from the script file and never from its arguments - edit",
          file=stream)
    print("that header to suit the job. Anything after <script> <input.yaml> is passed",
          file=stream)
    print("through to the workflow script by the slurm script.", file=stream)


def _looks_like_script_token(arg: str) -> bool:
    """Whether an argument could be a mistyped workflow script name.

    Used only to choose between two error messages when no workflow script was found, so
    it is a deliberately loose test: workflow names are dash/underscore separated words,
    which rules out the values sbatch options take (``12:00:00``, ``medium``) and file
    paths (``inputs/md/x.yaml``).

    :param arg: one of the arguments given to `rmlt slurm`.
    :return: True if the argument looks like a workflow script name.
    """
    return (
        bool(arg)
        and not arg.startswith("-")
        and not set("/.:") & set(arg)
        and ("-" in arg or "_" in arg)
    )


def _split_slurm_args(args: list[str]) -> tuple[dict, list[str], list[str]]:
    """Split `rmlt slurm` arguments into (options, sbatch options, workflow argv).

    Mirrors how sbatch reads its own command line - options first, then the script, then
    the script's arguments - so everything before the workflow script name is handed to
    sbatch and everything from it onwards is handed to `rmlt` inside the job.

    The workflow script is the first argument naming a known workflow module (dashes or
    underscores). Recognising it by name, rather than taking the first non-option
    argument, is what allows sbatch options to be written in either the ``--time=X`` or
    the ``--time X`` form: in the latter the value would otherwise look like the script.

    :param args: the arguments given after the `rmlt slurm` subcommand.
    :return: (options, sbatch options, workflow argv)
    :raises _SlurmUsageError: if the arguments name no workflow script.
    """
    options = {"script": None, "dry_run": False, "help": False}
    sbatch_options: list[str] = []
    not_options: list[str] = []
    index = 0

    while index < len(args):
        arg = args[index]

        if arg in ("-h", "--help"):
            # only before the workflow script, so that `rmlt slurm <script> <yaml> --help`
            # still passes --help through to the workflow script
            options["help"] = True
            return options, sbatch_options, []

        if arg == "--script" or arg.startswith("--script="):
            if arg == "--script":
                index += 1
                if index >= len(args):
                    raise _SlurmUsageError("--script needs the path of a slurm script")
                options["script"] = args[index]
            else:
                options["script"] = arg.split("=", 1)[1]
            index += 1
            continue

        if arg in ("-n", "--dry-run"):
            options["dry_run"] = True
            index += 1
            continue

        if _module_name(arg) is not None:
            # the workflow script: it and everything after it are the job's arguments
            return options, sbatch_options, args[index:]

        if arg.endswith(".sh"):
            options["script"] = arg
        else:
            # an sbatch option, or the value of one - neither of which this can tell
            # apart from a mistyped script name, so keep them for the error below
            if not arg.startswith("-"):
                not_options.append(arg)
            sbatch_options.append(arg)
        index += 1

    mistyped = next((arg for arg in not_options if _looks_like_script_token(arg)), None)
    if mistyped is not None:
        raise _SlurmUsageError(
            f"unknown script '{mistyped}'\n"
            "Run 'rmlt list' to see the available scripts."
        )
    raise _SlurmUsageError("no workflow script in the arguments")


def _sbatch_directives(slurm_script: str) -> list[str]:
    """The active '#SBATCH' lines of a slurm script, for display.

    Only directives before the first executable line count for SLURM, so reading stops
    there - which is also what keeps a commented-out directive in the script's own notes
    from being reported as active.

    :param slurm_script: path to the slurm script.
    :return: the '#SBATCH' lines, in file order (empty if the file cannot be read).
    """
    directives = []
    try:
        with open(slurm_script, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.startswith("#SBATCH"):
                    directives.append(stripped)
                elif stripped and not stripped.startswith("#"):
                    break
    except (OSError, UnicodeDecodeError):
        return []
    return directives


def _run_slurm(args: list[str]) -> int:
    """Submit a workflow script to SLURM with sbatch; returns the process exit code."""
    if not args:
        _print_slurm_help()
        return 0

    try:
        options, sbatch_options, workflow = _split_slurm_args(args)
    except _SlurmUsageError as exc:
        print(f"rmlt slurm: {exc}", file=sys.stderr)
        print(_SLURM_USAGE, file=sys.stderr)
        return 2

    if options["help"]:
        _print_slurm_help()
        return 0

    # every workflow script so far takes its input YAML as a single positional argument,
    # and the slurm script requires exactly that shape, so catch a missing input here
    # rather than in a job that dies seconds after it starts
    if len(workflow) < 2:
        print(f"rmlt slurm: no input file given for '{workflow[0]}'", file=sys.stderr)
        print(_SLURM_USAGE, file=sys.stderr)
        return 2

    slurm_script = options["script"] or _DEFAULT_SLURM_SCRIPT
    if not os.path.isfile(slurm_script) and not os.path.dirname(slurm_script):
        # a bare name like `other.sh` means the one in slurm/ - the command echoed below
        # always shows the path that was actually used
        candidate = os.path.join(os.path.dirname(_DEFAULT_SLURM_SCRIPT), slurm_script)
        if os.path.isfile(candidate):
            slurm_script = candidate

    if not os.path.isfile(slurm_script):
        print(f"rmlt slurm: slurm script not found: {slurm_script}", file=sys.stderr)
        print("Submit from the repository root - the slurm script and the input YAML are "
              "both resolved relative to the working directory - or name one with --script.",
              file=sys.stderr)
        return 2

    # sbatch takes its options before the script, and the script's arguments after it
    command = ["sbatch", *sbatch_options, slurm_script, *workflow]

    if options["dry_run"]:
        # all on stdout, command first, so the output can also be piped straight into a
        # shell - the resource notes are '#' comments
        print(shlex.join(command))
        directives = _sbatch_directives(slurm_script)
        if directives:
            print()
            print(f"# resources requested by {slurm_script} - edit its '#SBATCH' lines to "
                  f"change them:")
            for directive in directives:
                print(f"#   {directive}")
        return 0

    if shutil.which("sbatch") is None:
        print("rmlt slurm: 'sbatch' is not on PATH - submit from a cluster login node.",
              file=sys.stderr)
        print("Use --dry-run to see the command without submitting anything.", file=sys.stderr)
        return 2

    print(shlex.join(command))
    return subprocess.run(command).returncode


def main(argv: list[str] | None = None) -> int | None:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help", "help"):
        _print_list()
        return 0

    if args[0] in ("list", "-l", "--list"):
        _print_list()
        return 0

    # `slurm` submits a workflow script instead of running one, so it is a subcommand
    # rather than a workflow module (special-cased like `list` above). NB: a workflow
    # module named `slurm` would be shadowed by this.
    if args[0] == "slurm":
        return _run_slurm(args[1:])

    name = _module_name(args[0])
    if name is None:
        print(f"rmlt: unknown script '{args[0]}'", file=sys.stderr)
        print("Run 'rmlt list' to see the available scripts.", file=sys.stderr)
        return 2

    # runpy leaves sys.argv alone, so reshape it to look like a direct launch:
    # argv[0] stays (the rmlt command) and everything after the script name is
    # passed through untouched.
    sys.argv = [sys.argv[0] if sys.argv else "rmlt"] + args[1:]

    runpy.run_module(f"{_SCRIPTS_PKG}.{name}", run_name="__main__")
    return None
