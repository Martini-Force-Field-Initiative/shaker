"""GROMACS simulation setup and execution."""

import contextlib
import glob
import json
import os
import re
import select
import shlex
import shutil
import signal
import subprocess
import sys
import time
import warnings
from datetime import datetime, timedelta
from importlib.resources import files
from pathlib import Path

import MDAnalysis as md
import numpy as np

_ITER_RE = re.compile(r"^iter_(\d+)_\d{8}_\d{6}$")


def prepare_setup_water(
    initial_structure,
    structure_itp="initial_CG.itp",
    n_mols=1,
    box_size=4,
    FFitp=None,
    SolvITP=None,
    IonsITP=None,
    resname=None,
    NaCL_Conc=0.15,
    gmx_loc="",
):
    """
    Prepare a solvated Martini 3 simulation system for a coarse-grained molecule.

    This function builds a solvent box, inserts `n_mols` copies of the solute,
    solvates with water, neutralizes the system, and adds NaCl to reach
    a target concentration. It writes a minimal `topol.top` and produces GROMACS
    coordinate/tpr suitable for subsequent MD.

    Parameters
    ----------
    initial_structure : str or Path
        Solute structure file (typically a CG .gro or .pdb) used as the template
        for insertion.
    structure_itp : str, optional
        ITP filename (or path) describing the solute topology included in `topol.top`.
        Default is "initial_CG.itp".
    n_mols : int, optional
        Number of solute copies to insert into the simulation box. Default is 1.
    box_size : float, optional
        Cubic box edge length in nm passed to `gmx insert-molecules -box`.
        Default is 4.
    FFitp : str or Path, optional
        Martini force-field ITP file. If None, the SHAKER-distributed default is used.
    SolvITP : str or Path, optional
        Martini solvent ITP file. If None, the SHAKER-distributed default is used.
    IonsITP : str or Path, optional
        Martini ions ITP file. If None, the SHAKER-distributed default is used.
    NaCL_Conc : float, optional
        Target NaCl concentration (mol/L). Default is 0.15.
    gmx_loc : str, optional
        Prefix/path to the GROMACS executable (e.g. "/usr/local/bin/" or "").

    Notes
    -----
    The setup procedure performs the following steps:

    1. Generate a topology file.
    2. Insert `n_mols` copies of `initial_structure` into a cubic box (`gmx insert-molecules`).
    3. Solvate the box with Martini water using ``gmx solvate``.
    4. Neutralize the system with counterions using ``gmx genion``.
    5. Add additional NaCl to reach the target concentration.

    All GROMACS stdout/stderr are appended to `gmx_setup_water.log`.
    """

    ## normalize paths
    initial_structure = Path(initial_structure).resolve()
    structure_itp = Path(structure_itp).resolve()
    gmx = str(Path(gmx_loc) / "gmx") if gmx_loc else "gmx"

    ## Pick itps and mdps.
    if FFitp is None:
        FFitp = files("shaker.data.itps") / "martini_v3.0.0.itp"
    if SolvITP is None:
        SolvITP = files("shaker.data.itps") / "martini_v3.0.0_solvents_v1.itp"
    if IonsITP is None:
        IonsITP = files("shaker.data.itps") / "martini_v3.0.0_ions_v1.itp"

    minMDP = files("shaker.data.mdps") / "min.mdp"
    waterbox = files("shaker.data.itps") / "water.gro"

    ## Prepare top file
    if not resname:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"Empty box", category=UserWarning
            )
            resname = np.unique(md.Universe(initial_structure).residues.resnames)[0]

    with open("topol.top", "w") as topinput:
        topinput.write(f'#include "{FFitp}"\n')
        topinput.write(f'#include "{structure_itp}"\n')
        topinput.write(f'#include "{SolvITP}"\n')
        topinput.write(f'#include "{IonsITP}"\n')
        topinput.write("[system]\n")
        topinput.write("Shaker approved!\n")
        topinput.write("\n")
        topinput.write("[ molecules ]\n")
        topinput.write(f"{resname}   {n_mols}\n")

    ## Run gmx
    env = os.environ.copy()
    ## Get rid of the old runs when rerunning.
    env["GMX_MAXBACKUP"] = "-1"
    with open("gmx_setup_water.log", "w") as log:
        ## Create box with molecules
        _run(
            [
                gmx,
                "insert-molecules",
                "-ci",
                str(initial_structure),
                "-box",
                str(box_size),
                str(box_size),
                str(box_size),
                "-nmol",
                str(n_mols),
                "-o",
                "box.gro",
            ],
            log=log,
            env=env,
        )

        ## Solvate the box
        _run(
            [
                gmx,
                "solvate",
                "-cp",
                "box.gro",
                "-cs",
                str(waterbox),
                "-o",
                "watered.gro",
                "-p",
                "topol.top",
            ],
            log=log,
            env=env,
        )

        ## Neutralize
        _run(
            [
                gmx,
                "grompp",
                "-f",
                str(minMDP),
                "-c",
                "watered.gro",
                "-p",
                "topol.top",
                "-o",
                "memion.tpr",
                "-maxwarn",
                "1",
            ],
            log=log,
            env=env,
        )

        _run(
            [
                gmx,
                "genion",
                "-s",
                "memion.tpr",
                "-o",
                "memion.gro",
                "-p",
                "topol.top",
                "-pname",
                "NA",
                "-pq",
                "+1",
                "-nname",
                "CL",
                "-nq",
                "-1",
                "-neutral",
            ],
            log=log,
            env=env,
            input_text="W\n",
        )

        ## Add NaCl
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"Empty box", category=UserWarning
            )
            u_mem = md.Universe("memion.gro")
        Waternumber = len(u_mem.select_atoms("resname W").residues)
        naclNUM = int((NaCL_Conc * Waternumber * 4) / 55.5)

        _run(
            [
                gmx,
                "grompp",
                "-f",
                str(minMDP),
                "-c",
                "memion.gro",
                "-p",
                "topol.top",
                "-o",
                "memion_2.tpr",
                "-maxwarn",
                "4",
            ],
            log=log,
            env=env,
        )

        _run(
            [
                gmx,
                "genion",
                "-s",
                "memion_2.tpr",
                "-o",
                "memion_2.gro",
                "-p",
                "topol.top",
                "-pname",
                "NA",
                "-pq",
                "+1",
                "-nname",
                "CL",
                "-nq",
                "-1",
                "-np",
                str(naclNUM),
                "-nn",
                str(naclNUM),
            ],
            log=log,
            env=env,
            input_text="W\n",
        )


def runSim(
    minMDP=None,
    relMDP=None,
    prodMDP=None,
    minOPT="-pin on -nt 8",
    relOPT="-pin on -nt 8",
    prodOPT="-pin on -nt 8",
    maxwarn=1,
    gmx_loc="",
    cleanTraj=True,
    cleanOldRun=True,
    mapping=None,
    keep_last=-1,
    hang_timeout=600,
    verbose=False,
):
    """
    Run a standard Martini simulation pipeline (minimization → relaxation → production).

    By default, the MDP files distributed with SHAKER are used.

    Parameters
    ----------
    minMDP, relMDP, prodMDP : str or Path, optional
        MDP parameter files for minimization, relaxation, and production MD.
        If None, the defaults distributed with SHAKER are used.
    minOPT, relOPT, prodOPT : str, optional
        Extra command-line arguments passed to `gmx mdrun` for each stage
        (e.g. "-pin on -nt 8"). GROMACS's own wall-time cap, `-maxh`, can
        be passed here too.
    maxwarn : int, optional
        Value passed to `gmx grompp -maxwarn`. Default is 1.
    gmx_loc : str, optional
        Prefix/path to the GROMACS executable (e.g. "/usr/local/bin/" or "").
    cleanTraj : bool, optional
        If True, post-process the production trajectory using `_traj_cleanup`.
    cleanOldRun : bool, optional
        If True, remove files from previous runs using `_clean_old_files()`.
    mapping : dict, optional
        SHAKER mapping dictionary for the simulated molecule. If provided
        (and checkpointing is enabled via `keep_last`), saved as
        `mapping.json` alongside the checkpointed iteration for later
        reference.
    keep_last : int or None, optional
        Controls iteration checkpointing after this run (only takes effect
        if `cleanTraj=True`); see `_checkpoint`:

        - None — do not checkpoint at all.
        - -1 (default) — checkpoint and keep every iteration.
        - a positive integer N — checkpoint and keep only the N most
          recent iterations, pruning older ones.

        Use `list_iterations` to inspect saved iterations.
    hang_timeout : float or None, optional
        Seconds without any `mdrun` output after which the run is treated
        as hung, killed, and a RuntimeError raised. Default is 600. None
        disables this check. Independently, an `mdrun` that prints a fatal
        error but does not exit is always killed after a short grace period.
    verbose : bool, optional
        If False (default), GROMACS output goes to `gmx_run.log` and only one
        line per stage is printed: progress (ETA, 25 · 50 · 75) while it runs,
        then its result. A failing command raises with the last lines of
        output. If True, stream the full GROMACS output instead.

    Notes
    -----
    The function assumes the following files exist in the working directory:

    - `memion_2.gro` : starting coordinates
    - `topol.top`    : system topology

    These are the final outputs of `prepare_setup_water`.

    Output files use the prefixes `m`, `r`, and `p` corresponding to
    minimization, relaxation, and production stages.
    """

    if minMDP is None:
        minMDP = files("shaker.data.mdps") / "min.mdp"
    if relMDP is None:
        relMDP = files("shaker.data.mdps") / "rel.mdp"
    if prodMDP is None:
        prodMDP = files("shaker.data.mdps") / "prod.mdp"

    if cleanOldRun:
        _clean_old_files()

    gmx = str(Path(gmx_loc) / "gmx") if gmx_loc else "gmx"
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"

    # Quiet: all gmx output goes to gmx_run.log; errors still show its tail.
    log_ctx = contextlib.nullcontext() if verbose else open("gmx_run.log", "w")
    # (prefix, mdp, input coordinates, mdrun options); _STAGES names them.
    stages = (
        ("m", minMDP, "memion_2.gro", minOPT),  # minimize
        ("r", relMDP, "m.gro", relOPT),  # relax
        ("p", prodMDP, "r.gro", prodOPT),  # production
    )
    if not verbose:
        print("runSim · output → gmx_run.log")
    with log_ctx as log:
        for prefix, mdp, coords, opt in stages:
            # Quiet: the stage's line is started now and grows as it runs
            # (ETA, 25 · 50 · 75), then gets its result appended.
            label = f"  {_STAGES[prefix]:<6} "
            progress = None if verbose or prefix == "m" else _StageProgress(f"{prefix}.log")
            if not verbose:
                print(label, end="", flush=True)
            try:
                _run(
                    [
                        gmx,
                        "grompp",
                        "-f",
                        str(mdp),
                        "-c",
                        coords,
                        "-p",
                        "topol.top",
                        "-o",
                        f"{prefix}.tpr",
                        "-maxwarn",
                        str(maxwarn),
                    ],
                    log=log,
                    env=env,
                )
                _run_mdrun(
                    [gmx, "mdrun", "-v", "-deffnm", prefix, *shlex.split(opt)],
                    log=log,
                    env=env,
                    hang_timeout=hang_timeout,
                    on_output=progress,
                )
            except BaseException:
                # End the half-written line, so the error starts on its own.
                if not verbose:
                    print((progress.pad() if progress else "") + "✗ failed", flush=True)
                raise
            # Result line, e.g. so a -maxh-truncated production doesn't go
            # unnoticed. In verbose mode it follows mdrun's own output.
            result = _stage_result(_stage_report(f"{prefix}.log"))
            pad = progress.pad() if progress else ""
            print(("" if not verbose else label) + pad + result)

    if cleanTraj:
        _traj_cleanup("p.gro", "p.xtc", "p.tpr", gmx_loc=gmx_loc)
        _checkpoint(mapping=mapping, keep_last=keep_last)


def list_iterations(run_dir):
    """
    List checkpointed simulation iterations in a run directory.

    Iterations are created automatically by `runSim` (when `keep_last` is
    not None) as `iter_<N>_<timestamp>` folders, each containing the cleaned
    trajectory from that run (`pbc.pdb`/`pbc.xtc`), any `.itp` file(s)
    simulated, and `mapping.json` if a mapping was provided to `runSim`.

    Parameters
    ----------
    run_dir : str or Path
        Directory that `runSim` was executed in (e.g. the `CG_WAT` folder
        used in the tutorials).

    Returns
    -------
    list of Path
        Iteration folders found in `run_dir`, sorted oldest to newest by
        iteration index. Empty if none exist.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []

    found = []
    for entry in run_dir.iterdir():
        if not entry.is_dir():
            continue
        match = _ITER_RE.match(entry.name)
        if match:
            found.append((int(match.group(1)), entry))

    found.sort(key=lambda pair: pair[0])
    return [path for _, path in found]


def run_status(run_dir=".", verbose=True):
    """
    Report on the last `runSim` in `run_dir`, read from its m/r/p .log files.

    Useful when you did not watch the run yourself (the kernel died, an old
    folder, an agent picking up where it left off). `runSim` prints the same
    per-stage lines when it finishes.

    Parameters
    ----------
    run_dir : str or Path, optional
        Directory `runSim` was executed in. Default is ".".
    verbose : bool, optional
        If True (default), print one line per stage plus a summary line.

    Returns
    -------
    dict
        {"min"/"rel"/"prod": stage dict or None if no log,
         "done": cleaned trajectory present, "crash_files": [...],
         "iterations": [...]}. Stage dicts hold: finished, integrator,
        nsteps, steps_done, dt, ns_per_day, wall_s, stopped_early,
        lincs_warnings, fmax and emtol (minimization only).
    """
    run_dir = Path(run_dir)
    status = {
        name: _stage_report(run_dir / f"{prefix}.log")
        for prefix, name in _STAGES.items()
    }
    status["done"] = (run_dir / "pbc.pdb").is_file() and (
        run_dir / "pbc.xtc"
    ).is_file()
    status["crash_files"] = sorted(
        p.name for pattern in ("step*.pdb", "crash*") for p in run_dir.glob(pattern)
    )
    status["iterations"] = [p.name for p in list_iterations(run_dir)]

    if verbose:
        for name in _STAGES.values():
            print(f"  {name:<6} {_stage_result(status[name])}")
        crash = ", ".join(status["crash_files"]) or "none"
        print(
            f"  cleaned traj: {'yes' if status['done'] else 'no'}   "
            f"iterations: {len(status['iterations'])}   crash files: {crash}"
        )
    return status


_STAGES = {"m": "min", "r": "rel", "p": "prod"}
_EM_INTEGRATORS = ("steep", "cg", "l-bfgs")


def _stage_report(log_path):
    """
    Pull the few numbers worth reporting out of an mdrun .log (None if missing).

    Log wording can change between GROMACS versions, so every field is
    optional: a pattern that stops matching leaves that field None and it is
    simply left out of the report, never misreported.
    """
    path = Path(log_path)
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")

    def last(pattern, cast=float):
        found = re.findall(pattern, text, re.M)
        return cast(found[-1]) if found else None

    integrator = last(r"^\s+integrator\s+=\s+(\S+)", str)
    minimization = integrator in _EM_INTEGRATORS
    # -nsteps on the command line overrides the mdp value.
    nsteps = last(r"Overriding nsteps with value passed on the command line: (-?\d+)", int)
    if nsteps is None:
        nsteps = last(r"^\s+nsteps\s+=\s+(-?\d+)", int)
    if minimization:
        steps_done = last(r" in (\d+) steps", int)
    else:
        # MD averages are over steps_done + 1 frames (step 0 included).
        stats = last(r"Statistics over (\d+) steps", int)
        steps_done = stats - 1 if stats else None

    finished = "Finished mdrun" in text
    return {
        "finished": finished,
        "integrator": integrator,
        "nsteps": nsteps,
        "steps_done": steps_done,
        "dt": last(r"^\s+dt\s+=\s+(\S+)"),
        # Only MD reports ns/day; minimization has no simulated time.
        "ns_per_day": None if minimization else last(r"^Performance:\s+(\S+)"),
        "wall_s": last(r"^\s+Time:\s+\S+\s+(\S+)"),
        # Finished short of nsteps (-maxh, a signal, ...). Minimization may
        # legitimately stop early, so MD only.
        "stopped_early": (
            not minimization
            and finished
            and None not in (steps_done, nsteps)
            and 0 <= steps_done < nsteps
        ),
        "lincs_warnings": text.count("LINCS WARNING"),
        # Minimization: final max force vs. the target. Stopping at machine
        # precision above emtol is normal for Martini, so no flag for it.
        "fmax": last(r"^Maximum force\s+=\s+(\S+)") if minimization else None,
        "emtol": last(r"^\s+emtol\s+=\s+(\S+)") if minimization else None,
    }


def _stage_result(rep):
    """
    One-line result for a `_stage_report` dict, ending in ✓ or ⚠ + flags.
    """
    if rep is None:
        return "no log"

    parts = []
    if rep["integrator"] in _EM_INTEGRATORS:
        if rep["steps_done"]:
            parts.append(f"{rep['steps_done']} steps")
        if rep["fmax"] is not None:
            parts.append(f"Fmax {rep['fmax']:.1f} (target {rep['emtol']:g})")
    elif rep["steps_done"] is not None and rep["dt"] and rep["nsteps"]:
        done_ns = rep["steps_done"] * rep["dt"] / 1000
        total_ns = rep["nsteps"] * rep["dt"] / 1000
        parts.append(f"{done_ns:.1f}/{total_ns:.1f} ns")
    if rep["ns_per_day"]:
        parts.append(f"{rep['ns_per_day']:.0f} ns/day")
    if rep["wall_s"]:
        parts.append(str(timedelta(seconds=round(rep["wall_s"]))))

    flags = []
    if not rep["finished"]:
        flags.append("did not finish")
    if rep["stopped_early"]:
        flags.append("stopped early")
    if rep["lincs_warnings"]:
        flags.append(f"{rep['lincs_warnings']} LINCS warnings")
    parts.append("⚠ " + " · ".join(flags) if flags else "✓")
    return "   ".join(parts)


class _StageProgress:
    """
    Append-only progress for one MD stage in quiet mode: an ETA once ~5% is
    done, then 25 · 50 · 75. Fed the `mdrun -v` output by `_run_mdrun`; the
    planned step count comes from the stage .log header. Never rewrites a
    line (no \\r), so agents reading raw output get a few tokens, not a flood.
    """

    WIDTH = 42  # pad to this so the result columns line up across stages

    def __init__(self, log_path):
        self.log_path = log_path
        self.start = time.monotonic()
        self.nsteps = None
        self.eta_shown = False
        self.next_pct = 25
        self.printed = 0

    def __call__(self, text):
        steps = re.findall(r"step (\d+)", text)
        if not steps:
            return
        if self.nsteps is None:
            rep = _stage_report(self.log_path)
            self.nsteps = (rep and rep["nsteps"]) or 0
        if self.nsteps <= 0:  # unknown or infinite run: no progress
            return

        frac = int(steps[-1]) / self.nsteps
        if not self.eta_shown and 0.05 <= frac < 1:
            elapsed = time.monotonic() - self.start
            left = elapsed * (1 - frac) / frac
            eta = datetime.now() + timedelta(seconds=left)
            self._emit(f"ETA {eta:%H:%M} ({_duration(left)})")
            self.eta_shown = True
        while self.next_pct < 100 and frac * 100 >= self.next_pct:
            self._emit(f"{self.next_pct}")
            self.next_pct += 25

    def _emit(self, text):
        if self.printed:
            text = " · " + text
        print(text, end="", flush=True)
        self.printed += len(text)

    def pad(self):
        """Spaces to put after the progress so the result column lines up."""
        return " " * max(3, self.WIDTH - self.printed) if self.printed else ""


def _duration(seconds):
    """Rough human duration: '<1 min', '~9 min', '~1 h 20 min'."""
    minutes = round(seconds / 60)
    if minutes < 1:
        return "<1 min"
    if minutes < 60:
        return f"~{minutes} min"
    return f"~{minutes // 60} h {minutes % 60:02d} min"


def _run(cmd, *, log=None, env=None, input_text=None, cwd=None):
    """
    Short helper to assist when using subprocess to run gmx.

    When output goes to `log`, a failure raises with the log's tail in the
    message, since it was never shown.
    """
    try:
        subprocess.run(
        cmd,
        input=input_text,
        # No input → give it EOF, so a tool that unexpectedly prompts fails
        # instead of waiting forever.
            stdin=subprocess.DEVNULL if input_text is None else None,
            text=input_text is not None,
            stdout=log,
            stderr=subprocess.STDOUT,  # if log is None goes to term.
            env=env,
            cwd=cwd,
            check=True,
        )
    except subprocess.CalledProcessError as err:
        if log is None:
            raise
        log.flush()
        with open(log.name, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 20000))
            tail = f.read().decode(errors="replace")
        raise _GmxError(err.returncode, cmd, output=tail) from None


class _GmxError(subprocess.CalledProcessError):
    """
    CalledProcessError whose message ends with the last lines of gmx output.
    """

    def __str__(self):
        return f"{super().__str__()}\nLast output:\n{_tail_lines(self.output or '')}"


def _run_mdrun(
    cmd,
    *,
    log=None,
    env=None,
    hang_timeout=600,
    fatal_grace=30,
    poll=1.0,
    on_output=None,
):
    """
    Run `gmx mdrun`, blocking until it exits — but never forever.

    mdrun can crash without exiting (e.g. one thread-MPI rank hits a fatal
    error and the others wait for it forever). Output is written to `log`
    (an open file) or, if None, echoed to stdout as it arrives. The process
    is killed and a RuntimeError raised if:

    - it prints "Fatal error" and is still alive `fatal_grace` s later, or
    - it prints nothing for `hang_timeout` s (None disables this check).

    This relies on `mdrun -v`, whose progress lines arrive every few
    seconds; the stage .log is written too rarely to be useful. A non-zero
    exit raises CalledProcessError (with the output tail), as `_run` does.
    `on_output`, if given, is called with each chunk of output (progress). mdrun runs in its own
    process group and is always killed on the way out, Ctrl-C included.
    """
    # Popen only starts mdrun; the loop below is what keeps the cell blocked.
    # Own session/process group → Ctrl-C in the notebook doesn't reach mdrun
    # directly, and we can kill mdrun plus all its threads/children at once.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,  # we read it ourselves (echo + watchdog)
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )
    fd = proc.stdout.fileno()
    out = log if log is not None else sys.stdout
    tail = ""  # last ~20 kB of output, for the error message
    last_output = time.monotonic()  # when mdrun last printed anything
    fatal_at = None  # when "Fatal error" first showed up
    eof = False  # mdrun closed its output (normally: it exited)
    try:
        # Loop until mdrun has both closed its output and exited.
        while not (eof and proc.poll() is not None):
            if eof:
                time.sleep(poll)
            # Wait up to `poll` s for output. Waking up even when there is none
            # is what lets the timeout checks below run on a silent mdrun.
            elif select.select([fd], [], [], poll)[0]:
                # os.read returns whatever is available (no waiting for a
                # newline) — mdrun -v progress uses \r, not \n.
                chunk = os.read(fd, 65536)
                if chunk:
                    text = chunk.decode(errors="replace")
                    # Echo to the cell (as the old subprocess.run did) or log.
                    out.write(text)
                    out.flush()
                    if on_output is not None:
                        on_output(text)
                    tail = (tail + text)[-20000:]
                    last_output = time.monotonic()
                    if fatal_at is None and "Fatal error" in tail:
                        fatal_at = last_output
                else:
                    eof = True  # empty read = pipe closed

            # Watchdog: raising here jumps to `finally`, which kills mdrun.
            now = time.monotonic()
            if fatal_at is not None and now - fatal_at > fatal_grace:
                reason = f"still running {fatal_grace} s after a fatal error"
            elif hang_timeout is not None and now - last_output > hang_timeout:
                reason = f"no output for {hang_timeout} s"
            else:
                continue
            raise RuntimeError(
                f"mdrun hung ({reason}) and was killed.\n"
                f"Command: {shlex.join(map(str, cmd))}\n"
                f"Last output:\n{_tail_lines(tail)}"
            )
    finally:
        # Runs on normal exit, watchdog raise, Ctrl-C or any other error.
        # If mdrun is still alive at this point, it must not outlive us.
        proc.stdout.close()
        if proc.poll() is None:
            _kill_group(proc)

    if proc.returncode != 0:
        raise _GmxError(proc.returncode, cmd, output=tail)


def _kill_group(proc, grace=10):
    """
    SIGTERM the process group started by `_run_mdrun`, SIGKILL if it lingers.
    """
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        proc.wait()


def _tail_lines(text, n=20):
    """
    Last `n` non-empty lines of `text`, as a terminal would show them: mdrun
    -v rewrites its progress line with `\\r`, so only the last rewrite counts.
    """
    lines = [line.rstrip("\r").split("\r")[-1] for line in text.split("\n")]
    return "\n".join([line for line in lines if line.strip()][-n:])


def _traj_cleanup(gro, xtc, tpr, outname="pbc", gmx_loc=""):
    """
    Remove water and fix PBC after running the production.
    """
    u = md.Universe(gro)
    clean = u.select_atoms("not resname W ION NA CL")
    clean.write("index_clean.ndx", mode="w", name="clean")
    sel_in = "clean\n"

    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"
    gmx = str(Path(gmx_loc) / "gmx") if gmx_loc else "gmx"

    with open("gmx_traj_cleanup.log", "w") as log:
        # PBC-fixed trajectory
        _run(
            [
                gmx,
                "trjconv",
                "-f",
                str(xtc),
                "-s",
                str(tpr),
                "-o",
                f"{outname}.xtc",
                "-pbc",
                "mol",
                "-n",
                "index_clean.ndx",
            ],
            log=log,
            env=env,
            input_text=sel_in,
        )

        # First frame PDB w/ connects
        _run(
            [
                gmx,
                "trjconv",
                "-f",
                str(xtc),
                "-s",
                str(tpr),
                "-o",
                f"{outname}.pdb",
                "-pbc",
                "mol",
                "-b",
                "0",
                "-e",
                "0",
                "-conect",
                "-n",
                "index_clean.ndx",
            ],
            log=log,
            env=env,
            input_text=sel_in,
        )

    # Remove ENDMDL
    pdb_path = Path(f"{outname}.pdb")
    lines = pdb_path.read_text().splitlines(True)
    pdb_path.write_text("".join(l for l in lines if "ENDMDL" not in l))


def _clean_old_files():
    """
    Small function to clean old simulation files from previous iterations.
    """
    for pattern in ("m.*", "r.*", "p.*", "pbc.*", "step*", "crash*"):
        for f in glob.glob(pattern):
            if os.path.isfile(f):
                os.remove(f)


def _checkpoint(mapping=None, itp_glob="*.itp", keep_last=-1):
    """
    Save a snapshot of the just-completed run into a new iteration folder in
    the current working directory, then prune old iterations beyond
    `keep_last`.

    Snapshots `pbc.pdb`/`pbc.xtc` (written by `_traj_cleanup`) and any
    `.itp` file(s) matching `itp_glob` into `./iter_<N>_<timestamp>/`, where
    `N` is one more than the highest existing iteration index (see
    `list_iterations`). If `pbc.pdb`/`pbc.xtc` are not present (e.g.
    `cleanTraj=False`), nothing is saved.

    Parameters
    ----------
    mapping : dict, optional
        SHAKER mapping dictionary to save as `mapping.json` alongside the
        snapshot. Skipped if None.
    itp_glob : str, optional
        Glob pattern (relative to the working directory) used to find the
        solute `.itp` file(s) to snapshot. Default is "*.itp".
    keep_last : int or None, optional
        Controls whether/how many iterations are retained:

        - None — checkpointing is disabled; this call is a no-op.
        - -1 (default) — keep every iteration, no pruning.
        - a positive integer N — after saving this snapshot, delete the
          oldest iteration folders so that at most N remain.

    Returns
    -------
    Path or None
        Path to the newly created iteration folder, or None if
        checkpointing is disabled (`keep_last=None`) or there was no
        cleaned trajectory to snapshot.

    Raises
    ------
    ValueError
        If `keep_last` is 0 or a negative integer other than -1.
    """
    if keep_last is None:
        return None

    if keep_last != -1 and keep_last < 1:
        raise ValueError(
            "keep_last must be None (disable), -1 (keep all), or a positive integer."
        )

    cwd = Path.cwd()
    pbc_pdb = cwd / "pbc.pdb"
    pbc_xtc = cwd / "pbc.xtc"
    if not (pbc_pdb.exists() and pbc_xtc.exists()):
        return None

    existing = list_iterations(cwd)
    next_idx = (int(_ITER_RE.match(existing[-1].name).group(1)) + 1) if existing else 0

    local_tz = datetime.now().astimezone().tzinfo
    stamp = datetime.now(local_tz).strftime("%Y%m%d_%H%M%S")
    iter_dir = cwd / f"iter_{next_idx}_{stamp}"
    iter_dir.mkdir()

    shutil.copy2(pbc_pdb, iter_dir / "pbc.pdb")
    shutil.copy2(pbc_xtc, iter_dir / "pbc.xtc")

    for itp in cwd.glob(itp_glob):
        shutil.copy2(itp, iter_dir / itp.name)

    if mapping is not None:
        with open(iter_dir / "mapping.json", "w") as f:
            json.dump(mapping, f, indent=2)

    if keep_last != -1:
        all_iters = list_iterations(cwd)
        n_to_delete = max(0, len(all_iters) - keep_last)
        for old_dir in all_iters[:n_to_delete]:
            shutil.rmtree(old_dir)

    return iter_dir
