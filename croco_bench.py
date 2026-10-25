#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
croco_bench.py -- Find the most efficient run configuration for a CROCO simulation.

The script reads the CPU topology of the machine (physical cores, sockets, hyperthreads), derives the
possible parallel layouts (core counts and their x/y distribution) and runs the simulations that use few
cores simultaneously, each pinned to its own block of cores; the next runs are launched as soon as enough
cores are released. Every run works in its own directory, so concurrent runs never share croco.in or outputs.

For every experiment the script
  1. edits cppdefs.h  -> parallel mode (OPENMP or MPI) inside the active configuration block,
  2. edits param.h    -> number of cores and their x/y distribution
                         (MPI: NP_XI x NP_ETA ; OpenMP: NPP threads, NSUB_X x NSUB_E tiles),
  3. edits croco.in   -> time step dt and NTIMES so that the run covers `sim_days`
                         (output periods are rescaled so they stay constant in physical time),
  4. compiles with jobcomp one executable per parallel layout, all of them before the runs start
     (several at a time, in sibling directories of run_dir),
  5. runs the model, measures wall-clock time and detects BLOW UP / errors / timeouts,
  6. estimates CFL numbers (barotropic, internal-wave and advective) for each dt.

With all the accumulated results it writes a CSV, figures (PNG) and a Markdown report
that includes the fastest configuration and the optimal CFL condition.

The original cppdefs.h, param.h, croco.in and croco executable are always restored at the end
(also on Ctrl-C / SIGTERM). Results are appended to a CSV, so an interrupted benchmark can be
resumed by launching the same command again.

Usage
-----
    python3 croco_bench.py --init-config bench.json   # write an example configuration
    python3 croco_bench.py -c bench.json --dry-run    # show plan + file edits, nothing is run
    python3 croco_bench.py -c bench.json              # run the benchmark
    python3 croco_bench.py -c bench.json --report-only

Requirements: Python >= 3.8, numpy, matplotlib. Optional: netCDF4 (or scipy) for the CFL
estimates from the grid and history files.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import platform
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

G = 9.81
DAY = 86400.0

# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # --- CROCO case -------------------------------------------------------------------------
    "run_dir": ".",                 # directory with cppdefs.h, param.h, croco.in, jobcomp
    "cppdefs": "cppdefs.h",
    "param": "param.h",
    "croco_in": "croco.in",
    "compile_cmd": "./jobcomp",     # run inside run_dir
    "executable": "croco",          # produced by compile_cmd inside run_dir
    "config_name": None,            # e.g. "REGIONAL"; None = auto-detect from cppdefs.h
    "grid_size": None,              # [LLm, MMm]; None = read from param.h

    # --- search space -----------------------------------------------------------------------
    "modes": ["openmp", "mpi"],
    "ref_cores": 12,                # stage 2 starts with this number of cores (OpenMP and MPI)
    "core_step": 2,                 # E > threshold -> + core_step cores; otherwise - core_step
    "decompositions": "auto",       # "auto" = every nx*ny factorisation, or {"4": [[1,4],[2,2]]}
    "omp_tile_factors": [1],        # OpenMP: tiles = cores*factor (e.g. [1,2] also tests 2 tiles/thread)
    "mpi_tiles": [1, 1],            # NSUB_X, NSUB_E used inside each MPI subdomain
    "require_divisible": False,     # MPI: keep only layouts with LLm%nx==0 and MMm%ny==0
    "min_points_per_subdomain": 4,  # skip layouts giving subdomains/tiles thinner than this
    "sim_days": 1.0,                # simulated period of every run (days)
    "repeat": 1,                    # repetitions of each timed run (median is used)

    # --- machine / concurrency ----------------------------------------------------------------
    "max_cores": None,              # maximum number of cores to use (set it by hand); None = all physical cores
    "use_hyperthreads": False,      # count SMT siblings as extra cores
    "reserve_cores": 0,             # leave the first N cores free (OS, other users)
    "cpu_list": None,               # explicit logical CPU ids to use, overrides the detection
    "max_parallel_jobs": None,      # None = as many simultaneous runs as cores allow; 1 = sequential
    "efficiency_study": True,       # speed-up / efficiency table and plot (NLHPC "Estudio de Eficiencia")
    "efficiency_threshold": 0.5,    # E = T(1)/(n T(n)) above this: try more cores; otherwise fewer
    "efficiency_exclusive": True,   # re-time alone the points of the efficiency curve measured sharing
    "build_mode": "parallel",       # "parallel": sibling build dirs; "inplace": one at a time in run_dir
    "build_jobs": 4,                # simultaneous compilations in parallel build mode
    "keep_outputs": False,          # keep the .nc outputs of every run in output_dir/work/<run>

    # --- launch order (stages 2-4) -------------------------------------------------------------
    # "large_first": the run with the most cores and the largest dt first, then decreasing
    # "small_first": runs with few cores first, several at once
    "launch_order": "large_first",
    "live_report": True,            # rebuild figures and report after every finished simulation
    "pdf_report": True,             # convert the final benchmark_report.md to benchmark_report.pdf
    "latex_report": True,           # also write benchmark_report.tex (same content, standard LaTeX packages)

    # --- stage 1: time-step search with the serial build (1 core, no OpenMP/MPI) ------------
    # dt_ref, doubled while stable; after the first BLOW UP the next dt is the midpoint between the
    # last stable dt and the last BLOW UP; stops after two BLOW UPs in a row. Several runs at once.
    "dt_ref": None,                 # reference dt (s); None = the dt in croco.in
    "min_steps": 10,                # never use a dt that leaves fewer time steps than this in sim_days
    "dt_tol": 0.01,                 # also stop when (dt_blowup - dt_stable)/dt_blowup is below this
    "dt_hour_divisor": True,        # optimal dt = largest integer <= the largest stable dt with 3600/dt integer
                                    # (stage 2 and T(1) run at that dt)
    # "strict": only runs that can be needed whatever the pending results are (never above a dt that may
    #           blow up, never below half of a stable dt whose half is stable): at most 2 serial runs at once
    # "aggressive": speculative look-ahead with as many runs as cores (unneeded runs are cancelled)
    "dt_speculation": "strict",
    "max_dt_runs": 30,              # safety limit on the number of serial runs of the search
    "serial_ref_exclusive": True,   # re-time T(1) alone if the serial run at the optimal dt shared the machine
    "explore_exclusive": True,      # stage 2 runs one at a time, so E(n) decisions are not biased by contention

    # --- stage 2: cores exploration (see ref_cores / core_step / efficiency_threshold) ------
    "blowup_is_global": True,       # a dt that blew up is not run with other layouts (running ones are killed)
    "backfill": True,               # fill idle cores with smaller runs that do not delay the queue head
    "backfill_safety": 1.25,        # a backfill run must end before the head can start, with this margin

    # --- croco.in handling ------------------------------------------------------------------
    "ndtfast_mode": "fixed",        # "fixed" keeps NDTFAST; "keep_dtfast" keeps dt/NDTFAST constant
    "scale_output_periods": True,   # keep output periods constant in seconds when dt changes
    "scale_keywords": {             # croco.in keyword -> indices (0-based) of step counts to rescale
        "history:": [1],            # LDEFHIS, NWRT, NRPFHIS
        "averages:": [0, 1],        # NTSAVG, NAVG, NRPFAVG
        "restart:": [0],            # NRST, NRPFRST
        "diagnostics:": [1],        # LDEFDIA, NWRTDIA, NRPFDIA
    },

    # croco.in keywords whose files are outputs ("all" or indices of the file names in the block);
    # every other existing file is linked read-only into each run directory
    "output_keywords": {
        "history:": "all", "averages:": "all", "restart:": "all", "diagnostics:": "all",
        "diagnosticsM:": "all", "diagnostics_bio:": "all", "diags_vrt:": "all", "diags_ek:": "all",
        "diags_pv:": "all", "diags_eddy:": "all", "surf:": "all", "floats:": [1], "stations:": [1],
    },

    # --- execution --------------------------------------------------------------------------
    # placeholders: {exe} {np} {nx} {ny} {croco_in} {cpus} (e.g. 4,5,6,7) {places} (e.g. {4},{5})
    "run_cmd": {
        "openmp": "taskset -c {cpus} {exe} {croco_in}",
        "mpi": "taskset -c {cpus} mpirun --bind-to none -np {np} {exe} {croco_in}",
        "serial": "taskset -c {cpus} {exe} {croco_in}",
    },
    "env": {
        "openmp": {"OMP_NUM_THREADS": "{np}", "OMP_PLACES": "{places}", "OMP_PROC_BIND": "close",
                   "OMP_SCHEDULE": "static", "OMP_DYNAMIC": "false", "OMP_STACKSIZE": "512M"},
        "mpi": {"OMP_NUM_THREADS": "1"},
        "serial": {"OMP_NUM_THREADS": "1"},
    },
    # --- HPC (Slurm) ----------------------------------------------------------------------------
    # "auto": Slurm when running inside a Slurm job (SLURM_JOB_ID set), local otherwise
    "launcher": "auto",
    # flags of every 'srun' job step; '--exact' lets several steps share the allocation without overlapping
    # (Slurm >= 21.08; use '--exclusive' on older versions)
    "slurm_step_flags": "--exact --cpu-bind=cores",
    "walltime_margin_s": 600,       # stop launching runs this long before the Slurm job time limit
    "timeout_s": 6 * 3600,          # per run
    "compile_timeout_s": 1800,
    "blowup_grace_s": 30,           # kill the run this long after BLOW UP appears in the log
    "poll_s": 2.0,

    # --- log parsing ------------------------------------------------------------------------
    "blowup_regex": r"BLOW\s*-?\s*UP|\bNaN\b",
    "done_regex": r"MAIN\s*:\s*DONE",
    "error_regex": r"Abnormal termination|Segmentation fault|forrtl: severe|MPI_ABORT|"
                   r"Program received signal|ERROR STOP",

    # --- CFL --------------------------------------------------------------------------------
    "grid_file": None,              # None = read 'grid:' entry of croco.in
    "internal_wave_speed": 2.0,     # m/s, assumed first baroclinic mode speed (internal CFL estimate)

    # --- output -----------------------------------------------------------------------------
    "output_dir": "bench_results",
}

EXAMPLE_COMMENT = (
    "Example configuration for croco_bench.py. Paths are relative to this file. "
    "Remove keys you do not need: missing keys take the defaults documented in the script."
)


def load_config(path: Path | None) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    base = Path.cwd()
    user = {}
    if path:
        with open(path) as f:
            user = json.load(f)
        user.pop("_comment", None)
        for k, v in user.items():
            if k not in cfg:
                print(f"[warn] unknown config key '{k}' (ignored)")
                continue
            if isinstance(cfg[k], dict) and isinstance(v, dict) and k not in ("scale_keywords",
                                                                               "output_keywords"):
                cfg[k].update(v)
            else:
                cfg[k] = v
        base = path.resolve().parent
    cfg["run_dir"] = str((base / cfg["run_dir"]).resolve())
    out = Path(cfg["output_dir"])
    cfg["output_dir"] = str(out if out.is_absolute() else (base / out).resolve())
    for m in cfg["modes"]:
        if m not in ("openmp", "mpi"):
            sys.exit(f"[error] unknown mode '{m}' (use 'openmp' and/or 'mpi')")
    # launcher: local (taskset/mpirun) or Slurm job steps (srun) inside one allocation
    lau = cfg["launcher"]
    if lau == "auto":
        lau = "slurm" if os.environ.get("SLURM_JOB_ID") else "local"
    if lau not in ("local", "slurm"):
        sys.exit("[error] launcher must be 'auto', 'local' or 'slurm'")
    cfg["launcher"] = lau
    if lau == "slurm":
        urc = user.get("run_cmd", {}) if isinstance(user.get("run_cmd"), dict) else {}
        for mode, tpl in SLURM_RUN_CMD.items():
            if mode not in urc:          # templates written by the user are kept as they are
                cfg["run_cmd"][mode] = tpl
        uenv = (user.get("env") or {}).get("openmp", {})
        if "OMP_PLACES" not in uenv:     # Slurm chooses the cores of each step
            cfg["env"]["openmp"]["OMP_PLACES"] = "cores"
    return cfg


# job-step templates used inside a Slurm allocation ({slurm_flags} = slurm_step_flags)
SLURM_RUN_CMD = {
    "openmp": "srun {slurm_flags} -N 1 -n 1 -c {np} {exe} {croco_in}",
    "mpi": "srun {slurm_flags} -N 1 -n {np} -c 1 {exe} {croco_in}",
    "serial": "srun {slurm_flags} -N 1 -n 1 -c 1 {exe} {croco_in}",
}


class WallTimeUp(Exception):
    """The Slurm job is close to its time limit: stop cleanly, results are kept for a resubmission."""


def slurm_time_left():
    """Seconds left in the current Slurm job (None if unknown or unlimited)."""
    jid = os.environ.get("SLURM_JOB_ID")
    if not jid or not shutil.which("squeue"):
        return None
    try:
        out = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%L"], capture_output=True, text=True,
                             timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)", out)
    if not m:
        return None  # e.g. UNLIMITED / NOT_SET
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + se


# ----------------------------------------------------------------------------------------------
# C-preprocessor helpers (cppdefs.h / param.h)
# ----------------------------------------------------------------------------------------------
PP_IF = re.compile(r"^\s*#\s*(if|ifdef|ifndef)\b(.*)$")
PP_ELIF = re.compile(r"^\s*#\s*elif\b(.*)$")
PP_ELSE = re.compile(r"^\s*#\s*else\b")
PP_ENDIF = re.compile(r"^\s*#\s*endif\b")
C_COMMENT = re.compile(r"/\*.*?\*/")


def _clean(s: str) -> str:
    return C_COMMENT.sub("", s).strip()


def if_chains(lines):
    """Return every #if/#elif/#else/#endif chain as
    {'depth': d, 'branches': [[cond, first_line, end_line_exclusive], ...], 'end': endif_line}."""
    stack, chains = [], []
    for i, ln in enumerate(lines):
        m = PP_IF.match(ln)
        if m:
            kind, rest = m.group(1), _clean(m.group(2))
            cond = {"if": rest, "ifdef": f"defined {rest}", "ifndef": f"!defined {rest}"}[kind]
            stack.append({"depth": len(stack), "branches": [[cond, i, None]]})
            continue
        m = PP_ELIF.match(ln)
        if m and stack:
            br = stack[-1]["branches"]
            br[-1][2] = i
            br.append([_clean(m.group(1)), i, None])
            continue
        if PP_ELSE.match(ln) and stack:
            br = stack[-1]["branches"]
            br[-1][2] = i
            br.append(["else", i, None])
            continue
        if PP_ENDIF.match(ln) and stack:
            ch = stack.pop()
            ch["branches"][-1][2] = i
            ch["end"] = i
            chains.append(ch)
    return chains


def cond_has(cond: str, name: str) -> bool:
    """True if `defined NAME` appears (not negated) in a preprocessor condition."""
    for m in re.finditer(r"(!?)\s*defined\s*\(?\s*(\w+)", cond):
        if m.group(2) == name and m.group(1) != "!":
            return True
    return False


def detect_config_name(cppdefs_text: str) -> str | None:
    lines = cppdefs_text.splitlines()
    cands = []
    for ln in lines:
        if PP_IF.match(ln):
            break
        m = re.match(r"^\s*#\s*define\s+(\w+)", ln)
        if m:
            cands.append(m.group(1))
    if not cands:
        return None
    conds = [b[0] for ch in if_chains(lines) for b in ch["branches"]]
    for c in cands:
        if any(cond_has(cd, c) for cd in conds):
            return c
    return cands[0]


def config_block(lines, name):
    """(start, end) line range of the branch `#if/#elif defined NAME` with the smallest depth."""
    best = None
    for ch in if_chains(lines):
        for cond, s, e in ch["branches"]:
            if cond != "else" and cond_has(cond, name):
                if best is None or ch["depth"] < best[0]:
                    best = (ch["depth"], s, e)
    return None if best is None else (best[1], best[2])


def set_parallel_mode(text: str, mode: str, config_name: str | None) -> tuple[str, list[str]]:
    """Switch OPENMP / MPI on or off in cppdefs.h (restricted to the active configuration block)."""
    notes = []
    lines = text.splitlines(keepends=True)
    rng = config_block(lines, config_name) if config_name else None
    if rng is None:
        rng = (0, len(lines))
        notes.append("active configuration block not found: edited OPENMP/MPI in the whole file")
    want = {"OPENMP": mode == "openmp", "MPI": mode == "mpi"}
    pat = re.compile(r"^(\s*#\s*)(define|undef)(\s+)(OPENMP|MPI)\b(.*)$", re.S)
    found = set()
    for i in range(rng[0], rng[1]):
        m = pat.match(lines[i])
        if m:
            name = m.group(4)
            found.add(name)
            kw = "define" if want[name] else "undef"
            sp = m.group(3) if kw == m.group(2) else " "
            lines[i] = f"{m.group(1)}{kw}{sp}{name}{m.group(5)}"
    missing = [n for n in ("OPENMP", "MPI") if n not in found]
    if missing:
        ins = "".join(f"# {'define' if want[n] else 'undef'} {n}\n" for n in missing)
        lines.insert(rng[0] + 1, ins)
        notes.append(f"added {', '.join(missing)} lines after the configuration header")
    return "".join(lines), notes


def _sub_count(pattern, repl, s):
    new, n = re.subn(pattern, repl, s)
    return new, n


def parallel_chain(lines):
    for ch in if_chains(lines):
        if any(cond_has(b[0], "MPI") for b in ch["branches"]):
            s, e = ch["branches"][0][1], ch["end"]
            if any(re.search(r"NP_XI\s*=", lines[k]) for k in range(s, e)):
                return ch
    return None


SERIAL = {"mode": "serial", "np": 1, "nx": 1, "ny": 1}  # 1 core, OPENMP and MPI undefined


def set_decomposition(text: str, mode: str, np_: int, nx: int, ny: int,
                      mpi_tiles=(1, 1)) -> tuple[str, list[str]]:
    """Set NP_XI/NP_ETA (MPI) or NPP/NSUB_X/NSUB_E (OpenMP) in param.h."""
    notes = []
    if mode == "serial":  # the '#else' branch of param.h (NPP=1, one tile) is used as it is
        return text, notes
    lines = text.splitlines(keepends=True)
    ch = parallel_chain(lines)
    if ch is None:
        raise RuntimeError("param.h: could not find the '#ifdef MPI ... NP_XI=' parallel block")
    counts = {}
    for cond, s, e in ch["branches"]:
        kind = "mpi" if cond_has(cond, "MPI") else "omp" if cond_has(cond, "OPENMP") else "other"
        if kind != ("mpi" if mode == "mpi" else "omp"):
            continue
        for k in range(s + 1, e):
            ln = lines[k]
            if re.match(r"^\s*[cC!*]", ln) and not ln.lstrip().startswith("#"):
                continue  # Fortran comment line
            if kind == "mpi":
                reps = [(r"(\bNP_XI\s*=\s*)\d+", nx, "NP_XI"), (r"(\bNP_ETA\s*=\s*)\d+", ny, "NP_ETA"),
                        (r"(\bNSUB_X\s*=\s*)\w+", mpi_tiles[0], "NSUB_X"),
                        (r"(\bNSUB_E\s*=\s*)\w+", mpi_tiles[1], "NSUB_E")]
            else:
                reps = [(r"(\bNPP\s*=\s*)\d+", np_, "NPP"), (r"(\bNSUB_X\s*=\s*)\w+", nx, "NSUB_X"),
                        (r"(\bNSUB_E\s*=\s*)\w+", ny, "NSUB_E")]
            for pat, val, key in reps:
                ln, n = _sub_count(pat, rf"\g<1>{val}", ln)
                counts[key] = counts.get(key, 0) + n
            if ln != lines[k] and len(ln.rstrip("\n")) > 72:
                notes.append(f"param.h line {k + 1} is longer than 72 columns (fixed-form Fortran)")
            lines[k] = ln
    need = ["NP_XI", "NP_ETA"] if mode == "mpi" else ["NPP", "NSUB_X", "NSUB_E"]
    miss = [k for k in need if not counts.get(k)]
    if miss:
        raise RuntimeError(f"param.h: could not set {', '.join(miss)} in the {mode} branch")
    if mode == "openmp":
        body = "".join(lines[ch["branches"][0][1]:ch["end"]])
        if re.search(r"AUTOTILING", body) and re.search(r"#\s*define\s+AUTOTILING", text):
            notes.append("AUTOTILING is defined: NSUB_X/NSUB_E are chosen by CROCO at run time")
    return "".join(lines), notes


class MiniCPP:
    """Small C-preprocessor evaluator: follows #if/#ifdef/#ifndef/#elif/#else/#endif and applies
    #define/#undef in the active branches (conditions with defined, !, &&, ||, parentheses and
    integer comparisons). #include is ignored. Enough to know which options of cppdefs.h are on and
    which lines of param.h are compiled."""

    def __init__(self, macros=None):
        self.macros = dict(macros or {})

    def eval_cond(self, expr):
        e = C_COMMENT.sub("", expr)
        e = re.sub(r"//.*", "", e)
        e = re.sub(r"defined\s*\(\s*(\w+)\s*\)", lambda m: " 1 " if m.group(1) in self.macros else " 0 ", e)
        e = re.sub(r"defined\s+(\w+)", lambda m: " 1 " if m.group(1) in self.macros else " 0 ", e)

        def ident(m):
            v = self.macros.get(m.group(0))
            if v is None:
                return " 0 "
            v = str(v).strip()
            return f" {v} " if re.fullmatch(r"-?\d+", v) else " 1 "
        e = re.sub(r"\b[A-Za-z_]\w*\b", ident, e)
        e = e.replace("&&", " and ").replace("||", " or ")
        e = re.sub(r"!(?!=)", " not ", e)
        try:
            return bool(eval(e, {"__builtins__": {}}, {}))  # noqa: S307 - only digits/operators remain
        except Exception:  # noqa: BLE001
            return False

    def process(self, text):
        """Returns, for every line, True if it is an active (compiled) non-directive line."""
        stack, active, flags = [], True, []
        for ln in text.splitlines():
            m = re.match(r"^\s*#\s*(\w+)\s*(.*)$", ln)
            if not m:
                flags.append(active)
                continue
            d, rest = m.group(1), C_COMMENT.sub("", m.group(2)).strip()
            flags.append(False)
            if d in ("if", "ifdef", "ifndef"):
                name = rest.split()[0] if rest.split() else ""
                c = (name in self.macros) if d == "ifdef" else (name not in self.macros) if d == "ifndef" \
                    else self.eval_cond(rest)
                c = c and active
                stack.append([active, c])
                active = c
            elif d == "elif" and stack:
                par, taken = stack[-1]
                c = par and not taken and self.eval_cond(rest)
                stack[-1][1] = taken or c
                active = c
            elif d == "else" and stack:
                par, taken = stack[-1]
                active = par and not taken
                stack[-1][1] = True
            elif d == "endif" and stack:
                active = stack.pop()[0]
            elif active and d == "define":
                parts = rest.split(None, 1)
                if parts:
                    self.macros[re.sub(r"\(.*", "", parts[0])] = parts[1] if len(parts) > 1 else "1"
            elif active and d == "undef":
                parts = rest.split()
                if parts:
                    self.macros.pop(parts[0], None)
        return flags


def case_name_in_block(cppdefs_text: str, block: str | None):
    """First option #defined in a compiled line of the configuration block, e.g. BENGUELA_LR under
    REGIONAL ("Configuration Name" in the standard cppdefs.h)."""
    if not block:
        return None
    lines = cppdefs_text.splitlines()
    rng = config_block(lines, block)
    if not rng:
        return None
    for k in range(rng[0] + 1, rng[1]):
        m = re.match(r"^\s*#\s*define\s+(\w+)", lines[k])
        if m and _line_compiled(lines, k):
            return m.group(1)
    return None


def _line_compiled(lines, k):
    """True if line k sits in a compiled region of the file."""
    probe = lines[:k] + ["__PROBE__"] + lines[k + 1:]
    return MiniCPP().process("\n".join(probe))[k]


def grid_from_sources(cppdefs_text: str, param_text: str):
    """Grid dimensions of the case actually selected by cppdefs.h: the active
    'parameter (LLm0=..., MMm0=..., N=...)' line of param.h and the options that select it."""
    pp = MiniCPP()
    pp.process(cppdefs_text)
    defined = set(pp.macros)
    flags = pp.process(param_text)
    lines = param_text.splitlines()
    for i, (ln, act) in enumerate(zip(lines, flags)):
        if not act or re.match(r"^\s*[!cC*]", ln):
            continue
        code = ln.split("!")[0]
        mL = re.search(r"\bLLm0?\s*=\s*(\d+)", code)
        mM = re.search(r"\bMMm0?\s*=\s*(\d+)", code)
        if not (mL and mM):
            continue
        mN = re.search(r"\bN\s*=\s*(\d+)", code)
        chain = []
        for ch in sorted(if_chains(lines), key=lambda c: c["depth"]):
            for cond, s, e in ch["branches"]:
                if s < i < e:
                    names = [n for n in re.findall(r"defined\s*\(?\s*(\w+)", cond) if n in defined]
                    chain.append(" ".join(names) if cond != "else" and names else
                                 ("#else" if cond == "else" else cond))
        return {"LLm": int(mL.group(1)), "MMm": int(mM.group(1)),
                "N": int(mN.group(1)) if mN else None, "line": i + 1, "chain": chain,
                "case": chain[-1] if chain else None, "defined": sorted(defined)}
    return None


def grid_size_from_param(text: str, config_name: str | None):
    lines = text.splitlines()
    pat = re.compile(r"LLm0?\s*=\s*(\d+)\s*,\s*MMm0?\s*=\s*(\d+)")
    if config_name:
        rng = config_block(lines, config_name)
        if rng:
            for k in range(*rng):
                m = pat.search(lines[k])
                if m:
                    return int(m.group(1)), int(m.group(2))
    hits = [pat.search(ln) for ln in lines]
    hits = [h for h in hits if h]
    if len(hits) == 1:
        return int(hits[0].group(1)), int(hits[0].group(2))
    return None


# ----------------------------------------------------------------------------------------------
# croco.in helpers
# ----------------------------------------------------------------------------------------------
def _is_blank_or_comment(ln: str) -> bool:
    s = ln.strip()
    return not s or s.startswith("!")


def keyword_lines(lines, keyword):
    """Index of the keyword line and of the following value lines (non blank)."""
    for i, ln in enumerate(lines):
        if ln.lstrip().lower().startswith(keyword.lower()):
            vals = []
            j = i + 1
            while j < len(lines) and len(vals) < 3:
                if not _is_blank_or_comment(lines[j]):
                    if re.match(r"^\s*[A-Za-z_]\w*\s*:", lines[j]):  # next keyword
                        break
                    vals.append(j)
                j += 1
            return i, vals
    return None, []


def replace_tokens(line: str, new: dict, right_align: bool = True) -> str:
    toks = list(re.finditer(r"\S+", line))
    out, pos = [], 0
    for idx, m in enumerate(toks):
        out.append(line[pos:m.start()])
        if idx in new and not right_align:
            out.append(str(new[idx]))
        elif idx in new:
            s = str(new[idx])
            # keep the column where the token ends (right aligned tables) when possible
            pad = len(m.group()) - len(s)
            if pad > 0 and out and out[-1].endswith(" "):
                s = " " * pad + s
            elif pad < 0 and out and len(out[-1]) > -pad + 1:
                out[-1] = out[-1][:pad]
            out.append(s)
        else:
            out.append(m.group())
        pos = m.end()
    out.append(line[pos:])
    return "".join(out)


def _num(tok: str) -> float:
    return float(tok.lower().replace("d", "e"))


def _fmt_dt(dt: float) -> str:
    return str(int(dt)) if float(dt).is_integer() else f"{dt:g}"


def read_time_stepping(text: str):
    lines = text.splitlines(keepends=True)
    i, vals = keyword_lines(lines, "time_stepping:")
    if i is None or not vals:
        raise RuntimeError("croco.in: 'time_stepping:' entry not found")
    toks = lines[vals[0]].split()
    return {"ntimes": int(_num(toks[0])), "dt": _num(toks[1]), "ndtfast": int(_num(toks[2])),
            "ninfo": int(_num(toks[3])) if len(toks) > 3 else 1}


def read_filename(text: str, keyword: str, after_values: bool):
    lines = text.splitlines()
    i, vals = keyword_lines(lines, keyword)
    if i is None or not vals:
        return None
    k = 1 if after_values else 0
    if len(vals) <= k:
        return None
    tok = lines[vals[k]].split()
    return tok[0] if tok else None


def set_croco_in(text: str, cfg: dict, dt: float) -> tuple[str, dict, list[str]]:
    notes = []
    lines = text.splitlines(keepends=True)
    ts0 = read_time_stepping(text)
    dt0 = ts0["dt"]
    ntimes = int(math.ceil(round(cfg["sim_days"] * DAY / dt, 6)))
    if cfg["ndtfast_mode"] == "keep_dtfast":
        ndtfast = max(1, int(round(ts0["ndtfast"] * dt / dt0)))
    else:
        ndtfast = ts0["ndtfast"]
    i, vals = keyword_lines(lines, "time_stepping:")
    lines[vals[0]] = replace_tokens(lines[vals[0]], {0: ntimes, 1: _fmt_dt(dt), 2: ndtfast})
    if abs(ntimes * dt - cfg["sim_days"] * DAY) > 1e-6:
        notes.append(f"dt={_fmt_dt(dt)} does not divide {cfg['sim_days']} days exactly; "
                     f"NTIMES={ntimes} simulates {ntimes * dt / DAY:.4f} days")
    if cfg["scale_output_periods"] and dt != dt0:
        for kw, idxs in cfg["scale_keywords"].items():
            k, v = keyword_lines(lines, kw)
            if k is None or not v:
                continue
            toks = lines[v[0]].split()
            new = {}
            for ix in idxs:
                if ix < len(toks):
                    try:
                        val = int(_num(toks[ix]))
                    except ValueError:
                        continue
                    if val > 1:
                        nv = max(1, int(round(val * dt0 / dt)))
                        if abs(nv * dt - val * dt0) > 1e-6:
                            notes.append(f"{kw} period {val * dt0:g}s -> {nv * dt:g}s (not an exact multiple)")
                        new[ix] = nv
            if new:
                lines[v[0]] = replace_tokens(lines[v[0]], new)
    # at least one history record (needed for the advective CFL) when the output period > run length
    k, v = keyword_lines(lines, "history:")
    if k is not None and v:
        toks = lines[v[0]].split()
        try:
            nwrt = int(_num(toks[1]))
        except (IndexError, ValueError):
            nwrt = 0
        if nwrt > ntimes:
            lines[v[0]] = replace_tokens(lines[v[0]], {1: ntimes})
            notes.append(f"history NWRT {nwrt} > NTIMES {ntimes}: set to {ntimes} (one record at the end)")
    return "".join(lines), {"ntimes": ntimes, "ndtfast": ndtfast, "dtfast": dt / ndtfast}, notes


# ----------------------------------------------------------------------------------------------
# NetCDF / CFL
# ----------------------------------------------------------------------------------------------
def _open_nc(path):
    try:
        import netCDF4
        return netCDF4.Dataset(path), "netCDF4"
    except ImportError:
        pass
    from scipy.io import netcdf_file
    return netcdf_file(path, "r", mmap=False), "scipy"


def _arr(var, idx=None):
    import numpy as np
    a = var[idx] if idx is not None else var[:]
    a = np.ma.filled(np.ma.asarray(a).astype(float), np.nan)
    fv = getattr(var, "_FillValue", None)
    if fv is not None:
        a[a == fv] = np.nan
    a[np.abs(a) > 1e30] = np.nan
    return a


def grid_metrics(path: Path, c_int: float):
    import numpy as np
    ds, _ = _open_nc(str(path))
    try:
        pm, pn, h = _arr(ds.variables["pm"]), _arr(ds.variables["pn"]), _arr(ds.variables["h"])
        mask = _arr(ds.variables["mask_rho"]) if "mask_rho" in ds.variables else np.ones_like(h)
    finally:
        ds.close()
    wet = mask > 0.5
    inv = np.sqrt(pm ** 2 + pn ** 2)
    ext = np.sqrt(G * np.maximum(h, 0.0)) * inv
    return {
        "ext_factor": float(np.nanmax(np.where(wet, ext, np.nan))),  # CFL_bt = dtfast * ext_factor
        "int_factor": float(c_int * np.nanmax(np.where(wet, inv, np.nan))),  # CFL_int = dt * int_factor
        "dx_min": float(1.0 / np.nanmax(np.where(wet, pm, np.nan))),
        "dy_min": float(1.0 / np.nanmax(np.where(wet, pn, np.nan))),
        "h_max": float(np.nanmax(np.where(wet, h, np.nan))),
        "pm": pm, "pn": pn,
    }


def advective_factors(his: Path, grid: dict | None):
    """Max over the history file of |u|/dx and |v|/dy (1/s); CFL_adv = dt * (fx + fy)."""
    import numpy as np
    ds, _ = _open_nc(str(his))
    try:
        V = ds.variables
        un = "u" if "u" in V else "ubar" if "ubar" in V else None
        vn = "v" if "v" in V else "vbar" if "vbar" in V else None
        if not un or not vn:
            return None
        nt = V[un].shape[0] if V[un].ndim >= 3 else 1
        if grid is not None:
            pm, pn = grid["pm"], grid["pn"]
            pm_u = 0.5 * (pm[:, 1:] + pm[:, :-1])
            pn_v = 0.5 * (pn[1:, :] + pn[:-1, :])
        fx = fy = umax = vmax = 0.0
        for t in range(nt):
            u = np.abs(_arr(V[un], t if V[un].ndim >= 3 else None))
            v = np.abs(_arr(V[vn], t if V[vn].ndim >= 3 else None))
            if not np.isfinite(u).any():
                continue
            umax = max(umax, float(np.nanmax(u)))
            vmax = max(vmax, float(np.nanmax(v)))
            if grid is not None and u.shape[-2:] == pm_u.shape and v.shape[-2:] == pn_v.shape:
                fx = max(fx, float(np.nanmax(u * pm_u)))
                fy = max(fy, float(np.nanmax(v * pn_v)))
            elif grid is not None:
                fx = max(fx, umax / grid["dx_min"])
                fy = max(fy, vmax / grid["dy_min"])
    finally:
        ds.close()
    return {"umax": umax, "vmax": vmax, "fx": fx, "fy": fy}


# ----------------------------------------------------------------------------------------------
# Log parsing
# ----------------------------------------------------------------------------------------------
STEP_LINE = re.compile(r"^\s*(\d+)\s+(-?\d+\.\d*(?:[EeDd][+-]?\d+)?)\s+(\S+)\s+(\S+)", re.M)


def parse_log(text: str, cfg: dict):
    steps = [(int(m.group(1)), float(m.group(2).lower().replace("d", "e")))
             for m in STEP_LINE.finditer(text)]
    days = (steps[-1][1] - steps[0][1]) if len(steps) > 1 else 0.0
    return {
        "blowup": bool(re.search(cfg["blowup_regex"], text, re.I)),
        "done": bool(re.search(cfg["done_regex"], text, re.I)),
        "error": bool(re.search(cfg["error_regex"], text, re.I)),
        "last_step": steps[-1][0] if steps else None,
        "days": days,
        "first_time": steps[0][1] if steps else None,
    }


# ----------------------------------------------------------------------------------------------
# Machine information
# ----------------------------------------------------------------------------------------------
def machine_info():
    info = {"hostname": socket.gethostname(), "os": platform.platform(),
            "logical_cpus": os.cpu_count(), "cpu": platform.processor() or "?"}
    try:
        txt = Path("/proc/cpuinfo").read_text()
        m = re.search(r"model name\s*:\s*(.+)", txt)
        if m:
            info["cpu"] = m.group(1).strip()
        phys = {(a, b) for a, b in re.findall(r"physical id\s*:\s*(\d+).*?core id\s*:\s*(\d+)", txt, re.S)}
        if phys:
            info["physical_cores"] = len(phys)
    except OSError:
        pass
    return info


# ----------------------------------------------------------------------------------------------
# Results storage
# ----------------------------------------------------------------------------------------------
FIELDS = ["key", "stage", "mode", "np", "nx", "ny", "dt", "ndtfast", "dtfast", "ntimes", "rep",
          "status", "wall_s", "sim_days", "days_per_hour", "compile_s",
          "exclusive", "peers", "other_cores", "cpus", "t_start",
          "cfl_bt", "cfl_int", "cfl_adv", "cfl_adv_x", "cfl_adv_y", "umax", "vmax",
          "notes", "log", "timestamp"]
NUMERIC = {"np": int, "nx": int, "ny": int, "dt": float, "ndtfast": int, "dtfast": float,
           "ntimes": int, "rep": int, "wall_s": float, "sim_days": float, "days_per_hour": float,
           "compile_s": float, "peers": int, "other_cores": int, "t_start": float,
           "cfl_bt": float, "cfl_int": float, "cfl_adv": float,
           "cfl_adv_x": float, "cfl_adv_y": float, "umax": float, "vmax": float}


def layout_tag(pc):
    return f"{pc['mode']}_np{pc['np']}_{pc['nx']}x{pc['ny']}"


def run_key(pc, dt, rep, exclusive=False):
    return f"{layout_tag(pc)}_dt{_fmt_dt(dt)}_r{rep}" + ("_x" if exclusive else "")


def load_results(path: Path):
    recs = []
    if not path.exists():
        return recs
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k, t in NUMERIC.items():
                v = row.get(k, "")
                try:
                    row[k] = t(float(v)) if v not in ("", None) else None
                except ValueError:
                    row[k] = None
            row["exclusive"] = str(row.get("exclusive", "")).strip().lower() == "true"
            recs.append(row)
    return recs


def append_result(path: Path, rec: dict):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: ("" if rec.get(k) is None else rec.get(k)) for k in FIELDS})


# ----------------------------------------------------------------------------------------------
# CPU topology and core slots
# ----------------------------------------------------------------------------------------------
def cpu_slots(cfg):
    """Logical CPU ids the scheduler may hand out, one per physical core by default,
    ordered socket by socket so that contiguous blocks share caches."""
    if cfg["cpu_list"]:
        slots = [int(c) for c in cfg["cpu_list"]]
        return slots, {"source": "cpu_list", "slots": len(slots)}
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except AttributeError:
        allowed = list(range(os.cpu_count() or 1))
    groups = {}
    for c in allowed:
        base = Path(f"/sys/devices/system/cpu/cpu{c}/topology")
        try:
            core = int((base / "core_id").read_text())
            pkg = int((base / "physical_package_id").read_text())
        except (OSError, ValueError):
            core, pkg = c, 0
        groups.setdefault((pkg, core), []).append(c)
    keys = sorted(groups)
    phys = [groups[k][0] for k in keys]
    ht = [c for k in keys for c in groups[k][1:]]
    r = int(cfg["reserve_cores"] or 0)
    phys_av = phys[r:]
    ht_av = [c for k in keys[r:] for c in groups[k][1:]]
    mc = cfg["max_cores"]
    if mc:  # number of cores fixed by the user: respected as given
        mc = int(mc)
        pool = phys_av + ht_av
        if mc > len(pool):
            print(f"[warn] max_cores={mc} but only {len(pool)} logical CPUs are available: using {len(pool)}")
        elif mc > len(phys_av):
            print(f"[warn] max_cores={mc} exceeds the {len(phys_av)} physical cores: "
                  f"{mc - len(phys_av)} hyperthread(s) will be used as cores")
        slots = pool[:mc]
    else:
        slots = phys_av + (ht_av if cfg["use_hyperthreads"] else [])
    if not slots:
        sys.exit("[error] no CPU left for the benchmark (check reserve_cores / max_cores)")
    return slots, {"source": "topology", "logical_allowed": len(allowed), "physical": len(phys),
                   "sockets": len({k[0] for k in keys}), "hyperthreads": len(ht), "slots": len(slots)}


# ----------------------------------------------------------------------------------------------
# Per-run sandbox (concurrent runs must not share croco.in nor output files)
# ----------------------------------------------------------------------------------------------
FILE_TOKEN = re.compile(r"^[^\s!*]+\.(nc|in|xml|dat|txt|def|cdf|nml)$|^[^\s!*]*/[^\s!*]+$", re.I)
KW_LINE = re.compile(r"^\s*([A-Za-z_][\w.-]*)\s*:")
NUMLIKE = re.compile(r"^[-+\d.eEdD*]+$")


def rewrite_croco_in_paths(text, run_dir: Path, out_kw: dict, keep_kw=()):
    """Inputs -> inputs/<name> (symlinked), outputs -> <name> in the sandbox.
    A file is an output if its keyword says so or if it does not exist yet.
    Blocks of keywords in keep_kw (e.g. 'online:' when ONLINE is defined) and directory paths (ending in '/')
    are left unchanged; their relative paths are returned in 'kept' so they can be made reachable from the
    sandbox with symlinks."""
    out_kw = {k.lower(): v for k, v in out_kw.items()}
    keep_kw = {k.lower() for k in keep_kw}
    lines = text.splitlines(keepends=True)
    kw, idx, links, outputs, kept = None, {}, {}, {}, []
    for i, ln in enumerate(lines):
        m = KW_LINE.match(ln)
        if m:
            kw = m.group(1).lower() + ":"
            idx[kw] = 0
            continue
        if kw is None or _is_blank_or_comment(ln):
            continue
        new = {}
        for j, t in enumerate(ln.split()):
            if kw in keep_kw or t.endswith("/"):
                if "/" in t and not NUMLIKE.match(t) and not os.path.isabs(t):
                    kept.append(t)       # text kept as it is; path linked into the sandbox
                continue
            if not FILE_TOKEN.match(t) or NUMLIKE.match(t):
                continue
            k = idx[kw]
            idx[kw] += 1
            src = Path(t) if os.path.isabs(t) else run_dir / t
            spec = out_kw.get(kw)
            is_out = spec == "all" or (isinstance(spec, list) and k in spec) or not src.exists()
            name = Path(t).name
            if is_out:
                new[j] = name
                outputs.setdefault(kw, []).append(name)
            else:
                real = src.resolve()
                if name in links and links[name] != real:
                    name = f"{len(links)}_{name}"
                links[name] = real
                new[j] = f"inputs/{name}"
        if new:
            lines[i] = replace_tokens(ln, new, right_align=False)
    return "".join(lines), links, outputs, kept


def link_kept_path(sb: Path, run_dir: Path, tok: str):
    """Make a relative path written in croco.in (e.g. 'DATA/CFSR_Benguela_LR/') resolve from the sandbox the
    same way it resolves from run_dir: symlink its first missing component to the real location."""
    parts = [p for p in Path(tok.rstrip("/")).parts if p != "."]
    for n in range(1, len(parts) + 1):
        rel = os.path.join(*parts[:n])
        at = Path(os.path.normpath(sb / rel))
        real = Path(os.path.normpath(run_dir / rel))
        if at.exists() or at.is_symlink():
            continue
        if real.exists():
            at.parent.mkdir(parents=True, exist_ok=True)
            at.symlink_to(real.resolve())
            return at
        return None
    return None


def missing_inputs(outputs: dict, out_kw: dict):
    """Files treated as outputs only because they do not exist (probably missing inputs)."""
    out_kw = {k.lower() for k in out_kw}
    return {k: v for k, v in outputs.items() if k not in out_kw}


# ----------------------------------------------------------------------------------------------
# Benchmark driver
# ----------------------------------------------------------------------------------------------
class Job:
    def __init__(self, pc, dt, stage, rep=0, exclusive=False):
        self.pc, self.dt, self.stage, self.rep, self.exclusive = pc, float(dt), stage, rep, exclusive
        self.np = pc["np"]
        self.key = run_key(pc, dt, rep, exclusive)
        self.proc = self.thread = self.fw = self.fr = None
        self.filler = stage.startswith("filler")
        self.est = None
        self.cpus, self.slot_idx = [], []
        self.t0 = self.t_end = self.t_epoch = None
        self.peers = self.other_cores = 0
        self.blow_t, self.timed_out, self.cancelled = None, False, False
        self.tail, self.notes = "", []
        self.sandbox = self.his = self.log = self.ts = None


class Bench:
    def __init__(self, cfg: dict, rerun=False, dry_run=False):
        self.cfg = cfg
        self.dry = dry_run
        self.run_dir = Path(cfg["run_dir"])
        self.out = Path(cfg["output_dir"])
        self.files = {k: self.run_dir / cfg[k] for k in ("cppdefs", "param", "croco_in")}
        for k, p in self.files.items():
            if not p.exists():
                sys.exit(f"[error] {k} not found: {p}")
        self.exe = self.run_dir / cfg["executable"]
        self.pristine = {k: p.read_text() for k, p in self.files.items()}
        self.restored = False
        self.touched_run_dir = False
        if not dry_run:
            for d in ("originals", "builds", "logs", "work"):
                (self.out / d).mkdir(parents=True, exist_ok=True)
            orig = self.out / "originals"
            for k, p in self.files.items():
                if not (orig / p.name).exists():
                    shutil.copy2(p, orig / p.name)
            if self.exe.exists() and not (orig / self.exe.name).exists():
                shutil.copy2(self.exe, orig / self.exe.name)
        self.csv = self.out / "results.csv"
        self.results = [] if (rerun or dry_run) else load_results(self.csv)
        if rerun and self.csv.exists() and not dry_run:
            self.csv.rename(self.csv.with_suffix(f".{int(time.time())}.bak"))
        self.done = {r["key"]: r for r in self.results}
        self.global_fail = None
        for r in self.results:  # knowledge from a previous (resumed) session
            if r["status"] == "BLOWUP":
                self.global_fail = r["dt"] if self.global_fail is None else min(self.global_fail, r["dt"])

        # machine / cores
        self.slots, self.topo = cpu_slots(cfg)
        self.ncores = len(self.slots)
        self.ref_cores = int(cfg["ref_cores"])
        if self.ref_cores > self.ncores or self.ref_cores < 2:
            nc = max(2, min(self.ref_cores, self.ncores))
            print(f"[warn] ref_cores={self.ref_cores} outside [2, {self.ncores}] usable cores: starting with {nc}")
            self.ref_cores = nc
        self.max_jobs = int(cfg["max_parallel_jobs"] or self.ncores)
        self.backfill = bool(cfg["backfill"]) and self.max_jobs > 1
        if cfg["launch_order"] not in ("large_first", "small_first"):
            sys.exit("[error] launch_order must be 'large_first' or 'small_first'")
        self.large_first = cfg["launch_order"] == "large_first"
        self.stable_list = []
        self.live = None
        self.dt_opt = None
        self.running, self.free, self.fillers = [], set(range(self.ncores)), []
        self.dt_floor, self.dt_phase = None, None
        self.taskset = shutil.which("taskset")
        self.launcher = cfg["launcher"]
        self.deadline = None
        if self.launcher == "slurm":
            left = slurm_time_left()
            if left is not None:
                self.deadline = time.time() + left
            nn = os.environ.get("SLURM_JOB_NUM_NODES") or os.environ.get("SLURM_NNODES")
            if nn and nn.isdigit() and int(nn) > 1:
                print(f"[warn] the Slurm job has {nn} nodes: only the cores of this node are used (request -N 1)")
            if not shutil.which("srun"):
                print("[warn] launcher 'slurm' but 'srun' was not found in PATH")
        self.wake = threading.Event()
        self.builds, self.compile_s, self.build_notes = {}, {}, {}
        self.compile_fail_recorded = set()

        self.config_name = cfg["config_name"] or detect_config_name(self.pristine["cppdefs"])
        self.grid_info = grid_from_sources(self.pristine["cppdefs"], self.pristine["param"])
        pp = MiniCPP()
        pp.process(self.pristine["cppdefs"])
        self.online = "ONLINE" in pp.macros
        self.keep_kw = ("online:",) if self.online else ()
        if cfg["grid_size"]:
            self.grid_size = tuple(cfg["grid_size"])
        elif self.grid_info:
            self.grid_size = (self.grid_info["LLm"], self.grid_info["MMm"])
        else:
            self.grid_size = grid_size_from_param(self.pristine["param"], self.config_name)
            print("[warn] the active LLm0/MMm0 line of param.h could not be determined from cppdefs.h"
                  + (f"; using {self.grid_size}" if self.grid_size else ""))
        self.ts0 = read_time_stepping(self.pristine["croco_in"])
        self.grid = None
        gf = cfg["grid_file"] or read_filename(self.pristine["croco_in"], "grid:", after_values=False)
        if gf:
            gpath = (self.run_dir / gf)
            try:
                self.grid = grid_metrics(gpath, cfg["internal_wave_speed"])
            except Exception as e:  # noqa: BLE001
                print(f"[warn] CFL from grid disabled ({gpath}: {e})")
        mi = machine_info()
        mi.update({"cores_used": self.ncores, "cpu_slots": self.slots, "topology": self.topo,
                   "max_parallel_jobs": self.max_jobs})
        gi = self.grid_info
        if gi and gi.get("case") and gi["case"] != "#else":
            self.case_name = gi["case"]          # option of param.h that selects the active dimensions
        else:                                    # no own entry in param.h: name defined in the block
            self.case_name = case_name_in_block(self.pristine["cppdefs"], self.config_name)
        self.meta = {"config_name": self.config_name, "grid_size": self.grid_size,
                     "case_name": self.case_name, "grid_info": self.grid_info, "online": self.online,
                     "dt0": self.ts0["dt"], "ndtfast0": self.ts0["ndtfast"], "ref_cores": self.ref_cores,
                     "grid": {k: v for k, v in (self.grid or {}).items() if k not in ("pm", "pn")},
                     "machine": mi}
        if not (rerun or dry_run):
            self.load_state()

    # ---------------------------------------------------------------- state (for --report-only / resume)
    STATE_KEYS = ("dt_path", "explore_path", "dt_opt", "dt_stable_max")

    def save_state(self):
        if self.dry:
            return
        st = {k: self.meta.get(k) for k in self.STATE_KEYS if self.meta.get(k) is not None}
        _write_atomic(self.out / "search_state.json", json.dumps(st, indent=1, default=str))

    def load_state(self):
        f = self.out / "search_state.json"
        if not f.exists():
            return
        try:
            st = json.loads(f.read_text())
        except (OSError, ValueError):
            return
        for m in (st.get("explore_path") or {}).values():  # JSON keys are strings
            m["tested"] = {int(n): v for n, v in m.get("tested", {}).items()}
        self.meta.update({k: v for k, v in st.items() if k in self.STATE_KEYS})

    # ---------------------------------------------------------------- restore
    def restore(self):
        if self.restored or self.dry:
            return
        if self.touched_run_dir:
            for k, p in self.files.items():
                p.write_text(self.pristine[k])
            orig_exe = self.out / "originals" / self.exe.name
            if orig_exe.exists():
                shutil.copy2(orig_exe, self.exe)
            print("[info] original cppdefs.h, param.h and executable restored in run_dir")
        self.restored = True

    # ---------------------------------------------------------------- layouts
    def decomps(self, mode, n):
        """Every valid x/y distribution of n cores for a parallel mode
        (MPI: NP_XI*NP_ETA = n; OpenMP: NSUB_X*NSUB_E = n*omp_tile_factor, NPP = n)."""
        cfg = self.cfg
        explicit = cfg["decompositions"] if isinstance(cfg["decompositions"], dict) else None
        if explicit is not None and str(n) in explicit:
            pairs = [tuple(p) for p in explicit[str(n)]]
        else:
            pairs = []
            for f in (cfg["omp_tile_factors"] if mode == "openmp" else [1]):
                t = n * f
                pairs += [(a, t // a) for a in range(1, t + 1) if t % a == 0]
        out = []
        for nx, ny in pairs:
            if mode == "openmp" and (nx * ny) % n:
                continue
            if mode == "mpi" and nx * ny != n:
                continue
            if self.grid_size:
                L, M = self.grid_size
                mp = cfg["min_points_per_subdomain"]
                if L / nx < mp or M / ny < mp:
                    continue
                if mode == "mpi" and cfg["require_divisible"] and (L % nx or M % ny):
                    continue
            pc = {"mode": mode, "np": n, "nx": nx, "ny": ny}
            if pc not in out:
                out.append(pc)
        return out

    def layouts(self, n=None):
        """Serial build + every decomposition at n cores (default: the reference core count)."""
        n = self.ref_cores if n is None else n
        out = [dict(SERIAL)]
        for mode in self.cfg["modes"]:
            out += self.decomps(mode, n)
        return out

    def _squareness(self, p):
        L, M = self.grid_size or (1, 1)
        return abs(math.log((L / p["nx"]) / (M / p["ny"])))

    # ---------------------------------------------------------------- builds
    def _cached(self, pc):
        return self.out / "builds" / f"croco_{layout_tag(pc)}"

    def _edited_sources(self, pc):
        cpp, n1 = set_parallel_mode(self.pristine["cppdefs"], pc["mode"], self.config_name)
        par, n2 = set_decomposition(self.pristine["param"], pc["mode"], pc["np"], pc["nx"], pc["ny"],
                                    self.cfg["mpi_tiles"])
        return cpp, par, n1 + n2

    def build_all(self, layouts):
        todo = []
        for pc in layouts:
            c = self._cached(pc)
            if c.exists():
                self.builds[layout_tag(pc)] = c
            else:
                todo.append(pc)
        if not todo:
            return
        print(f"\n=== compiling {len(todo)} executables "
              f"({self.cfg['build_mode']}, up to {self.cfg['build_jobs']} at a time) ===")
        failed = todo
        if self.cfg["build_mode"] == "parallel":
            failed = self._build_parallel(todo)
            if failed:
                print(f"[build] {len(failed)} parallel build(s) failed: retrying in run_dir, one at a time")
        for pc in failed:
            self.builds[layout_tag(pc)] = self._build_in_place(pc)

    def _build_parallel(self, todo):
        prefix = f".{self.run_dir.name}.bench_build."
        queue, running, failed = list(todo), [], []
        nb = max(1, int(self.cfg["build_jobs"]))
        try:
            while queue or running:
                while queue and len(running) < nb:
                    pc = queue.pop(0)
                    tag = layout_tag(pc)
                    bdir = self.run_dir.parent / (prefix + tag)  # sibling: relative paths in jobcomp still work
                    try:
                        if bdir.exists():
                            shutil.rmtree(bdir)
                        bdir.mkdir()
                        for p in self.run_dir.iterdir():
                            if p.is_file() and p.suffix != ".nc" and p.name != self.exe.name \
                                    and p.stat().st_size < 50e6:
                                shutil.copy2(p, bdir / p.name)
                        cpp, par, notes = self._edited_sources(pc)
                        self.build_notes[tag] = notes
                        (bdir / self.cfg["cppdefs"]).write_text(cpp)
                        (bdir / self.cfg["param"]).write_text(par)
                        log = open(self.out / "logs" / f"compile_{tag}.log", "w")
                        p = subprocess.Popen(self.cfg["compile_cmd"], shell=True, cwd=bdir, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                        running.append((pc, bdir, p, log, time.perf_counter()))
                        print(f"[build] {tag} started")
                    except (OSError, RuntimeError) as e:
                        print(f"[build] {tag}: cannot prepare build dir ({e})")
                        failed.append(pc)
                time.sleep(0.5)
                for item in list(running):
                    pc, bdir, p, log, t0 = item
                    el = time.perf_counter() - t0
                    if p.poll() is None and el < self.cfg["compile_timeout_s"]:
                        continue
                    if p.poll() is None:
                        self._kill(p)
                    log.close()
                    running.remove(item)
                    tag = layout_tag(pc)
                    exe = bdir / self.exe.name
                    if p.returncode == 0 and exe.exists():
                        shutil.copy2(exe, self._cached(pc))
                        self.builds[tag] = self._cached(pc)
                        self.compile_s[tag] = el
                        print(f"[build] {tag} ok ({el:.0f} s)")
                    else:
                        failed.append(pc)
                        print(f"[build] {tag} FAILED (see logs/compile_{tag}.log)")
                    if bdir.name.startswith(prefix):
                        shutil.rmtree(bdir, ignore_errors=True)
        except BaseException:
            for pc, bdir, p, log, _ in running:
                self._kill(p)
                log.close()
                if bdir.name.startswith(prefix):
                    shutil.rmtree(bdir, ignore_errors=True)
            raise
        return failed

    def _build_in_place(self, pc):
        tag = layout_tag(pc)
        try:
            cpp, par, notes = self._edited_sources(pc)
        except RuntimeError as e:
            print(f"[build] {tag}: {e}")
            return None
        self.build_notes[tag] = notes
        self.touched_run_dir = True
        self.files["cppdefs"].write_text(cpp)
        self.files["param"].write_text(par)
        if self.exe.exists():
            self.exe.unlink()
        log = self.out / "logs" / f"compile_{tag}.log"
        print(f"[build] {tag} (in run_dir) ...", flush=True)
        t0 = time.perf_counter()
        with open(log, "w") as f:
            try:
                rc = subprocess.run(self.cfg["compile_cmd"], shell=True, cwd=self.run_dir, stdout=f,
                                    stderr=subprocess.STDOUT, timeout=self.cfg["compile_timeout_s"]).returncode
            except subprocess.TimeoutExpired:
                rc = -999
        self.compile_s[tag] = time.perf_counter() - t0
        if rc != 0 or not self.exe.exists():
            print(f"[build] {tag} FAILED, see {log}")
            return None
        shutil.copy2(self.exe, self._cached(pc))
        print(f"[build] {tag} ok ({self.compile_s[tag]:.0f} s)")
        return self._cached(pc)

    # ---------------------------------------------------------------- one job
    def _kill(self, p):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(p.pid, sig)
            except (ProcessLookupError, PermissionError):
                return
            try:
                p.wait(timeout=15)
                return
            except subprocess.TimeoutExpired:
                continue

    def make_sandbox(self, key, cin_text):
        sb = self.out / "work" / key
        if sb.exists():
            shutil.rmtree(sb)
        (sb / "inputs").mkdir(parents=True)
        text, links, outputs, kept = rewrite_croco_in_paths(cin_text, self.run_dir, self.cfg["output_keywords"],
                                                            self.keep_kw)
        for name, src in links.items():
            (sb / "inputs" / name).symlink_to(src)
        for tok in kept:  # e.g. the data path of the 'online:' section, kept unchanged
            link_kept_path(sb, self.run_dir, tok)
        out_names = {n for v in outputs.values() for n in v}
        skip = out_names | {Path(self.cfg["croco_in"]).name, self.exe.name}
        for p in self.run_dir.iterdir():  # namelists, iodef.xml, AGRIF_FixedGrids.in ... read from cwd
            if p.is_file() and p.name not in skip and p.suffix != ".nc":
                (sb / p.name).symlink_to(p.resolve())
        (sb / Path(self.cfg["croco_in"]).name).write_text(text)
        his = sb / outputs["history:"][0] if outputs.get("history:") else None
        return sb, his

    def _command(self, j: Job, exe: Path):
        cfg = self.cfg
        fmt = {"exe": str(exe), "np": j.np, "nx": j.pc["nx"], "ny": j.pc["ny"],
               "croco_in": Path(cfg["croco_in"]).name, "cpus": ",".join(map(str, j.cpus)),
               "places": ",".join("{%d}" % c for c in j.cpus), "slurm_flags": cfg["slurm_step_flags"]}
        tpl = cfg["run_cmd"][j.pc["mode"]]
        if "taskset" in tpl and not self.taskset:
            tpl = re.sub(r"taskset\s+(-c\s+)?\{cpus\}\s*", "", tpl)
            j.notes.append("taskset not found: run not pinned to its cores")
        env = os.environ.copy()
        for k, v in cfg["env"].get(j.pc["mode"], {}).items():
            env[k] = str(v).format(**fmt)
        return tpl.format(**fmt), env

    def _start(self, j: Job, exe: Path):
        cin, ts, notes = set_croco_in(self.pristine["croco_in"], self.cfg, j.dt)
        j.ts = ts
        j.notes += notes + self.build_notes.get(layout_tag(j.pc), [])
        j.sandbox, j.his = self.make_sandbox(j.key, cin)
        cmd, env = self._command(j, exe)
        j.log = self.out / "logs" / f"run_{j.key}.log"
        j.fw = open(j.log, "w")
        j.fw.write(f"# command: {cmd}\n# cwd: {j.sandbox}\n# cpus: {j.cpus}\n"
                   f"# started: {_dt.datetime.now().isoformat()}\n")
        j.fw.flush()
        j.fr = open(j.log, "r", errors="replace")
        j.t_epoch = time.time()
        j.t0 = time.perf_counter()
        j.proc = subprocess.Popen(cmd, shell=True, cwd=j.sandbox, stdout=j.fw, stderr=subprocess.STDOUT,
                                  env=env, start_new_session=True)

        def _waiter(job=j):
            job.proc.wait()
            job.t_end = time.perf_counter()
            self.wake.set()
        j.thread = threading.Thread(target=_waiter, daemon=True)
        j.thread.start()
        print(f"[run]   {j.key:34s} cpus={_cpu_ranges(j.cpus):>10s}  NTIMES={ts['ntimes']}", flush=True)

    def _monitor(self, j: Job, blow_re):
        if j.t_end is not None:
            return
        chunk = j.fr.read()
        if chunk:
            j.tail = j.tail[-2000:] + chunk
            if j.blow_t is None and blow_re.search(j.tail):
                j.blow_t = time.perf_counter()
        now = time.perf_counter()
        if j.blow_t is not None and now - j.blow_t > self.cfg["blowup_grace_s"]:
            self._kill(j.proc)
        elif now - j.t0 > self.cfg["timeout_s"]:
            j.timed_out = True
            self._kill(j.proc)

    def _close(self, j: Job):
        if j.thread:
            j.thread.join(timeout=30)
        for fh in (j.fw, j.fr):
            if fh:
                fh.close()

    def _cleanup_sandbox(self, j: Job):
        if self.cfg["keep_outputs"] or not j.sandbox or not j.sandbox.exists():
            return
        for f in j.sandbox.iterdir():
            if f.is_file() and not f.is_symlink() and f.suffix == ".nc":
                f.unlink()

    def _finish(self, j: Job):
        self._close(j)
        cfg, dt = self.cfg, j.dt
        wall = (j.t_end or time.perf_counter()) - j.t0
        rc = j.proc.returncode
        info = parse_log(j.log.read_text(errors="replace"), cfg)
        sim_days = info["days"] + (dt / DAY if info["last_step"] else 0.0)
        notes = list(j.notes)
        if info["blowup"]:
            status = "BLOWUP"
        elif j.timed_out:
            status = "TIMEOUT"
        elif rc == 0 and (info["done"] or sim_days >= 0.99 * cfg["sim_days"]) and not info["error"]:
            status = "OK"
            if not info["done"]:
                notes.append("no 'MAIN: DONE' in log; accepted from simulated days")
        else:
            status = "ERROR"
            notes.append(f"exit code {rc}")
        tag = layout_tag(j.pc)
        rec = {"key": j.key, "stage": j.stage, **j.pc, "dt": dt, "rep": j.rep, **j.ts,
               "status": status, "wall_s": round(wall, 3), "sim_days": round(sim_days, 4),
               "compile_s": round(self.compile_s.get(tag, 0.0), 1), "exclusive": j.exclusive,
               "peers": j.peers, "other_cores": j.other_cores, "cpus": _cpu_ranges(j.cpus),
               "t_start": round(j.t_epoch, 2), "log": str(j.log),
               "timestamp": _dt.datetime.fromtimestamp(j.t_epoch).isoformat(timespec="seconds")}
        if status == "OK":
            rec["days_per_hour"] = round(sim_days / (wall / 3600.0), 3)
        if self.grid:
            rec["cfl_bt"] = j.ts["dtfast"] * self.grid["ext_factor"]
            rec["cfl_int"] = dt * self.grid["int_factor"]
        if j.his and j.his.exists() and status in ("OK", "BLOWUP"):
            try:
                adv = advective_factors(j.his, self.grid)
                if adv:
                    rec.update(umax=adv["umax"], vmax=adv["vmax"])
                    if self.grid:
                        rec.update(cfl_adv_x=dt * adv["fx"], cfl_adv_y=dt * adv["fy"],
                                   cfl_adv=dt * (adv["fx"] + adv["fy"]))
            except Exception as e:  # noqa: BLE001
                notes.append(f"advective CFL not computed: {e}")
        rec["notes"] = "; ".join(dict.fromkeys(notes))
        for k in ("cfl_bt", "cfl_int", "cfl_adv", "cfl_adv_x", "cfl_adv_y", "dtfast"):
            if rec.get(k) is not None:
                rec[k] = round(rec[k], 5)
        msg = f"{status:8s} wall={wall:9.1f}s  days={sim_days:.3f}  with {j.peers} other run(s)"
        if rec.get("cfl_bt") is not None:
            msg += f"  CFL_bt={rec['cfl_bt']:.3f}"
        print(f"[done]  {j.key:34s} {msg}", flush=True)
        self._cleanup_sandbox(j)
        self._record(rec)
        return rec

    def _record(self, rec):
        append_result(self.csv, rec)
        self.results.append(rec)
        self.done[rec["key"]] = rec
        if self.live:
            self.live.notify()

    # ---------------------------------------------------------------- scheduler
    def _alloc(self, free: set, n: int):
        """n free slot indices, preferring a contiguous block (same socket / shared cache)."""
        if len(free) < n:
            return None
        order = sorted(free)
        for s in range(len(order) - n + 1):
            blk = order[s:s + n]
            if blk[-1] - blk[0] == n - 1:
                return blk
        return order[:n]

    def _pruned(self, j: Job):
        if j.exclusive:
            return False
        if self.cfg["blowup_is_global"] and self.global_fail is not None and j.dt >= self.global_fail:
            return True   # rule B: never above (or at) a dt that blew up
        return (j.stage == "dt-search" and self.dt_floor is not None and j.dt < self.dt_floor)

    def _order(self, j: Job):
        """Queue priority: large_first = most cores and largest dt first; small_first = the reverse
        in cores (smallest dt first, i.e. the longest runs of a core count start first)."""
        return (-j.np, -j.dt, j.rep) if self.large_first else (j.np, j.dt, j.rep)

    def _admit(self, j: Job, recs, pending, running):
        if j.key in self.done:
            recs[j.key] = self.done[j.key]
            return False
        if self._pruned(j):
            print(f"[skip]  {j.key}: dt >= {_fmt_dt(self.global_fail)} s already blew up")
            return False
        if j.np > self.ncores:
            print(f"[skip]  {j.key}: needs {j.np} cores, only {self.ncores} usable")
            return False
        tag = layout_tag(j.pc)
        if not self.builds.get(tag):
            if tag not in self.compile_fail_recorded:
                self.compile_fail_recorded.add(tag)
                self._record({"key": f"{tag}_compile", "stage": j.stage, **j.pc, "dt": j.dt, "rep": 0,
                              "status": "COMPILE_FAIL", "log": str(self.out / "logs" / f"compile_{tag}.log"),
                              "timestamp": _dt.datetime.now().isoformat(timespec="seconds")})
            return False
        if any(p.key == j.key for p in pending + running):
            return False
        pending.append(j)
        return True

    # ---- runtime estimates (used to backfill without delaying the job at the head of the queue)
    def _estimate(self, j: Job):
        """Expected wall time of a job from runs already measured, scaled by NTIMES (1/dt) and cores."""
        ok = [r for r in self.results if r["status"] == "OK" and r.get("wall_s")]
        if not ok:
            return None
        same = [r for r in ok if layout_tag(r) == layout_tag(j.pc)]
        if same:
            return statistics.median(r["wall_s"] * r["dt"] / j.dt for r in same)
        pool = [r for r in ok if r["mode"] == j.pc["mode"]] or ok
        r = min(pool, key=lambda r: (abs(math.log(r["np"] / j.np)), -r["dt"]))
        t = r["wall_s"] * r["dt"] / j.dt
        # conservative: linear slow-down with fewer cores, sqrt speed-up with more cores
        return t * (r["np"] / j.np if j.np < r["np"] else (r["np"] / j.np) ** 0.5)

    def _est_end(self, j: Job):
        """Expected end (perf_counter time) of a running job: log progress first, model otherwise."""
        now = time.perf_counter()
        el = now - j.t0
        steps = STEP_LINE.findall(j.tail)
        if steps and j.ts and j.ts.get("ntimes"):
            f = int(steps[-1][0]) / float(j.ts["ntimes"])
            if f >= 0.02:
                return now + el * (1.0 - f) / f
        if j.est is not None:
            return j.t0 + max(j.est, el)
        return now if not j.filler else float("inf")

    def _shadow(self, need):
        """Earliest time at which `need` cores will be free, and cores left over at that time."""
        now = time.perf_counter()
        avail = len(self.free)
        if avail >= need:
            return now, avail - need
        for t, n in sorted((self._est_end(r), r.np) for r in self.running):
            avail += n
            if avail >= need:
                return t, avail - need
        return float("inf"), 0

    def _check_walltime(self, j: Job):
        """Refuse to start a run that cannot finish before the Slurm time limit (minus the margin)."""
        if not self.deadline:
            return
        left = self.deadline - time.time() - float(self.cfg["walltime_margin_s"])
        est = j.est if j.est is not None else (self._estimate(j) or 0.0)
        if left <= 0 or est * 1.2 > left:
            raise WallTimeUp(f"not enough wall time left for {j.key} (expected {_tfmt(est)}, "
                             f"{_tfmt(max(left, 0))} available)")

    def _launch(self, j: Job, idx):
        self._check_walltime(j)
        self.free -= set(idx)
        j.slot_idx, j.cpus = idx, [self.slots[i] for i in idx]
        if j.est is None:
            j.est = self._estimate(j)
        self._start(j, self.builds[layout_tag(j.pc)])
        self.running.append(j)
        busy = sum(r.np for r in self.running)
        for r in self.running:
            r.peers = max(r.peers, len(self.running) - 1)
            r.other_cores = max(r.other_cores, busy - r.np)

    def _pick_backfill(self, pending, batch_keys):
        """A smaller job (later main job or filler of any stage) that fits in the free cores now and
        does not delay the job at the head of the queue (EASY backfilling)."""
        if not self.backfill or not self.free:
            return None, None
        now = time.perf_counter()
        if pending:
            shadow_t, extra = self._shadow(pending[0].np)
        else:  # nothing queued: fill only until the last main run of this batch ends
            ends = [self._est_end(r) for r in self.running if r.key in batch_keys]
            if not ends:
                return None, None
            shadow_t, extra = max(ends), 0
        busy_keys = {r.key for r in self.running} | {p.key for p in pending}
        safety = float(self.cfg["backfill_safety"])
        cands = [(p, pending) for p in pending[1:]] + [(f, self.fillers) for f in self.fillers]
        for j, src in cands:
            if j.np > len(self.free) or j.key in self.done or (src is self.fillers and j.key in busy_keys):
                continue
            if self._pruned(j) or not self.builds.get(layout_tag(j.pc)):
                continue
            est = self._estimate(j)
            if est is None:
                continue
            if now + est * safety <= shadow_t or j.np <= extra:
                j.est = est
                return j, src
        return None, None

    def _drop_dead_fillers(self):
        keep = []
        for f in self.fillers:
            if f.key in self.done or self._pruned(f):
                continue
            keep.append(f)
        self.fillers = keep

    def _reap(self, recs, blow_re, on_done):
        """Monitor running jobs; finish those that ended."""
        for j in list(self.running):
            if j.cancelled or j not in self.running:
                continue  # cancelled by a BLOW UP handled earlier in this same pass
            self._monitor(j, blow_re)
            if j.t_end is None:
                continue
            self.running.remove(j)
            self.free |= set(j.slot_idx)
            rec = self._finish(j)
            recs[j.key] = rec
            if rec["status"] == "BLOWUP":
                self.global_fail = j.dt if self.global_fail is None else min(self.global_fail, j.dt)
            elif rec["status"] == "OK" and j.stage == "dt-search" and self.dt_phase == "bisect":
                self.dt_floor = max(self.dt_floor or 0.0, j.dt)  # runs below a stable dt are not needed
            for r in list(self.running):  # cancel the runs that the new result made unnecessary
                if self._pruned(r):
                    self._cancel(r)
                    self.running.remove(r)
                    self.free |= set(r.slot_idx)
            on_done(rec)

    def drain(self):
        """Wait for the backfill runs still going on (nothing new is launched)."""
        if not self.running:
            return
        print(f"[sched] waiting for {len(self.running)} backfill run(s) to finish")
        blow_re = re.compile(self.cfg["blowup_regex"], re.I)
        try:
            while self.running:
                self.wake.wait(timeout=self.cfg["poll_s"])
                self.wake.clear()
                self._reap({}, blow_re, lambda rec: None)
        except BaseException:
            self.cancel_all()
            raise

    def cancel_all(self):
        for r in list(self.running):
            self._cancel(r)
        self.running.clear()
        self.free = set(range(self.ncores))

    def run_batch(self, jobs, exclusive=False, on_finish=None):
        """Run jobs packing them on the free cores: small jobs first and simultaneously, the next ones
        as soon as enough cores are released. When the head of the queue does not fit, smaller jobs
        (later ones or fillers from other stages) backfill the idle cores if they do not delay it.
        on_finish(rec, running, pending) may return new jobs, which join the queue (pipelining)."""
        cfg = self.cfg
        if exclusive:
            self.drain()
        recs, pending = {}, []
        for j in jobs:
            self._admit(j, recs, pending, self.running)

        def _more(rec):
            if on_finish is None:
                return
            for nj in on_finish(rec, self.running, pending) or []:
                self._admit(nj, recs, pending, self.running)
            pending.sort(key=self._order)
        _more(None)  # lets the controller react to results already known (resumed benchmark)
        if not pending:
            return recs
        pending.sort(key=self._order)
        batch_keys = set()
        max_jobs = 1 if exclusive else self.max_jobs
        blow_re = re.compile(cfg["blowup_regex"], re.I)
        print(f"[sched] {len(pending)} run(s) queued on {self.ncores} cores, "
              f"{'alone on the machine' if exclusive else 'one at a time' if max_jobs == 1 else f'up to {max_jobs} simultaneous'}"
              + (f", {len(self.fillers)} backfill candidate(s)" if self.fillers and not exclusive else ""))
        try:
            while pending or any(r.key in batch_keys for r in self.running):
                while len(self.running) < max_jobs:
                    while pending and self._pruned(pending[0]):
                        j = pending.pop(0)
                        print(f"[skip]  {j.key}: dt >= {_fmt_dt(self.global_fail)} s already blew up")
                    if pending:
                        idx = self._alloc(self.free, pending[0].np)
                        if idx is not None:  # the head of the queue fits
                            j = pending.pop(0)
                            batch_keys.add(j.key)
                            self._launch(j, idx)
                            continue
                    if exclusive:
                        break
                    self._drop_dead_fillers()
                    j, src = self._pick_backfill(pending, batch_keys)
                    if j is None:
                        break
                    src.remove(j)
                    if src is pending:
                        batch_keys.add(j.key)
                    print(f"[fill]  {j.key} uses idle cores (expected {_tfmt(j.est)})")
                    self._launch(j, self._alloc(self.free, j.np))
                self.wake.wait(timeout=cfg["poll_s"])
                self.wake.clear()
                self._reap(recs, blow_re, _more)
                if self.deadline and time.time() > self.deadline - 0.5 * self.cfg["walltime_margin_s"]:
                    raise WallTimeUp("Slurm job time limit reached")
        except BaseException:
            self.cancel_all()
            raise
        return recs

    def _cancel(self, j: Job):
        self._kill(j.proc)
        self._close(j)
        j.cancelled = True
        if j.sandbox and j.sandbox.exists():
            shutil.rmtree(j.sandbox, ignore_errors=True)
        print(f"[cancel] {j.key}")

    # ---------------------------------------------------------------- strategies
    def _jobs(self, pcs, dts, stage, exclusive=False):
        return [Job(pc, dt, stage, r, exclusive) for pc in pcs for dt in dts
                for r in range(int(self.cfg["repeat"]))]

    # ---------------------------------------------------------------- stage 1: dt search (serial)
    def _round_dt(self, x):
        return float(round(x)) if x >= 20 else round(x, 2)

    def _serial_status(self, dt, infer=False):
        sts = [r["status"] for r in self.results if layout_tag(r) == layout_tag(SERIAL) and r["dt"] == dt]
        if "BLOWUP" in sts:
            return "BLOWUP"
        if "OK" in sts:
            return "OK"
        if self.global_fail is not None and dt >= self.global_fail:
            return "BLOWUP"  # a smaller or equal dt already blew up
        if sts:
            return sts[-1]
        if infer and any(r["status"] == "OK" and r["dt"] > dt for r in self.results
                         if layout_tag(r) == layout_tag(SERIAL)):
            return "OK (inferred)"  # below a stable dt: stable (monotonicity), the run was not needed
        return None

    def _serial_batch(self, dts, stage):
        todo = sorted({d for d in dts if self._serial_status(d) is None})
        if todo:
            print(f"[dt] running {len(todo)} serial run(s) at once: "
                  f"{', '.join(_fmt_dt(d) for d in todo)} s")
            self.run_batch([Job(dict(SERIAL), d, stage, 0) for d in todo])
        self.dt_runs += len(todo)

    def _bisect_nodes(self, lo, hi, count, depth, tol):
        """Speculative decision tree of the halving phase: the next midpoint and, below it, the
        midpoints that would follow a BLOW UP (lower half) or a stable run (upper half)."""
        out = []

        def rec(lo, hi, count, d):
            m = self._round_dt((lo + hi) / 2.0)
            if d == 0 or not (lo < m < hi) or (hi - lo) / hi <= tol:
                return
            out.append(m)
            if count + 1 < 2:          # a second consecutive BLOW UP ends the search
                rec(lo, m, count + 1, d - 1)
            rec(m, hi, 0, d - 1)
        rec(lo, hi, count, depth)
        return out

    def dt_search(self):
        """Serial (1 core) time-step search.
        Up:     dt_ref, doubled while stable. Rule B: never above a dt that blew up.
        Down:   if dt_ref blows up, dt is halved until a dt is stable AND its half is stable (rule A:
                never decrease further than that).
        Bisect: then the next dt is the midpoint between the last stable dt and the last BLOW UP;
                stop after two BLOW UPs in a row (or interval < dt_tol, or max_dt_runs).
        Several serial runs run at once, but in "strict" mode only runs that are needed whatever the
        pending results are: doubling one at a time, going down d and d/2 together, bisecting the
        midpoint together with the next lower midpoint (cancelled if the midpoint is stable)."""
        cfg = self.cfg
        strict = cfg["dt_speculation"] != "aggressive"
        dt0 = float(cfg["dt_ref"] or self.ts0["dt"])
        dt_cap = float(cfg["sim_days"]) * DAY / max(1, int(cfg["min_steps"]))
        if dt0 > dt_cap:
            print(f"[dt] dt_ref {_fmt_dt(dt0)} s leaves fewer than {cfg['min_steps']} steps: using {_fmt_dt(dt_cap)} s")
            dt0 = dt_cap
        k = max(1, min(self.ncores, self.max_jobs))
        tol = float(cfg["dt_tol"])
        maxr = int(cfg["max_dt_runs"])
        path, self.dt_runs = [], 0
        self.dt_floor, self.dt_phase = None, "up"
        st = self._serial_status

        def log(dt, s, phase):
            path.append({"dt": dt, "status": s, "phase": phase})
            print(f"[dt] {phase:8s} dt = {_fmt_dt(dt):>8s} s -> {s}")

        def need(cands):
            if self.dt_runs >= maxr:
                return False
            self._serial_batch(cands, "dt-search")
            return True

        lo = hi = None
        # ---- up: dt_ref, then doubling while stable
        dt = dt0
        while True:
            if st(dt) is None:
                ahead = [dt] if strict else [dt * 2 ** i for i in range(k) if dt * 2 ** i <= dt_cap]
                if not need(ahead) or st(dt) is None:
                    print("[dt] search stopped (max_dt_runs reached or run not completed)")
                    break
            s = st(dt)
            log(dt, s, "up")
            if s == "OK":
                lo = dt
                if dt * 2 > dt_cap:
                    print(f"[dt] doubling to {_fmt_dt(dt * 2)} s would leave fewer than {cfg['min_steps']} steps: stop")
                    break
                dt *= 2
                continue
            if s == "BLOWUP":
                hi = dt
            else:
                print(f"[dt] run ended with {s} (not a BLOW UP): search stopped")
            break
        # ---- down: dt_ref blew up -> halve until d stable and d/2 stable (rule A)
        if lo is None and hi is not None:
            self.dt_phase = "down"
            d = hi / 2.0
            while True:
                if st(d) is None or st(d / 2.0) is None:
                    if not need([d, d / 2.0] if strict else [d / 2 ** i for i in range(min(k, 2))]):
                        print(f"[dt] max_dt_runs ({maxr}) reached")
                        break
                s = st(d)
                if s is None:
                    print("[dt] run not completed: search stopped")
                    break
                log(d, s, "down")
                if s == "BLOWUP":
                    hi, d = d, d / 2.0
                    continue
                if s != "OK":
                    print(f"[dt] run ended with {s} (not a BLOW UP): search stopped")
                    break
                sh = st(d / 2.0)
                if sh is not None:
                    log(d / 2.0, sh, "down")
                lo = d
                if sh != "OK":
                    print(f"[dt] warning: {_fmt_dt(d)} s stable but {_fmt_dt(d / 2)} s ended with {sh}")
                break
        # ---- bisect between [lo stable, hi BLOW UP]; stop after two BLOW UPs in a row
        if lo is not None and hi is not None:
            self.dt_phase, self.dt_floor = "bisect", lo
            depth = max(1, int(math.floor(math.log2(k + 1))))
            count, reason = 0, None
            while reason is None:
                m = self._round_dt((lo + hi) / 2.0)
                if not (lo < m < hi) or (hi - lo) / hi <= tol:
                    reason = f"interval below dt_tol ({tol:g})"
                    break
                if st(m, True) is None:
                    if strict:
                        nodes = [m]
                        c = self._round_dt((lo + m) / 2.0)
                        if count == 0 and lo < c < m and (m - lo) / m > tol:
                            nodes.append(c)    # needed if m blows up; cancelled if m is stable
                    else:
                        nodes = self._bisect_nodes(lo, hi, count, depth, tol)
                    if not need(nodes):
                        reason = f"max_dt_runs ({maxr}) reached"
                        break
                    if st(m, True) is None:
                        reason = "run could not be completed"
                        break
                s = st(m, True)
                log(m, s, "bisect")
                s = "OK" if s == "OK (inferred)" else s
                if s == "BLOWUP":
                    hi, count = m, count + 1
                    if count >= 2:
                        reason = "two BLOW UPs in a row"
                elif s == "OK":
                    lo, count = m, 0
                    self.dt_floor = max(self.dt_floor or 0.0, lo)
                else:
                    reason = f"run ended with {s}"
            print(f"[dt] search finished ({reason}): stable {_fmt_dt(lo)} s, BLOW UP {_fmt_dt(hi)} s")
        self.dt_phase, self.dt_floor = None, None
        self.meta["dt_path"] = path
        self.save_state()
        return lo, hi

    # ---------------------------------------------------------------- stage 2: cores exploration
    def _best_at(self, mode, n, dt):
        pts = [a for a in aggregate(self.results) if a["mode"] == mode and a["np"] == n and a["dt"] == dt
               and a["status"] == "OK" and a["wall_s"]]
        return min(pts, key=lambda a: a["wall_s"]) if pts else None

    def explore_cores(self, dt, t1):
        """Per mode: start at ref_cores, E(n) = T(1)/(n T(n)) with the best decomposition;
        E > threshold -> n + core_step, otherwise n - core_step; stop when the next n was already
        tested (direction reversed) or is outside [2, usable cores]."""
        cfg = self.cfg
        thr, step = float(cfg["efficiency_threshold"]), int(cfg["core_step"])
        st = {m: {"n": self.ref_cores, "tested": {}, "active": True, "why": ""}
              for m in cfg["modes"]}
        rnd = 0
        while any(s["active"] for s in st.values()):
            rnd += 1
            lay = []
            for mode, s in st.items():
                if s["active"]:
                    d = self.decomps(mode, s["n"])
                    if not d:
                        s["active"], s["why"] = False, f"no valid decomposition for {s['n']} cores"
                    lay += d
            if not lay:
                break
            print(f"\n--- exploration round {rnd}: "
                  + ", ".join(f"{m.upper()} {s['n']} cores" for m, s in st.items() if s["active"]) + " ---")
            self.build_all(lay)
            self.run_batch([Job(pc, dt, "explore", r, exclusive=bool(cfg["explore_exclusive"]))
                            for pc in lay if self.builds.get(layout_tag(pc)) for r in range(int(cfg["repeat"]))],
                           exclusive=bool(cfg["explore_exclusive"]))
            for mode, s in st.items():
                if not s["active"]:
                    continue
                n = s["n"]
                b = self._best_at(mode, n, dt)
                if b is None:
                    s["active"], s["why"] = False, f"no successful run with {n} cores"
                    continue
                E = t1 / (n * b["wall_s"])
                s["tested"][n] = {"E": E, "layout": f"{b['nx']}x{b['ny']}", "wall": b["wall_s"]}
                nxt = n + step if E > thr else n - step
                print(f"[cores] {mode.upper():6s} n={n:<3d} best {b['nx']}x{b['ny']}  T={_tfmt(b['wall_s'])}  "
                      f"E={E:.3f} {'>' if E > thr else '<='} {thr:g} -> next {nxt}")
                if nxt in s["tested"]:
                    s["active"], s["why"] = False, f"direction reversed at {n} cores"
                elif nxt < 2 or nxt > self.ncores:
                    s["active"], s["why"] = False, f"{nxt} cores outside [2, {self.ncores}]"
                else:
                    s["n"] = nxt
        self.meta["explore_path"] = {m: {"tested": s["tested"], "why": s["why"]}
                                     for m, s in st.items()}
        self.save_state()
        for m, s in st.items():
            ok = [n for n, v in s["tested"].items() if v["E"] > thr]
            print(f"[cores] {m.upper()}: stop ({s['why']}); "
                  + (f"largest core count with E > {thr:g}: {max(ok)}" if ok else f"no core count with E > {thr:g}"))

    def go(self):
        self.plan()
        self.build_all([dict(SERIAL)])
        if not self.builds.get(layout_tag(SERIAL)):
            print("[error] the serial executable could not be compiled")
            return
        print("\n=== stage 1: time-step search, serial build (1 core, no OpenMP/MPI), several runs at once ===")
        lo, hi = self.dt_search()
        if lo is None:
            print("[error] no stable dt found; stopping")
            return
        dt_opt = hour_divisor_floor(lo) if self.cfg["dt_hour_divisor"] else lo
        self.meta["dt_stable_max"], self.meta["dt_opt"] = lo, dt_opt
        self.save_state()
        self.dt_opt = dt_opt
        if dt_opt != lo:
            print(f"[stage] largest stable dt = {_fmt_dt(lo)} s -> optimal dt rounded down to {_fmt_dt(dt_opt)} s "
                  f"(3600/{_fmt_dt(dt_opt)} = {3600 / dt_opt:g} steps per hour)")
        else:
            print(f"[stage] optimal dt = {_fmt_dt(dt_opt)} s")
        # clean serial reference T(1) at dt_opt
        key = run_key(SERIAL, dt_opt, 0)
        r = self.done.get(key)
        if r is None:
            print(f"\n=== serial reference T(1) at dt = {_fmt_dt(dt_opt)} s, alone on the machine ===")
            self.run_batch([Job(dict(SERIAL), dt_opt, "serial-ref", 0)], exclusive=True)
            r = self.done.get(key)
            if r is None or r["status"] != "OK":
                print(f"[warn] the serial run at {_fmt_dt(dt_opt)} s did not end OK: using {_fmt_dt(lo)} s")
                self.dt_opt = dt_opt = lo
                self.meta["dt_opt"] = lo
                key = run_key(SERIAL, dt_opt, 0)
                r = self.done.get(key)
        if self.cfg["serial_ref_exclusive"] and r and (r.get("peers") or 0) > 0:
            print(f"\n=== serial reference T(1) at dt = {_fmt_dt(dt_opt)} s, alone on the machine ===")
            self.run_batch([Job(dict(SERIAL), dt_opt, "serial-ref", 0, exclusive=True)], exclusive=True)
        ser = self._best_at("serial", 1, dt_opt)
        if ser is None:
            print("[error] no serial reference time; stopping")
            return
        t1 = ser["wall_s"]
        print(f"[stage] T(1) = {_tfmt(t1)}" + ("" if ser["clean"] else " (measured sharing the machine)"))
        print(f"\n=== stage 2: cores exploration from {self.ref_cores} cores "
              f"(E > {self.cfg['efficiency_threshold']:g} -> +{self.cfg['core_step']}, otherwise "
              f"-{self.cfg['core_step']}) ===")
        self.explore_cores(dt_opt, t1)
        self.drain()
        # stage 3: re-time alone the points of the efficiency curve measured sharing the machine
        if self.cfg["efficiency_exclusive"] and self.max_jobs > 1:
            at = [a for a in aggregate(self.results) if a["dt"] == dt_opt and a["status"] == "OK"]
            need = [{k: a[k] for k in ("mode", "np", "nx", "ny")}
                    for pts in efficiency_points(at).values() for a in pts if not a["clean"]]
            if need:
                print(f"\n=== stage 3: re-timing {len(need)} efficiency point(s) alone on the machine ===")
                self.run_batch([Job(pc, dt_opt, "exclusive", r, exclusive=True) for pc in need
                                for r in range(int(self.cfg["repeat"]))], exclusive=True)

    def plan(self):
        t = self.topo
        print(f"[plan] machine: {self.meta['machine'].get('cpu')}  ->  {self.ncores} cores usable "
              f"(physical={t.get('physical', '?')}, hyperthreads={t.get('hyperthreads', '?')}, "
              f"sockets={t.get('sockets', '?')}); CPU ids {_cpu_ranges(self.slots)}")
        gi = self.grid_info or {}
        print(f"[plan] cppdefs.h: block {self.config_name}, case {self.case_name or '?'}   param.h: "
              + (f"{' / '.join(gi['chain'])} (line {gi['line']}) -> LLm x MMm x N = {gi['LLm']} x {gi['MMm']} x "
                 f"{gi['N']}" if gi else f"grid {self.grid_size} (not determined from cppdefs.h)"))
        if gi and gi.get("case") == "#else":
            print(f"[warn] case {self.case_name} has no entry of its own in param.h: the '#else' branch "
                  f"({gi['LLm']} x {gi['MMm']} x {gi['N']}) is used; check param.h")
        print(f"[plan] dt in croco.in = {_fmt_dt(self.ts0['dt'])} s  NDTFAST = {self.ts0['ndtfast']}   "
              f"{_days(self.cfg['sim_days'])} per run")
        dtr = float(self.cfg["dt_ref"] or self.ts0["dt"])
        print(f"[plan] stage 1: serial dt search from {_fmt_dt(dtr)} s (x2 while stable, then halving the "
              f"interval until two BLOW UPs in a row), up to {min(self.ncores, self.max_jobs)} runs at once")
        print(f"[plan] stage 2: {', '.join(m.upper() for m in self.cfg['modes'])} from {self.ref_cores} cores, "
              f"step {self.cfg['core_step']}, efficiency threshold {self.cfg['efficiency_threshold']:g}; "
              f"pinning: {'taskset' if self.taskset else 'none (taskset not found)'}")
        for m in self.cfg["modes"]:
            d = self.decomps(m, self.ref_cores)
            print(f"        {m.upper():6s} {self.ref_cores} cores: " + "  ".join(f"{p['nx']}x{p['ny']}" for p in d))
        if self.launcher == "slurm":
            left = (self.deadline - time.time()) if self.deadline else None
            print(f"[plan] launcher: Slurm job {os.environ.get('SLURM_JOB_ID', '?')} (srun job steps, "
                  f"'{self.cfg['slurm_step_flags']}'); wall time left: "
                  f"{_tfmt(left) if left is not None else 'unknown'}; runs stop "
                  f"{_tfmt(self.cfg['walltime_margin_s'])} before the limit")
        else:
            print("[plan] launcher: local (taskset / mpirun)")
        ex = {"serial": dict(SERIAL), "openmp": {"mode": "openmp", "np": self.ref_cores, "nx": 1, "ny": self.ref_cores},
              "mpi": {"mode": "mpi", "np": self.ref_cores, "nx": 1, "ny": self.ref_cores}}
        for mode in ["serial"] + list(self.cfg["modes"]):
            j = Job(ex[mode], self.ts0["dt"], "plan")
            j.cpus = self.slots[:j.np]
            cmd, env = self._command(j, Path("<exe>"))
            extra = "  OMP_NUM_THREADS=" + env.get("OMP_NUM_THREADS", "") if mode == "openmp" else ""
            print(f"        {mode:6s} command: {cmd}{extra}")
        return self.layouts()

    def dry_run(self):
        lay = self.plan()
        pc = next((p for p in lay if p["mode"] != "serial"), lay[0])
        import difflib
        cpp, par, n12 = self._edited_sources(pc)
        dtr = float(self.cfg["dt_ref"] or self.ts0["dt"])
        cin, ts, n3 = set_croco_in(self.pristine["croco_in"], self.cfg, dtr)
        cin2, links, outputs, kept = rewrite_croco_in_paths(cin, self.run_dir, self.cfg["output_keywords"],
                                                            self.keep_kw)
        for name, a, b in (("cppdefs.h", self.pristine["cppdefs"], cpp),
                           ("param.h", self.pristine["param"], par),
                           ("croco.in (run sandbox)", self.pristine["croco_in"], cin2)):
            print(f"\n--- edits in {name} for {layout_tag(pc)}, dt={_fmt_dt(dtr)} ---")
            sys.stdout.writelines(difflib.unified_diff(a.splitlines(True), b.splitlines(True), n=0))
        cpp_s, _, _ = self._edited_sources(dict(SERIAL))
        print("\n--- edits in cppdefs.h for the serial build ---")
        sys.stdout.writelines(difflib.unified_diff(self.pristine["cppdefs"].splitlines(True),
                                                   cpp_s.splitlines(True), n=0))
        if self.online:
            print("\n[sandbox] ONLINE is defined in cppdefs.h: the 'online:' section of croco.in is left unchanged")
        print("\n[sandbox] inputs linked read-only:")
        for n, s in links.items():
            print(f"          inputs/{n} -> {s}")
        for t in kept:
            first = [p for p in Path(t).parts if p != "."][0]
            real = os.path.normpath(self.run_dir / first)
            print(f"          {t}  (unchanged in croco.in; '{first}' linked to {real}"
                  + ("" if os.path.exists(real) else "  -- WARNING: not found") + ")")
        miss = missing_inputs(outputs, self.cfg["output_keywords"])
        print("[sandbox] outputs written inside each run directory: "
              + ", ".join(f"{k} {v}" for k, v in outputs.items() if k not in miss))
        for k, v in miss.items():
            print(f"[warn] '{k}' file(s) {v} do not exist in run_dir: they are treated as outputs; "
                  f"if they are inputs CROCO will stop with 'file not found'")
        for n in n12 + n3:
            print(f"[note] {n}")
        if self.grid:
            g = self.grid
            print(f"\n[cfl] h_max={g['h_max']:.1f} m dx_min={g['dx_min']:.0f} m dy_min={g['dy_min']:.0f} m")
            for e in range(-3, 6):
                dt = dtr * 2.0 ** e
                _, t, _ = set_croco_in(self.pristine["croco_in"], self.cfg, dt)
                print(f"      dt={_fmt_dt(dt):>8s}  dtfast={t['dtfast']:.2f}  "
                      f"CFL_bt={t['dtfast'] * g['ext_factor']:.3f}  CFL_int={dt * g['int_factor']:.3f}")


def _cpu_ranges(cpus):
    """[0,1,2,3,8] -> '0-3,8'"""
    cpus = sorted(cpus)
    out, s = [], 0
    for i in range(1, len(cpus) + 1):
        if i == len(cpus) or cpus[i] != cpus[i - 1] + 1:
            out.append(f"{cpus[s]}" if s == i - 1 else f"{cpus[s]}-{cpus[i - 1]}")
            s = i
    return ",".join(out)

# ----------------------------------------------------------------------------------------------
# Analysis, figures and report
# ----------------------------------------------------------------------------------------------
def aggregate(results):
    """One entry per (layout, dt): status and median wall time."""
    groups = {}
    for r in results:
        k = (r["mode"], int(r["np"]), int(r["nx"]), int(r["ny"]), float(r["dt"]))
        groups.setdefault(k, []).append(r)
    agg = []
    for (mode, n, nx, ny, dt), rs in groups.items():
        sts = [r["status"] for r in rs]
        status = "BLOWUP" if "BLOWUP" in sts else "OK" if "OK" in sts else sts[-1]
        ok_all = [r for r in rs if r["status"] == "OK" and r.get("wall_s") is not None]
        clean = [r for r in ok_all if r.get("exclusive") or not (r.get("peers") or 0)]
        conc = [r for r in ok_all if r not in clean]
        ok = clean or ok_all  # a run measured alone on the machine is preferred
        wall = statistics.median([r["wall_s"] for r in ok]) if ok else None
        first = (ok or rs)[0]
        sd = max([r["sim_days"] or 0 for r in rs] or [0])
        agg.append({"mode": mode, "np": n, "nx": nx, "ny": ny, "dt": dt, "status": status,
                    "wall_s": wall, "nrep": len(ok), "spread": (max(r["wall_s"] for r in ok) -
                                                                min(r["wall_s"] for r in ok)) if len(ok) > 1 else 0.0,
                    "sim_days": sd, "ndtfast": first.get("ndtfast"), "dtfast": first.get("dtfast"),
                    "cfl_bt": first.get("cfl_bt"), "cfl_int": first.get("cfl_int"),
                    "cfl_adv": max([r["cfl_adv"] for r in rs if r.get("cfl_adv") is not None], default=None),
                    "days_per_hour": (sd / (wall / 3600.0)) if wall else None,
                    "clean": bool(clean), "peers": max([r.get("peers") or 0 for r in ok], default=0),
                    "wall_clean": statistics.median([r["wall_s"] for r in clean]) if clean else None,
                    "wall_conc": statistics.median([r["wall_s"] for r in conc]) if conc else None,
                    "tag": f"{mode}_np{n}_{nx}x{ny}"})
    return agg


def stability_summary(agg):
    by_dt = {}
    for a in agg:
        by_dt.setdefault(a["dt"], []).append(a)
    rows = []
    for dt in sorted(by_dt):
        g = by_dt[dt]
        sts = {a["status"] for a in g}
        status = "BLOWUP" if "BLOWUP" in sts else "OK" if "OK" in sts else "/".join(sorted(sts))
        pick = lambda k: max([a[k] for a in g if a.get(k) is not None], default=None)  # noqa: E731
        rows.append({"dt": dt, "status": status, "cfl_bt": pick("cfl_bt"), "cfl_int": pick("cfl_int"),
                     "cfl_adv": pick("cfl_adv"), "dtfast": pick("dtfast"), "ndtfast": pick("ndtfast"),
                     "n": len(g), "best_wall": min([a["wall_s"] for a in g if a["wall_s"]], default=None)})
    first_fail = min([r["dt"] for r in rows if r["status"] == "BLOWUP"], default=None)
    stable = [r for r in rows if r["status"] == "OK" and (first_fail is None or r["dt"] < first_fail)]
    return rows, (stable[-1] if stable else None), \
        (next(r for r in rows if r["dt"] == first_fail) if first_fail else None)


def hour_divisor_floor(dt):
    """Largest integer dt <= the given one with 3600/dt an integer (whole number of steps per hour)."""
    if dt is None or dt < 1:
        return dt
    return float(max(d for d in range(1, 3601) if 3600 % d == 0 and d <= dt))


def _days(d):
    return f"{d:g} day" + ("" if float(d) == 1.0 else "s")


def _tfmt(sec):
    if sec is None:
        return "–"
    if sec < 120:
        return f"{sec:.1f} s"
    if sec < 7200:
        return f"{sec / 60:.1f} min"
    return f"{sec / 3600:.2f} h"


def _tunit(values):
    m = max([v for v in values if v] or [1])
    return (1.0, "s") if m < 300 else (60.0, "min") if m < 3 * 3600 else (3600.0, "h")


def _short_notes(notes, n=70):
    """Notes for the report table: rounding remarks dropped (they stay in results.csv), truncated."""
    keep = [x.strip() for x in str(notes or "").split(";")
            if x.strip() and not re.search(r"does not divide|not an exact multiple|simulates [\d.]+ days", x)]
    t = "; ".join(keep)
    return t if len(t) <= n else t[:n - 1] + "…"


def _measured(a):
    if not a.get("wall_s"):
        return "–"
    return "alone" if a["clean"] else f"with {a['peers']} concurrent"


def _f(x, fmt="{:.3f}", na="–"):
    return na if x is None else fmt.format(x)


def make_figures(agg, out: Path, dt_opt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"openmp": "#2a6fdb", "mpi": "#d9572b", "serial": "#343a40"}
    figs = []
    U, un = _tunit([a["wall_s"] for a in agg])

    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
    # (a) wall time vs dt
    ax = axs[0, 0]
    tags = sorted({a["tag"] for a in agg})
    multi = [t for t in tags if len({a["dt"] for a in agg if a["tag"] == t}) > 1]
    for t in multi:
        pts = sorted([a for a in agg if a["tag"] == t], key=lambda a: a["dt"])
        ok = [a for a in pts if a["status"] == "OK"]
        mode = pts[0]["mode"]
        ln, = ax.plot([a["dt"] for a in ok], [a["wall_s"] / U for a in ok], "o-",
                      color=colors[mode], alpha=0.85, label=t)
        ls = "-" if mode == "openmp" else "--"
        ln.set_linestyle(ls)
        bu = [a["dt"] for a in pts if a["status"] == "BLOWUP"]
        if bu:
            ax.scatter(bu, [0.97] * len(bu), marker="X", s=110, color="crimson", zorder=5,
                       transform=ax.get_xaxis_transform())
    ax.scatter([], [], marker="X", s=80, color="crimson", label="BLOW UP")
    xs = [a["dt"] for a in agg if a["tag"] in multi]
    if xs and max(xs) / min(xs) > 8:  # wide dt range (doubling search): log2 axis
        ax.set_xscale("log", base=2)
        ax.set_xlim(min(xs) / 1.3, max(xs) * 1.3)
    elif xs:
        pad = 0.05 * (max(xs) - min(xs) or max(xs))
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_xlabel("Baroclinic time step dt [s]")
    ax.set_ylabel(f"Wall-clock time [{un}]")
    ax.set_title("(a) Run time vs. time step")
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)

    # (b) wall time vs cores at dt_opt (best layout per np and mode)
    ax = axs[0, 1]
    at = [a for a in agg if a["dt"] == dt_opt and a["status"] == "OK"]
    base = None
    best_np = {}
    for a in at:
        k = (a["mode"], a["np"])
        if k not in best_np or a["wall_s"] < best_np[k]["wall_s"]:
            best_np[k] = a
    if best_np:
        n_min = min(n for _, n in best_np)
        base = min(a["wall_s"] * a["np"] / n_min for (m, n), a in best_np.items() if n == n_min)
    ser = [best_np[("serial", 1)]] if ("serial", 1) in best_np else []
    for mode in ("openmp", "mpi"):
        pts = sorted([a for (m, _), a in best_np.items() if m == mode], key=lambda a: a["np"])
        pts = ser + [a for a in pts if not (ser and a["np"] == 1)] if pts else []
        if pts:
            ax.plot([a["np"] for a in pts], [a["wall_s"] / U for a in pts], "o-", color=colors[mode],
                    label=f"{mode.upper()} (best layout{', 1 core = serial' if ser else ''})")
            for a in pts:
                if a["mode"] == "serial" and mode != "openmp":
                    continue
                ax.annotate("serial" if a["mode"] == "serial" else f"{a['nx']}x{a['ny']}",
                            (a["np"], a["wall_s"] / U), fontsize=7,
                            color=colors[mode], textcoords="offset points",
                            xytext=(5, 5) if mode == "openmp" else (5, -11))
    if best_np:
        ns = sorted({n for _, n in best_np})
        n_min = ns[0]
        ax.plot(ns, [base * n_min / n / U for n in ns], "k:", label="ideal scaling")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Cores")
    ax.set_ylabel(f"Wall-clock time [{un}]")
    ax.set_title(f"(b) Scaling at dt = {_fmt_dt(dt_opt) if dt_opt else '?'} s")
    ax.grid(alpha=0.3, which="both")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)

    # (c) parallel efficiency
    ax = axs[1, 0]
    for mode in ("openmp", "mpi"):
        pts = sorted([a for (m, _), a in best_np.items() if m == mode], key=lambda a: a["np"])
        pts = ser + [a for a in pts if not (ser and a["np"] == 1)] if pts else []
        if pts and base:
            n_min = min(n for _, n in best_np)
            sp = [base / a["wall_s"] for a in pts]
            eff = [s / (a["np"] / n_min) * 100 for s, a in zip(sp, pts)]
            ax.plot([a["np"] for a in pts], eff, "o-", color=colors[mode], label=mode.upper())
    ax.axhline(100, color="k", ls=":", lw=1)
    ax.set_xscale("log", base=2)
    if best_np:
        ns = sorted({n for _, n in best_np})
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
    ax.set_ylim(0, 120)
    ax.set_xlabel("Cores")
    ax.set_ylabel("Parallel efficiency [%]")
    ax.set_title("(c) Efficiency relative to the smallest core count")
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)

    # (d) all layouts at dt_opt
    ax = axs[1, 1]
    at_s = sorted(at, key=lambda a: a["wall_s"])[:30]
    y = np.arange(len(at_s))
    ax.barh(y, [a["wall_s"] / U for a in at_s], color=[colors[a["mode"]] for a in at_s])
    ax.set_yticks(y)
    ax.set_yticklabels([a["tag"] for a in at_s], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"Wall-clock time [{un}]")
    ax.set_title(f"(d) All layouts at dt = {_fmt_dt(dt_opt) if dt_opt else '?'} s (fastest first)")
    ax.grid(alpha=0.3, axis="x")
    fig.suptitle("CROCO benchmark: time to simulate the target period", fontsize=13)
    fig.tight_layout()
    p = out / "benchmark_summary.png"
    _save_fig(fig, p)
    plt.close(fig)
    figs.append(p)

    # CFL figure
    rows, last_ok, first_fail = stability_summary(agg)
    if any(r["cfl_bt"] is not None or r["cfl_adv"] is not None for r in rows):
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, lab, mk in (("cfl_bt", "barotropic (dt/NDTFAST)", "o"),
                             ("cfl_int", "internal waves", "s"),
                             ("cfl_adv", "advective (max |u|/dx+|v|/dy)", "^")):
            pts = [r for r in rows if r[key] is not None]
            if pts:
                ax.plot([r["dt"] for r in pts], [r[key] for r in pts], marker=mk, label=lab)
        bu = [r["dt"] for r in rows if r["status"] == "BLOWUP"]
        if bu:
            ax.scatter(bu, [0.03] * len(bu), marker="X", s=90, color="crimson", zorder=5,
                       transform=ax.get_xaxis_transform(), label="BLOW UP")
        if first_fail:
            ax.axvline(first_fail["dt"], color="crimson", ls="--", label="first BLOW UP")
        if last_ok:
            ax.axvline(last_ok["dt"], color="green", ls="--", label="largest stable dt")
        dts_ = [r["dt"] for r in rows]
        if dts_ and max(dts_) / min(dts_) > 8:
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
        ax.set_xlabel("dt [s]")
        ax.set_ylabel("Courant number")
        ax.set_title("CFL numbers vs. time step")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        fig.tight_layout()
        p = out / "cfl_vs_dt.png"
        _save_fig(fig, p)
        plt.close(fig)
        figs.append(p)
    return figs


def _save_fig(fig, p: Path):
    """Write the figure to a temporary file and rename it, so a viewer never sees a half-written PNG."""
    tmp = p.with_name(p.stem + ".tmp" + p.suffix)
    fig.savefig(tmp, dpi=130)
    os.replace(tmp, p)


def _write_atomic(p: Path, text: str):
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, p)


class LiveReport:
    """Rebuilds figures and report in a background thread after every finished simulation
    (several finishes close together are coalesced into one update)."""

    def __init__(self, bench):
        self.b = bench
        self.ev, self.stop_ev = threading.Event(), threading.Event()
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.n = 0

    def start(self):
        self.t.start()
        return self

    def notify(self):
        self.ev.set()

    def _loop(self):
        while not self.stop_ev.is_set():
            if not self.ev.wait(timeout=0.5):
                continue
            self.ev.clear()
            try:
                write_report(list(self.b.results), self.b.cfg, self.b.meta, Path(self.b.cfg["output_dir"]),
                             partial=True)
                self.n += 1
            except Exception as e:  # noqa: BLE001
                print(f"[warn] live report update failed: {e}")

    def stop(self):
        self.stop_ev.set()
        self.t.join(timeout=300)


def _parse_ranges(s):
    out = []
    for part in str(s or "").split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip().isdigit():
            out.append(int(part))
    return out


def efficiency_points(agg_at_dt):
    """Best (fastest) layout for each (mode, cores) among the OK runs at one dt."""
    best = {}
    for a in agg_at_dt:
        if a["status"] != "OK" or not a["wall_s"]:
            continue
        k = (a["mode"], a["np"])
        if k not in best or a["wall_s"] < best[k]["wall_s"]:
            best[k] = a
    out = {}
    for (mode, n), a in sorted(best.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        out.setdefault(mode, []).append(a)
    return out


def efficiency_study(agg, dt, threshold):
    """NLHPC efficiency study: S(n) = T(1)/T(n), E(n) = S(n)/n, per parallel mode."""
    res = {}
    allpts = efficiency_points([a for a in agg if a["dt"] == dt])
    serial = (allpts.pop("serial", None) or [None])[0]
    for mode, pts in allpts.items():
        if serial is not None:  # the serial build (no OpenMP/MPI) is T(1) for every mode
            pts = [serial] + [a for a in pts if a["np"] > 1]
        base = pts[0]
        n0, t0 = base["np"], base["wall_s"]
        rows = []
        for a in pts:
            # without a 1-core run, T(1) is estimated as n0*T(n0) (ideal scaling up to n0 cores)
            S = n0 * t0 / a["wall_s"]
            E = S / a["np"]
            rows.append({"mode": mode, "cores": a["np"],
                         "layout": "serial" if a["mode"] == "serial" else f"{a['nx']}x{a['ny']}",
                         "time_s": a["wall_s"], "speedup": S, "efficiency": E, "alone": a["clean"],
                         "ok": E > threshold})
        # recommended: largest core count with efficiency above the threshold
        okr = [r for r in rows if r["ok"] and r["cores"] > 1]
        rec = max(okr, key=lambda r: r["cores"]) if okr else None
        res[mode] = {"rows": rows, "baseline_cores": n0, "recommended": rec,
                     "serial_baseline": base["mode"] == "serial"}
    return res


def make_efficiency_figure(study, out: Path, dt, threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not study:
        return None
    colors = {"openmp": "#2a6fdb", "mpi": "#d9572b", "serial": "#343a40"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    allc = sorted({r["cores"] for st in study.values() for r in st["rows"]})
    for mode, st in study.items():
        rows = st["rows"]
        c = [r["cores"] for r in rows]
        a1.plot(c, [r["speedup"] for r in rows], "o-", color=colors[mode], label=mode.upper())
        a2.plot(c, [r["efficiency"] for r in rows], "o-", color=colors[mode], label=mode.upper())
        if st["recommended"]:
            rr = st["recommended"]
            a2.plot(rr["cores"], rr["efficiency"], "*", ms=16, color=colors[mode], mec="k",
                    label=f"{mode.upper()}: recommended {rr['cores']} cores")
    a1.plot(allc, allc, "k:", label="ideal (S = n)")
    a1.set_xlabel("Number of cores")
    a1.set_ylabel("Speed-up  S(n) = T(1) / T(n)")
    a1.set_title("Speed-up")
    a2.axhline(threshold, color="crimson", ls="--", label=f"minimum efficiency {threshold:g}")
    a2.axhline(1.0, color="k", ls=":", lw=1)
    a2.set_ylim(0, 1.15)
    a2.set_xlabel("Number of cores")
    a2.set_ylabel("Efficiency  E(n) = S(n) / n")
    a2.set_title("Parallel efficiency")
    for ax in (a1, a2):
        ax.set_xticks(allc)
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle(f"CROCO efficiency study (dt = {_fmt_dt(dt)} s, best layout per core count)", fontsize=12)
    fig.tight_layout()
    p = out / "efficiency_study.png"
    _save_fig(fig, p)
    plt.close(fig)
    return p


def make_schedule_figure(results, out: Path):
    """Gantt chart: which runs occupied which cores and when."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    rs = [r for r in results if r.get("t_start") and r.get("wall_s") and r.get("cpus")]
    if not rs:
        return None
    t0 = min(r["t_start"] for r in rs)
    tend = max(r["t_start"] + r["wall_s"] for r in rs)
    U, un = _tunit([tend - t0])
    cpus = sorted({c for r in rs for c in _parse_ranges(r["cpus"])})
    row = {c: i for i, c in enumerate(cpus)}
    fixed = {"dt-search": "#2a6fdb", "dt-refine": "#d9572b", "serial-ref": "#343a40", "parallel": "#2f9e44",
             "full": "#2f9e44", "filler-dt": "#e8a33d", "filler-rep": "#9c36b5", "dt-confirm": "#1098ad",
             "exclusive": "#868e96", "explore": "#2f9e44"}
    stages = [st for st in fixed if any(r["stage"] == st for r in rs)]
    stages += [st for st in dict.fromkeys(r["stage"] for r in rs) if st not in fixed]
    col = {st: fixed.get(st, "#495057") for st in stages}
    fig, ax = plt.subplots(figsize=(12, max(3.4, 0.32 * len(cpus) + 2.0)))
    for r in rs:
        x0, w = (r["t_start"] - t0) / U, r["wall_s"] / U
        ys = [row[c] for c in _parse_ranges(r["cpus"])]
        for y in ys:
            ax.barh(y, w, left=x0, height=0.8, color=col[r["stage"]],
                    edgecolor="crimson" if r["status"] == "BLOWUP" else "white",
                    linewidth=1.5 if r["status"] == "BLOWUP" else 0.6)
        if w > 0.04 * (tend - t0) / U:
            ax.text(x0 + w / 2, (min(ys) + max(ys)) / 2, f"{ {'openmp': 'omp', 'mpi': 'mpi'}.get(r['mode'], 'ser')} {r['nx']}x{r['ny']}\n"
                    f"dt={_fmt_dt(r['dt'])}", ha="center", va="center", fontsize=6, color="white")
    ax.set_yticks(range(len(cpus)))
    ax.set_yticklabels([f"cpu {c}" for c in cpus], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"Time since start of benchmark [{un}]")
    ax.set_title("Run schedule on the machine cores (red edge = BLOW UP)")
    ax.legend(handles=[Patch(color=col[s], label=s) for s in stages], fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=min(7, len(stages)), frameon=False)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    p = out / "schedule.png"
    _save_fig(fig, p)
    plt.close(fig)
    return p


_TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
_TEX_UNICODE = {"×": r"$\times$", "·": r"$\cdot$", "›": r"$\triangleright$", "→": r"$\rightarrow$",
                "≈": r"$\approx$", "≤": r"$\leq$", "≥": r"$\geq$", "√": r"$\surd$", "₁": r"$_1$", "²": r"$^2$",
                "★": r"$\star$", "−": r"$-$", "–": "--", "—": "---", "…": r"\ldots{}", "“": "``", "”": "''",
                "±": r"$\pm$", "°": r"$^\circ$", "✓": r"$\checkmark$", "✗": r"$\times$", "€": "EUR"}


def _tex_escape(t: str) -> str:
    out = "".join(_TEX_SPECIAL.get(ch, ch) for ch in t)
    return "".join(_TEX_UNICODE.get(ch, ch) for ch in out)


def _tex_inline(text: str) -> str:
    """Markdown inline markup (**bold**, *italic*, `code`, [text](url)) -> LaTeX, with escaping."""
    keep = []

    def stash(s):
        keep.append(s)
        return f"\x00{len(keep) - 1}\x00"
    t = re.sub(r"`([^`]+)`", lambda m: stash(r"\texttt{" + _tex_escape(m.group(1)) + "}"), str(text))
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
               lambda m: stash(r"\href{" + m.group(2).replace("%", r"\%").replace("#", r"\#") + "}{"
                               + _tex_escape(m.group(1)) + "}"), t)
    t = _tex_escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", t)
    return re.sub("\x00(\\d+)\x00", lambda m: keep[int(m.group(1))], t)


class ReportDoc:
    """Structured report: the same blocks are rendered to Markdown and to PDF."""

    def __init__(self):
        self.blocks, self.nfig, self.ntab = [], 0, 0

    def title(self, text, subtitle=""):
        self.blocks.append(("title", text, subtitle))

    def h(self, level, text):
        self.blocks.append(("h", level, text))

    def p(self, text):
        self.blocks.append(("p", text))

    def note(self, text, kind="info"):
        self.blocks.append(("note", kind, text))

    def ul(self, items):
        if items:
            self.blocks.append(("ul", list(items)))

    def kv(self, rows):
        self.blocks.append(("kv", list(rows)))

    def kpis(self, cards):
        if cards:
            self.blocks.append(("kpi", list(cards)))

    def table(self, header, rows, align=None, caption=None):
        if not rows:
            return
        self.ntab += 1
        self.blocks.append(("table", header, rows, align or ["l"] * len(header), caption, self.ntab))

    def image(self, path, caption):
        if path and Path(path).exists():
            self.nfig += 1
            self.blocks.append(("img", Path(path), caption, self.nfig))

    def pagebreak(self):
        self.blocks.append(("pb",))

    # ------------------------------------------------------------------ Markdown
    def markdown(self):
        L = []
        for b in self.blocks:
            k = b[0]
            if k == "title":
                L += [f"# {b[1]}", ""] + ([f"*{b[2]}*", ""] if b[2] else [])
            elif k == "h":
                L += ["#" * b[1] + " " + b[2], ""]
            elif k == "p":
                L += [b[1], ""]
            elif k == "note":
                L += [f"> **{'Warning' if b[1] == 'warn' else 'Note'}:** {b[2]}", ""]
            elif k == "ul":
                L += [f"- {x}" for x in b[1]] + [""]
            elif k == "kv":
                L += ["| | |", "|:---|:---|"] + [f"| **{a}** | {v} |" for a, v in b[1]] + [""]
            elif k == "kpi":
                L += ["| Result | Value | Detail |", "|:---|:---|:---|"]
                L += [f"| {a} | **{v}** | {s} |" for a, v, s in b[1]] + [""]
            elif k == "table":
                _, header, rows, align, cap, n = b
                if cap:
                    L += [f"*Table {n}. {cap}*", ""]
                sep = {"l": ":---", "r": "---:", "c": ":---:"}
                L += ["| " + " | ".join(header) + " |", "|" + "|".join(sep[a] for a in align) + "|"]
                L += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]
            elif k == "img":
                L += [f"![{b[2]}]({b[1].name})", "", f"*Figure {b[3]}. {b[2]}*", ""]
        return "\n".join(L) + "\n"

    # ------------------------------------------------------------------ LaTeX
    def latex(self):
        """Standalone LaTeX source (pdflatex or xelatex; standard packages only)."""
        L = [r"\documentclass[10pt,a4paper]{article}",
             r"\usepackage[T1]{fontenc}", r"\usepackage[utf8]{inputenc}",
             r"\IfFileExists{lmodern.sty}{\usepackage{lmodern}}{}",
             r"\usepackage[margin=18mm,headheight=14pt]{geometry}", r"\usepackage[table]{xcolor}",
             r"\usepackage{graphicx}", r"\usepackage{booktabs}", r"\usepackage{longtable}", r"\usepackage{array}",
             r"\usepackage{amssymb}", r"\usepackage{fancyhdr}",
             r"\IfFileExists{lastpage.sty}{\usepackage{lastpage}\newcommand{\ofpages}{\ of \pageref{LastPage}}}"
             r"{\newcommand{\ofpages}{}}",
             r"\IfFileExists{caption.sty}{\usepackage{caption}"
             r"\captionsetup{font=small,labelfont={bf,color=navy}}}{}",
             r"\IfFileExists{enumitem.sty}{\usepackage{enumitem}"
             r"\setlist[itemize]{leftmargin=14pt,itemsep=2pt,topsep=2pt}}{}",
             r"\usepackage[hidelinks]{hyperref}",
             r"\definecolor{navy}{HTML}{16324F}", r"\definecolor{accent}{HTML}{2A6FDB}",
             r"\definecolor{zebra}{HTML}{F4F6F9}", r"\definecolor{noteinfo}{HTML}{EEF4FD}",
             r"\definecolor{notewarn}{HTML}{FFF6E5}", r"\definecolor{muted}{HTML}{6B7685}",
             r"\setlength{\parindent}{0pt}", r"\setlength{\parskip}{4pt}",
             r"\renewcommand{\arraystretch}{1.15}",
             r"\pagestyle{fancy}\fancyhf{}", r"\renewcommand{\headrulewidth}{0.3pt}",
             r"\fancyfoot[L]{\color{muted}\footnotesize Generated by croco\_bench.py}",
             r"\fancyfoot[R]{\color{muted}\footnotesize Page \thepage\ofpages}"]
        hdr = next((b for b in self.blocks if b[0] == "title"), None)
        if hdr:
            L.append(r"\fancyhead[L]{\color{muted}\footnotesize " + _tex_inline(hdr[1] + " · " + hdr[2]) + "}")
        L += [r"\begin{document}"]
        for b in self.blocks:
            k = b[0]
            if k == "title":
                L += [r"\noindent\colorbox{navy}{\parbox{\dimexpr\linewidth-2\fboxsep}{\vspace{8pt}",
                      r"{\color{white}\LARGE\bfseries " + _tex_inline(b[1]) + r"}\\[4pt]",
                      r"{\color{white!80!accent}\small " + _tex_inline(b[2]) + r"}\vspace{6pt}}}",
                      r"\par\noindent{\color{accent}\rule{\linewidth}{2pt}}\par\medskip"]
            elif k == "h":
                L.append((r"\section{" if b[1] <= 2 else r"\subsection*{") + _tex_inline(b[2]) + "}")
            elif k == "p":
                L += [_tex_inline(b[1]), ""]
            elif k == "note":
                col = "notewarn" if b[1] == "warn" else "noteinfo"
                L += [r"\par\noindent\colorbox{" + col + r"}{\parbox{\dimexpr\linewidth-2\fboxsep}{\small "
                      + (r"\textbf{Warning:} " if b[1] == "warn" else r"\textbf{Note:} ") + _tex_inline(b[2])
                      + r"}}\par\medskip"]
            elif k == "ul":
                L += [r"\begin{itemize}"] + [r"\item " + _tex_inline(x) for x in b[1]] + [r"\end{itemize}"]
            elif k == "kv":
                L += [r"{\small\noindent\begin{tabular}{@{}p{0.24\linewidth}p{0.73\linewidth}@{}}"]
                L += [r"\textbf{\textcolor{navy}{" + _tex_inline(a) + "}} & " + _tex_inline(v) + r" \\[1pt]"
                      for a, v in b[1]]
                L += [r"\end{tabular}}\par\medskip"]
            elif k == "kpi":
                L += [r"\begin{center}\small\rowcolors{1}{zebra}{zebra}",
                      r"\begin{tabular}{!{\color{accent}\vrule width 2pt}>{\raggedright\arraybackslash}p{0.29\linewidth}"
                      r"!{\color{accent}\vrule width 2pt}>{\raggedright\arraybackslash}p{0.29\linewidth}"
                      r"!{\color{accent}\vrule width 2pt}>{\raggedright\arraybackslash}p{0.29\linewidth}}"]
                cells = [r"{\scriptsize\color{muted}" + _tex_inline(a.upper()) + r"}\newline"
                         r"{\large\bfseries\color{navy} " + _tex_inline(v) + r"}\newline{\scriptsize " +
                         _tex_inline(s_) + "}" for a, v, s_ in b[1]]
                while len(cells) % 3:
                    cells.append("")
                for i in range(0, len(cells), 3):
                    L.append(" & ".join(cells[i:i + 3]) + r" \\[3pt]")
                L += [r"\end{tabular}\end{center}"]
            elif k == "table":
                _, header, rows, align, caption, n = b
                size = r"\scriptsize" if len(header) > 8 else r"\footnotesize"
                spec = "".join({"l": "l", "r": "r", "c": "c"}[a] for a in align)
                if len(header) > 8:  # wide table: let the last text column wrap
                    spec = spec[:-1] + r"p{0.16\linewidth}" if spec[-1] == "l" else spec
                L += ["{" + size + r"\rowcolors{2}{white}{zebra}",
                      r"\begin{longtable}{" + spec + "}"]
                if caption:
                    L.append(r"\caption{" + _tex_inline(caption) + r"}\\")
                head = " & ".join(r"\textbf{" + _tex_inline(h) + "}" for h in header) + r" \\"
                L += [r"\toprule", head, r"\midrule", r"\endfirsthead", r"\toprule", head, r"\midrule", r"\endhead",
                      r"\bottomrule", r"\endfoot"]
                L += [" & ".join(_tex_inline(str(c)) for c in r) + r" \\" for r in rows]
                L += [r"\end{longtable}}"]
            elif k == "img":
                L += [r"\begin{figure}[htbp]\centering",
                      r"\includegraphics[width=\linewidth,height=0.55\textheight,keepaspectratio]{"
                      + b[1].name + "}", r"\caption{" + _tex_inline(b[2]) + "}", r"\end{figure}"]
            elif k == "pb":
                L.append(r"\clearpage")
        L += [r"\end{document}", ""]
        return "\n".join(L)

    # ------------------------------------------------------------------ PDF
    def pdf(self, out_pdf: Path, header_text=""):
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        from reportlab.lib.fonts import addMapping
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.platypus import (CondPageBreak, Image, KeepTogether, ListFlowable, ListItem, PageBreak,
                                        Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)
        from reportlab.platypus.flowables import HRFlowable
        import matplotlib
        ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        for name, f in (("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
                        ("DejaVu-Oblique", "DejaVuSans-Oblique.ttf"),
                        ("DejaVu-BoldOblique", "DejaVuSans-BoldOblique.ttf"), ("DejaVuMono", "DejaVuSansMono.ttf")):
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, str(ttf / f)))
        addMapping("DejaVu", 0, 0, "DejaVu")
        addMapping("DejaVu", 1, 0, "DejaVu-Bold")
        addMapping("DejaVu", 0, 1, "DejaVu-Oblique")
        addMapping("DejaVu", 1, 1, "DejaVu-BoldOblique")

        NAVY, ACCENT, INK = colors.HexColor("#16324f"), colors.HexColor("#2a6fdb"), colors.HexColor("#1f2933")
        MUTED, LINE, ZEBRA = colors.HexColor("#6b7685"), colors.HexColor("#cfd6df"), colors.HexColor("#f4f6f9")
        body = ParagraphStyle("body", fontName="DejaVu", fontSize=9.2, leading=13, textColor=INK,
                              alignment=TA_LEFT, spaceAfter=5)
        small = ParagraphStyle("small", parent=body, fontSize=7.6, leading=10, textColor=MUTED, spaceAfter=0)
        cap = ParagraphStyle("cap", parent=body, fontName="DejaVu-Oblique", fontSize=8, leading=10.5,
                             textColor=MUTED, spaceBefore=2, spaceAfter=8)
        h2 = ParagraphStyle("h2", parent=body, fontName="DejaVu-Bold", fontSize=13.5, leading=17, textColor=NAVY,
                            spaceBefore=4, spaceAfter=2, keepWithNext=1)
        h3 = ParagraphStyle("h3", parent=body, fontName="DejaVu-Bold", fontSize=10.5, leading=14, textColor=NAVY,
                            spaceBefore=6, spaceAfter=3, keepWithNext=1)
        page_w, page_h = A4
        margin = 16 * mm
        avail = page_w - 2 * margin
        story, sec = [], 0

        def colwidths(rows_plain, header_bold=True, fs=7.4, pad=10.0):
            ncol = max(len(r) for r in rows_plain)

            def wd(t, k, size):  # measured in bold (cells may be bold) plus 3 % margin
                return 1.03 * pdfmetrics.stringWidth(t, "DejaVu-Bold", size)
            while True:
                wmin = [max(max((wd(x, k, fs) for x in r[c].split()), default=0.0)
                            for k, r in enumerate(rows_plain)) + pad for c in range(ncol)]
                if sum(wmin) <= avail or fs <= 5.0:
                    break
                fs -= 0.4
            wtgt = [max(wmin[c], max(wd(r[c][:48], k, fs) for k, r in enumerate(rows_plain)) + pad)
                    for c in range(ncol)]
            if sum(wtgt) <= avail:
                return wtgt, fs
            extra = max(0.0, avail - sum(wmin))
            grow = [wtgt[c] - wmin[c] for c in range(ncol)]
            g = float(sum(grow)) or 1.0
            return [wmin[c] + extra * grow[c] / g for c in range(ncol)], fs

        def plain(t):
            return re.sub(r"[*`]", "", re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", str(t)))

        for b in self.blocks:
            k = b[0]
            if k == "title":
                band = Table([[Paragraph(f'<font name="DejaVu-Bold" size="20" color="white">{_md_inline(b[1])}</font>',
                                         body)],
                              [Paragraph(f'<font size="10" color="#d6e2f2">{_md_inline(b[2])}</font>', body)]],
                             colWidths=[avail])
                band.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY),
                                          ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                                          ("TOPPADDING", (0, 0), (0, 0), 14), ("BOTTOMPADDING", (0, -1), (-1, -1), 12),
                                          ("LINEBELOW", (0, -1), (-1, -1), 3, ACCENT)]))
                story += [band, Spacer(1, 10)]
            elif k == "h":
                if b[1] <= 2:
                    sec += 1
                    story += [CondPageBreak(60 * mm), Spacer(1, 6),
                              Paragraph(f"{sec}. {_md_inline(b[2])}", h2),
                              HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceBefore=1, spaceAfter=7)]
                else:
                    story.append(Paragraph(_md_inline(b[2]), h3))
            elif k == "p":
                story.append(Paragraph(_md_inline(b[1]), body))
            elif k == "note":
                col = colors.HexColor("#e8a33d") if b[1] == "warn" else ACCENT
                bg = colors.HexColor("#fff6e5") if b[1] == "warn" else colors.HexColor("#eef4fd")
                t = Table([[Paragraph(_md_inline(b[2]), body)]], colWidths=[avail])
                t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, col),
                                       ("LEFTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 5),
                                       ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
                story += [t, Spacer(1, 7)]
            elif k == "ul":
                story.append(ListFlowable([ListItem(Paragraph(_md_inline(x), body), leftIndent=11) for x in b[1]],
                                          bulletType="bullet", start="•", leftIndent=12, bulletFontName="DejaVu",
                                          bulletColor=ACCENT))
                story.append(Spacer(1, 3))
            elif k == "kv":
                data = [[Paragraph(f'<font name="DejaVu-Bold" color="#16324f">{_md_inline(a)}</font>', body),
                         Paragraph(_md_inline(v), body)] for a, v in b[1]]
                t = Table(data, colWidths=[avail * 0.27, avail * 0.73])
                t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("LINEBELOW", (0, 0), (-1, -2), 0.3, LINE),
                                       ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
                story += [t, Spacer(1, 8)]
            elif k == "kpi":
                cards = b[1]
                per = 3
                cw = avail / per
                cells = []
                for a, v, s_ in cards:
                    cells.append([Paragraph(f'<font size="7.5" color="#6b7685">{_md_inline(a).upper()}</font>', body),
                                  Paragraph(f'<font name="DejaVu-Bold" size="14" color="#16324f">{_md_inline(v)}</font>',
                                            ParagraphStyle("kv", parent=body, leading=18, spaceAfter=1)),
                                  Paragraph(_md_inline(s_), small)])
                while len(cells) % per:
                    cells.append("")
                grid = [cells[i:i + per] for i in range(0, len(cells), per)]
                t = Table(grid, colWidths=[cw] * per)
                st_ = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 9),
                       ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 6),
                       ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]
                for i in range(len(cards)):
                    r, c = divmod(i, per)
                    st_ += [("BACKGROUND", (c, r), (c, r), ZEBRA), ("LINEBEFORE", (c, r), (c, r), 3, ACCENT),
                            ("LINEAFTER", (c, r), (c, r), 4, colors.white),
                            ("LINEBELOW", (c, r), (c, r), 4, colors.white)]
                t.setStyle(TableStyle(st_))
                story += [t, Spacer(1, 8)]
            elif k == "table":
                _, header, rows, align, caption, n = b
                rp = [[plain(c) for c in header]] + [[plain(c) for c in r] for r in rows]
                widths, fs = colwidths(rp)
                cst = ParagraphStyle("c", parent=body, fontSize=fs, leading=fs * 1.25, spaceAfter=0)
                hst = ParagraphStyle("hc", parent=cst, fontName="DejaVu-Bold", textColor=colors.white)
                amap = {"l": TA_LEFT, "r": 2, "c": TA_CENTER}

                def cellp(txt, j, style):
                    return Paragraph(_md_inline(txt), ParagraphStyle(f"a{j}{style.name}", parent=style,
                                                                     alignment=amap.get(align[j], TA_LEFT)))
                data = [[cellp(c, j, hst) for j, c in enumerate(header)]] + \
                       [[cellp(str(c), j, cst) for j, c in enumerate(r)] for r in rows]
                t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
                t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY),
                                       ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ZEBRA]),
                                       ("LINEBELOW", (0, 0), (-1, -1), 0.25, LINE),
                                       ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                       ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
                                       ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2)]))
                items = []
                if caption:
                    items.append(Paragraph(f'<font name="DejaVu-Bold" color="#16324f">Table {n}.</font> '
                                           f"{_md_inline(caption)}", ParagraphStyle("tc", parent=cap,
                                                                                    fontName="DejaVu", spaceAfter=3)))
                items.append(t)
                story += [KeepTogether(items) if len(rows) < 25 else items[0], *([] if len(rows) < 25 else [t]),
                          Spacer(1, 9)]
            elif k == "img":
                _, path, caption, n = b
                iw, ih = ImageReader(str(path)).getSize()
                w_ = avail
                hh = w_ * ih / iw
                if hh > 0.58 * page_h:
                    hh = 0.58 * page_h
                    w_ = hh * iw / ih
                img = Image(str(path), width=w_, height=hh)
                frame = Table([[img]], colWidths=[avail])
                frame.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("BOX", (0, 0), (-1, -1), 0.4, LINE),
                                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
                story.append(KeepTogether([frame, Paragraph(
                    f'<font name="DejaVu-Bold" color="#16324f">Figure {n}.</font> {_md_inline(caption)}', cap)]))
            elif k == "pb":
                story.append(PageBreak())

        class Numbered(rl_canvas.Canvas):  # "page X of Y"
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self._pages = []

            def showPage(self):
                self._pages.append(dict(self.__dict__))
                self._startPage()

            def save(self):
                n = len(self._pages)
                for st in self._pages:
                    self.__dict__.update(st)
                    self.setFont("DejaVu", 7)
                    self.setFillColor(MUTED)
                    if self._pageNumber > 1:
                        self.drawString(margin, page_h - 10 * mm, header_text)
                        self.setStrokeColor(LINE)
                        self.setLineWidth(0.4)
                        self.line(margin, page_h - 11.5 * mm, page_w - margin, page_h - 11.5 * mm)
                    self.drawString(margin, 9 * mm, "Generated by croco_bench.py")
                    self.drawRightString(page_w - margin, 9 * mm, f"Page {self._pageNumber} of {n}")
                    super().showPage()
                super().save()

        tmp = out_pdf.with_name(out_pdf.stem + ".tmp.pdf")
        doc = SimpleDocTemplate(str(tmp), pagesize=A4, leftMargin=margin, rightMargin=margin,
                                topMargin=16 * mm, bottomMargin=16 * mm, title="CROCO Benchmark Report",
                                author="croco_bench.py")
        doc.build(story, canvasmaker=Numbered)
        os.replace(tmp, out_pdf)


def write_report(results, cfg, meta, out: Path, partial=False):
    agg = aggregate(results)
    if not agg:
        if not partial:
            print("[report] no results yet")
        return None
    rows, last_ok, first_fail = stability_summary(agg)
    ok = [a for a in agg if a["status"] == "OK"]
    dt_opt = None
    if ok:  # dt at which most layouts were compared (stage 2); ties -> larger dt
        cnt = {}
        for a in ok:
            cnt[a["dt"]] = cnt.get(a["dt"], 0) + 1
        dt_opt = max(cnt, key=lambda d: (cnt[d], d))
    elif last_ok:
        dt_opt = last_ok["dt"]
    at_opt = [a for a in ok if a["dt"] == dt_opt]
    best = min(at_opt, key=lambda a: a["wall_s"]) if at_opt else None
    make_figures(agg, out, dt_opt)
    sch = make_schedule_figure(results, out)
    sim_days = cfg["sim_days"]
    mach = meta["machine"]
    gi = meta.get("grid_info") or {}
    thr = float(cfg["efficiency_threshold"])
    dt_rec = meta.get("dt_opt")
    rec_row = next((r for r in rows if dt_rec is not None and r["dt"] == dt_rec), None)
    study = efficiency_study(agg, dt_opt, thr) if (cfg["efficiency_study"] and dt_opt is not None) else {}
    effig = make_efficiency_figure(study, out, dt_opt, thr) if study else None
    case = f"{meta['config_name']} › {meta.get('case_name') or '?'}"
    grid_txt = f"{gi['LLm']} × {gi['MMm']} × {gi['N']}" if gi else (
        f"{meta['grid_size'][0]} × {meta['grid_size'][1]}" if meta.get("grid_size") else "unknown")

    D = ReportDoc()
    D.title("CROCO Benchmark Report",
            f"{case}  ·  grid {grid_txt}  ·  {_days(sim_days)} per run  ·  {_dt.datetime.now():%Y-%m-%d %H:%M}")
    if partial:
        D.note(f"**Partial report**, updated {_dt.datetime.now():%Y-%m-%d %H:%M:%S} with {len(results)} finished "
               "run(s); the benchmark is still running.", "warn")
    D.kv([
        ("Configuration", f"`cppdefs.h` block `{meta['config_name']}`, case `{meta.get('case_name') or '?'}`"),
        ("Grid (param.h)", (f"LLm × MMm × N = **{grid_txt}** (active branch `{' › '.join(gi['chain'])}`, line "
                            f"{gi['line']})") if gi else f"{grid_txt} (active LLm0/MMm0 line not determined)"),
        ("Machine", f"`{mach.get('hostname')}`: {mach.get('cpu')}; {mach.get('physical_cores', '?')} physical / "
                    f"{mach.get('logical_cpus')} logical CPUs"),
        ("Cores used", f"{mach.get('cores_used', '?')} (CPU {_cpu_ranges(mach.get('cpu_slots') or [])}), up to "
                       f"{mach.get('max_parallel_jobs', '?')} simultaneous runs"),
        ("Run directory", f"`{cfg['run_dir']}`"),
        *([("ONLINE forcing", "`ONLINE` defined in `cppdefs.h`: the `online:` section of `croco.in` is used unchanged "
                              "(its data path is linked into every run directory)")] if meta.get("online") else []),
        ("Simulated period", f"{_days(sim_days)} per run; original croco.in dt = {_fmt_dt(meta['dt0'])} s, "
                             f"NDTFAST = {meta['ndtfast0']} (`{cfg['ndtfast_mode']}`)"),
    ])
    if gi and gi.get("case") == "#else":
        D.note(f"Case `{meta.get('case_name')}` has no entry of its own in `param.h`: the `#else` branch "
               f"({grid_txt}) is compiled. Check that these dimensions match your grid.", "warn")

    # ---------------------------------------------------------------- 1. key results
    D.h(2, "Key results")
    cards = []
    if rec_row:
        cards.append(("Recommended dt", f"{_fmt_dt(rec_row['dt'])} s",
                      f"3600 / {3600 / rec_row['dt']:g} steps per hour; largest stable "
                      f"{_fmt_dt(last_ok['dt']) if last_ok else '?'} s"))
    elif last_ok:
        cards.append(("Optimal dt", f"{_fmt_dt(last_ok['dt'])} s", "largest stable dt"))
    for mode, st in study.items():
        r = st["recommended"]
        cards.append((f"Recommended cores · {mode.upper()}", f"{r['cores']} cores" if r else "none",
                      f"layout {r['layout']}, E = {r['efficiency']:.3f}" if r else f"no core count with E > {thr:g}"))
    if best:
        cards.append(("Fastest run", f"{best['mode'].upper()} {best['np']} cores",
                      f"{best['nx']} × {best['ny']}, {_tfmt(best['wall_s'])} per {_days(sim_days)}"))
    crow = rec_row or last_ok
    if crow and crow.get("cfl_bt") is not None:
        cards.append(("Barotropic CFL at recommended dt", f"{crow['cfl_bt']:.3f}",
                      f"advective {_f(crow.get('cfl_adv'))}" + (f"; first BLOW UP at {_fmt_dt(first_fail['dt'])} s"
                                                                if first_fail else "")))
    ser = next((a for a in agg if a["mode"] == "serial" and a["dt"] == dt_opt and a["wall_s"]), None)
    if ser:
        cards.append(("Serial reference T(1)", _tfmt(ser["wall_s"]), f"1 core, no OpenMP/MPI, dt = {_fmt_dt(dt_opt)} s"))
    D.kpis(cards)

    items = []
    if rec_row:
        items.append(f"**Recommended time step: {_fmt_dt(rec_row['dt'])} s** = 3600 / {3600 / rec_row['dt']:g}, the "
                     "largest integer dt that divides 3600 without exceeding the largest stable dt. All times and "
                     "efficiencies in this report were measured with it.")
    if last_ok:
        items.append(f"**Largest stable dt found:** {_fmt_dt(last_ok['dt'])} s (NDTFAST = {last_ok['ndtfast']}, "
                     f"barotropic dt = {_f(last_ok['dtfast'], '{:.2f}')} s).")
    if first_fail and last_ok:
        items.append(f"**Stability limit** between {_fmt_dt(last_ok['dt'])} s (stable) and "
                     f"{_fmt_dt(first_fail['dt'])} s (first BLOW UP).")
    elif first_fail:
        items.append(f"**No tested dt was stable:** even the smallest, {_fmt_dt(min(r['dt'] for r in rows))} s, failed. "
                     "Try a smaller `dt_ref`; if it still fails, check the run log: the problem may not be the time "
                     "step (initial conditions, forcing, viscosity, NDTFAST).")
    else:
        items.append("No tested dt blew up: the stability limit is above the largest dt tested (the search stopped "
                     "at `min_steps` or `max_dt_runs`).")
    for mode, st in study.items():
        r = st["recommended"]
        items.append(f"**{mode.upper()}:** " + (f"up to **{r['cores']} cores** (layout {r['layout']}, efficiency "
                                                 f"{r['efficiency']:.3f} > {thr:g})." if r else
                                                 f"no core count keeps the efficiency above {thr:g}."))
    if best:
        items.append(f"**Fastest configuration:** {best['mode'].upper()} with {best['np']} cores, layout "
                     f"{best['nx']} × {best['ny']} ({'NP_XI × NP_ETA' if best['mode'] == 'mpi' else 'NSUB_X × NSUB_E'}), "
                     f"{_tfmt(best['wall_s'])} per {_days(sim_days)} ({_f(best['days_per_hour'], '{:.1f}')} simulated "
                     "days per wall-clock hour), " + ("measured alone on the machine." if best["clean"] else
                                                       f"measured while sharing the machine with {best['peers']} run(s)."))
    D.ul(items)

    # ---------------------------------------------------------------- 2. time step
    D.h(2, "Time-step search (serial run)")
    D.p("All runs of this stage use the serial build (1 core, `OPENMP` and `MPI` undefined). Starting from the "
        f"reference dt ({_fmt_dt(float(cfg['dt_ref'] or meta['dt0']))} s), dt is doubled while the run is stable; "
        "after the first BLOW UP the next dt is the midpoint between the last stable dt and the last BLOW UP, and the "
        "search stops after two BLOW UPs in a row. Rules: a dt larger than one that blew up is never tested, and when "
        "going down the search never goes below a stable dt whose half is also stable.")
    path = meta.get("dt_path") or []
    if path:
        lab = {"OK": "stable", "BLOWUP": "BLOW UP", "OK (inferred)": "stable (inferred)"}
        D.table(["Step", "Phase", "dt [s]", "Result"],
                [[i + 1, r["phase"], _fmt_dt(r["dt"]), lab.get(r["status"], r["status"])] for i, r in enumerate(path)],
                ["r", "l", "r", "l"], "Decision sequence of the time-step search.")
    D.table(["dt [s]", "NDTFAST", "dt_fast [s]", "Status", "CFL barotropic", "CFL internal", "CFL advective",
             "Best time"],
            [[_fmt_dt(r["dt"]) + (" ★" if rec_row and r is rec_row else ""), _f(r["ndtfast"], "{}"),
              _f(r["dtfast"], "{:.2f}"), r["status"], _f(r["cfl_bt"]), _f(r["cfl_int"]), _f(r["cfl_adv"]),
              _tfmt(r["best_wall"])] for r in rows],
            ["r", "r", "r", "l", "r", "r", "r", "r"],
            "Stability and Courant numbers for every dt tested (★ = recommended dt). Internal-wave CFL estimated "
            f"with an assumed c₁ = {cfg['internal_wave_speed']} m/s.")
    D.image(out / "cfl_vs_dt.png", "Courant numbers versus time step; the dashed lines mark the largest stable dt "
                                   "and the first BLOW UP.")

    # ---------------------------------------------------------------- 3. cores
    D.h(2, "Core-count exploration")
    D.p(f"OpenMP and MPI start at {meta.get('ref_cores')} cores. For each core count every valid nx × ny "
        "decomposition is run and the fastest is kept. With E(n) = T(1) / (n · T(n)): if E > "
        f"{thr:g} the next test uses {cfg['core_step']} more cores, otherwise {cfg['core_step']} fewer; each mode "
        "stops when the direction reverses or the core limit is reached.")
    for mode, e in (meta.get("explore_path") or {}).items():
        t = e["tested"]
        if not t:
            continue
        D.table(["Cores", "Best layout", "Time", "Efficiency E", "Decision"],
                [[n, v["layout"], _tfmt(v["wall"]), f"{v['E']:.3f}",
                  f"E > {thr:g} → +{cfg['core_step']}" if v["E"] > thr else f"E ≤ {thr:g} → −{cfg['core_step']}"]
                 for n, v in t.items()],
                ["r", "c", "r", "r", "l"], f"{mode.upper()} exploration path (stop: {e['why']}).")
    D.image(out / "benchmark_summary.png", f"(a) run time versus dt, (b) scaling and (c) parallel efficiency at dt = "
                                           f"{_fmt_dt(dt_opt) if dt_opt else '?'} s, (d) every layout tested.")

    # ---------------------------------------------------------------- 4. efficiency study
    if study:
        D.h(2, "Efficiency study (NLHPC format)")
        D.p(f"Format of the [NLHPC efficiency study](https://wiki.nlhpc.cl/Estudio_de_Eficiencia), at dt = "
            f"{_fmt_dt(dt_opt)} s and {_days(sim_days)} simulated, with the best layout of each core count: "
            "**speed-up** S(n) = T(1) / T(n), **efficiency** E(n) = S(n) / n. T(1) is the serial run (1 core, built "
            f"without OpenMP or MPI). Recommended: the largest core count with E > {thr:g}.")
        with open(out / "efficiency_study.csv", "w", newline="") as fcsv:
            cw = csv.writer(fcsv)
            cw.writerow(["mode", "cores", "layout", "time_s", "speedup", "efficiency", "measured_alone"])
            for mode, st in study.items():
                rr = st["rows"]
                for r in rr:
                    cw.writerow([mode, r["cores"], r["layout"], f"{r['time_s']:.2f}", f"{r['speedup']:.2f}",
                                 f"{r['efficiency']:.2f}", r["alone"]])
                D.table(["No. of cores"] + [str(r["cores"]) for r in rr],
                        [["Layout"] + [r["layout"] for r in rr],
                         ["Time [s]"] + [f"{r['time_s']:.2f}" for r in rr],
                         ["Speed-up"] + [f"{r['speedup']:.2f}" for r in rr],
                         ["Efficiency"] + [f"**{r['efficiency']:.2f}**" if r["ok"] else f"{r['efficiency']:.2f}"
                                           for r in rr],
                         ["Measured alone"] + ["yes" if r["alone"] else "**no**" for r in rr]],
                        ["l"] + ["r"] * len(rr),
                        f"{mode.upper()}: " + (f"recommended **{st['recommended']['cores']} cores**."
                                               if st["recommended"] else f"no core count with E > {thr:g}."))
                if st["baseline_cores"] > 1:
                    D.note(f"No 1-core run: T(1) estimated as {st['baseline_cores']}·T({st['baseline_cores']}), "
                           "assuming ideal scaling up to that core count.")
        D.image(effig, "Speed-up (left, with the ideal line) and parallel efficiency (right, with the threshold); "
                       "stars mark the recommended core count of each mode.")
        if any(not r["alone"] for st in study.values() for r in st["rows"]):
            D.note("Some points were measured while sharing the machine with other runs (row “Measured alone” = no). "
                   "For a formal efficiency study use `efficiency_exclusive: true`.", "warn")

    # ---------------------------------------------------------------- 5. all layouts
    if dt_opt is not None:
        at = sorted([a for a in agg if a["dt"] == dt_opt], key=lambda a: (a["wall_s"] is None, a["wall_s"] or 0))
        refr = [a for a in at if a["wall_s"]]
        nmin = min([a["np"] for a in refr], default=1)
        tref = min([a["wall_s"] * a["np"] / nmin for a in refr if a["np"] == nmin], default=None)
        D.h(2, f"All layouts at dt = {_fmt_dt(dt_opt)} s")
        trs = []
        for a in at:
            sp = tref / a["wall_s"] if a["wall_s"] and tref else None
            eff = sp / (a["np"] / nmin) * 100 if sp else None
            trs.append([a["mode"].upper(), a["np"], f"{a['nx']} × {a['ny']}", _tfmt(a["wall_s"]),
                        _f(a["days_per_hour"], "{:.1f}"), _f(sp, "{:.2f}"), _f(eff, "{:.0f} %"), _measured(a),
                        a["status"]])
        D.table(["Mode", "Cores", "Layout", "Time", "Days/hour", "Speed-up", "Efficiency", "Measured", "Status"],
                trs, ["l", "r", "c", "r", "r", "r", "r", "l", "l"],
                f"Every layout at dt = {_fmt_dt(dt_opt)} s, fastest first; speed-up and efficiency relative to the "
                f"fastest run with {nmin} core(s).")
        both = [a for a in agg if a["wall_clean"] and a["wall_conc"]]
        if both:
            D.table(["Layout", "dt [s]", "Time alone", "Time sharing", "Ratio"],
                    [[a["tag"], _fmt_dt(a["dt"]), _tfmt(a["wall_clean"]), _tfmt(a["wall_conc"]),
                      f"{a['wall_conc'] / a['wall_clean']:.2f}×"] for a in sorted(both, key=lambda a: a["wall_clean"])],
                    ["l", "r", "r", "r", "r"],
                    "Effect of sharing the machine (ratio > 1: slower when other runs compete for memory bandwidth "
                    "and cache).")

    # ---------------------------------------------------------------- 6. schedule + appendix
    D.h(2, "Run schedule")
    D.image(sch, "Which run used which cores and when (red edge = BLOW UP).")
    D.h(2, "Methodology")
    D.ul([
        "Every run starts from the original `cppdefs.h`, `param.h` and `croco.in`. `cppdefs.h`: `OPENMP` or `MPI` "
        "switched inside the active configuration block (both undefined for the serial build); `param.h`: "
        "`NP_XI`/`NP_ETA` (MPI) or `NPP`/`NSUB_X`/`NSUB_E` (OpenMP); `croco.in`: `dt` and "
        f"`NTIMES = ceil({sim_days:g}·86400/dt)`. Output periods are rescaled to stay constant in seconds.",
        "Case and grid are not assumed: the preprocessor directives of `cppdefs.h` are evaluated and then used to find "
        "the `parameter (LLm0=…, MMm0=…, N=…)` line of `param.h` that is actually compiled.",
        "One executable per layout, compiled before the runs that use it, several at a time in sibling directories "
        "of `run_dir`; changing dt does not recompile. Every run works in its own directory (`work/<run>`) with its own "
        "`croco.in`, inputs linked read-only, and is pinned to its own block of cores (`taskset`, `OMP_PLACES`).",
        "Measured time: wall-clock time from launch to process end. BLOW UP is detected in the log (`BLOW UP` or "
        "`NaN`); the run is killed after a grace period and runs with a larger dt are cancelled.",
        "CFL barotropic = dt_fast·√(g·h)·√(1/dx²+1/dy²) (maximum over wet points, dt_fast = dt/NDTFAST); CFL internal "
        "= dt·c₁·√(1/dx²+1/dy²) with an assumed c₁; CFL advective = dt·max(|u|/dx + |v|/dy) from the history file.",
        "Stability over the simulated period does not guarantee stability under stronger forcing: for production, "
        "keep a margin (≈ 10–20 % below the largest stable dt).",
    ])
    D.h(2, "Appendix: all runs")
    D.table(["Stage", "Mode", "Cores", "Layout", "dt", "Rep", "CPUs", "Concurrent", "Status", "Time [s]",
             "Sim. days", "Notes"],
            [[r["stage"], r["mode"], r["np"], f"{r['nx']}×{r['ny']}", _fmt_dt(r["dt"]), r["rep"], r.get("cpus") or "",
              _f(r.get("peers"), "{}"), r["status"], _f(r["wall_s"], "{:.1f}"), _f(r["sim_days"], "{:.3f}"),
              _short_notes(r.get("notes"))] for r in results],
            ["l", "l", "r", "c", "r", "r", "l", "r", "l", "r", "r", "l"], "Every run of the benchmark, in order.")

    rep = out / "benchmark_report.md"
    _write_atomic(rep, D.markdown())
    _write_atomic(out / "benchmark_summary.json", json.dumps(
        {"best": best, "largest_stable_dt": last_ok, "first_blowup": first_fail, "recommended_dt": dt_rec,
         "stability": rows, "meta": meta}, indent=2, default=str))
    if partial:
        return rep
    print(f"[report] {rep}")
    if cfg["latex_report"]:
        tex = out / "benchmark_report.tex"
        _write_atomic(tex, D.latex())
        print(f"[report] {tex}  (compile in {out} with: pdflatex benchmark_report.tex, twice)")
    if cfg["pdf_report"]:
        pdf, how = report_to_pdf(rep, D, f"CROCO Benchmark Report  ·  {case}  ·  grid {grid_txt}")
        if pdf:
            print(f"[report] {pdf}  (via {how})")
    return rep


# ----------------------------------------------------------------------------------------------
# Markdown report -> PDF
# ----------------------------------------------------------------------------------------------
def _md_inline(text: str) -> str:
    """Markdown inline markup -> reportlab paragraph markup."""
    t = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#1c5fb8">\1</link>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"`([^`]+)`", r'<font face="DejaVuMono">\1</font>', t)
    return t


def _pdf_matplotlib(md: str, base: Path, pdf: Path):
    """Last resort: plain text pages + one page per figure (needs only matplotlib)."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    text_lines, images = [], []
    for ln in md.splitlines():
        m = re.match(r"^\s*!\[[^\]]*\]\(([^)]+)\)", ln)
        if m:
            images.append(base / m.group(1))
            continue
        clean = re.sub(r"\*\*|`", "", ln)
        clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)
        text_lines += textwrap.wrap(clean, 110) or [""]
    tmp = pdf.with_name(pdf.stem + ".tmp.pdf")
    with PdfPages(tmp) as pp:
        per = 70
        for k in range(0, len(text_lines), per):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.05, 0.97, "\n".join(text_lines[k:k + per]), va="top", ha="left", family="DejaVu Sans Mono",
                     fontsize=7)
            pp.savefig(fig)
            plt.close(fig)
        for p in images:
            if p.exists():
                img = plt.imread(str(p))
                fig = plt.figure(figsize=(11.69, 8.27))
                ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
                ax.imshow(img)
                ax.axis("off")
                pp.savefig(fig)
                plt.close(fig)
    os.replace(tmp, pdf)


def report_to_pdf(md_path: Path, doc=None, header_text=""):
    """benchmark_report.md -> benchmark_report.pdf (reportlab from the structured report; else pandoc;
    else matplotlib)."""
    pdf = md_path.with_suffix(".pdf")
    md = md_path.read_text()
    errors = []
    try:
        if doc is None:
            raise ImportError("structured report not available")
        doc.pdf(pdf, header_text)
        return pdf, "reportlab"
    except ImportError as e:
        errors.append(f"reportlab not available ({e})")
    except Exception as e:  # noqa: BLE001
        errors.append(f"reportlab failed ({e})")
    if shutil.which("pandoc"):
        for eng in ("xelatex", "wkhtmltopdf", "weasyprint", "pdflatex"):
            if shutil.which(eng):
                r = subprocess.run(["pandoc", md_path.name, "-o", pdf.name, f"--pdf-engine={eng}",
                                    "-V", "geometry:margin=1.5cm"], cwd=md_path.parent,
                                   capture_output=True, text=True)
                if r.returncode == 0 and pdf.exists():
                    return pdf, f"pandoc ({eng})"
                errors.append(f"pandoc/{eng} failed")
    try:
        _pdf_matplotlib(md, md_path.parent, pdf)
        return pdf, "matplotlib (basic layout; install reportlab for a formatted PDF)"
    except Exception as e:  # noqa: BLE001
        errors.append(f"matplotlib failed ({e})")
    print("[warn] PDF not generated: " + "; ".join(errors))
    return None, None


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", type=Path, help="JSON configuration file")
    ap.add_argument("--init-config", type=Path, metavar="FILE", help="write an example configuration and exit")
    ap.add_argument("--dry-run", action="store_true", help="show the plan and the file edits, run nothing")
    ap.add_argument("--report-only", action="store_true", help="only rebuild report and figures from results.csv")
    ap.add_argument("--rerun", action="store_true", help="ignore previous results.csv (it is kept as .bak)")
    ap.add_argument("--max-cores", type=int, metavar="N",
                    help="maximum number of cores to use (overrides max_cores in the config)")
    args = ap.parse_args()

    if args.init_config:
        ex = {"_comment": EXAMPLE_COMMENT, **json.loads(json.dumps(DEFAULT_CONFIG))}
        args.init_config.write_text(json.dumps(ex, indent=2) + "\n")
        print(f"[info] example configuration written to {args.init_config}")
        return

    cfg = load_config(args.config)
    if args.max_cores is not None:
        if args.max_cores < 1:
            sys.exit("[error] --max-cores must be >= 1")
        cfg["max_cores"] = args.max_cores
    bench = Bench(cfg, rerun=args.rerun, dry_run=args.dry_run)
    if args.dry_run:
        bench.dry_run()
        return
    out = Path(cfg["output_dir"])
    (out / "config_used.json").write_text(json.dumps(cfg, indent=2))
    if args.report_only:
        write_report(bench.results, cfg, bench.meta, out)
        return

    def _term(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    t0 = time.time()
    if cfg["live_report"]:
        bench.live = LiveReport(bench).start()
        print(f"[info] figures and report in {out} are updated after every simulation")
    try:
        bench.go()
    except KeyboardInterrupt:
        print("\n[info] interrupted: partial results kept; run again to resume")
    except WallTimeUp as e:
        print(f"\n[info] {e}: stopping before the Slurm time limit. Results are kept; "
              "submit the same job again to resume.")
    finally:
        bench.cancel_all()
        bench.restore()
        if bench.live:
            bench.live.stop()
            bench.live = None
        write_report(bench.results, cfg, bench.meta, out)
        print(f"[info] total benchmark time {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
