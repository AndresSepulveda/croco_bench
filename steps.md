# croco_bench — prompts, sources and step-by-step development

This document records how `croco_bench.py` was produced: the prompts that defined it, the sources used, and the
sequence of design and testing steps. The tool was written by an AI assistant (see [How to cite](#how-to-cite)) in
a conversation with the user between 22 and 24 October 2026.

## 1. Prompts

The work was driven by the user's messages below, quoted verbatim in the order they were sent (the first ones in
Spanish, later ones in English).

1. **Initial request (22 Oct 2026):**
   > Trabajaras en un ambiente linux escribiendo un script para encontrar la configuración mas eficiente para una
   > simulacion con el modelo de oceano CROCO. Para esto tienes que medir el tiempo que demora una simulación al
   > cambiar el metodo de paralelizacion (openmp, o mpi) en el cppdefs.h, el numero de cores y su distribucion en x e
   > y en el param.h y tambien el paso del tiempo en el croco.in. Eso significa poder cambiar esos archivos,
   > recompilar el ejecutable, lanzar la simulación y medir el tiempo en que termina. Mide el tiempo que demora en
   > simular una semana. Con la información acumulada genera un grafico y un reporte. Al cambiar el paso del tiempo
   > es posible que el modelo no termine al generar un error de BLOW UP. Reporta la condición CFL optima.
2. > Mejora el script para que, considerando el número de cores de una maquina se determinen las posibles
   > combinaciones de paralelismo y se lancen las simulaciones que usan pocos cores de forma simultanea, y se vayan
   > lanzando las siguientes segun la disponibilidad de nucleos
3. > como calculas el numero de cores disponibles?
4. > prefiero definir de forma manual el numero maximo de cores a usar
5. > cual es el orden en que son lanzadas las simulaciones_
6. > rellena de forma que las simulaciones mas chicas de otras etapas tambien entren
7. > Al final de los calculos genera tambien una tabla y un gr[afico como el descrito en
   > https://wiki.nlhpc.cl/Estudio_de_Eficiencia
8. > si el dt minimo falla, sigue disminuyendo este en mitades
9. > cambia la estrategia de lanzamiento de simulaciones. Comienza con la simulacion que usa el numero maximo de
   > cores y el dt mas alto y disminuye progresivamente. Despues de que termine cada simulacion, actualiza los
   > graficos con los resultados
10. > que es omp_tile_factors
11. > change the way you consider the changes in time step. As for a reference time step. if the run is stable,
    > double it. If the run is unstable, decrease it by half. Do this processs a maximum number of times, set it at
    > 4. Also, always start a run with 1 core (no openmp or mpi parallelization) and use it as a reference for the
    > eficiency calcularions
12. > Set as a default a run of 1 day not one week
13. > So. First to the serial run and explore the best time step. Use the reference time step. If it runs ok, double
    > it. stop when the model blows up two times in a row after decreasing the time step in half. Since these runs use
    > a single core, run several of them at the same time. Once you have defined the time step explore de
    > paralelization. Use also a reference number of cores, start with 12 (try both openmp and mpi). Calculate the
    > eficiency considering the run with 1 core. If the efficiency is above 0.5, increase the number of cores to try by
    > 2. If it is below 0.5, decrease the number of cores by 2.
14. > if a time step is stable and the value half of it is stable, do not decrease any further. If a time step is not
    > stable, do not increase any further
15. > el bloque de configuracion aparece reportado en el archivo benchmark_report.md como "REGIONAL", pero este bloque
    > tiene varias opciones, y hay que reportar las dimensiones, especificadas en el param.h para la opcion
    > seleccionada despues del IFDEFINE REGIONAL. El caso por defecto es BENGUELA_LR con grilla LLmxMMm = (41x42) pero
    > eso no debe ser reportado por defecto sino debe ser verificado segun el cppdefs.h y param.h que sea leido.
    > Ademas, al final, convierta el archivo benchmark_report.md a formato pdd

    followed by the correction “formato pdf.”
16. > when reporting the optima time step round this number so when dividing the number of secons in an hour, 3600,
    > the result is an integer
17. > what is omp_tile_factors?
18. > Escribe todos los textos en ingles. Mejora la presentacion del PDF. Escribe un archivo llamado steps.md donde
    > describas el prompt, las fuentes, y el paso a paso para la solucion que entregas y citate.

19. > Adapt the so it can be also used in a HPC environment using a script similar to the one attached

    The attached file was an NLHPC Slurm job script: `#SBATCH -J CROCO`, `-p general`, `-n 4`,
    `--ntasks-per-node=4`, mail and output options, `ml purge`, `ml intel/2022.00`, `ml croco/2.0.0`,
    `srun ./croco croco.in`.
20. > create also a LaTeX file with the report information

21. > if ONLINE is activated in cppdefs.h don't change the online section in the croco.in

Messages 3, 5, 10 and 17 were questions; they were answered in the conversation and led to no code change, except
message 5, whose answer led to request 6.

## 2. Sources

### Consulted during the work
- **NLHPC — Estudio de Eficiencia.** Laboratorio Nacional de Computación de Alto Rendimiento (NLHPC), Chile.
  <https://wiki.nlhpc.cl/Estudio_de_Eficiencia> (accessed 23 Oct 2026). It defines speed-up S(n) = T(1)/T(n),
  efficiency E(n) = S(n)/n, a table with rows *No. of cores / Time / Speed-up / Efficiency*, and the acceptance
  threshold E > 0.5. It is the basis of the efficiency study section and of the 0.5 threshold of the core
  exploration.

### Background knowledge (not fetched during the session)
- **CROCO** (Coastal and Regional Ocean COmmunity model), <https://www.croco-ocean.org>. The structure of the files
  the script edits came from prior knowledge of standard CROCO distributions:
  - `cppdefs.h`: configuration blocks such as `REGIONAL` › `BENGUELA_LR`, with `OPENMP`/`MPI` switches;
  - `param.h`: `LLm0/MMm0/N`, and `NP_XI/NP_ETA/NPP/NSUB_X/NSUB_E` in the `#ifdef MPI … #elif defined OPENMP … #else`
    block;
  - `croco.in`: keywords `time_stepping:`, `history:`, `averages:`, `restart:`;
  - `jobcomp`, and the `BLOW UP` / `MAIN: DONE` log messages.

  The script was not run against a real CROCO installation in this session (see step 3).
- **Split-explicit time stepping and Courant limits** in ROMS/CROCO: Shchepetkin, A. F. and McWilliams, J. C. (2005).
  The regional oceanic modeling system (ROMS): a split-explicit, free-surface, topography-following-coordinate oceanic
  model. *Ocean Modelling*, 9(4), 347–404. It is the background for the barotropic Courant number
  dt_fast·√(gh)·√(1/dx²+1/dy²), with dt_fast = dt/NDTFAST.
- **Backfilling** (starting smaller jobs on idle cores only if they do not delay the job at the head of the queue),
  the idea behind EASY backfilling: Lifka, D. A. (1995). The ANL/IBM SP scheduling system. In *Job Scheduling
  Strategies for Parallel Processing*, LNCS 949, Springer, 295–303.
- **Software:**
  - Python standard library;
  - NumPy and Matplotlib (figures, and the DejaVu fonts it bundles, used in the PDF);
  - netCDF4 or SciPy (reading grid and history files);
  - ReportLab (PDF);
  - Linux `taskset` and `/sys/devices/system/cpu` topology files.

## 3. Step-by-step development

### Step 1 — Environment check
The assistant's cloud workspace had no CROCO source, Fortran compiler or MPI, and only 2 cores. The user's linked
computer was a Windows laptop without a CROCO build. The tool was therefore written to run on the user's Linux
modelling machine and tested here against a **mock CROCO**:
- realistic `cppdefs.h`, `param.h`, `croco.in` and a netCDF grid;
- a fake `jobcomp` that "compiles" by reading the parallel settings;
- a fake `croco` executable that reads `croco.in`, sleeps according to a cost model (parallel efficiency, halo
  overhead, memory contention between simultaneous runs), writes a history file, and blows up when the barotropic
  CFL exceeds 0.85 (true limit ≈ 1799 s).

### Step 2 — First version: edit, compile, run, time, report (prompt 1)
1. **Editing the three files, always from pristine copies:**
   - `cppdefs.h`: `OPENMP`/`MPI` switched only inside the active configuration block, located with a parser of
     `#if/#elif/#else/#endif` chains;
   - `param.h`: `NP_XI/NP_ETA` or `NPP/NSUB_X/NSUB_E` set in the right branch of the parallel block;
   - `croco.in`: `dt`, `NTIMES` and output periods rescaled so they stay constant in seconds.
2. **One executable per parallel layout,** cached; runs launched with process-group control and BLOW UP detection
   in the log (`BLOW UP`/`NaN`); classification as OK / BLOWUP / ERROR / TIMEOUT / COMPILE_FAIL.
3. **CFL numbers:** barotropic and internal-wave from the grid file; advective from the history file of each run.
4. **Outputs:** CSV of results (resumable), figures, Markdown report. The original files are always restored,
   including on Ctrl-C or SIGTERM.
5. **Tests:** dry-run diffs, staged and full strategies, resume, compile failure, crash, SIGTERM. A bug where the
   model process survived an interruption was found and fixed.

### Step 3 — Concurrency by available cores (prompts 2–6)
1. **CPU topology:** read from `/sys/devices/system/cpu`, one slot per physical core, socket by socket; manual
   `max_cores` / `--max-cores`, with hyperthreads only when explicitly requested.
2. **A per-run sandbox directory,** so simultaneous runs never share `croco.in` or output files: inputs are
   symlinked read-only, outputs stay local. Each run is pinned to its own block of cores (`taskset`,
   `OMP_PLACES`).
3. **Parallel compilation** in sibling directories of `run_dir`, so relative `SOURCE` paths in `jobcomp` keep working.
4. **A scheduler** with a core-based queue, pipelined stages, and EASY-style backfilling. Run durations are estimated
   from log progress (step / NTIMES) or from measured runs scaled by dt and cores.
5. **Contention handling:** runs measured while sharing the machine are flagged, and the best ones are re-timed alone.
6. **Bugs found and fixed:**
   - a premature "stability resolved" while a larger dt was still running;
   - a double-removal crash when two runs ended in the same pass.

### Step 4 — Efficiency study (prompt 7)
The NLHPC page was read, and the report gained a table per mode in its format (cores / layout / time / speed-up /
efficiency / measured alone), an efficiency CSV, and a two-panel figure (speed-up with the ideal line, efficiency
with the 0.5 threshold), plus a recommended core count. Every point of the study is re-timed alone so that T(1) is
not inflated.

### Step 5 — Time-step search and core exploration (prompts 8, 11–14, 16)
The time-step strategy was redefined several times. The final design:
1. **Serial build first** (1 core, no OpenMP/MPI). dt is doubled from `dt_ref` while the run is stable. After the
   first BLOW UP, the next dt is the midpoint between the last stable dt and the last BLOW UP, until two BLOW UPs in a
   row. If `dt_ref` blows up, dt is halved until a dt and its half are both stable.
2. **Rules enforced with live cancellation:** never test above a dt that blew up; never go below a stable dt whose
   half is stable. In the default *strict* mode, simultaneous serial runs are limited to runs that are needed whatever
   the pending results, which means at most 2 at a time. An *aggressive* look-ahead mode remains available.
3. **Optimal dt rounded down** to the largest integer that divides 3600. The serial T(1) reference and stage 2 run
   at that dt.
4. **Core exploration** for OpenMP and MPI from 12 cores. E(n) = T(1)/(n·T(n)) with the best decomposition at each
   core count; ±2 cores depending on E vs 0.5; stop when the direction reverses or at the core limit. Runs one at a
   time by default, so contention cannot flip a decision.
5. **Live report:** figures and report are rebuilt after every finished simulation, in a background thread with
   atomic file writes.
6. **Tests:** mock scenarios for every branch (doubling only, halving from an unstable reference, two BLOW UPs in a
   row, no stable dt, huge `dt_ref` capped by `min_steps`). Compliance with "never above a known BLOW UP" was checked
   from the run timestamps.

### Step 6 — Case and grid from the sources, PDF (prompt 15)
1. **A small C-preprocessor evaluator:**
   - it applies `#define/#undef` in the active branches of `cppdefs.h`;
   - it evaluates `param.h` with the resulting macros;
   - it reports the `LLm0/MMm0/N` line actually compiled and the option that selects it.

   Tested on CROCO-like files: BENGUELA_LR → 41×42×32, BENGUELA_HR → 83×85×32, an unknown case → the `#else`
   branch with a warning, BASIN → 60×50×10.
2. **PDF conversion,** with fallbacks to pandoc and to a basic Matplotlib PDF.

### Step 7 — English texts and a better PDF (prompt 18)
1. **English throughout:** every text of the report, PDF, README and code.
2. **One source for both formats:** the report is now built as a structured document (title, key-value table,
   key-result cards, numbered sections, captioned tables and figures, notes) and rendered both to Markdown and to
   PDF.
3. **PDF design:**
   - cover band and key-result cards;
   - numbered sections with rules;
   - tables with navy headers and zebra rows;
   - column widths measured with the real font, so words are never split;
   - framed figures with numbered captions;
   - running header and "Page X of Y".
4. **Search state** is saved to `search_state.json`, so `--report-only` rebuilds the full report.
5. **This `steps.md`.**

### Step 8 — HPC (Slurm / NLHPC) support (prompt 19)
1. **One allocation, many job steps.** The design runs the whole benchmark inside one Slurm allocation, with every
   simulation as an `srun --exact` job step. Submitting each run as its own `sbatch` job would put every run through
   the queue and make timings depend on which node each job lands on.
2. **Automatic detection:** a Slurm job is recognised from `SLURM_JOB_ID`, and switches the run templates (MPI
   `-n {np} -c 1`, OpenMP `-n 1 -c {np}` with `OMP_PLACES=cores`, serial `-n 1 -c 1`).
3. **Wall-time guard:** time left is read from `squeue -o %L`; no run starts if it cannot finish before the limit
   minus a margin; the benchmark stops cleanly with its report; resubmitting resumes.
4. **Job script:** `run_bench_nlhpc.bash` follows the structure of the attached script (same `#SBATCH` options and
   module lines, plus `-N 1`, `--exclusive`, `-t`), and `bench_hpc.json` is an example configuration for it.
5. **Tested with a fake `srun`** that runs the command and audits the CPUs used by simultaneous steps (maximum 16 of
   16 allocated, 0 violations), and a fake `squeue`. A job given 40 s of wall time stopped cleanly after one run, and
   the "resubmitted" job finished the benchmark without repeating any run.

### Step 9 — LaTeX report (prompt 20)
1. **A third renderer** of the structured report writes `benchmark_report.tex`: cover band, key-value table,
   key-result boxes, numbered sections, `longtable`/`booktabs` tables with captions, figures with captions, notes,
   and "Page X of Y".
2. **Character handling:** LaTeX special characters are escaped, and symbols are converted to commands
   (× → `\times`, → → `\rightarrow`, ≈ → `\approx`, √ → `\surd`, …).
3. **Portability:** optional packages (`lmodern`, `lastpage`, `caption`, `enumitem`) are loaded only if installed.
   The file compiles with `pdflatex` without errors.
4. **Bugs found while compiling and fixed:**
   - a symbol command upper-cased in the result cards;
   - the key-value block taking a table number.

### Step 10 — ONLINE forcing (prompt 21)
1. **Bug found in the review.** The per-run path rewriting treated the `online:` data path (`DATA/CFSR_Benguela_LR/`)
   as an input file and rewrote it as `inputs/CFSR_Benguela_LR`, without the trailing slash. CROCO concatenates path
   and file name, so it would not have found the forcing.
2. **Fix:**
   - `ONLINE` is detected with the preprocessor evaluator;
   - when it is defined, the `online:` block is copied unchanged;
   - its relative path is made reachable from each run directory by symlinking its first component to the original;
   - directory paths in any other section are also kept and linked.
3. **Tested** with copies of the mock case with `# define ONLINE` and `# undef ONLINE`, a real-style `online:`
   section and a data directory. The fake model now fails if it cannot find the data path:
   - `online:` block byte-identical in 100 % of the run directories;
   - every run found the data;
   - no errors, and `run_dir` untouched.

### Verification performed at each step
- Static checks with `pyflakes`.
- End-to-end runs against the mock on 2 real cores and on 8 and 16 virtual cores.
- Resume tests: no run repeated.
- Interruption tests: no orphan processes, `run_dir` byte-identical afterwards (checksums).
- Visual inspection of every figure and of the rendered PDF pages.

**Not done:** a run against a real CROCO build. The first real run should start with `--dry-run` to check the file
edits, the detected case and grid, and the input/output classification of `croco.in`.

## 4. Files delivered

| File | Content |
|---|---|
| `croco_bench.py` | The benchmark tool (single Python file). |
| `bench_example.json` | Example configuration (16-core workstation, `ref_cores` 12, `dt_ref` 300 s, 1 simulated day). |
| `run_bench_nlhpc.bash` | Slurm job script for NLHPC (one node; the benchmark runs every simulation as an `srun` step). |
| `bench_hpc.json` | Example configuration for the Slurm job. |
| `README.md` | User documentation. |
| `steps.md` | This document. |
| `example_output/` | Report (`.md`, `.pdf`, `.tex`), figures and efficiency CSV produced against the mock CROCO — format only, not real results. |

## How to cite

This tool and its documentation were generated with an AI assistant:

> Anthropic. (2026). *Claude* [Large language model]. Conversation of 22–24 October 2026 producing `croco_bench.py`,
> `run_bench_nlhpc.bash`, `README.md` and `steps.md`. Model identifier configured for the session: `claude-sonnet-5-5`; parts of the
> session may have been served by another Claude model (`claude-opus-5-5`). <https://claude.ai>

Suggested acknowledgement in a report or account request:

> The CROCO benchmark (time-step and core-count search, efficiency study in the NLHPC format) was carried out with
> `croco_bench.py`, a script generated with Claude (Anthropic, 2026) and reviewed by the user.

The assistant's code and text were produced from the prompts above and the sources listed; responsibility for
running the tool on a real model, checking its results and their use remains with the user.
