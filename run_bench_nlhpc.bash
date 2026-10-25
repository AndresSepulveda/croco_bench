#!/bin/bash
# ---------------- Script SBATCH - NLHPC - Slurm Parameters ----------------
# CROCO benchmark (croco_bench.py) inside ONE Slurm allocation: every simulation is launched
# by croco_bench.py as an 'srun' job step, so no run waits in the queue and the timings are comparable.
#SBATCH -J CROCO_BENCH
#SBATCH -p general
#SBATCH -N 1                       # one node: all runs share this node's cores
#SBATCH -n 16                      # cores available to the benchmark (>= ref_cores in the JSON)
#SBATCH --ntasks-per-node=16
#SBATCH -c 1
#SBATCH --exclusive                # whole node: no other user's job disturbs the timings
#SBATCH -t 3-00:00:00              # wall time; the benchmark stops cleanly before it (walltime_margin_s)
#SBATCH --mail-user=your.email@institution.cl
#SBATCH --mail-type=ALL
#SBATCH -o CROCO_BENCH_%j.out
#SBATCH -e CROCO_BENCH_%j.err
#-----------------Toolchain---------------------------
ml purge
ml intel/2022.00
# ----------------Modulos----------------------------
ml croco/2.0.0
# Python 3 with numpy, matplotlib, netCDF4 (and reportlab for the formatted PDF).
# Load the cluster's Python module if 'python3' is not available, e.g.:  ml Python
# and install the packages once with:  python3 -m pip install --user numpy matplotlib netCDF4 reportlab
#-------------------Comandos--------------------------
cd "$SLURM_SUBMIT_DIR"
python3 croco_bench.py -c bench_hpc.json
# If the wall time runs out, results are kept: submit this same script again to resume.
