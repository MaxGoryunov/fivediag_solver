#!/bin/bash
#SBATCH --job-name=fivediag
#SBATCH --output=fivediag.txt
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --partition=cascade

module purge
module load mpi/openmpi/4.1.6/gcc/11
module load python/3.11
PYTHON_BIN=/opt/software/python/3.11.8/bin/python3
pip install mpi4py
module load dev/cmake/3.23.1
THREADS=(2 4 8 16 28 48)
PROCS=(2 4 8 16 28 56 84 112)


export OMPI_MCA_btl="tcp,self"

echo "===== Job Info ====="
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "Nodes Allocated   = $SLURM_JOB_NUM_NODES"
echo "Tasks Allocated   = $SLURM_NTASKS"
echo "CPUs per Task     = $SLURM_CPUS_PER_TASK"
echo "==================="

# ----------------------------
# Compile all programs
# ----------------------------
rm -rf build
mkdir build
cd build
cmake ..
make 
cd ../

# ----------------------------
# Run posix
# ----------------------------
echo
echo "=== RUN pthreads ==="
for t in "${THREADS[@]}"; do
  echo "Threads = $t"
  bash posix_run.sh $t
done

# ----------------------------
# Run OpenMP
# ----------------------------
echo
echo "=== RUN OpenMP ==="
for t in "${THREADS[@]}"; do
  echo "Threads = $t"
  bash openmp_run.sh $t
done

# ----------------------------
# Run MPI C
# ----------------------------
echo
echo "=== RUN MPI C ==="
for p in "${PROCS[@]}"; do
  echo "Procs = $p"
  bash mpi_run.sh $p
done

# ----------------------------
# Run MPI Python
# ----------------------------
echo
echo "=== RUN MPI Python ==="
for p in "${PROCS[@]}"; do
  echo "Procs = $p"
  bash py_mpi_run.sh $p
done