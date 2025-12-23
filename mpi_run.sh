#!/bin/bash

# echo 0 | sudo tee /proc/sys/kernel/randomize_va_space ##-- may change parameter to fix it ( running under gdb has no segfaults)
if [ $# -eq 0 ]; then
    echo "Usage: $0 <num_threads>"
    echo "Error: Number of threads required"
    exit 1
fi

# Check if argument is a positive integer
if ! [[ $1 =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Processes must be a positive integer"
    echo "Received: $1"
    exit 1
fi

echo "Number of processes requested: $1"
echo $SLURM_NTASKS
echo $PBS_NP  
echo $OMPI_COMM_WORLD_SIZE

time mpirun -np $1 ./build/main_executable_mpi