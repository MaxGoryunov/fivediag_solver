#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <num_threads>"
    echo "Error: Number of threads required"
    exit 1
fi

# Check if argument is a positive integer
if ! [[ $1 =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: OMP_NUM_THREADS must be a positive integer"
    echo "Received: $1"
    exit 1
fi
time OMP_NUM_THREADS=$1 ./build/main_executable_openmp
