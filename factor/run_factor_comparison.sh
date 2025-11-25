#!/bin/bash
set -e

# Compile Serial
echo "Compiling Serial..."
gcc -o serial_lu serial_lu.c -O3

# Compile MPI
echo "Compiling MPI..."
mpicc -o mpi_lu mpi_lu.c -O3

# Compile CUDA
echo "Compiling CUDA..."
nvcc -o cuda_lu cuda_lu.cu -O3

echo "--------------------------------------------------"
echo "Running Benchmarks (LU Decomposition)..."
echo "--------------------------------------------------"

# Run Serial
echo "Running Serial..."
./serial_lu

# Run MPI
echo "Running MPI (4 processes)..."
mpiexec -n 4 ./mpi_lu

# Run CUDA with ZLUDA
echo "Running CUDA (with ZLUDA)..."
# Assuming run_with_zluda.sh is in the parent directory
../run_with_zluda.sh ./cuda_lu

echo "--------------------------------------------------"
echo "Done."
