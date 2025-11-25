#!/bin/bash
set -e

# Compile Serial
echo "Compiling Serial..."
gcc -o serial_matmul serial_matmul.c -O3

# Compile MPI
echo "Compiling MPI..."
mpicc -o mpi_matmul mpi_matmul.c -O3

# Compile CUDA
echo "Compiling CUDA..."
nvcc -o cuda_matmul cuda_matmul.cu -O3

echo "--------------------------------------------------"
echo "Running Benchmarks..."
echo "--------------------------------------------------"

# Run Serial
echo "Running Serial..."
./serial_matmul

# Run MPI
echo "Running MPI (8 processes)..."
mpiexec -n 8 ./mpi_matmul

# Run CUDA with ZLUDA
echo "Running CUDA (with ZLUDA)..."
# Assuming run_with_zluda.sh is in the parent directory
../run_with_zluda.sh ./cuda_matmul

echo "--------------------------------------------------"
echo "Done."
