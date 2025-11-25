# CUDA on AMD Benchmarks

This repository contains benchmark comparisons for Matrix Multiplication and Matrix Factorization (LU Decomposition) using three different approaches:
1.  **Serial**: Single-threaded CPU implementation.
2.  **MPI**: Parallel CPU implementation using Message Passing Interface.
3.  **CUDA (via ZLUDA)**: GPU implementation running on AMD hardware using ZLUDA.

## Prerequisites

-   **Linux OS** (Tested on Ubuntu)
-   **GCC**
-   **MPI** (e.g., MPICH or OpenMPI)
-   **CUDA Toolkit** (for `nvcc` compiler) [Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
-   **ZLUDA**: For running CUDA applications on AMD GPUs.

## Directory Structure

-   `multiply/`: Contains Matrix Multiplication benchmarks.
-   `factor/`: Contains Matrix Factorization (LU Decomposition) benchmarks.
-   `check_version/`: Simple utility to check CUDA version.
-   `zluda/`: (Excluded from git) Directory containing ZLUDA binaries.

## Running Benchmarks

### Matrix Multiplication

Navigate to the `multiply` directory and run the comparison script:

```bash
cd multiply
./run_comparison.sh
```

This will compile and run:
-   `serial_matmul.c`
-   `mpi_matmul.c`
-   `cuda_matmul.cu` (via ZLUDA)

### Matrix Factorization (LU Decomposition)

Navigate to the `factor` directory and run the comparison script:

```bash
cd factor
./run_factor_comparison.sh
```

This will compile and run:
-   `serial_lu.c`
-   `mpi_lu.c`
-   `cuda_lu.cu` (via ZLUDA)

## Notes on ZLUDA

This project uses [ZLUDA](https://github.com/vosen/ZLUDA) to run unmodified CUDA binaries on AMD ROCm. The `run_with_zluda.sh` helper script sets up the necessary environment variables (`LD_LIBRARY_PATH`) to inject ZLUDA libraries.
