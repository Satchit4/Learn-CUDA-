#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 2048
#define BLOCK_SIZE 256

// Kernel to scale the column (calculate L factors)
__global__ void scale_column_kernel(double *A, int k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = k + 1 + idx;
    
    if (i < n) {
        A[i * n + k] /= A[k * n + k];
    }
}

// Kernel to update the submatrix (Schur complement)
__global__ void update_submatrix_kernel(double *A, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // We are updating A[i][j] for i > k, j > k
    int i = k + 1 + row;
    int j = k + 1 + col;

    if (i < n && j < n) {
        A[i * n + j] -= A[i * n + k] * A[k * n + j];
    }
}

int main() {
    double *h_A;
    double *d_A;
    size_t size = N * N * sizeof(double);

    // Allocate host memory
    h_A = (double *)malloc(size);

    // Initialize matrix
    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < N; i++) {
        h_A[i * N + i] += N;
    }

    clock_t start_cpu = clock();

    // Allocate device memory
    cudaMalloc(&d_A, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Main loop
    for (int k = 0; k < N - 1; k++) {
        // 1. Scale column k (L factors)
        // Number of elements to scale: N - 1 - k
        int num_elements = N - 1 - k;
        int blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scale_column_kernel<<<blocks, BLOCK_SIZE>>>(d_A, k, N);
        
        // 2. Update submatrix
        // Size of submatrix: (N - 1 - k) x (N - 1 - k)
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((num_elements + 15) / 16, (num_elements + 15) / 16);
        update_submatrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, k, N);
    }

    // Copy result back to host
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

    clock_t end_cpu = clock();
    double time_spent = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

    printf("CUDA LU Decomposition (N=%d): %f seconds\n", N, time_spent);
    printf("Verification: A[N-1][N-1] = %f\n", h_A[(N-1)*N + (N-1)]);

    // Free memory
    cudaFree(d_A);
    free(h_A);

    return 0;
}
