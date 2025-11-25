#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 8192
#define BLOCK_SIZE 16

__global__ void matmul_kernel(double *A, double *B, double *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    double *h_A, *h_B, *h_C;
    double *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(double);

    // Allocate host memory
    h_A = (double *)malloc(size);
    h_B = (double *)malloc(size);
    h_C = (double *)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (double)rand() / RAND_MAX;
        h_B[i] = (double)rand() / RAND_MAX;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // cudaEventRecord(start);
    clock_t start_cpu = clock();

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaError_t err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA Error (Memcpy D2H): %s\n", cudaGetErrorString(err));
    }

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    
    clock_t end_cpu = clock();

    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    double time_spent = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error (Kernel/Other): %s\n", cudaGetErrorString(err));
    }

    printf("CUDA Matrix Multiplication (N=%d): %f seconds\n", N, time_spent);
    printf("Verification: C[0] = %f\n", h_C[0]);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
