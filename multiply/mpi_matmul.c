#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8192

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) {
            printf("Matrix size %d must be divisible by number of processes %d\n", N, size);
        }
        MPI_Finalize();
        return 1;
    }

    double *A = NULL;
    double *B = (double *)malloc(N * N * sizeof(double));
    double *C = NULL;
    double *local_A = (double *)malloc((N * N / size) * sizeof(double));
    double *local_C = (double *)malloc((N * N / size) * sizeof(double));

    if (rank == 0) {
        A = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));
        for (int i = 0; i < N * N; i++) {
            A[i] = (double)rand() / RAND_MAX;
            B[i] = (double)rand() / RAND_MAX;
        }
    }

    // Broadcast B to all processes
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter A to all processes
    MPI_Scatter(A, N * N / size, MPI_DOUBLE, local_A, N * N / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();

    // Perform matrix multiplication on local chunk
    int rows_per_proc = N / size;
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += local_A[i * N + k] * B[k * N + j];
            }
            local_C[i * N + j] = sum;
        }
    }

    // Gather results into C
    MPI_Gather(local_C, N * N / size, MPI_DOUBLE, C, N * N / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("MPI Matrix Multiplication (N=%d, Processes=%d): %f seconds\n", N, size, end_time - start_time);
        free(A);
        free(C);
    }

    free(B);
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}
