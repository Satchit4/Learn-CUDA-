#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 2048

void lu_decompose(double *A, int n) {
    for (int k = 0; k < n; k++) {
        // Pivot is A[k][k]
        // Calculate L part (stored in lower triangle of A)
        for (int i = k + 1; i < n; i++) {
            A[i * n + k] /= A[k * n + k];
        }

        // Update U part (stored in upper triangle of A)
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
        }
    }
}

int main() {
    double *A = (double *)malloc(N * N * sizeof(double));

    // Initialize matrix
    srand(42);
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)rand() / RAND_MAX;
    }
    // Make it diagonally dominant to avoid zero pivots
    for (int i = 0; i < N; i++) {
        A[i * N + i] += N;
    }

    clock_t start = clock();
    lu_decompose(A, N);
    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Serial LU Decomposition (N=%d): %f seconds\n", N, time_spent);
    
    // Verification (check a random element or sum)
    printf("Verification: A[N-1][N-1] = %f\n", A[(N-1)*N + (N-1)]);

    free(A);
    return 0;
}
