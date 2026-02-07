%%writefile matrix.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <ctime>

using namespace std;

__global__ void matrixMul(float *A, float *B, float *R, int M, int N, int P, int batchOffset) {
    int k = threadIdx.x + batchOffset;
    float *a = A + k * M * N;
    float *b = B + k * N * P;
    float *r = R + k * M * P;

    for(int outer = 0; outer < 100; outer++) {
        for(int i = 0; i < M; i++) {
            for(int l = 0; l < P; l++) {
                r[i * P + l] = 0.0f;
                for(int j = 0; j < N; j++) {
                    r[i * P + l] += a[i * N + j] * b[j * P + l];
                }
            }
        }
    }
}

// FULL matrix print করবে
void printMatrix(float *A, int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("%.0f ", A[i * cols + j]);
        }
        cout << endl;
    }
    cout << endl;
}

int main(int argc, char* argv[]) {

  srand(time(NULL));   // Random seed

    if(argc != 6) {
        cerr << "Usage: " << argv[0] << " <threads> <k> <m> <n> <p>\n";
        return 1;
    }

    int threads = atoi(argv[1]);
    int K = atoi(argv[2]);
    int M = atoi(argv[3]);
    int N = atoi(argv[4]);
    int P = atoi(argv[5]);

    int size_of_a = K * M * N;
    int size_of_b = K * N * P;
    int size_of_r = K * M * P;

    float *h_A = (float*)malloc(size_of_a * sizeof(float));
    float *h_B = (float*)malloc(size_of_b * sizeof(float));
    float *h_R = (float*)malloc(size_of_r * sizeof(float));

    for(int i = 0; i < size_of_a; i++) h_A[i] = rand() % 10;
    for(int i = 0; i < size_of_b; i++) h_B[i] = rand() % 10;

    float *d_A, *d_B, *d_R;
    cudaMalloc(&d_A, size_of_a * sizeof(float));
    cudaMalloc(&d_B, size_of_b * sizeof(float));
    cudaMalloc(&d_R, size_of_r * sizeof(float));

    cudaMemcpy(d_A, h_A, size_of_a * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_of_b * sizeof(float), cudaMemcpyHostToDevice);

    int remainingMatrices = K;
    int batchOffset = 0;

    auto start = chrono::high_resolution_clock::now();

    while(remainingMatrices > 0) {
        int currentBatchSize = min(remainingMatrices, threads);
        matrixMul<<<1, currentBatchSize>>>(d_A, d_B, d_R, M, N, P, batchOffset);
        cudaDeviceSynchronize();
        remainingMatrices -= currentBatchSize;
        batchOffset += currentBatchSize;
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cudaMemcpy(h_R, d_R, size_of_r * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Execution Time: " << elapsed.count() << " seconds\n\n";

    // First Matrix
    cout << "A[0]:\n";
    printMatrix(h_A, M, N);

    cout << "B[0]:\n";
    printMatrix(h_B, N, P);

    cout << "C[0]:\n";
    printMatrix(h_R, M, P);

    // Batch 9
    if(K > 9) {
        int batch9 = 9;
        cout << "A[9]:\n";
        printMatrix(h_A + batch9 * M * N, M, N);

        cout << "B[9]:\n";
        printMatrix(h_B + batch9 * N * P, N, P);

        cout << "C[9]:\n";
        printMatrix(h_R + batch9 * M * P, M, P);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_R);
    free(h_A);
    free(h_B);
    free(h_R);

    return 0;
}

//!nvcc -arch=sm_75 matrix.cu -o matrix
//!time ./matrix 25 120 5 5 5 > output.txt