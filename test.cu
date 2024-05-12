#include <cmath>
#include <iostream>
#include "cuda_runtime.h"
#include <vector>

using namespace std;

// Function to initialize DCT matrix on GPU
__global__ void initDCTMatrix(float *d_DCT) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 8 && j < 8) {
        float alpha_i = (i == 0) ? sqrtf(1.0 / 8) : sqrtf(2) / sqrtf(8);
        d_DCT[i * 8 + j] = alpha_i * cos((2 * j + 1) * i * M_PI / 16);
    }
}

// Function to transpose matrix on GPU
__global__ void transposeMatrix(float *input, float *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 8 && j < 8) {
        output[j * 8 + i] = input[i * 8 + j];
    }
}

// Function to print matrix
void printMatrix(vector<float> matrix, int N, string label) {
    cout << label << " Matrix:" << endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << matrix[i * N + j] << "\t";
        }
        cout << endl;
    }
    cout << endl;
}

int main() {
    int N = 8;
    int SIZE = N * N;

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    uint32_t numSMs = prop.multiProcessorCount;
    cout<<numSMs<<endl<<endl;

    // Initialize matrices on the host
    vector<float> h_A(SIZE);
    vector<float> h_B(SIZE);
    vector<float> h_C(SIZE);

    // Initialize DCT matrix on GPU
    float *d_DCT;
    cudaMalloc((void **)&d_DCT, SIZE * sizeof(float));
    initDCTMatrix<<<1, dim3(8, 8)>>>(d_DCT);
    cudaDeviceSynchronize();

    // Transpose DCT matrix on GPU
    float *d_DCT_transpose;
    cudaMalloc((void **)&d_DCT_transpose, SIZE * sizeof(float));
    transposeMatrix<<<1, dim3(8, 8)>>>(d_DCT, d_DCT_transpose);
    cudaDeviceSynchronize();

    // Copy DCT matrices to host for printing
    cudaMemcpy(&h_A[0], d_DCT, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_B[0], d_DCT_transpose, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Print DCT matrices
    printMatrix(h_A, N, "DCT");
    printMatrix(h_B, N, "Transposed DCT");

    // Free allocated memory
    cudaFree(d_DCT);
    cudaFree(d_DCT_transpose);

    return 0;
}
