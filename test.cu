// #include <math.h>
#include <cmath>
#include <iostream>
#include "cuda_runtime.h"
// #include <stdlib.h>
#include <vector>
// #include <time.h>
#include "dev_array.h"

using namespace std;

__global__ void matrixMultiplicationKernel(double* A, double* B, double* C, int N) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    double tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
}


void matrixMultiplication(double *A, double *B, double *C, int N){

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
        if (N*N > 512){
            threadsPerBlock.x = 512;
            threadsPerBlock.y = 512;
            blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
        }

    matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, N);
}

void print(vector<double> h_C, int N){
    cout << "Result Matrix (GPU):" << endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << h_C[i * N + j] << "\t";
        }
        cout << endl;
    }
}


int main()
{
    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int N = 8;
    int SIZE = N*N;


    // Allocate memory on the host
    vector<double> h_A(SIZE);
    vector<double> h_B(SIZE);
    vector<double> h_C(SIZE);

    // Initialize matrices on the host
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            h_A[i*N+j] = sin(i);
            h_B[i*N+j] = cos(j);
        }
    }

    // Allocate memory on the device
    dev_array<double> d_A(SIZE);
    dev_array<double> d_B(SIZE);
    dev_array<double> d_C(SIZE);

    d_A.set(&h_A[0], SIZE);
    d_B.set(&h_B[0], SIZE);

    matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
    cudaDeviceSynchronize();

    d_C.get(&h_C[0], SIZE);
    cudaDeviceSynchronize();

    print(h_C,N);

    double *cpu_C;
    cpu_C=new double[SIZE];

    // Now do the matrix multiplication on the CPU
    double sum;
    for (int row=0; row<N; row++){
        for (int col=0; col<N; col++){
            sum = 0.f;
            for (int n=0; n<N; n++){
                sum += h_A[row*N+n]*h_B[n*N+col];
            }
            cpu_C[row*N+col] = sum;
        }
    }

    double err = 0;
    // Check the result and make sure it is correct
    for (int ROW=0; ROW < N; ROW++){
        for (int COL=0; COL < N; COL++){
            err += cpu_C[ROW * N + COL] - h_C[ROW * N + COL];
        }
    }

    cout << "Error: " << err << endl;

    return 0;
}