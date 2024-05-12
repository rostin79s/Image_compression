#include <cmath>
#include <iostream>
#include "cuda_runtime.h"
#include <vector>
#include <string>
#include <chrono>
#include <opencv4/opencv2/opencv.hpp>
#include "dev_array.h"

using namespace std;
using namespace cv;

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
void printMatrix(vector<float> matrix, int N) {
    cout << "Matrix:" << endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << matrix[i * N + j] << "\t";
        }
        cout << endl;
    }
    cout << endl;
}

__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int N) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
}


void matrixMultiplication(float *A, float *B, float *C, int N){

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
        if (N*N > 512){
            threadsPerBlock.x = 512;
            threadsPerBlock.y = 512;
            blocksPerGrid.x = ceil(float(N)/float(threadsPerBlock.x));
            blocksPerGrid.y = ceil(float(N)/float(threadsPerBlock.y));
        }

    matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, N);
}


void image_compression(string filename){
    int N = 8;
    int size = N * N;

    Mat image = imread("images/stone.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error: Unable to read the image file." << endl;
        return ;
    }

    int numBlocksX = image.cols / 8;
    int numBlocksY = image.rows / 8;


    // Initialize matrices on the host
    vector<float> h_block(size);
    vector<float> quantizationMatrix = {
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99
    };
    float quantizationScalar = 5.0;


    dev_array<float> d_block(size);
    dev_array<float> d_DCT(size);
    dev_array<float> d_DCT_transpose(size);
    dev_array<float> d_result(size);
    dev_array<float> d_Q(size);

    initDCTMatrix<<<1, dim3(8, 8)>>>(d_DCT.getData());
    cudaDeviceSynchronize();
    transposeMatrix<<<1, dim3(8, 8)>>>(d_DCT.getData(), d_DCT_transpose.getData());
    cudaDeviceSynchronize();

    

    d_DCT_transpose.get(&h_block[0],size);
    printMatrix(h_block,N);

    for (int i = 0; i < numBlocksY; ++i) {
        for (int j = 0; j < numBlocksX; ++j) {
            // Get the 8x8 block from the image
            Mat block = image(Rect(j * 8, i * 8, 8, 8));

            // Convert the block to float and store it in h_A
            for (int y = 0; y < 8; ++y) {
                for (int x = 0; x < 8; ++x) {
                    h_block[y * 8 + x] = static_cast<float>(block.at<uchar>(y, x)) - 128;
                }
            }
            printMatrix(h_block,N);

            d_block.set(&h_block[0],size);
            matrixMultiplication(d_DCT.getData(), d_block.getData(), d_result.getData(), N);
            cudaDeviceSynchronize();
            matrixMultiplication(d_result.getData(), d_DCT_transpose.getData(), d_block.getData(), N);
            cudaDeviceSynchronize();


            d_block.get(&h_block[0],size);
            
            
            printMatrix(h_block, N);
            return;
        }
    }
    

}

int main() {

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    uint32_t numSMs = prop.multiProcessorCount;
    cout<<numSMs<<endl<<endl;

    string filename = "images/stone.jpg";

    auto start = std::chrono::high_resolution_clock::now();

    image_compression(filename);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;


    return 0;
}
