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
__global__ void initDCTMatrix(float *d_DCT, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        float alpha_i = (i == 0) ? sqrtf(1.0 / N) : sqrtf(2) / sqrtf(N);
        d_DCT[i * N + j] = alpha_i * cos((2 * j + 1) * i * M_PI / 16);
    }
}

// Function to transpose matrix on GPU
__global__ void transposeMatrix(float *input, float *output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        output[j * N + i] = input[i * N + j];
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

__global__ void quantizeBlock(float* block, float* Q, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Quantize the block element-wise
        block[idx] = round(block[idx] / Q[idx]);
    }
}

// Function to perform quantization on the GPU
void quantizeOnGPU(dev_array<float>& d_block, dev_array<float>& d_Q, int size) {
    dim3 threadsPerBlock(256); // 256 threads per block
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    quantizeBlock<<<blocksPerGrid, threadsPerBlock>>>(d_block.getData(), d_Q.getData(), size);
}


// Kernel function for element-wise matrix multiplication
__global__ void matrixElementWiseMultiplication(float* A, float* B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        A[idx] *= B[idx];
    }
}

// Function to perform element-wise matrix multiplication on the GPU
void elementWiseMatrixMultiplicationOnGPU(dev_array<float>& A, dev_array<float>& B, int size) {
    dim3 threadsPerBlock(256); // 256 threads per block
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    matrixElementWiseMultiplication<<<blocksPerGrid, threadsPerBlock>>>(A.getData(), B.getData(), size);
}

// Kernel function to round each element and add 128
__global__ void roundAndAdd128(float* block, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Round the element
        block[idx] = roundf(block[idx]) + 128;

        // Ensure the value is within the pixel range
        block[idx] = max(0.0f, min(255.0f, block[idx]));
    }
}

// Function to perform rounding and adding 128 on the GPU
void roundAndAdd128OnGPU(dev_array<float>& d_block, int size) {
    dim3 threadsPerBlock(256); // 256 threads per block
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    roundAndAdd128<<<blocksPerGrid, threadsPerBlock>>>(d_block.getData(), size);
}


void updateImageFromBlock(const vector<float>& h_block, Mat& image, int block_row, int block_col, int N) {
    int image_row = block_row * N;
    int image_col = block_col * N;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int pixel_row = image_row + i;
            int pixel_col = image_col + j;

            // Ensure the pixel coordinates are within the image bounds
            if (pixel_row < image.rows && pixel_col < image.cols) {
                // Update the pixel value by adding the corresponding value from h_block
                image.at<uchar>(pixel_row, pixel_col) = static_cast<uchar>(h_block[i * N + j]);
            }
        }
    }
}

void image_compression(string filename){
    int N = 8;
    int size = N * N;

    Mat image = imread("images/stone.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error: Unable to read the image file." << endl;
        return ;
    }

    int numBlocksX = image.cols / N;
    int numBlocksY = image.rows / N;
    cout<<image.cols<<" "<<image.rows<<endl;


    // Initialize matrices on the host
    vector<float> h_block(size);
    vector<float> h_Q = {
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99
    };
    float q = 5.0;


    dev_array<float> d_block(size);
    dev_array<float> d_DCT(size);
    dev_array<float> d_DCT_transpose(size);
    dev_array<float> d_result(size);
    dev_array<float> d_Q(size);

    initDCTMatrix<<<1, dim3(N, N)>>>(d_DCT.getData(), N);
    cudaDeviceSynchronize();
    transposeMatrix<<<1, dim3(N, N)>>>(d_DCT.getData(), d_DCT_transpose.getData(), N);
    cudaDeviceSynchronize();

    for (int i = 0; i < size; i++){
        h_Q[i] = min((float)255, h_Q[i] * q);
    }
    d_Q.set(&h_Q[0],size);

    Mat modified_image = image.clone();

    for (int i = 0; i < numBlocksY; ++i) {
        for (int j = 0; j < numBlocksX; ++j) {
            // Get the 8x8 block from the image
            Mat block = image(Rect(j * N, i * N, N, N));

            // Convert the block to float and store it in h_A
            for (int y = 0; y < N; ++y) {
                for (int x = 0; x < N; ++x) {
                    h_block[y * N + x] = static_cast<float>(block.at<uchar>(y, x)) - 128;
                }
            }
            // printMatrix(h_block,N);

            d_block.set(&h_block[0],size);
            matrixMultiplication(d_DCT.getData(), d_block.getData(), d_result.getData(), N);
            cudaDeviceSynchronize();
            matrixMultiplication(d_result.getData(), d_DCT_transpose.getData(), d_block.getData(), N);
            cudaDeviceSynchronize();

            quantizeOnGPU(d_block,d_Q,size);
            cudaDeviceSynchronize();

            elementWiseMatrixMultiplicationOnGPU(d_block,d_Q,size);
            cudaDeviceSynchronize();

            matrixMultiplication(d_DCT_transpose.getData(), d_block.getData(), d_result.getData(), N);
            cudaDeviceSynchronize();
            matrixMultiplication(d_result.getData(), d_DCT.getData(), d_block.getData(), N);
            cudaDeviceSynchronize();

            roundAndAdd128OnGPU(d_block,size);
            cudaDeviceSynchronize();


            d_block.get(&h_block[0],size);

            updateImageFromBlock(h_block, modified_image, i, j, N);


            
            
            
            // printMatrix(h_block, N);
            // return;
        }
    }

    imwrite("parallel_" + filename, modified_image);
    

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