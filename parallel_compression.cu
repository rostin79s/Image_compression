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

__global__ void processBlock(float* d_image, float * d_imageres, float* d_DCT, float* d_DCT_transpose, float* d_Q, int N, int numBlocksX, int numBlocksY) {
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    if (block_row < numBlocksY && block_col < numBlocksX) {
        int start_idx = (block_row * numBlocksX + block_col) * N * N;
        float tmpSum = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int pixel_idx = i * N + j;
                tmpSum = 0;
                for (int k = 0; k < N; ++k) {
                    tmpSum += d_DCT[i * N + k] * d_image[start_idx + k * N + j];
                }
                d_imageres[start_idx + pixel_idx] = tmpSum;
            }
        }
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int pixel_idx = i * N + j;
                tmpSum = 0;
                for (int k = 0; k < N; ++k) {
                    tmpSum += d_imageres[start_idx + i * N + k] *d_DCT_transpose[k * N + j];
                }
                d_image[start_idx + pixel_idx] = tmpSum;
            }
        }

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int pixel_idx = i * N + j;
                d_image[start_idx + pixel_idx] = roundf(d_image[start_idx + pixel_idx] / d_Q[pixel_idx]);
                d_image[start_idx + pixel_idx] *= d_Q[pixel_idx];
            }
        }

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int pixel_idx = i * N + j;
                tmpSum = 0;
                for (int k = 0; k < N; ++k) {
                    tmpSum += d_DCT_transpose[i * N + k] * d_image[start_idx + k * N + j];
                }
                d_imageres[start_idx + pixel_idx] = tmpSum;
            }
        }

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int pixel_idx = i * N + j;
                tmpSum = 0;
                for (int k = 0; k < N; ++k) {
                    tmpSum += d_imageres[start_idx + i * N + k] *d_DCT[k * N + j];
                }
                d_image[start_idx + pixel_idx] = tmpSum;
            }
        }

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int pixel_idx = i * N + j;
                d_image[start_idx + pixel_idx] = roundf(d_image[start_idx + pixel_idx]) + 128;
                d_image[start_idx + pixel_idx] = max(0.0f, min(255.0f, d_image[start_idx + pixel_idx]));
            }
        }
    }
}



// Function to print matrix. for debugging
void printMatrix(vector<float> matrix, int N, int M) {
    cout << "Matrix:" << endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            cout << matrix[i * M + j] << "\t";
        }
        cout << endl;
    }
    cout << endl;
}


void image_compression(string filename){
    int N = 8;

    Mat image = imread(filename, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error: Unable to read the image file." << endl;
        return ;
    }

    int numBlocksX = image.cols / N;
    int numBlocksY = image.rows / N;

    // Initialize matrices on the host
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

    // Quantize the quantization matrix
    for (int i = 0; i < N * N; i++){
        h_Q[i] = min(255.0f, h_Q[i] * q);
    }

    // Calculate quantization matrix
    dev_array<float> d_Q(N * N);
    d_Q.set(&h_Q[0], N * N);

    // Calculate the DCT matrix
    dev_array<float> d_DCT(N * N);
    initDCTMatrix<<<1, dim3(N, N)>>>(d_DCT.getData(), N);
    cudaDeviceSynchronize();

    // Calculate the transpose of the DCT matrix
    dev_array<float> d_DCT_transpose(N * N);
    transposeMatrix<<<1, dim3(N, N)>>>(d_DCT.getData(), d_DCT_transpose.getData(), N);
    cudaDeviceSynchronize();

    vector<float> h_image(image.cols * image.rows);

    for (int i = 0; i < numBlocksY; ++i) {
        for (int j = 0; j < numBlocksX; ++j) {
            Mat block = image(Rect(j * N, i * N, N, N));
            int blockStartIndex = (i * numBlocksX + j) * N * N;

            for (int y = 0; y < N; ++y) {
                for (int x = 0; x < N; ++x) {
                    int blockIndex = y * N + x;
                    h_image[blockStartIndex + blockIndex] = static_cast<float>(block.at<uchar>(y, x)) - 128;
                }
            }
        }
    }   

    dev_array<float> d_image(image.cols * image.rows);
    // this is needed for temp matrix mul,
    dev_array<float> d_imageres(image.cols * image.rows);
    d_image.set(&h_image[0], image.cols * image.rows);

    // Set up grid and block dimensions
    dim3 threadsPerBlock(1, 1); // One thread per block
    dim3 blocksPerGrid(numBlocksX, numBlocksY); // One block per 8x8 block in the image

    // Process each block in parallel
    processBlock<<<blocksPerGrid, threadsPerBlock>>>(d_image.getData(),d_imageres.getData(), d_DCT.getData(), d_DCT_transpose.getData(), d_Q.getData(), N, numBlocksX, numBlocksY);
    cudaDeviceSynchronize();

    d_image.get(&h_image[0], image.cols * image.rows);

    for (int i = 0; i < numBlocksY; ++i) {
        for (int j = 0; j < numBlocksX; ++j) {
            Mat block = image(Rect(j * N, i * N, N, N));
            int blockStartIndex = (i * numBlocksX + j) * N * N;

            for (int y = 0; y < N; ++y) {
                for (int x = 0; x < N; ++x) {
                    int blockIndex = y * N + x;
                    block.at<uchar>(y, x) = h_image[blockStartIndex + blockIndex];
                }
            }
        }
    }   
    imwrite("parallel_" + filename, image);

    cout <<"Modified image saved" << endl;
}

int main() {
    string filename = "images/stone.jpg";

    auto start = std::chrono::high_resolution_clock::now();

    image_compression(filename);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}