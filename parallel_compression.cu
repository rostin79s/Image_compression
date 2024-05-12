#include <iostream>
#include <vector>
#include <cmath>
#include <opencv4/opencv2/opencv.hpp>

// CUDA kernel for DCT calculation
__global__ void dct_kernel(double *T, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        if (i == 0) {
            T[i * N + j] = 1 / sqrtf(N);
        } else {
            double tmp = ((2 * j + 1) * i * M_PI) / (2 * N);
            T[i * N + j] = sqrtf(2) / sqrtf(N) * cos(tmp);
        }
    }
}

// CUDA kernel for matrix multiplication
__global__ void mat_mul_kernel(double *A, double *B, double *C, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

int main() {
    // Load image and convert to grayscale
    cv::Mat image = cv::imread("images/stone.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Unable to read the image file." << std::endl;
        return -1;
    }

    int N = 8; // Size of the DCT matrix (assuming 8x8 blocks)
    int rows = image.rows;
    int cols = image.cols;

    // Allocate memory for DCT matrix on GPU
    double *d_T;
    cudaMalloc(&d_T, N * N * sizeof(double));

    // Launch DCT kernel
    dim3 blockSize(8, 8);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    dct_kernel<<<gridSize, blockSize>>>(d_T, N);
    cudaDeviceSynchronize(); // Ensure all threads have finished execution

    // Allocate memory for image data on GPU
    double *d_image;
    size_t image_size = rows * cols * sizeof(double);
    cudaMalloc(&d_image, image_size);

    // Copy image data from CPU to GPU
    cudaMemcpy(d_image, image.data, image_size, cudaMemcpyHostToDevice);

    // Allocate memory for result matrix on GPU
    double *d_result;
    cudaMalloc(&d_result, image_size);

    // Calculate grid and block dimensions for matrix multiplication kernel
    int block_dim = 16; // Adjust this based on your GPU architecture
    dim3 block(block_dim, block_dim);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    // Launch matrix multiplication kernel
    mat_mul_kernel<<<grid, block>>>(d_T, d_image, d_result, N);
    cudaDeviceSynchronize();

    // Copy result back from GPU to CPU
    cv::Mat result(rows, cols, CV_64F);
    cudaMemcpy(result.data, d_result, image_size, cudaMemcpyDeviceToHost);

    // Free allocated memory on GPU
    cudaFree(d_T);
    cudaFree(d_image);
    cudaFree(d_result);

    // Save or display the result
    cv::imwrite("modified_stone.jpg", result);
    std::cout << "Modified image saved" << std::endl;

    return 0;
}
