#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv4/opencv2/opencv.hpp>
    
using namespace std;
using namespace cv;


vector<vector<double>> dct(int N) {
    vector<vector<double>> T(N, vector<double>(N, 0.0));
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            if (i == 0){
                T[i][j] = 1/sqrt(N);
            }
            else{
                double tmp = ((2*j+1)*i*M_PI)/(2*N);
                T[i][j] = sqrt(2)/sqrt(N) * cos(tmp);
            }
        }
    }
    return T;
}

vector<vector<double>> transpose(vector<vector<double>> &matrix , int N){
    vector<vector<double>> mat_tran (N, vector<double>(N, 0.0));
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            mat_tran[j][i] = matrix[i][j];
        }
    }
    return mat_tran;
}


vector<vector<double>> mat_mul(vector<vector<double>> &matrix1, vector<vector<double>> &matrix2, int N){
    vector<vector<double>> res(N, vector<double>(N,0));
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            for (int k = 0; k < N; k ++){
                res[i][j] += matrix1[i][k] + matrix2[k][j];
            }
        }
    }
    return res;
}

vector<vector<double>> muls(vector<vector<double>> &M,vector<vector<double>> &T,vector<vector<double>> &T_tran,int N){
    vector<vector<double>> tmp = mat_mul(M,T,N);
    vector<vector<double>> D = mat_mul(tmp,T_tran,N);
    return D;
}

void print(vector<vector<double>> &matrix, int N){
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << matrix[i][j] << "\t"; // Print each element followed by a tab
        }
        cout << endl; // Move to the next line after printing each row
    }
}

vector<vector<double>> compress(vector<vector<double>> &Q, vector<vector<double>> &D, int N){
    vector<vector<double>> C (N, vector<double>(N, 0.0));
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            C[i][j] = round(D[i][j]/Q[i][j]);
        }
    }
    return C;
}

vector<vector<double>> image_compression(vector<vector<double>> &M , int N){
    vector<vector<double>> T =  dct(N);
    vector<vector<double>> T_tran = transpose(T,N);
    vector<vector<double>> Q = {
        {16, 11, 10, 16, 24, 40, 51, 61},
        {12, 12, 14, 19, 26, 58, 60, 55},
        {14, 13, 16, 24, 40, 57, 69, 56},
        {14, 17, 22, 29, 51, 87, 80, 62},
        {18, 22, 37, 56, 68, 109, 103, 77},
        {24, 35, 55, 64, 81, 104, 113, 92},
        {49, 64, 78, 87, 103, 121, 120, 101},
        {72, 92, 95, 98, 112, 100, 103, 99}
    };
    vector<vector<double>> D = muls(M,T,T_tran,N);
    vector<vector<double>> C = compress(Q,D,N);
    // print(C,N);
    return C;
}

void read_image(string filename){
    Mat image = imread(filename);
     if (image.empty()) {
        cout << "Error: Unable to read the image file." << endl;
        return;
    }
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Get image dimensions
    int rows = grayImage.rows;
    int cols = grayImage.cols;
    
    int block_size = 8;

    // Vector to store 8x8 regions as vectors of vectors
    vector<vector<vector<double>>> blockVectors;

    // Loop through the image and convert each 8x8 block to vector of vectors
    for (int y = 0; y < rows; y += block_size) {
        for (int x = 0; x < cols; x += block_size) {
            // Extract an 8x8 region from the grayscale image
            Rect roi(x, y, block_size, block_size);
            Mat region = grayImage(roi);

            // Convert Mat region to a vector<vector<double>>
            vector<vector<double>> blockVector;
            for (int i = 0; i < region.rows; ++i) {
                vector<double> rowVector;
                for (int j = 0; j < region.cols; ++j) {
                    rowVector.push_back(static_cast<double>(region.at<uchar>(i, j)));
                }
                blockVector.push_back(rowVector);
            }

            vector<vector<double>> updated_blockVector = image_compression(blockVector,block_size);

            // Store the block vector
            blockVectors.push_back(blockVector);
        }
    }

    Mat newImage(rows, cols, CV_8U, Scalar(0)); // Create a new grayscale image
    int index = 0;
    for (int y = 0; y < rows; y += block_size) {
        for (int x = 0; x < cols; x += block_size) {
            // Get the block vector
            vector<vector<double>> blockVector = blockVectors[index++];

            // Convert the block vector back to Mat format
            for (int i = 0; i < block_size; ++i) {
                for (int j = 0; j < block_size; ++j) {
                    newImage.at<uchar>(y + i, x + j) = static_cast<uchar>(blockVector[i][j]);
                }
            }
        }
    }

    imwrite("modified_paris.jpg", newImage);

    cout << "Modified image saved as modified_paris.jpg" << endl;
}

int main(){
    string filename = "images/paris.jpg";
    read_image(filename);
    return 0;
}