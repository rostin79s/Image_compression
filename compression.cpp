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
            mat_tran[i][j] = matrix[j][i];
        }
    }
    return mat_tran;
}


vector<vector<double>> mat_mul(vector<vector<double>> &matrix1, vector<vector<double>> &matrix2, int N){
    vector<vector<double>> res(N, vector<double>(N,0));
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            for (int k = 0; k < N; k ++){
                res[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return res;
}

vector<vector<double>> muls(vector<vector<double>> &M,vector<vector<double>> &T,vector<vector<double>> &T_tran,int N){
    vector<vector<double>> tmp = mat_mul(T,M,N);
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

vector<vector<double>> decompress(vector<vector<double>> &Q, vector<vector<double>> &C, int N){
    vector<vector<double>> R (N, vector<double>(N, 0.0));
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            R[i][j] = round(Q[i][j] * C[i][j]);
        }
    }
    return R;
}
void sub128(vector<vector<double>> &M, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            M[i][j] -= 128;
        }
    }
}
void add128(vector<vector<double>> &M, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            M[i][j] += 128;
        }
    }
}

void quant(vector<vector<double>> &M, int N, int q){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            M[i][j] *= q;
        }
    }
}

void mat_round(vector<vector<double>> &M, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            M[i][j] = round(M[i][j]);
        }
    }
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
    quant(Q,N,5);
    sub128(M,N);
    vector<vector<double>> D = muls(M,T,T_tran,N);
    vector<vector<double>> C = compress(Q,D,N);
    // we need to decode C
    vector<vector<double>> R = decompress(Q,C,N);

    vector<vector<double>> res = muls(R,T_tran,T,N);
    mat_round(res,N);
    add128(res,N);

    // print(res,N);
    return res;
}

void read_image(string filename){
    Mat image = imread(filename);
     if (image.empty()) {
        cout << "Error: Unable to read the image file." << endl;
        return;
    }
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    imwrite("gray_"+filename, grayImage);

    // Get image dimensions
    int rows = grayImage.rows;
    int cols = grayImage.cols;
    cout<<rows<<" "<<cols<<endl;
    
    int block_size = 8;

    // Vector to store 8x8 regions as vectors of vectors
    vector<vector<vector<double>>> blockVectors;

    // Loop through the image and convert each 8x8 block to vector of vectors
    for (int y = 0; y < rows; y += block_size) {
        for (int x = 0; x < cols; x += block_size) {
            // Extract an 8x8 region from the grayscale image
            // cout<<x<<" "<<y<<endl;
            Rect roi(x, y, min(block_size,cols-x), min(block_size,rows-y));
            Mat region = grayImage(roi);
            // cout<<region.rows<< " "<<region.cols<<endl;
            // Convert Mat region to a vector<vector<double>>
            vector<vector<double>> blockVector;
            for (int i = 0; i < region.rows; ++i) {
                vector<double> rowVector;
                for (int j = 0; j < region.cols; ++j) {
                    rowVector.push_back(static_cast<double>(region.at<uchar>(i, j)));
                }
                blockVector.push_back(rowVector);
            }
            if (region.rows!= block_size || region.cols != block_size){
                // cout<<x<<" "<<y<<endl;
                blockVectors.push_back(blockVector);
                continue;
            }
            vector<vector<double>> updated_blockVector = image_compression(blockVector,block_size);
            // return;
            // Store the block vector
            blockVectors.push_back(updated_blockVector);
        }
    }

    Mat newImage(rows, cols, CV_8U, Scalar(0)); // Create a new grayscale image
    int index = 0;
    for (int y = 0; y < rows; y += block_size) {
        for (int x = 0; x < cols; x += block_size) {
            // Get the block vector
            vector<vector<double>> blockVector = blockVectors[index++];

            // Convert the block vector back to Mat format
            for (int i = 0; i < (int)blockVector.size(); ++i) {
                for (int j = 0; j < (int)blockVector[0].size(); ++j) {
                    newImage.at<uchar>(y + i, x + j) = static_cast<uchar>(blockVector[i][j]);
                }
            }
        }
    }

    imwrite("modified_"+filename, newImage);

    cout <<"Modified image saved" << endl;
}

int main(){
    string filename = "images/stone.jpg";
    read_image(filename);
    return 0;
}