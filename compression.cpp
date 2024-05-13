#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <opencv4/opencv2/opencv.hpp>
    
using namespace std;
using namespace cv;


vector<vector<float>> dct(int N) {
    vector<vector<float>> T(N, vector<float>(N, 0.0));
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            if (i == 0){
                T[i][j] = 1/sqrt(N);
            }
            else{
                float tmp = ((2*j+1)*i*M_PI)/(2*N);
                T[i][j] = sqrt(2)/sqrt(N) * cos(tmp);
            }
        }
    }
    return T;
}

vector<vector<float>> transpose(vector<vector<float>> &matrix , int N){
    vector<vector<float>> mat_tran (N, vector<float>(N, 0.0));
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            mat_tran[i][j] = matrix[j][i];
        }
    }
    return mat_tran;
}


vector<vector<float>> mat_mul(vector<vector<float>> &matrix1, vector<vector<float>> &matrix2, int N){
    vector<vector<float>> res(N, vector<float>(N,0));
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            for (int k = 0; k < N; k ++){
                res[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return res;
}

vector<vector<float>> muls(vector<vector<float>> &M,vector<vector<float>> &T,vector<vector<float>> &T_tran,int N){
    vector<vector<float>> tmp = mat_mul(T,M,N);
    vector<vector<float>> D = mat_mul(tmp,T_tran,N);
    return D;
}


vector<vector<float>> compress(vector<vector<float>> &Q, vector<vector<float>> &D, int N){
    vector<vector<float>> C (N, vector<float>(N, 0.0));
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            C[i][j] = round(D[i][j]/Q[i][j]);
        }
    }
    return C;
}

vector<vector<float>> decompress(vector<vector<float>> &Q, vector<vector<float>> &C, int N){
    vector<vector<float>> R (N, vector<float>(N, 0.0));
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            R[i][j] = round(Q[i][j] * C[i][j]);
        }
    }
    return R;
}
void sub128(vector<vector<float>> &M, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            M[i][j] -= 128;
        }
    }
}
void add128(vector<vector<float>> &M, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            M[i][j] = max((float)0, min((float)255, M[i][j] + 128));
        }
    }
}

void quant(vector<vector<float>> &M, int N, int q){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            M[i][j] = min((float)255, M[i][j] * q);
        }
    }
}

void mat_round(vector<vector<float>> &M, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            M[i][j] = round(M[i][j]);
        }
    }
}

// print matrix for debugging
void print(vector<vector<float>> &matrix, int N){
    cout<<"Matrix: "<<endl<<endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << matrix[i][j] << "\t"; 
        }
        cout << endl; 
    }
    cout<<endl<<endl;
}

vector<vector<float>> image_compression(vector<vector<float>> &M , int N){
    vector<vector<float>> T =  dct(N);
    vector<vector<float>> T_tran = transpose(T,N);
    vector<vector<float>> Q = {
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
    vector<vector<float>> tmp = mat_mul(T,M,N);
    vector<vector<float>> D = muls(M,T,T_tran,N);
    vector<vector<float>> C = compress(Q,D,N);

    // In JPEG, C matrix will be encoded to save space and remove 0s.

    vector<vector<float>> R = decompress(Q,C,N);

    vector<vector<float>> res = muls(R,T_tran,T,N);
    mat_round(res,N);
    add128(res,N);
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

    vector<vector<vector<float>>> blockVectors;

    for (int y = 0; y < rows; y += block_size) {
        for (int x = 0; x < cols; x += block_size) {
            Rect roi(x, y, min(block_size,cols-x), min(block_size,rows-y));
            Mat region = grayImage(roi);
            vector<vector<float>> blockVector;
            for (int i = 0; i < region.rows; ++i) {
                vector<float> rowVector;
                for (int j = 0; j < region.cols; ++j) {
                    rowVector.push_back(static_cast<float>(region.at<uchar>(i, j)));
                }
                blockVector.push_back(rowVector);
            }
            if (region.rows!= block_size || region.cols != block_size){
                blockVectors.push_back(blockVector);
                continue;
            }
            vector<vector<float>> updated_blockVector = image_compression(blockVector,block_size);
            blockVectors.push_back(updated_blockVector);
        }
    }

    Mat newImage(rows, cols, CV_8U, Scalar(0));
    int index = 0;
    for (int y = 0; y < rows; y += block_size) {
        for (int x = 0; x < cols; x += block_size) {
            vector<vector<float>> blockVector = blockVectors[index++];
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
    string filename = "images/nature.jpg";
    auto start = std::chrono::high_resolution_clock::now();

    read_image(filename);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;
    return 0;
}