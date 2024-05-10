#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv4/opencv2/opencv.hpp>
    
using namespace std;



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


int main(){
    int N = 8;
    vector<vector<double>> M(N, vector<double>(N,0));
    vector<vector<double>> T =  dct(N);
    vector<vector<double>> T_tran = transpose(T,N);
    vector<vector<double>> D = muls(M,T,T_tran,N);
    print(T,N);
    return 0;
}