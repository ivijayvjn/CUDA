#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<iostream>
using namespace std;

#define M 2
#define K 3
#define N 2

int main() {
    float h_A[M * K] = {
        20 , 30 , 40 ,
        50 , 60 , 70
    };
    float h_B[K * N] = {
        25 , 35 , 45 , 
        55 , 65 , 75
    };
    float h_C[M * N];
    float *d_A , *d_B , *d_C;

    cudaMalloc(&d_A , M * K * sizeof(float));
    cudaMalloc(&d_B , K * N * sizeof(float));
    cudaMalloc(&d_C , M * N * sizeof(float));

    cudaMemcpy(d_A , h_A , M * K * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_B , h_B , K * N * sizeof(float) , cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N , N , M ,K , &alpha , d_B , N , d_A , K , &beta , d_C , N);

    cudaMemcpy(h_C , d_C , M * N * sizeof(float) , cudaMemcpyDeviceToHost);

    cout << "Result Matrix C :\n";

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << h_C[i * N + j] << " ";
        }
        cout << endl;
    }
    cublasDestroy(handle);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
