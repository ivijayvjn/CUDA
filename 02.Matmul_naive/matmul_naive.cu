#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


#define M 2 
#define K 3
#define N 2
#define BLOCK_SIZE 16 


__global__ void matmul_naive(float *A , float *B , float *C , int m , int k , int n ) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        float sum = 0.0f;
        for (int l = 0; l < k; l++)
        {
            sum += A[row * k + l ] * B[l * n + col];
        }
        C[row * n + col] = sum;
    }
    
}

int main() {
    float *h_A , *h_B , *h_C;
    float *d_A , *d_B , *d_C;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);

    // A = 2 by 3 matrix

    h_A[0] = 23 ; h_A[1] = 38 ; h_A[2] = 54;
    h_A[3] = 32 ; h_A[4] = 83 ; h_A[5] = 45;


    // B = 3 by 2 matrix
    h_B[0] = 43 ; h_B[1] = 65;
    h_B[2] = 56 ; h_B[3] = 47;
    h_B[4] = 76 ; h_B[5] = 87;
    
    cudaMalloc(&d_A , size_A);
    cudaMalloc(&d_B , size_B);
    cudaMalloc(&d_C , size_C);


    cudaMemcpy(d_A , h_A , size_A , cudaMemcpyHostToDevice);
    cudaMemcpy(d_B , h_B ,size_B , cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE , BLOCK_SIZE); 

    dim3 blocksPerGrid(
        (N + BLOCK_SIZE - 1 ) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1 ) / BLOCK_SIZE 
    );
    
    matmul_naive<<<blocksPerGrid , threadsPerBlock >>>(d_A , d_B , d_C , M , K , N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C , d_C , size_C , cudaMemcpyDeviceToHost);


    printf("C Matrix : \n" );
    printf("%f %f\n" , h_C[0] , h_C[1]);
    printf("%f %f\n" , h_C[2] , h_C[3]);

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
 }
