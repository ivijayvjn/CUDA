
//including the library files
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

//defining elelment size and number of threads

#define N 1024
#define BLOCK_SIZE 256


//GPU kernel

__global__ void vector_add(float *a , float *b , float *c , int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i]; 
    }
    
}

//Memory allocation

int main() {

    float *h_a ; 
    float *h_b ; 
    float *h_c ;
    float *d_a ;
    float *d_b ; 
    float *d_c ;
//bytes needed
    size_t size = N * sizeof(float);
//host side memory allocation
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
//initializing the vector
    for (int i = 0; i < N; i++)
    {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
//device side memory allocation
    cudaMalloc(&d_a , size);    
    cudaMalloc(&d_b , size);    
    cudaMalloc(&d_c , size);

//move data from host to device 
    cudaMemcpy(d_a , h_a , size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_b , h_b , size , cudaMemcpyHostToDevice);

//number of blocks
    int num_blocks = (N + BLOCK_SIZE - 1 ) / BLOCK_SIZE;

//launch kernel
    vector_add<<<num_blocks, BLOCK_SIZE>>>(d_a , d_b , d_c , N);

//let CPU wait until GPU finishes 

    cudaDeviceSynchronize();

//Move result from device to host

    cudaMemcpy(h_c , d_c , size , cudaMemcpyDeviceToHost);

//print few results 

    printf("h_c[0] = %f \n" , h_c[0]);
    printf("h_c[10] = %f \n" , h_c[10]);

//freeing the memory

    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
      
    
}
