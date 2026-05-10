# 03 - cuBLAS Matrix Multiplication

## Purpose

Implemented matrix multiplication using NVIDIA cuBLAS library to understand:

- cuBLAS GEMM workflow

- Optimized CUDA library usage

- Host ↔ Device memory flow

- cuBLAS handle creation

- SGEMM API structure

- Column-major behavior in cuBLAS

- Difference between handwritten CUDA kernels and optimized libraries

----------------------------------------------------------------------------------------

## cuBLAS GEMM Formula

C = alpha * AB + beta * C

For this implementation:

```cpp

alpha = 1.0f

beta  = 0.0f

which becomes C=AB

```
----------------------------------------------------------------------------------------
## CUDA/cuBLAS Execution Flow

Host matrix allocation
→ Device memory allocation
→ Host → Device copy
→ Create cuBLAS handle
→ Execute cublasSgemm()
→ Device → Host copy
→ Print output
→ Free memory
---------------------------------------------------------------------------------
## Execution Environment

| Component | Details |
|---|---|
| GPU | NVIDIA L4 |
| NVIDIA Driver Version | 580.126.20 |
| CUDA Runtime Version | 13.0 |
| NVCC Version | CUDA 12.6 |
| Operating Environment | Lightning AI GPU Instance |
---------------------------------------------------------------------------------

## Compilation

``` Bash 
nvcc mat_mul_cublas.cu -o matmul_cublas -lcublas
./matmul_cublas

```
lcublas --> link cublas 
----------------------------------------------------------------------------------

## Ouput 

``` Bash 

Result Matrix C :
4450 5350
8500 10300

```

![GPU-Execution](a.CUBLAS/Assets/output.png)
------------------------------------------------------------------------------------

## Key Learning Notes

* cuBLAS internally launches highly optimized GEMM kernels.
* Manual CUDA indexing logic is abstracted away by the library.
* cuBLAS assumes column-major ordering internally.
* cublasSgemm() performs FP32 matrix multiplication.
* alpha and beta control GEMM scaling behavior.
* CUDA memory management flow remains the same even when using libraries.

--------------------------------------------------------------------------------------
