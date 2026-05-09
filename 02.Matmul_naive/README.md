# 02 - CUDA Matrix Multiplication (Naive)

## Purpose

Implemented a basic CUDA matrix multiplication kernel to understand:

- 2D thread indexing
- Matrix memory layout in flattened 1D memory
- Row and column mapping
- GPU parallel computation for matrix multiplication
- Host/device memory allocation
- CPU ↔ GPU memory transfer
- Kernel launch configuration
- Synchronization flow

---

## Kernel Logic

Each GPU thread computes one output element of the result matrix.

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

if (row < m && col < n) {

    float sum = 0.0f;

    for (int l = 0; l < k; l++) {
        sum += A[row * k + l] * B[l * n + col];
    }

    C[row * n + col] = sum;
}
```

---

## CUDA Execution Flow

```text
Host memory allocation
→ Initialize matrices
→ Device memory allocation
→ Copy host → device
→ Configure 2D grid and block dimensions
→ Launch matrix multiplication kernel
→ Synchronize
→ Copy device → host
→ Print output matrix
→ Free memory
```

---

## Execution Environment

| Component | Details |
|---|---|
| GPU | NVIDIA L4 |
| NVIDIA Driver Version | 580.126.20 |
| CUDA Runtime Version | 13.0 |
| NVCC Version | CUDA 12.6 (V12.6.77) |
| Operating Environment | Lightning AI GPU Instance |

---

## Output

```bash
C Matrix:
7221.000000 7979.000000
9444.000000 9896.000000
```

---

## GPU Execution

![GPU Execution](02.Matmul_naive/Assets/output.png)

---

## Learning Notes

- Matrix multiplication uses 2D thread indexing.
- Each GPU thread computes one output matrix element.
- Matrices are stored in flattened 1D memory.
- `row * n + col` computes the correct memory offset for output storage.
- `A[row * k + l]` traverses across a row of matrix A.
- `B[l * n + col]` traverses down a column of matrix B.
- `dim3` is used to configure 2D thread blocks and grids.
- `cudaDeviceSynchronize()` ensures GPU execution completes before CPU continues.
