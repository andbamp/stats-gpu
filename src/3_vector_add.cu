// File: vector_add.cu
#include <stdio.h>
#include <stdlib.h>  // For malloc/free
#include <stdbool.h> // For bool type

// The vectorAdd kernel from the previous section.
__global__ void vectorAdd(const float *d_A, const float *d_B, float *d_C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        d_C[idx] = d_A[idx] + d_B[idx];
    }
}

/*
 * Host: The main function that runs on the CPU.
 */
int main()
{
    // --- 1. Host-side Setup ---
    int N = 1024 * 1024;
    size_t size = N * sizeof(float);

    // Allocate host memory for vectors A, B, and C
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize host vectors
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // --- 2. Device-side Memory Allocation ---
    // Declare device pointers
    float *d_A, *d_B, *d_C;
    // Allocate memory on the GPU for each vector
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // --- 3. Copy Input Data from Host to Device ---
    printf("Copying input data from Host to Device...\n");
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // --- 4. Launch the Kernel ---
    int THREADS_PER_BLOCK = 256;
    int BLOCKS_PER_GRID = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf("Launching vectorAdd kernel...\n");
    vectorAdd<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);

    // Block host execution until the device kernel finishes
    cudaDeviceSynchronize();
    printf("Kernel execution finished.\n");

    // --- 5. Copy Result Data from Device to Host ---
    printf("Copying result data from Device to Host...\n");
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // --- 6. Verification (on the host) ---
    bool success = true;
    // Check all elements for correctness
    for (int i = 0; i < N; ++i)
    {
        if (h_C[i] != 3.0f)
        {
            printf("Error at index %d: Expected 3.0, got %f\n", i, h_C[i]);
            success = false;
            break;
        }
    }
    if (success)
    {
        // Print one of the results to show it worked
        printf("Verification successful! e.g., h_C[100] = %f\n", h_C[100]);
    }

    // --- 7. Cleanup ---
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
