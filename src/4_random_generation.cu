// File: random_generation.cu
#include <stdio.h>
#include <stdlib.h>
#include <time.h>          // For time()
#include <curand_kernel.h> // Required for cuRAND device functions

/*
 * Kernel 1: Initializes the cuRAND state for each thread.
 * Each thread gets a unique state based on its ID and a seed.
 */
__global__ void setup_kernel(curandState_t *states, unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialize the generator state for this thread
    curand_init(seed,          // The seed for the generator
                idx,           // A unique sequence number for each thread
                0,             // A 0 offset
                &states[idx]); // The address of the state to initialize
}

/*
 * Kernel 2: Uses the initialized states to generate random numbers.
 */
__global__ void generate_kernel(float *output, curandState_t *states, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        // Copy the state from global memory to a register for this thread
        curandState_t localState = states[idx];

        // Generate a random float between 0.0 and 1.0
        output[idx] = curand_uniform(&localState);

        // Copy the updated state back to global memory
        states[idx] = localState;
    }
}

// The main host function
int main()
{
    int N = 1024;
    size_t size = N * sizeof(float);

    // --- 1. Host-side Setup ---
    float *h_output = (float *)malloc(size);

    // --- 2. Device-side Memory Allocation ---
    float *d_output;
    curandState_t *d_states;
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_states, N * sizeof(curandState_t));

    // --- 3. Kernel Configuration ---
    int THREADS_PER_BLOCK = 256;
    int BLOCKS_PER_GRID = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // --- 4. Launch Setup Kernel ---
    // Use the current time as a seed for the random number generator
    setup_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_states, time(NULL));

    // --- 5. Launch Generation Kernel ---
    generate_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_output, d_states, N);
    cudaDeviceSynchronize();

    // --- 6. Copy results back to host ---
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // --- 7. Verification ---
    printf("Generated random numbers (first 10):\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("h_output[%d] = %f\n", i, h_output[i]);
    }

    // --- 8. Cleanup ---
    cudaFree(d_states);
    cudaFree(d_output);
    free(h_output);

    return 0;
}
