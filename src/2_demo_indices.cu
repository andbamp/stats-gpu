// File: demo_indices.cu
#include <stdio.h>
#include <cuda_runtime.h>

/*
 * Kernel to demonstrate thread indexing.
 * Each thread calculates its global index, and the first thread
 * in each block prints its identity to show the hierarchy.
 */
__global__ void demoIndices(int N)
{

    // 1. Calculate the unique global index for this thread.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Bounds check to ensure we don't do work for padded threads.
    if (idx < N)
    {
        // 3. To keep output clean, we will only have the FIRST thread
        // of each block print its information. This confirms that
        // multiple blocks are being launched and are aware of their IDs.
        if (threadIdx.x == 0)
        {
            printf("GPU: Block ID = %d, Thread ID = %d -> Calculated Global Index = %d\n",
                   blockIdx.x, threadIdx.x, idx);
        }
    }
}

/*
 * Host: The main function that runs on the CPU.
 */
int main()
{
    // Define the size of our conceptual data.
    // We'll use a small N that is not a perfect multiple of THREADS_PER_BLOCK
    // to show that the bounds check and grid calculation work correctly.
    int N = 500;

    // Define the number of threads per block. 256 is a common, efficient choice.
    int THREADS_PER_BLOCK = 256;

    // Calculate the number of blocks needed in the grid, using the
    // integer arithmetic trick for ceiling division.
    // For N=500 and Threads=256, this will be ceil(500/256) = 2 blocks.
    int BLOCKS_PER_GRID = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf("Host: Problem size N = %d\n", N);
    printf("Host: Threads per Block = %d\n", THREADS_PER_BLOCK);
    printf("Host: Calculated Blocks in Grid = %d\n\n", BLOCKS_PER_GRID);

    // Launch the kernel with our calculated grid and block dimensions.
    demoIndices<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(N);

    // Synchronize to make sure the kernel finishes and we see its output.
    cudaDeviceSynchronize();

    printf("\nHost: Kernel finished.\n");

    return 0;
}
