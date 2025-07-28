// File: hello_gpu.cu
#include <stdio.h>

/*
 * Kernel: A function that runs on the device (GPU).
 * The __global__ specifier marks it as such.
 * This kernel will be executed by many threads in parallel.
 */
__global__ void hello_from_gpu()
{
    printf("Hello, World! from the GPU!\n");
}

/*
 * Host: The main function that runs on the CPU.
 * It orchestrates the program and launches kernels on the device.
 */
int main()
{
    // 1. A message from the host CPU.
    printf("Hello from the host CPU before launching the kernel.\n");

    // 2. The Host launches the kernel on the Device.
    // We launch a "grid" of 1 block, containing 4 threads.
    hello_from_gpu<<<1, 4>>>();

    // 3. The host must wait for the device to finish its work before exiting.
    // cudaDeviceSynchronize() is a barrier that pauses the host
    // until all previously launched device tasks are complete.
    cudaDeviceSynchronize();

    // 4. A final message from the host CPU.
    printf("Kernel launch finished. Hello from the host CPU again.\n");

    return 0;
}
