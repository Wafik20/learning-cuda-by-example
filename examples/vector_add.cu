#include <stdio.h>
#include <stdlib.h>

// Error checking wrapper for CUDA calls
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


/**
 * Kernel: Corrected for proper parallel execution.
 * Each thread now calculates its own unique index `i` and performs one addition.
 * We also add a bounds check to ensure we don't write past the end of the arrays.
 */
__global__ void vec_add(int *c, int *a, int *b, int n) {
    // Calculate the global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check: Since we may launch more threads than n,
    // ensure this thread is within the bounds of the array.
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

/**
 * Host Function: Modified to launch the kernel efficiently.
 */
int* parallel_vector_add(int *a, int *b, int n) {
    int *dev_c, *dev_a, *dev_b;

    // Allocate memory on the GPU device
    gpuErrchk(cudaMalloc((void**)&dev_c, n * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_a, n * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_b, n * sizeof(int)));

    // Copy host arrays to the device
    gpuErrchk(cudaMemcpy(dev_a, a, n * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_b, b, n * sizeof(int), cudaMemcpyHostToDevice));

    // --- KERNEL LAUNCH CORRECTION ---
    // Define the number of threads per block (a multiple of 32 is good, 256 is common)
    int threadsPerBlock = 256;
    // Calculate the number of blocks needed in the grid to cover all 'n' elements
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    printf("GPU Launch Config: %d blocks, %d threads/block\n", blocksPerGrid, threadsPerBlock);

    // Launch the kernel with the new, efficient configuration
    vec_add<<<blocksPerGrid, threadsPerBlock>>>(dev_c, dev_a, dev_b, n);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Allocate memory on the host for the result
    int *c = (int*)malloc(n * sizeof(int));
    if (c == NULL) {
        fprintf(stderr, "Failed to allocate host memory for result!\n");
        exit(1);
    }
    
    // Copy the result from the device back to the host
    gpuErrchk(cudaMemcpy(c, dev_c, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Free GPU memory
    gpuErrchk(cudaFree(dev_c));
    gpuErrchk(cudaFree(dev_a));
    gpuErrchk(cudaFree(dev_b));
    
    return c;
}

/**
 * Main Function: Modified to use large vectors from the heap.
 */
int main(void) {
    // Define a large vector size, e.g., 20 million elements
    const int N = 20 * 1000 * 1000;
    printf("Vector size (N): %d\n", N);

    // --- HEAP ALLOCATION ---
    // Allocate host vectors on the heap to avoid stack overflow
    int *a = (int*)malloc(N * sizeof(int));
    int *b = (int*)malloc(N * sizeof(int));

    if (a == NULL || b == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        return 1;
    }

    // Initialize the large vectors with some data
    for(int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = 10;
    }

    // Run the parallel addition
    int *c = parallel_vector_add(a, b, N);

    // Verification: Check a few values instead of printing millions of lines
    printf("\n--- Verification ---\n");
    printf("First element:  %d + %d = %d\n", a[0], b[0], c[0]);
    printf("Middle element: %d + %d = %d\n", a[N/2], b[N/2], c[N/2]);
    printf("Last element:   %d + %d = %d\n", a[N-1], b[N-1], c[N-1]);

    // --- CLEANUP ---
    // Free all heap-allocated memory
    free(a);
    free(b);
    free(c);

    return 0;
}