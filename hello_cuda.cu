#include <stdio.h>

__global__ void hello_from_gpu() {
    printf("Hello from GPU thread!\n");
}

int main() {
    // Launch 1 block of 1 thread
    hello_from_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
