#include <iostream> 

__global__ void add(int a, int b, int *c){
    *c = a + b;
}

int main( void ){
    int c;

    // Pointer to be allocated on the GPU
    int *dev_c;

    // Allocate memory on the GPU for an integer
    cudaMalloc( (void**)&dev_c, sizeof(int) );

    // Launch the kernel with 1 block and 1 thread
    // The kernel will add 2 and 7, storing the result in dev_c
    add<<< 1, 1 >>>(2, 7, dev_c);

    // Copy the result back from the GPU to the CPU
    // The result will be stored in the variable c
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    printf( "2 + 7 = %d\n", c );

    // Free the memory allocated on the GPU
    cudaFree( dev_c );

    return 0;
}