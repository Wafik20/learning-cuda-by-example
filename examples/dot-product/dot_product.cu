#include <iostream>
#include <random>
#include <cuda_runtime.h>
using namespace std;

#define THREADS_PER_BLOCK 256

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void dot_product_kernel(const float *__restrict__ a, const float *__restrict__ b, int n, float *__restrict__ result) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
        sum += a[i] * b[i];

    // Reduce within warp
    sum = warpReduceSum(sum);

    // First thread of each warp writes to shared memory
    constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;
    __shared__ float shared[WARPS_PER_BLOCK];

    if (threadIdx.x % warpSize == 0)
        shared[threadIdx.x / warpSize] = sum;

    __syncthreads();

    // Final reduction by first warp
    if (threadIdx.x < THREADS_PER_BLOCK / warpSize) {
        sum = shared[threadIdx.x];
        sum = warpReduceSum(sum);
    }

    if (threadIdx.x == 0)
        atomicAdd(result, sum);
}

float compute_dot_product(const float *a, const float *b, int n) {
    float *dev_a, *dev_b, *dev_result;
    float result = 0.0f;

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&dev_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_result, sizeof(float)));
    CUDA_CHECK(cudaMemset(dev_result, 0, sizeof(float)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

    // Tune number of blocks
    int blocks = min(1024, (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // Launch kernel
    dot_product_kernel<<<blocks, THREADS_PER_BLOCK>>>(dev_a, dev_b, n, dev_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(&result, dev_result, sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);

    return result;
}

int main() {
    int N = 50'000'000;

    // Use pinned memory for faster transfer
    float *a, *b;
    CUDA_CHECK(cudaMallocHost(&a, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&b, N * sizeof(float)));

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(1.0f, 100.0f);

    for (int i = 0; i < N; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }

    // Measure time
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    float result = compute_dot_product(a, b, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    cout << "Dot product: " << result << endl;
    cout << "Time: " << milliseconds << " ms" << endl;

    CUDA_CHECK(cudaFreeHost(a));
    CUDA_CHECK(cudaFreeHost(b));
    return 0;
}
