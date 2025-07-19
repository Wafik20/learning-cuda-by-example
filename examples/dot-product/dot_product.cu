#include <iostream>
#include <vector>
#include <chrono>
#include <random>
using namespace std;

#define CUDA_CHECK(call)                                                                                         \
    do                                                                                                           \
    {                                                                                                            \
        cudaError_t err = call;                                                                                  \
        if (err != cudaSuccess)                                                                                  \
        {                                                                                                        \
            cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << endl; \
            exit(EXIT_FAILURE);                                                                                  \
        }                                                                                                        \
    } while (0);

#define imin(a, b) (a < b ? a : b)

#define N 50'000'000                                                              // 50 million elements, adjust as needed
#define THREADS_PER_BLOCK 256                                                     // divisible by 32 for warp size
#define BLOCKS_PER_GRID imin(32, (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) // block_per_grid = min(32, ceil(N / THREADS_PER_BLOCK))

__global__ void dot_product(const float *a, const float *b, int n, float *c)
{
    __shared__ float cache[THREADS_PER_BLOCK];

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // flatten 2D grid to 1D
    int tid = threadIdx.x;
    float temp = 0.0f;

    // A grid-stride loop to cover all indices i ∈ [0, n)
    while (idx < n)
    {
        temp += a[idx] * b[idx];
        idx += blockDim.x * gridDim.x;
    }

    cache[tid] = temp;
    __syncthreads();

    // prefix sum reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            cache[tid] += cache[tid + s];
        __syncthreads();
    }

    // Thread 0 of each block writes block sum to global memory
    if (tid == 0)
        c[blockIdx.x] = cache[0];
}

int main()
{
    float *a = new float[N];
    float *b = new float[N];

    // Fill arrays with random float data
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(1.0f, 100.0f);

    for (int i = 0; i < N; ++i)
    {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }

    float *c = new float[BLOCKS_PER_GRID];
    float *dev_c, *dev_a, *dev_b;

    CUDA_CHECK(cudaMalloc((void **)&dev_c, BLOCKS_PER_GRID * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_b, N * sizeof(float)));

    CUDA_CHECK(cudaMemset(dev_c, 0, BLOCKS_PER_GRID * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

    dot_product<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_a, dev_b, N, dev_c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(c, dev_c, BLOCKS_PER_GRID * sizeof(float), cudaMemcpyDeviceToHost));

    // Final reduction on the host
    float result = 0.0f;
    for (int i = 0; i < BLOCKS_PER_GRID; ++i)
    {
        result += c[i];
    }

    cout << "Dot product result: " << result << endl;

    // Free device memory
    CUDA_CHECK(cudaFree(dev_c));
    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));
    // Free host memory
    delete[] c;
    delete[] a;
    delete[] b;

    return 0;
}
