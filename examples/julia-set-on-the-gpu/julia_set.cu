#include <SDL2/SDL.h>
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define DIM_X 1920
#define DIM_Y 1080
uint32_t bitmap[DIM_X * DIM_Y];

// Complex number struct
struct cuComplex
{
    float r, i;
    __host__ __device__ cuComplex(float a, float b) : r(a), i(b) {}
    __host__ __device__ float magnitude2() const { return r * r + i * i; }
    __host__ __device__ cuComplex operator*(const cuComplex &a) const
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    __host__ __device__ cuComplex operator+(const cuComplex &a) const
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

// Set a pixel in the bitmap (GPU-side)
__device__ void setPixel(uint32_t *bitmap, int x, int y, uint8_t R, uint8_t G, uint8_t B, uint8_t A)
{
    if (x >= 0 && x < DIM_X && y >= 0 && y < DIM_Y)
    {
        bitmap[y * DIM_X + x] = (A << 24) | (R << 16) | (G << 8) | B; // ARGB
    }
}

// Julia set calculation
__device__ int julia(int x, int y, float zoom)
{
    float scale = 1.5f / zoom;
    float jx = scale * (DIM_X / 2.0f - x) / (DIM_X / 2.0f);
    float jy = scale * (DIM_Y / 2.0f - y) / (DIM_Y / 2.0f);
    cuComplex c(-0.8f, 0.156f);
    cuComplex a(jx, jy);
    int i;
    for (i = 0; i < 200; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            break;
    }
    return i;
}


// CUDA kernel to compute the Julia set
__global__ void drawJuliaKernel(float zoom, uint32_t *bitmap)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < DIM_X && y < DIM_Y)
    {
        int iterations = julia(x, y, zoom);
        uint8_t gray = (iterations == 200) ? 0 : (uint8_t)(255.0f * iterations / 200.0f);
        setPixel(bitmap, x, y, gray, gray, gray, 255);
    }
}


int main()
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        cerr << "SDL_Init Error: " << SDL_GetError() << endl;
        return 1;
    }

    SDL_Window *win = SDL_CreateWindow("Julia Set",
                                       SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                       DIM_X, DIM_Y, SDL_WINDOW_SHOWN);

    if (!win)
    {
        cerr << "SDL_CreateWindow Error: " << SDL_GetError() << endl;
        SDL_Quit();
        return 1;
    }

    SDL_Surface *screen = SDL_GetWindowSurface(win);

    SDL_Surface *image = SDL_CreateRGBSurfaceFrom(
        bitmap, DIM_X, DIM_Y, 32, DIM_X * sizeof(uint32_t),
        0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000);

    if (!image)
    {
        cerr << "SDL_CreateRGBSurfaceFrom Error: " << SDL_GetError() << endl;
        SDL_DestroyWindow(win);
        SDL_Quit();
        return 1;
    }

    SDL_SetSurfaceBlendMode(image, SDL_BLENDMODE_NONE);

    float zoom = 1.0f;
    bool running = true;

    // Allocate GPU memory once, reuse
    uint32_t *d_bitmap;
    cudaMalloc(&d_bitmap, DIM_X * DIM_Y * sizeof(uint32_t));

    while (running)
    {
        SDL_Event e;
        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_QUIT)
                running = false;
            if (e.type == SDL_KEYDOWN)
            {
                if (e.key.keysym.sym == SDLK_EQUALS || e.key.keysym.sym == SDLK_PLUS)
                    zoom *= 1.1f; // Zoom in
                if (e.key.keysym.sym == SDLK_MINUS)
                    zoom /= 1.1f; // Zoom out
            }
        }

        // Launch kernel
        dim3 blockSize(16, 16);
        dim3 gridSize((DIM_X + blockSize.x - 1) / blockSize.x,
                      (DIM_Y + blockSize.y - 1) / blockSize.y);
        drawJuliaKernel<<<gridSize, blockSize>>>(zoom, d_bitmap);
        cudaDeviceSynchronize(); // Wait for GPU to finish

        // Copy result to host
        cudaMemcpy(bitmap, d_bitmap, DIM_X * DIM_Y * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Update SDL surface
        SDL_LockSurface(image);
        SDL_UnlockSurface(image);
        SDL_BlitSurface(image, NULL, screen, NULL);
        SDL_UpdateWindowSurface(win);
    }

    cudaFree(d_bitmap);
    SDL_FreeSurface(image);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
