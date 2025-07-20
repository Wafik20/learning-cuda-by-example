#include <SDL2/SDL.h>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define DIM_X 1920
#define DIM_Y 1080
#define PI 3.1415926535897932f

uint32_t bitmap[DIM_X * DIM_Y];

__device__ void setPixel(uint32_t *bitmap, int x, int y, uint8_t color)
{
    if (x >= 0 && x < DIM_X && y >= 0 && y < DIM_Y)
    {
        bitmap[y * DIM_X + x] = (255 << 24) | (color << 16) | (color << 8) | color; // grayscale
    }
}

__device__ void setPixelRGB(uint32_t *bitmap, int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
    if (x >= 0 && x < DIM_X && y >= 0 && y < DIM_Y)
    {
        bitmap[y * DIM_X + x] = (255 << 24) | (r << 16) | (g << 8) | b;
    }
}

__device__ void valueToRainbow(float value, uint8_t &r, uint8_t &g, uint8_t &b)
{
    float hue = value * 6.0f;
    float c = 255.0f;
    int h = (int)hue;
    float f = hue - h;
    uint8_t p = 0;
    uint8_t q = static_cast<uint8_t>(c * (1 - f));
    uint8_t t = static_cast<uint8_t>(c * f);

    switch (h % 6)
    {
    case 0: r = c; g = t; b = p; break;
    case 1: r = q; g = c; b = p; break;
    case 2: r = p; g = c; b = t; break;
    case 3: r = p; g = q; b = c; break;
    case 4: r = t; g = p; b = c; break;
    case 5: r = c; g = p; b = q; break;
    }
}

__global__ void draw_grid(uint32_t *bitmap, float zoom, float offsetX, float offsetY, float time)
{
    int globalX = threadIdx.x + blockIdx.x * blockDim.x;
    int globalY = threadIdx.y + blockIdx.y * blockDim.y;

    if (globalX >= DIM_X || globalY >= DIM_Y)
        return;

    __shared__ float tile[16][16];

    float normX = (globalX + offsetX - DIM_X / 2.0f) / zoom;
    float normY = (globalY + offsetY - DIM_Y / 2.0f) / zoom;

    float value = sinf(normX * 0.5f + time) * sinf(normY * 0.5f + time);
    value = (value + 1.0f) / 2.0f;

    // Store into shared memory tile
    tile[threadIdx.y][threadIdx.x] = value;

    __syncthreads(); // Wait for all threads to write

    // Read from transposed shared memory position
    float transposed_value = tile[threadIdx.x][threadIdx.y];

    uint8_t r, g, b;
    valueToRainbow(transposed_value, r, g, b);
    setPixelRGB(bitmap, globalX, globalY, r, g, b);
}

int main()
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        cerr << "SDL_Init Error: " << SDL_GetError() << endl;
        return 1;
    }

    SDL_Window *win = SDL_CreateWindow("CUDA Transposed Tile Effect",
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

    uint32_t *d_bitmap;
    cudaMalloc(&d_bitmap, DIM_X * DIM_Y * sizeof(uint32_t));

    float zoom = 10.0f;
    float offsetX = 0.0f, offsetY = 0.0f;
    float time = 0.0f;

    bool running = true;
    while (running)
    {
        SDL_Event e;
        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_QUIT)
                running = false;
            else if (e.type == SDL_KEYDOWN)
            {
                switch (e.key.keysym.sym)
                {
                    case SDLK_UP:    offsetY -= 50; break;
                    case SDLK_DOWN:  offsetY += 50; break;
                    case SDLK_LEFT:  offsetX -= 50; break;
                    case SDLK_RIGHT: offsetX += 50; break;
                    case SDLK_EQUALS: zoom *= 1.1f; break;
                    case SDLK_MINUS:  zoom /= 1.1f; break;
                }
            }
        }

        dim3 threads(16, 16);
        dim3 blocks((DIM_X + 15) / 16, (DIM_Y + 15) / 16);
        draw_grid<<<blocks, threads>>>(d_bitmap, zoom, offsetX, offsetY, time);
        cudaDeviceSynchronize();

        cudaMemcpy(bitmap, d_bitmap, DIM_X * DIM_Y * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        SDL_BlitSurface(image, NULL, screen, NULL);
        SDL_UpdateWindowSurface(win);

        time += 0.01f; // animate
    }

    cudaFree(d_bitmap);
    SDL_FreeSurface(image);
    SDL_DestroyWindow(win);
    SDL_Quit();

    return 0;
}
