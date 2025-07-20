#include <SDL2/SDL.h>
#include <iostream>
#include <cuda_runtime.h>
#include <ctime>   // For time()
#include <cstdlib> // For srand(), rand()
#include <cmath>   // For sqrtf

using namespace std;

#define DIM_X 1920
#define DIM_Y 1080
#define PI 3.1415926535897932f
#define INF 2e10f
#define CUDA_CHECK(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cudaError_t err = call;                                                                                 \
        if (err != cudaSuccess)                                                                                 \
        {                                                                                                       \
            cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
    } while (0)
#define rnd(x) (x * (float)rand() / RAND_MAX)
#define SPHERES 25

struct Sphere
{
    float r, b, g;
    float radius;
    float x, y, z;
    __device__ float hit(float ox, float oy, float *n)
    {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius)
        {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            // This 'n' is effectively the normalized Z-component of the vector from sphere center to hit point
            *n = dz / radius; // Normalized depth component
            return dz + z;    // Return the absolute Z-coordinate of the hit point
        }
        return -INF;
    }
};

// Host-side bitmap for SDL
uint32_t bitmap[DIM_X * DIM_Y];
// Device-side constant memory for spheres
__constant__ Sphere spheres[SPHERES];

__device__ void setPixel(uint32_t *bitmap, int x, int y, uint8_t color)
{
    // No need for redundant bounds check here if kernel already checks
    bitmap[y * DIM_X + x] = (255 << 24) | (color << 16) | (color << 8) | color; // grayscale
}

__global__ void drawSpheres(uint32_t *d_bitmap, float zoom, float offsetX, float offsetY)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Early exit for threads outside the image bounds
    if (x >= DIM_X || y >= DIM_Y)
    {
        return;
    }

    // Apply zoom and offset to the ray origin
    // Note: The /2.0f and /zoom should be applied to the relative coordinates
    float ox = ((float)x - DIM_X / 2.0f) / zoom - offsetX;
    float oy = ((float)y - DIM_Y / 2.0f) / zoom - offsetY;


    float r = 0, g = 0, b = 0;
    float maxz = -INF; // Initialize maxz for each pixel

    for (int i = 0; i < SPHERES; i++)
    {
        float n;
        float t = spheres[i].hit(ox, oy, &n);
        if (t > maxz)
        {
            maxz = t; // IMPORTANT: Update maxz to the new closest hit
            float fscale = n; // Use 'n' for intensity scaling
            r = spheres[i].r * fscale;
            g = spheres[i].g * fscale;
            b = spheres[i].b * fscale;
        }
    }

    // Clamp color components to [0, 1] before scaling to 255
    r = fmaxf(0.0f, fminf(1.0f, r));
    g = fmaxf(0.0f, fminf(1.0f, g));
    b = fmaxf(0.0f, fminf(1.0f, b));


    uint8_t grayscale = (uint8_t)(((r + g + b) / 3.0f) * 255.0f);
    setPixel(d_bitmap, x, y, grayscale);
}

int main()
{
    srand(time(NULL)); // Seed the random number generator

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
    if (!screen)
    {
        cerr << "SDL_GetWindowSurface Error: " << SDL_GetError() << endl;
        SDL_DestroyWindow(win);
        SDL_Quit();
        return 1;
    }

    // Create a surface from the host bitmap
    SDL_Surface *image = SDL_CreateRGBSurfaceFrom(
        bitmap, DIM_X, DIM_Y, 32, DIM_X * sizeof(uint32_t),
        0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000); // RGBA masks (adjust if your SDL build is different)

    if (!image)
    {
        cerr << "SDL_CreateRGBSurfaceFrom Error: " << SDL_GetError() << endl;
        SDL_DestroyWindow(win);
        SDL_Quit();
        return 1;
    }

    // Allocate and initialize spheres on host
    Sphere *temp_s = (Sphere *)malloc(sizeof(Sphere) * SPHERES);
    if (!temp_s)
    {
        cerr << "Failed to allocate host memory for spheres!" << endl;
        SDL_FreeSurface(image);
        SDL_DestroyWindow(win);
        SDL_Quit();
        return 1;
    }

    for (int i = 0; i < SPHERES; i++)
    {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500.0f;
        temp_s[i].y = rnd(1000.0f) - 500.0f;
        temp_s[i].z = rnd(1000.0f) - 500.0f;
        temp_s[i].radius = rnd(100.0f) + 20.0f;
    }

    uint32_t *d_bitmap;

    CUDA_CHECK(cudaMalloc(&d_bitmap, DIM_X * DIM_Y * sizeof(uint32_t)));
    // Copy spheres data from host to device constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(spheres, temp_s, SPHERES * sizeof(Sphere)));
    free(temp_s); // Free host memory for spheres after copying

    float zoom = 10.0f;
    float offsetX = 0.0f, offsetY = 0.0f;
    float time_val = 0.0f; // Renamed 'time' to 'time_val' to avoid conflict with ctime's time()

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
                case SDLK_UP:
                    offsetY -= 50.0f;
                    break;
                case SDLK_DOWN:
                    offsetY += 50.0f;
                    break;
                case SDLK_LEFT:
                    offsetX -= 50.0f;
                    break;
                case SDLK_RIGHT:
                    offsetX += 50.0f;
                    break;
                case SDLK_EQUALS: // '+' key
                    zoom *= 1.1f;
                    break;
                case SDLK_MINUS: // '-' key
                    zoom /= 1.1f;
                    break;
                }
            }
        }

        dim3 grids((DIM_X + 15) / 16, (DIM_Y + 15) / 16); // Corrected grid calculation for full coverage
        dim3 threads(16, 16);

        // Pass zoom, offsetX, offsetY to the kernel
        drawSpheres<<<grids, threads>>>(d_bitmap, zoom, offsetX, offsetY);

        CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to complete

        // Copy rendered bitmap from device to host
        CUDA_CHECK(cudaMemcpy(bitmap, d_bitmap, DIM_X * DIM_Y * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        SDL_BlitSurface(image, NULL, screen, NULL);
        SDL_UpdateWindowSurface(win);

        time_val += 0.01f;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_bitmap));

    SDL_FreeSurface(image);
    SDL_DestroyWindow(win);
    SDL_Quit();

    return 0;
}