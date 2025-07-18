#include <SDL2/SDL.h>
#include <iostream>
#include <cmath>
using namespace std;

#define DIM_X 1920
#define DIM_Y 1080
uint32_t bitmap[DIM_X * DIM_Y];

void setPixel(uint32_t *bitmap, int x, int y, uint8_t R, uint8_t G, uint8_t B, uint8_t A)
{
    if (x >= 0 && x < DIM_X && y >= 0 && y < DIM_Y)
    {
        bitmap[y * DIM_X + x] = (A << 24) | (R << 16) | (G << 8) | B;
    }
}

void drawRipple(uint32_t *bitmap, int ticks)
{
    const float cx = DIM_X / 2.0f;
    const float cy = DIM_Y / 2.0f;

    for (int y = 0; y < DIM_Y; y++)
    {
        for (int x = 0; x < DIM_X; x++)
        {
            float fx = x - cx;
            float fy = y - cy;
            float d = sqrtf(fx * fx + fy * fy);

            // More waves, nested cos/sin
            float ripple = cosf(d / 10.0f - ticks / 7.0f)
                         + 0.5f * sinf(d / 5.0f - ticks / 3.5f)
                         + 0.25f * cosf(d / 2.0f - ticks / 1.2f)
                         + 0.15f * sinf(logf(d + 1.0f) + ticks / 10.0f);

            // Simulate lighting: compute gradient
            float grad_x = cosf((fx + 1) / 10.0f - ticks / 7.0f) - cosf((fx - 1) / 10.0f - ticks / 7.0f);
            float grad_y = cosf((fy + 1) / 10.0f - ticks / 7.0f) - cosf((fy - 1) / 10.0f - ticks / 7.0f);

            float norm = sqrtf(grad_x * grad_x + grad_y * grad_y + 1);
            float light = (grad_x + grad_y + 1.0f) / norm;

            ripple *= light;

            // Fake blur based on distance from center (slower falloff)
            float falloff = expf(-d / 300.0f);

            int intensity = (int)(128.0f + 127.0f * ripple * falloff);

            if (intensity < 0) intensity = 0;
            if (intensity > 255) intensity = 255;

            setPixel(bitmap, x, y, intensity, intensity, intensity, 255);
        }
    }
}

int main(void)
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        cerr << "SDL_Init Error: " << SDL_GetError() << endl;
        return 1;
    }

    SDL_Window *win = SDL_CreateWindow("Ripple Effect",
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
    bool running = true;
    int ticks = 0;

    while (running)
    {
        SDL_Event e;
        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_QUIT)
                running = false;
        }

        // Draw new ripple frame
        drawRipple(bitmap, ticks++);

        // Refresh screen
        SDL_BlitSurface(image, NULL, screen, NULL);
        SDL_UpdateWindowSurface(win);

        // Cap frame rate (≈ 60 FPS)
        SDL_Delay(16);
    }

    SDL_FreeSurface(image);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
