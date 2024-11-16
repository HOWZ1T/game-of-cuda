#include <iostream>
#include <string>
#include <algorithm>

#include <GLFW/glfw3.h>  // defines openGL types
#include <raylib.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_gl_interop.h>
#include <surface_indirect_functions.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void
golPopulate(unsigned int *grid, const float *randomVals, int width, int height,
            float threshold) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        unsigned int index = y * width + x;
        grid[index] = (randomVals[index] < threshold);
    }
}

__global__ void
gol(cudaSurfaceObject_t outSurf, const unsigned int *gridA, unsigned int *gridB, int width,
    int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        // simulate 1 step of game of life for this cell

        // calculate neighbourhood score
        unsigned int neighbourHoodSum = 0;

        // discrete index performed for efficiency
        // TODO: further efficiency gains using dynamic parallelism here?
        // top row
        neighbourHoodSum += gridA[((y - 1 + height) % height) * width +
                                  ((x - 1 + width) % width)];
        neighbourHoodSum += gridA[((y - 1 + height) % height) * width +
                                  ((x + width) % width)];
        neighbourHoodSum += gridA[((y - 1 + height) % height) * width +
                                  ((x + 1 + width) % width)];

        // sides
        neighbourHoodSum += gridA[((y + height) % height) * width +
                                  ((x - 1 + width) % width)];
        neighbourHoodSum += gridA[((y + height) % height) * width +
                                  ((x + 1 + width) % width)];

        // bottom row
        neighbourHoodSum += gridA[((y + 1 + height) % height) * width +
                                  ((x - 1 + width) % width)];
        neighbourHoodSum += gridA[((y + 1 + height) % height) * width +
                                  ((x + width) % width)];
        neighbourHoodSum += gridA[((y + 1 + height) % height) * width +
                                  ((x + 1 + width) % width)];

        // apply rules
        unsigned int newCellState = 0;
        unsigned int index = y * width + x;
        if (gridA[index] == 1) {
            // if cell alive and has 2 or 3 neighbours, it stays alive (rule 2)
            if (neighbourHoodSum == 2 || neighbourHoodSum == 3) {
                newCellState = 1;
            }
            // else it dies (rules 1 and 3)
        } else {
            // if cell dead and has 3 neighbours, it becomes alive (rule 4)
            if (neighbourHoodSum == 3) {
                newCellState = 1;
            }
        }

        // update gridB with new cell state
        gridB[index] = newCellState;

        // render to surface
        // adjust for difference between cuda and opengl coordinate systems
        int flippedY = height - int(y) - 1;

        // assign black or white based on cell state
        uchar4 col;
        if (newCellState == 1) {
            col.x = 0;
            col.y = 254;
            col.z = 252;
            col.w = 255;
        } else {
            col.x = 0;
            col.y = 0;
            col.z = 0;
            col.w = 255;
        }

        surf2Dwrite(col, outSurf, int(x * sizeof(uchar4)), flippedY);
    }
}

__global__ void
gaussianBlur(cudaSurfaceObject_t inSurf, cudaSurfaceObject_t outSurf, const float *kernel,
             int kernelWidth, int kernelHeight, int width, int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // read pixel
        uchar4 col;
        surf2Dread(&col, inSurf, int(x * sizeof(uchar4)), int(y));

        // apply kernel
        float4 newCol = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int ix = -kernelWidth / 2; ix <= kernelWidth / 2; ix++) {
            for (int iy = -kernelHeight / 2; iy <= kernelHeight / 2; iy++) {
                // handle edges by wrapping
                int nx = (int(x) + ix + width) % width;
                int ny = (int(y) + iy + height) % height;

                // read neighbour pixel
                uchar4 neighbourCol;
                surf2Dread(&neighbourCol, inSurf, int(nx * sizeof(uchar4)), ny);

                // apply kernel
                float kernelVal = kernel[(iy + kernelHeight / 2) * kernelWidth +
                                         (ix + kernelWidth / 2)];
                newCol.x += float(neighbourCol.x) * kernelVal;
                newCol.y += float(neighbourCol.y) * kernelVal;
                newCol.z += float(neighbourCol.z) * kernelVal;
                newCol.w += float(neighbourCol.w) * kernelVal;
            }
        }
        uchar4 newPixel = make_uchar4(__float2uint_rz(newCol.x),
                                      __float2uint_rz(newCol.y),
                                      __float2uint_rz(newCol.z),
                                      __float2uint_rz(newCol.w));

        // write pixel
        surf2Dwrite(newPixel, outSurf, int(x * sizeof(uchar4)), int(y));
    }
}

__global__ void
lumaFilter(cudaSurfaceObject_t inSurf, cudaSurfaceObject_t outSurf, int width, int height,
           float threshold) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // read pixel
        uchar4 col;
        surf2Dread(&col, inSurf, int(x * sizeof(uchar4)), int(y));

        // calculate luma
        float luma =
                0.299f * float(col.x) + 0.587f * float(col.y) + 0.114f * float(col.z);

        // apply threshold
        if (luma > threshold) {
            // write pixel
            surf2Dwrite(col, outSurf, int(x * sizeof(uchar4)), int(y));
        } else {
            surf2Dwrite(make_uchar4(0, 0, 0, 0), outSurf, int(x * sizeof(uchar4)),
                        int(y));
        }
    }
}

__global__ void
combine(cudaSurfaceObject_t inSurf, cudaSurfaceObject_t outSurf, int width, int height,
        float exposure, float gamma) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // read pixel
        uchar4 bloomCol, inCol;
        surf2Dread(&bloomCol, inSurf, int(x * sizeof(uchar4)), int(y));
        surf2Dread(&inCol, outSurf, int(x * sizeof(uchar4)), int(y));

        // tone mapping
        float3 hdrColor = make_float3(float(inCol.x), float(inCol.y), float(inCol.z));
        float3 bloomColor = make_float3(float(bloomCol.x), float(bloomCol.y),
                                        float(bloomCol.z));

        // normalize to 0 - 1
        hdrColor.x /= 255.0f;
        hdrColor.y /= 255.0f;
        hdrColor.z /= 255.0f;

        bloomColor.x /= 255.0f;
        bloomColor.y /= 255.0f;
        bloomColor.z /= 255.0f;

        hdrColor.x += bloomColor.x;
        hdrColor.y += bloomColor.y;
        hdrColor.z += bloomColor.z;

        float3 result = make_float3(
                1.0f - exp(-hdrColor.x * exposure),
                1.0f - exp(-hdrColor.y * exposure),
                1.0f - exp(-hdrColor.z * exposure)
        );

        result.x = pow(result.x, 1.0f / gamma);
        result.y = pow(result.y, 1.0f / gamma);
        result.z = pow(result.z, 1.0f / gamma);

        // denormalize to 0 - 255
        result.x *= 255.0f;
        result.y *= 255.0f;
        result.z *= 255.0f;

        surf2Dwrite(make_uchar4(__float2uint_rz(result.x), __float2uint_rz(result.y),
                                __float2uint_rz(result.z), inCol.w), outSurf,
                    int(x * sizeof(uchar4)), int(y));
    }
}

__global__ void
surf2SurfWrite(cudaSurfaceObject_t srcSurf, cudaSurfaceObject_t destSurf, int width,
               int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // read pixel
        uchar4 col;
        surf2Dread(&col, srcSurf, int(x * sizeof(uchar4)), int(y));

        // write pixel
        surf2Dwrite(col, destSurf, int(x * sizeof(uchar4)), int(y));
    }
}

int main(int argc, char *argv[]) {
    // ensure the screen size and grid size is given as arguments
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <screen_width> <screen_height> <grid_width> <grid_height>"
                  << std::endl;
        return 1;
    }

    // parse arguments
    const int screenWidth = std::stoi(argv[1]);
    const int screenHeight = std::stoi(argv[2]);
    const int gridWidth = std::stoi(argv[3]);
    const int gridHeight = std::stoi(argv[4]);

    InitWindow(screenWidth, screenHeight, "Game of Cuda - A Song of Blocks and Threads");
    SetTargetFPS(60);

    // TODO check for cuda device and openGL compatibility

    std::cout << "startup with configuration: " << screenWidth << "x" << screenHeight
              << " screen, " << gridWidth << "x" << gridHeight << " grid" << std::endl;

    // initialize camera
    Camera2D cam = {0};
    cam.target = Vector2{float(gridWidth) / 2.0f, float(gridHeight) / 2.0f};
    cam.offset = Vector2{float(screenWidth) / 2.0f, float(screenHeight) / 2.0f};
    cam.rotation = 0.0f;
    cam.zoom = 1.0f;

    // create blank raylib image as output for processed texture
    Image rayImgOut = GenImageColor(gridWidth, gridHeight, BLANK);
    ImageFormat(&rayImgOut, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    Texture2D rayTexOut = LoadTextureFromImage(rayImgOut);
    UnloadImage(rayImgOut);

#ifdef GAME_OF_CUDA__DEBUG
    Image rayImgOut2 = GenImageColor(gridWidth, gridHeight, BLANK);
    ImageFormat(&rayImgOut2, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    Texture2D rayTexOut2 = LoadTextureFromImage(rayImgOut2);
    UnloadImage(rayImgOut2);
#endif

    // TODO ensure optimized block and grid size
    dim3 blockSize(32, 32);
    dim3 gridSize((rayTexOut.width + blockSize.x - 1) / blockSize.x,
                  (rayTexOut.height + blockSize.y - 1) / blockSize.y);

    // register openGL texture 2D with cuda
#ifdef GAME_OF_CUDA__DEBUG
    cudaGraphicsResource* cudaResources[2];
    gpuErrchk(cudaGraphicsGLRegisterImage(&cudaResources[0], rayTexOut.id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore))
    gpuErrchk(cudaGraphicsGLRegisterImage(&cudaResources[1], rayTexOut2.id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore))
#else
    cudaGraphicsResource *cudaResources[1];
    gpuErrchk(cudaGraphicsGLRegisterImage(&cudaResources[0], rayTexOut.id, GL_TEXTURE_2D,
                                          cudaGraphicsRegisterFlagsSurfaceLoadStore))
#endif

    float *kernel;
    int kernelSize = 11;
    gpuErrchk(cudaMalloc(&kernel, kernelSize * kernelSize * sizeof(float)))
    float kernelH[11][11] = {
            {1.0f,   10.0f,   45.0f,    120.0f,   210.0f,   252.0f,   210.0f,   120.0f,   45.0f,    10.0f,   1.0f},
            {10.0f,  100.0f,  450.0f,   1200.0f,  2100.0f,  2520.0f,  2100.0f,  1200.0f,  450.0f,   100.0f,  10.0f},
            {45.0f,  450.0f,  2025.0f,  5400.0f,  9450.0f,  11340.0f, 9450.0f,  5400.0f,  2025.0f,  450.0f,  45.0f},
            {120.0f, 1200.0f, 5400.0f,  14400.0f, 25200.0f, 30240.0f, 25200.0f, 14400.0f, 5400.0f,  1200.0f, 120.0f},
            {210.0f, 2100.0f, 9450.0f,  25200.0f, 44100.0f, 52920.0f, 44100.0f, 25200.0f, 9450.0f,  2100.0f, 210.0f},
            {252.0f, 2520.0f, 11340.0f, 30240.0f, 52920.0f, 63504.0f, 52920.0f, 30240.0f, 11340.0f, 2520.0f, 252.0f},
            {210.0f, 2100.0f, 9450.0f,  25200.0f, 44100.0f, 52920.0f, 44100.0f, 25200.0f, 9450.0f,  2100.0f, 210.0f},
            {120.0f, 1200.0f, 5400.0f,  14400.0f, 25200.0f, 30240.0f, 25200.0f, 14400.0f, 5400.0f,  1200.0f, 120.0f},
            {45.0f,  450.0f,  2025.0f,  5400.0f,  9450.0f,  11340.0f, 9450.0f,  5400.0f,  2025.0f,  450.0f,  45.0f},
            {10.0f,  100.0f,  450.0f,   1200.0f,  2100.0f,  2520.0f,  2100.0f,  1200.0f,  450.0f,   100.0f,  10.0f},
            {1.0f,   10.0f,   45.0f,    120.0f,   210.0f,   252.0f,   210.0f,   120.0f,   45.0f,    10.0f,   1.0f}
    };

    // normalize kernel
    float kernel_sum = 0.0f;
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel_sum += kernelH[i][j];
        }
    }

    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernelH[i][j] /= kernel_sum;
        }
    }
    gpuErrchk(cudaMemcpy(kernel, kernelH, kernelSize * kernelSize * sizeof(float),
                         cudaMemcpyHostToDevice))

    // holds conway game of life grid
    unsigned int *devGridA, *devGridB;
    gpuErrchk(cudaMalloc(&devGridA,
                         rayTexOut.width * rayTexOut.height * sizeof(unsigned int)))
    gpuErrchk(cudaMalloc(&devGridB,
                         rayTexOut.width * rayTexOut.height * sizeof(unsigned int)))

    // generate random initial state
    curandGenerator_t gen;

    // allocate space for random state on device
    float *devRandomData;
    gpuErrchk(
            cudaMalloc((void **) &devRandomData, gridWidth * gridHeight * sizeof(float)))

    // create the pseudo-random number generator
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    // set the seed for the random number generator
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // generate gridWidth * gridHeight random floats
    curandGenerateNormal(gen, devRandomData, gridWidth * gridHeight, 0.5f, 1.2f);

    // destroy the random number generator
    curandDestroyGenerator(gen);

    // populate grid with random data
    golPopulate<<<gridSize, blockSize>>>(devGridA, devRandomData, rayTexOut.width,
                                         rayTexOut.height, 0.5f);
    gpuErrchk(cudaFree(devRandomData)) // free random data

    int blurPasses = 1;
    int threshold = 150;
    float exposure = 1.0f;
    float gamma = 2.2f;
    while (!WindowShouldClose()) {
        Vector2 mouseDelta = GetMouseDelta();
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            cam.target.x -= mouseDelta.x / cam.zoom;
            cam.target.y -= mouseDelta.y / cam.zoom;
        }

        float mouseScroll = GetMouseWheelMove();
        if (mouseScroll != 0) {
            cam.zoom += mouseScroll * 0.5f;
            cam.zoom = std::min(100.0f, std::max(0.1f, cam.zoom));
        }

        if (IsKeyPressed(KEY_UP)) {
            blurPasses++;
        }

        if (IsKeyPressed(KEY_DOWN)) {
            blurPasses--;
            blurPasses = std::max(1, blurPasses);
        }

        if (IsKeyPressed(KEY_LEFT)) {
            threshold -= 10;
            threshold = std::max(0, threshold);
        }

        if (IsKeyPressed(KEY_RIGHT)) {
            threshold += 10;
            threshold = std::min(255, threshold);
        }

        if (IsKeyPressed(KEY_W)) {
            exposure += 0.1f;
        }

        if (IsKeyPressed(KEY_S)) {
            exposure -= 0.1f;
            exposure = std::max(0.0f, exposure);
        }

        if (IsKeyPressed(KEY_A)) {
            gamma -= 0.1f;
            gamma = std::max(0.1f, gamma);
        }

        if (IsKeyPressed(KEY_D)) {
            gamma += 0.1f;
        }

            // prepare devSurface for kernels
            // map openGL resources to cuda
            // openGL shouldn't use cuda mapped resources until they are unmapped
#ifdef GAME_OF_CUDA__DEBUG
            gpuErrchk(cudaGraphicsMapResources(2, cudaResources, nullptr))
#else
        gpuErrchk(cudaGraphicsMapResources(1, cudaResources, nullptr))
#endif

#ifdef GAME_OF_CUDA__DEBUG
        cudaArray_t arrOut, arrOut2;
        gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&arrOut, cudaResources[0], 0, 0))
        gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&arrOut2, cudaResources[1], 0, 0))
#else
        cudaArray_t arrOut;
        gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&arrOut, cudaResources[0], 0, 0))
#endif

#ifdef GAME_OF_CUDA__DEBUG
        // create render surface (linked to openGL texture)
        cudaSurfaceObject_t renderSurface, renderSurface2;
        {
            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = arrOut;

            gpuErrchk(cudaCreateSurfaceObject(&renderSurface, &resDesc))
        }
        {
            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = arrOut2;

            gpuErrchk(cudaCreateSurfaceObject(&renderSurface2, &resDesc))
        }
#else
        // create render surface (linked to openGL texture)
        cudaSurfaceObject_t renderSurface;
        {
            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = arrOut;

            gpuErrchk(cudaCreateSurfaceObject(&renderSurface, &resDesc))
        }
#endif

        // create in device surface for post processing
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8,
                                                                  cudaChannelFormatKindUnsigned);  // define color channel format
        cudaArray *cArray, *cArray2, *cArray3;
        gpuErrchk(
                cudaMallocArray(&cArray, &channelDesc, rayTexOut.width, rayTexOut.height,
                                cudaArraySurfaceLoadStore))
        gpuErrchk(
                cudaMallocArray(&cArray2, &channelDesc, rayTexOut.width, rayTexOut.height,
                                cudaArraySurfaceLoadStore))
        gpuErrchk(
                cudaMallocArray(&cArray3, &channelDesc, rayTexOut.width, rayTexOut.height,
                                cudaArraySurfaceLoadStore))

        cudaResourceDesc resDesc{};
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cArray;

        cudaSurfaceObject_t devSurface;
        gpuErrchk(cudaCreateSurfaceObject(&devSurface, &resDesc))

        resDesc.res.array.array = cArray2;
        cudaSurfaceObject_t devSurface2;
        gpuErrchk(cudaCreateSurfaceObject(&devSurface2, &resDesc))

        resDesc.res.array.array = cArray3;
        cudaSurfaceObject_t devSurface3;
        gpuErrchk(cudaCreateSurfaceObject(&devSurface3, &resDesc))

        // launch kernels
        gol<<<gridSize, blockSize>>>(devSurface, devGridA, devGridB, rayTexOut.width,
                                     rayTexOut.height);

        // swap grids for rendering and next simulation step
        unsigned int *temp = devGridA;
        devGridA = devGridB;
        devGridB = temp;

        // apply luma filter
        lumaFilter<<<gridSize, blockSize>>>(devSurface, devSurface2, rayTexOut.width,
                                            rayTexOut.height, float(threshold));

        // blur
        for (int i = 1; i <= blurPasses; i++) {
            gaussianBlur<<<gridSize, blockSize>>>(devSurface2, devSurface3, kernel,
                                                  kernelSize, kernelSize,
                                                  rayTexOut.width, rayTexOut.height);
            surf2SurfWrite<<<gridSize, blockSize>>>(devSurface3, devSurface2,
                                                    rayTexOut.width, rayTexOut.height);
        }

        // combine
        surf2SurfWrite<<<gridSize, blockSize>>>(devSurface, renderSurface,
                                                rayTexOut.width, rayTexOut.height);
        combine<<<gridSize, blockSize>>>(devSurface2, renderSurface, rayTexOut.width,
                                         rayTexOut.height, exposure, gamma);

#ifdef GAME_OF_CUDA__DEBUG
        // write unaltered to second texture for comparison
        surf2SurfWrite<<<gridSize, blockSize>>>(devSurface, renderSurface2,
                                                rayTexOut.width, rayTexOut.height);
#endif
        // destroy & unmap openGL resources from cuda
        gpuErrchk(cudaDestroySurfaceObject(renderSurface))

#ifdef GAME_OF_CUDA__DEBUG
        gpuErrchk(cudaDestroySurfaceObject(renderSurface2))
#endif

        gpuErrchk(cudaDestroySurfaceObject(devSurface))
        gpuErrchk(cudaDestroySurfaceObject(devSurface2))
        gpuErrchk(cudaDestroySurfaceObject(devSurface3))

        gpuErrchk(cudaFreeArray(cArray))
        gpuErrchk(cudaFreeArray(cArray2))
        gpuErrchk(cudaFreeArray(cArray3))

#ifdef GAME_OF_CUDA__DEBUG
        gpuErrchk(cudaGraphicsUnmapResources(2, cudaResources, nullptr))
#else
        gpuErrchk(cudaGraphicsUnmapResources(1, cudaResources, nullptr))
#endif

        BeginDrawing();
        ClearBackground(BLACK);

        BeginMode2D(cam);
        // draw texture
        DrawTexturePro(
                rayTexOut,
                Rectangle{0, 0, (float) rayTexOut.width, (float) rayTexOut.height},
                Rectangle{0, 0, (float) rayTexOut.width, (float) rayTexOut.height},
                Vector2{0, 0},
                0.0f,
                WHITE
        );

#ifdef GAME_OF_CUDA__DEBUG
        DrawTexturePro(
                rayTexOut2,
                Rectangle{0, 0, (float)rayTexOut2.width, (float)rayTexOut2.height},
                Rectangle{0, 0, (float)rayTexOut2.width, (float)rayTexOut2.height},
                Vector2{-(float)rayTexOut.width, 0},
                0.0f,
                WHITE
        );
#endif
        EndMode2D();

#ifdef GAME_OF_CUDA__UI
        int x = 10;
        int y = 10;
        int fontSize = 45;
        DrawText(("FPS: " + std::to_string(GetFPS())).c_str(), x, y, fontSize, GREEN);
        DrawText(("Blur Passes: " + std::to_string(blurPasses)).c_str(), x, y + fontSize + 10, fontSize, GREEN);
        DrawText(("Threshold: " + std::to_string(threshold)).c_str(), x, y + ((fontSize + 10) * 2), fontSize, GREEN);
        DrawText(("Exposure: " + std::to_string(exposure)).c_str(), x, y + ((fontSize + 10) * 3), fontSize, GREEN);
        DrawText(("Gamma: " + std::to_string(gamma)).c_str(), x, y + ((fontSize + 10) * 4), fontSize, GREEN);
#endif

        EndDrawing();
    }

    // unregister openGL resources from cuda
    gpuErrchk(cudaGraphicsUnregisterResource(cudaResources[0]))

#ifdef GAME_OF_CUDA__DEBUG
    gpuErrchk(cudaGraphicsUnregisterResource(cudaResources[1]))
#endif

    gpuErrchk(cudaFree(kernel))
    gpuErrchk(cudaFree(devGridA))
    gpuErrchk(cudaFree(devGridB))

    // unload textures from OpenGL
    UnloadTexture(rayTexOut);

#ifdef GAME_OF_CUDA__DEBUG
    UnloadTexture(rayTexOut2);
#endif

    CloseWindow();
    return 0;
}
