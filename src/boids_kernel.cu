#include <cassert>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include "boids_kernel.cuh"
#include <random>
#include <thrust/device_vector.h>

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

namespace GPUSim
{

    __device__ void limitVelocity(glm::vec4 *vel)
    {
        float speed = glm::length(*vel);
        if (speed > MAX_SPEED)
        {
            *vel = *vel / speed * MAX_SPEED;
        }
    }

    __global__ void initCurandStates(curandState *states, unsigned long long seed, int numStates)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < numStates)
        {
            curand_init(seed, index, 0, &states[index]);
        }
    }

    __device__ glm::vec3 randVec4(curandState *state)
    {
        float3 randFloats;
        // go from random unif 0,1 to -BOUND, BOUND
        randFloats.x = curand_uniform(state) * 2.0f - 1.0f;
        randFloats.y = curand_uniform(state) * 2.0f - 1.0f;
        randFloats.z = curand_uniform(state) * 2.0f - 1.0f;

        return glm::vec3(randFloats.x, randFloats.y, randFloats.z);
    }

    __global__ void initBoidsKernel(glm::vec4 *boidPositions, glm::vec4 *boidVelPing, glm::vec4 *boidVelPong, int numBoids, curandState *states)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index >= numBoids)
        {
            return;
        }

        boidPositions[index] = BOUND * glm::vec4(randVec4(&states[index]), 1.0f);
        boidVelPing[index] = glm::vec4(randVec4(&states[index]), 0.0f);
        boidVelPong[index] = glm::vec4(randVec4(&states[index]), 0.0f);
    }

    void initCuda(GLuint *bPosVBO, GLuint *bVelVBO)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Device: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;

        gpuErrchk(cudaPeekAtLastError());

        gpuErrchk(cudaDeviceSynchronize());
    }

    void initVanillaBoids(glm::vec4 **boidPositions, glm::vec4 **boidVelPing, glm::vec4 **boidVelPong, int numBoids)
    {
        gpuErrchk(cudaMalloc((void **)boidPositions, numBoids * sizeof(glm::vec4)));

        gpuErrchk(cudaMalloc((void **)boidVelPing, numBoids * sizeof(glm::vec4)));

        gpuErrchk(cudaMalloc((void **)boidVelPong, numBoids * sizeof(glm::vec4)));
        cudaPointerAttributes attributes;
        cudaError_t err = cudaPointerGetAttributes(&attributes, boidVelPong);

        curandState *devStates;

        gpuErrchk(cudaMalloc((void **)&devStates, numBoids * 2 * sizeof(curandState)));

        initCurandStates<<<NUM_BLOCKS, NUM_THREADS>>>(devStates, 1234, numBoids);

        gpuErrchk(cudaDeviceSynchronize());

        initBoidsKernel<<<NUM_BLOCKS, NUM_THREADS>>>(*boidPositions, *boidVelPing, *boidVelPong, numBoids, devStates);
        gpuErrchk(cudaDeviceSynchronize());
    }

    __device__ glm::vec4 changeVanillaVelocity(int numBoids, int i, glm::vec4 *pos, glm::vec4 *vel)
    {
        glm::vec4 cohesion(0.0f);
        glm::vec4 separation(0.0f);
        glm::vec4 alignment(0.0f);
        glm::vec4 dV(0.0f);

        int cohesionNeighbors = 0;
        int alignmentNeighbors = 0;

        for (int j = 0; j < numBoids; j++)
        {
            if (i == j)
            {
                continue;
            }

            float distance = glm::distance(pos[i], pos[j]);

            // printf("Boid %d distance: %f\n", i, distance);

            if (distance < COHERENCE_RADIUS)
            {
                cohesion += pos[j];
                cohesionNeighbors++;
            }

            if (distance < SEPARATION_RADIUS)
            {
                separation -= (pos[j] - pos[i]);
            }

            if (distance < ALIGNMENT_RADIUS)
            {
                alignment += vel[j];
                alignmentNeighbors++;
            }
        }

        if (cohesionNeighbors > 0)
        {
            cohesion /= (float)cohesionNeighbors;
            cohesion = (cohesion - pos[i]) * COHESION_FACTOR;
        }
        if (alignmentNeighbors > 0)
        {
            alignment /= (float)alignmentNeighbors;
            alignment *= ALIGNMENT_FACTOR;
        }

        separation *= SEPARATION_FACTOR;

        dV = cohesion + separation + alignment;

        return dV;
    }

    __global__ void vanillaVelocityUpdate(glm::vec4 *boidPos, glm::vec4 *boidVelPing, glm::vec4 *boidVelPong, int numBoids)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if (index >= numBoids)
        {
            return;
        }
        // update velocity
        glm::vec4 updated = boidVelPing[index] + changeVanillaVelocity(numBoids, index, boidPos, boidVelPing);

        limitVelocity(&updated);
        boidVelPong[index] = updated;
    }

    __device__ void dbound(glm::vec4 *boid)
    {
        if (boid->x > BOUND)
        {
            boid->x = -BOUND;
        }
        if (boid->x < -BOUND)
        {
            boid->x = BOUND;
        }
        if (boid->y > BOUND)
        {
            boid->y = -BOUND;
        }
        if (boid->y < -BOUND)
        {
            boid->y = BOUND;
        }
        if (boid->z > BOUND)
        {
            boid->z = -BOUND;
        }
        if (boid->z < -BOUND)
        {
            boid->z = BOUND;
        }
    }
    // kernel to update vanilla sim
    __global__ void vanillaPositionUpdate(int numBoids, float dt, glm::vec4 *boidPos, glm::vec4 *boidVelPing, glm::vec4 *boidVelPong)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        // printf("index, threadidx, blockidx: %d %d %d\n", index, threadIdx.x, blockIdx.x);
        if (index >= numBoids)
        {
            return;
        }
        glm::vec4 pos = boidPos[index];

        pos += boidVelPong[index] * dt;

        dbound(&pos);

        boidPos[index] = pos;
    }

    __global__ void copyPosToVBOKer(int numBoids, glm::vec4 *boidPos, float *bPosVBO)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < numBoids)
        {
            bPosVBO[4 * index] = boidPos[index].x * 1.0f / SCALING;
            bPosVBO[4 * index + 1] = boidPos[index].y * 1.0f / SCALING;
            bPosVBO[4 * index + 2] = boidPos[index].z * 1.0f / SCALING;

            bPosVBO[4 * index + 3] = 1.0f;
        }
    }

    void copyPosToVBO(glm::vec4 *boidPos, float *bPosVBO, int numBoids)
    {
        dim3 numBlocks((numBoids + NUM_THREADS - 1) / NUM_THREADS);

        copyPosToVBOKer<<<numBlocks, NUM_THREADS>>>(numBoids, boidPos, bPosVBO);

        gpuErrchk(cudaDeviceSynchronize());
    }

    __global__ void copyVelToVBOKer(glm::vec4 *boidVel, float *bVelVBO, int numBoids)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index >= numBoids)
        {
            return;
        }

        bVelVBO[4 * index] = boidVel[index].x;
        bVelVBO[4 * index + 1] = boidVel[index].y;
        bVelVBO[4 * index + 2] = boidVel[index].z;

        bVelVBO[4 * index + 3] = 0.0f;
    }

    void copyVelToVBO(glm::vec4 *boidVel, float *bVelVBO, int numBoids)
    {

        dim3 numBlocks((numBoids + NUM_THREADS - 1) / NUM_THREADS);

        copyVelToVBOKer<<<numBlocks, NUM_THREADS>>>(boidVel, bVelVBO, numBoids);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    void runVanillaSim(glm::vec4 *boidPositions, glm::vec4 **boidVelPing, glm::vec4 **boidVelPong, int numBoids, float dt, GLuint *bPosVBO, GLuint *bVelVBO)
    {
        dim3 numBlocks((NUM_BOIDS + NUM_THREADS - 1) / NUM_THREADS);

        vanillaVelocityUpdate<<<numBlocks, NUM_THREADS>>>(boidPositions, *boidVelPing, *boidVelPong, NUM_BOIDS);

        gpuErrchk(cudaGetLastError());

        gpuErrchk(cudaDeviceSynchronize());

        vanillaPositionUpdate<<<numBlocks, NUM_THREADS>>>(numBoids, dt, boidPositions, *boidVelPing, *boidVelPong);

        gpuErrchk(cudaGetLastError());

        gpuErrchk(cudaDeviceSynchronize());

        // swap the references
        // ping pong ping
        glm::vec4 *temp = *boidVelPing;
        *boidVelPing = *boidVelPong;
        *boidVelPong = temp;

        gpuErrchk(cudaDeviceSynchronize());
    }

    // grid Sim accel.

    const int CUBE_INT = GRID_SIZE * GRID_SIZE * GRID_SIZE;

    __device__ int getGridIndex(glm::vec4 pos)
    {
        int x = (int)((pos.x + BOUND) / (2 * BOUND) * GRID_SIZE) % GRID_SIZE;
        int y = (int)((pos.y + BOUND) / (2 * BOUND) * GRID_SIZE) % GRID_SIZE;
        int z = (int)((pos.z + BOUND) / (2 * BOUND) * GRID_SIZE) % GRID_SIZE;

        return x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
    }

    __global__ void gridHashKernel(int numBoids, glm::vec4 *boidPos, int *gridHash)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index >= numBoids)
        {
            return;
        }

        int gridIndex = getGridIndex(boidPos[index]);

        gridHash[index] = gridIndex;
    }

    __global__ void gridIndexKernel(int numBoids, int *gridHash, int *cellStart, int *cellEnd)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index >= numBoids)
        {
            return;
        }

        int hash = gridHash[index];

        if (index == 0 || hash != gridHash[index - 1])
        {
            cellStart[hash] = index;
        }
        if (index == numBoids - 1 || hash != gridHash[index + 1])
        {
            cellEnd[hash] = index;
        }
    }

    __device__ glm::vec4 changeGridVelocity(int numBoids, int i, glm::vec4 *pos, glm::vec4 *vel, int *gridHash, int *cellStart, int *cellEnd)
    {
        glm::vec4 cohesion(0.0f);
        glm::vec4 separation(0.0f);
        glm::vec4 alignment(0.0f);
        glm::vec4 dV(0.0f);

        int cohesionNeighbors = 0;
        int alignmentNeighbors = 0;

        int gridIndex = gridHash[i];

        for (int j = cellStart[gridIndex]; j <= cellEnd[gridIndex]; j++)
        {
            if (i == j)
            {
                continue;
            }

            float distance = glm::distance(pos[i], pos[j]);

            if (distance < COHERENCE_RADIUS)
            {
                cohesion += pos[j];
                cohesionNeighbors++;
            }

            if (distance < SEPARATION_RADIUS)
            {
                separation -= (pos[j] - pos[i]);
            }

            if (distance < ALIGNMENT_RADIUS)
            {
                alignment += vel[j];
                alignmentNeighbors++;
            }
        }

        if (cohesionNeighbors > 0)
        {
            cohesion /= (float)cohesionNeighbors;
            cohesion = (cohesion - pos[i]) * COHESION_FACTOR;
        }
        if (alignmentNeighbors > 0)
        {
            alignment /= (float)alignmentNeighbors;
            alignment *= ALIGNMENT_FACTOR;
        }

        separation *= SEPARATION_FACTOR;

        dV = cohesion + separation + alignment;

        return dV;
    }

    __global__ void gridVelocityUpdate(int numBoids, glm::vec4 *boidPos, glm::vec4 *boidVelPing, glm::vec4 *boidVelPong, int *gridHash, int *cellStart, int *cellEnd)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if (index >= numBoids)
        {
            return;
        }
        glm::vec4 updated = boidVelPing[index] + changeGridVelocity(numBoids, index, boidPos, boidVelPing, gridHash, cellStart, cellEnd);

        limitVelocity(&updated);

        boidVelPong[index] = updated;
    }

    __global__ void gridPositionUpdate(int numBoids, float dt, glm::vec4 *boidPos, glm::vec4 *boidVelPing, glm::vec4 *boidVelPong)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        // printf("index, threadidx, blockidx: %d %d %d\n", index, threadIdx.x, blockIdx.x);
        if (index >= numBoids)
        {
            return;
        }
        glm::vec4 pos = boidPos[index];

        pos += boidVelPong[index] * dt;

        dbound(&pos);

        boidPos[index] = pos;
    }

    void initGridBoids(glm::vec4 **boidPositions, glm::vec4 **boidVelPing, glm::vec4 **boidVelPong, int numBoids)
    {
        gpuErrchk(cudaMalloc((void **)boidPositions, numBoids * sizeof(glm::vec4)));

        gpuErrchk(cudaMalloc((void **)boidVelPing, numBoids * sizeof(glm::vec4)));

        gpuErrchk(cudaMalloc((void **)boidVelPong, numBoids * sizeof(glm::vec4)));
        cudaPointerAttributes attributes;
        cudaError_t err = cudaPointerGetAttributes(&attributes, boidVelPong);

        curandState *devStates;

        gpuErrchk(cudaMalloc((void **)&devStates, numBoids * 2 * sizeof(curandState)));

        initCurandStates<<<NUM_BLOCKS, NUM_THREADS>>>(devStates, 1337, numBoids);

        gpuErrchk(cudaDeviceSynchronize());

        initBoidsKernel<<<NUM_BLOCKS, NUM_THREADS>>>(*boidPositions, *boidVelPing, *boidVelPong, numBoids, devStates);
        gpuErrchk(cudaDeviceSynchronize());
    }

    void runGridSim(glm::vec4 *boidPositions, glm::vec4 **boidVelPing, glm::vec4 **boidVelPong, int numBoids, float dt, GLuint *bPosVBO, GLuint *bVelVBO)
    {
        dim3 numBlocks((NUM_BOIDS + NUM_THREADS - 1) / NUM_THREADS);

        int *gridHash;
        int *cellStart;
        int *cellEnd;

        gpuErrchk(cudaMalloc((void **)&gridHash, numBoids * sizeof(int)));
        gpuErrchk(cudaMalloc((void **)&cellStart, CUBE_INT * sizeof(int)));
        gpuErrchk(cudaMalloc((void **)&cellEnd, CUBE_INT * sizeof(int)));

        gridHashKernel<<<numBlocks, NUM_THREADS>>>(numBoids, boidPositions, gridHash);

        gpuErrchk(cudaGetLastError());

        gpuErrchk(cudaDeviceSynchronize());

        thrust::device_vector<int> thrustGridHash(gridHash, gridHash + numBoids);
        thrust::device_vector<int> thrustCellStart(CUBE_INT, -1);
        thrust::device_vector<int> thrustCellEnd(CUBE_INT, -1);

        gpuErrchk(cudaGetLastError());

        thrust::fill(thrustCellStart.begin(), thrustCellStart.end(), -1);
        thrust::fill(thrustCellEnd.begin(), thrustCellEnd.end(), -1);

        thrust::device_vector<int> thrustCellStartD = thrustCellStart;
        thrust::device_vector<int> thrustCellEndD = thrustCellEnd;

        gridIndexKernel<<<numBlocks, NUM_THREADS>>>(numBoids, gridHash, thrust::raw_pointer_cast(thrustCellStartD.data()), thrust::raw_pointer_cast(thrustCellEndD.data()));

        gpuErrchk(cudaGetLastError());

        gpuErrchk(cudaDeviceSynchronize());

        thrust::copy(thrustCellStartD.begin(), thrustCellStartD.end(), thrustCellStart.begin());
        thrust::copy(thrustCellEndD.begin(), thrustCellEndD.end(), thrustCellEnd.begin());

        gridVelocityUpdate<<<numBlocks, NUM_THREADS>>>(numBoids, boidPositions, *boidVelPing, *boidVelPong, thrust::raw_pointer_cast(thrustGridHash.data()), thrust::raw_pointer_cast(thrustCellStart.data()), thrust::raw_pointer_cast(thrustCellEnd.data()));

        gpuErrchk(cudaDeviceSynchronize());

        vanillaPositionUpdate<<<numBlocks, NUM_THREADS>>>(numBoids, dt, boidPositions, *boidVelPing, *boidVelPong);

        gpuErrchk(cudaDeviceSynchronize());

        glm::vec4 *temp = *boidVelPing;
        *boidVelPing = *boidVelPong;
        *boidVelPong = temp;

        gpuErrchk(cudaFree(gridHash));
        gpuErrchk(cudaFree(cellStart));
        gpuErrchk(cudaFree(cellEnd));
    }
}
