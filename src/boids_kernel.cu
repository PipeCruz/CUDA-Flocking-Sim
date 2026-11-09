#include <cassert>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include "boids_kernel.cuh"
#include <random>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

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

        curandState *devStates;

        gpuErrchk(cudaMalloc((void **)&devStates, numBoids * sizeof(curandState)));

        dim3 numBlocks((numBoids + NUM_THREADS - 1) / NUM_THREADS);

        initCurandStates<<<numBlocks, NUM_THREADS>>>(devStates, 1234, numBoids);

        gpuErrchk(cudaDeviceSynchronize());

        initBoidsKernel<<<numBlocks, NUM_THREADS>>>(*boidPositions, *boidVelPing, *boidVelPong, numBoids, devStates);
        gpuErrchk(cudaDeviceSynchronize());

        // curand states are only needed for initialization
        gpuErrchk(cudaFree(devStates));
    }

    // Naive all-pairs velocity update, tiled through shared memory: each block
    // streams the full boid set in NUM_THREADS-sized tiles so every position/
    // velocity read from global memory is reused by all threads in the block.
    __global__ void vanillaVelocityUpdate(glm::vec4 *boidPos, glm::vec4 *boidVelPing, glm::vec4 *boidVelPong, int numBoids)
    {
        __shared__ glm::vec4 sharedPos[NUM_THREADS];
        __shared__ glm::vec4 sharedVel[NUM_THREADS];

        int index = threadIdx.x + blockIdx.x * blockDim.x;

        // Cache this boid's own state in registers (0,0,0,0 for out-of-range
        // threads, which still have to help load shared memory below).
        glm::vec4 posI = (index < numBoids) ? boidPos[index] : glm::vec4(0.0f);
        glm::vec4 velI = (index < numBoids) ? boidVelPing[index] : glm::vec4(0.0f);

        glm::vec4 cohesion(0.0f);
        glm::vec4 separation(0.0f);
        glm::vec4 alignment(0.0f);
        int cohesionNeighbors = 0;
        int alignmentNeighbors = 0;

        for (int tile = 0; tile < numBoids; tile += blockDim.x)
        {
            int j = tile + threadIdx.x;
            if (j < numBoids)
            {
                sharedPos[threadIdx.x] = boidPos[j];
                sharedVel[threadIdx.x] = boidVelPing[j];
            }
            __syncthreads();

            int tileCount = min((int)blockDim.x, numBoids - tile);
            for (int k = 0; k < tileCount; k++)
            {
                int jj = tile + k;
                if (jj == index)
                {
                    continue;
                }

                glm::vec4 diff = sharedPos[k] - posI;
                float dist2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

                if (dist2 < COHERENCE_RADIUS * COHERENCE_RADIUS)
                {
                    cohesion += sharedPos[k];
                    cohesionNeighbors++;
                }
                if (dist2 < SEPARATION_RADIUS * SEPARATION_RADIUS)
                {
                    separation -= diff;
                }
                if (dist2 < ALIGNMENT_RADIUS * ALIGNMENT_RADIUS)
                {
                    alignment += sharedVel[k];
                    alignmentNeighbors++;
                }
            }
            __syncthreads();
        }

        if (index >= numBoids)
        {
            return;
        }

        if (cohesionNeighbors > 0)
        {
            cohesion /= (float)cohesionNeighbors;
            cohesion = (cohesion - posI) * COHESION_FACTOR;
        }
        if (alignmentNeighbors > 0)
        {
            alignment /= (float)alignmentNeighbors;
            alignment *= ALIGNMENT_FACTOR;
        }
        separation *= SEPARATION_FACTOR;

        glm::vec4 updated = velI + cohesion + separation + alignment;
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
    // kernel to update position from the freshly computed velocity (pong)
    __global__ void vanillaPositionUpdate(int numBoids, float dt, glm::vec4 *boidPos, glm::vec4 *boidVelPing, glm::vec4 *boidVelPong)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
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
        dim3 numBlocks((numBoids + NUM_THREADS - 1) / NUM_THREADS);

        // Kernels on the default stream run in order, so we only need one sync
        // at the end (before the host maps the VBOs) instead of after each.
        vanillaVelocityUpdate<<<numBlocks, NUM_THREADS>>>(boidPositions, *boidVelPing, *boidVelPong, numBoids);
        gpuErrchk(cudaGetLastError());

        vanillaPositionUpdate<<<numBlocks, NUM_THREADS>>>(numBoids, dt, boidPositions, *boidVelPing, *boidVelPong);
        gpuErrchk(cudaGetLastError());

        gpuErrchk(cudaDeviceSynchronize());

        // ping-pong the velocity buffers
        glm::vec4 *temp = *boidVelPing;
        *boidVelPing = *boidVelPong;
        *boidVelPong = temp;
    }

    // ---- grid-accelerated sim -------------------------------------------------

    const int CUBE_INT = GRID_SIZE * GRID_SIZE * GRID_SIZE;

    // How many cells out to search per axis. A fixed +/-1 (3x3x3) is only
    // enough when a cell is at least as wide as the largest neighbour radius;
    // if the grid is refined so cells get smaller than the radius, a fixed
    // search silently misses cross-cell neighbours. Derived from the config
    // (computed in-kernel from the inlined constants) so it is always correct.
    __device__ int cellSearchRadius()
    {
        float cellW = 2.0f * BOUND / GRID_SIZE;
        float maxR = COHERENCE_RADIUS > ALIGNMENT_RADIUS ? COHERENCE_RADIUS : ALIGNMENT_RADIUS;
        int r = (int)ceilf(maxR / cellW);
        return r < 1 ? 1 : r;
    }

    // Persistent scratch buffers, allocated once in initGridBoids and reused
    // every frame (the old code malloc/free'd these per frame).
    namespace
    {
        int *dGridHash = nullptr;      // cell hash of each (sorted) boid
        int *dBoidIndex = nullptr;     // sorted-slot -> original boid index
        int *dCellStart = nullptr;     // first sorted slot in each cell
        int *dCellEnd = nullptr;       // one-past last sorted slot in each cell
        glm::vec4 *dSortedPos = nullptr;
        glm::vec4 *dSortedVel = nullptr;
    }

    __device__ int cellCoord(float p)
    {
        int c = (int)((p + BOUND) / (2.0f * BOUND) * GRID_SIZE);
        // wrap into [0, GRID_SIZE) (domain is toroidal, see dbound)
        c %= GRID_SIZE;
        if (c < 0)
        {
            c += GRID_SIZE;
        }
        return c;
    }

    __device__ int getGridIndex(glm::vec4 pos)
    {
        int x = cellCoord(pos.x);
        int y = cellCoord(pos.y);
        int z = cellCoord(pos.z);

        return x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
    }

    // hash each boid to a cell and seed the identity permutation
    __global__ void gridHashKernel(int numBoids, glm::vec4 *boidPos, int *gridHash, int *boidIndex)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index >= numBoids)
        {
            return;
        }

        gridHash[index] = getGridIndex(boidPos[index]);
        boidIndex[index] = index;
    }

    // gather positions/velocities into cell-sorted order using the permutation
    // produced by sort_by_key
    __global__ void reorderKernel(int numBoids, int *boidIndex, glm::vec4 *pos, glm::vec4 *vel, glm::vec4 *sortedPos, glm::vec4 *sortedVel)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index >= numBoids)
        {
            return;
        }

        int src = boidIndex[index];
        sortedPos[index] = pos[src];
        sortedVel[index] = vel[src];
    }

    // find the [start, end) slot range of each cell within the sorted hash array
    __global__ void cellStartEndKernel(int numBoids, int *sortedHash, int *cellStart, int *cellEnd)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index >= numBoids)
        {
            return;
        }

        int hash = sortedHash[index];

        if (index == 0 || hash != sortedHash[index - 1])
        {
            cellStart[hash] = index;
        }
        if (index == numBoids - 1 || hash != sortedHash[index + 1])
        {
            cellEnd[hash] = index + 1;
        }
    }

    // velocity update over the 3x3x3 neighbourhood of cells, reading cell-sorted
    // arrays; writes the result back into the original boid ordering.
    __global__ void gridVelocityUpdate(int numBoids, int *boidIndex, int *cellStart, int *cellEnd,
                                       glm::vec4 *sortedPos, glm::vec4 *sortedVel, glm::vec4 *boidVelPong)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index >= numBoids)
        {
            return;
        }

        glm::vec4 posI = sortedPos[index];
        glm::vec4 velI = sortedVel[index];

        int cx = cellCoord(posI.x);
        int cy = cellCoord(posI.y);
        int cz = cellCoord(posI.z);

        glm::vec4 cohesion(0.0f);
        glm::vec4 separation(0.0f);
        glm::vec4 alignment(0.0f);
        int cohesionNeighbors = 0;
        int alignmentNeighbors = 0;

        // Visit `searchR` cells out in each direction. When the search span
        // reaches the whole grid we walk every cell exactly once (absolute
        // coords) so wrap-around can't double-count a cell; otherwise the
        // wrapped window is already distinct.
        const int searchR = cellSearchRadius();
        const int span = 2 * searchR + 1;
        const int axisCount = span < GRID_SIZE ? span : GRID_SIZE;
        for (int iz = 0; iz < axisCount; iz++)
        {
            for (int iy = 0; iy < axisCount; iy++)
            {
                for (int ix = 0; ix < axisCount; ix++)
                {
                    int nx = (span < GRID_SIZE) ? ((cx - searchR + ix) % GRID_SIZE + GRID_SIZE) % GRID_SIZE : ix;
                    int ny = (span < GRID_SIZE) ? ((cy - searchR + iy) % GRID_SIZE + GRID_SIZE) % GRID_SIZE : iy;
                    int nz = (span < GRID_SIZE) ? ((cz - searchR + iz) % GRID_SIZE + GRID_SIZE) % GRID_SIZE : iz;

                    int cell = nx + ny * GRID_SIZE + nz * GRID_SIZE * GRID_SIZE;

                    int start = cellStart[cell];
                    if (start < 0)
                    {
                        continue; // empty cell
                    }
                    int end = cellEnd[cell];

                    for (int j = start; j < end; j++)
                    {
                        if (j == index)
                        {
                            continue;
                        }

                        glm::vec4 diff = sortedPos[j] - posI;
                        float dist2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

                        if (dist2 < COHERENCE_RADIUS * COHERENCE_RADIUS)
                        {
                            cohesion += sortedPos[j];
                            cohesionNeighbors++;
                        }
                        if (dist2 < SEPARATION_RADIUS * SEPARATION_RADIUS)
                        {
                            separation -= diff;
                        }
                        if (dist2 < ALIGNMENT_RADIUS * ALIGNMENT_RADIUS)
                        {
                            alignment += sortedVel[j];
                            alignmentNeighbors++;
                        }
                    }
                }
            }
        }

        if (cohesionNeighbors > 0)
        {
            cohesion /= (float)cohesionNeighbors;
            cohesion = (cohesion - posI) * COHESION_FACTOR;
        }
        if (alignmentNeighbors > 0)
        {
            alignment /= (float)alignmentNeighbors;
            alignment *= ALIGNMENT_FACTOR;
        }
        separation *= SEPARATION_FACTOR;

        glm::vec4 updated = velI + cohesion + separation + alignment;
        limitVelocity(&updated);

        // scatter back to the original ordering so position update / ping-pong
        // / VBO copy all stay index-consistent
        boidVelPong[boidIndex[index]] = updated;
    }

    void initGridBoids(glm::vec4 **boidPositions, glm::vec4 **boidVelPing, glm::vec4 **boidVelPong, int numBoids)
    {
        gpuErrchk(cudaMalloc((void **)boidPositions, numBoids * sizeof(glm::vec4)));
        gpuErrchk(cudaMalloc((void **)boidVelPing, numBoids * sizeof(glm::vec4)));
        gpuErrchk(cudaMalloc((void **)boidVelPong, numBoids * sizeof(glm::vec4)));

        // persistent grid scratch (reused every frame)
        gpuErrchk(cudaMalloc((void **)&dGridHash, numBoids * sizeof(int)));
        gpuErrchk(cudaMalloc((void **)&dBoidIndex, numBoids * sizeof(int)));
        gpuErrchk(cudaMalloc((void **)&dCellStart, CUBE_INT * sizeof(int)));
        gpuErrchk(cudaMalloc((void **)&dCellEnd, CUBE_INT * sizeof(int)));
        gpuErrchk(cudaMalloc((void **)&dSortedPos, numBoids * sizeof(glm::vec4)));
        gpuErrchk(cudaMalloc((void **)&dSortedVel, numBoids * sizeof(glm::vec4)));

        curandState *devStates;
        gpuErrchk(cudaMalloc((void **)&devStates, numBoids * sizeof(curandState)));

        dim3 numBlocks((numBoids + NUM_THREADS - 1) / NUM_THREADS);

        initCurandStates<<<numBlocks, NUM_THREADS>>>(devStates, 1337, numBoids);
        gpuErrchk(cudaDeviceSynchronize());

        initBoidsKernel<<<numBlocks, NUM_THREADS>>>(*boidPositions, *boidVelPing, *boidVelPong, numBoids, devStates);
        gpuErrchk(cudaDeviceSynchronize());

        gpuErrchk(cudaFree(devStates));
    }

    void freeGridBoids()
    {
        cudaFree(dGridHash);
        cudaFree(dBoidIndex);
        cudaFree(dCellStart);
        cudaFree(dCellEnd);
        cudaFree(dSortedPos);
        cudaFree(dSortedVel);
    }

    void runGridSim(glm::vec4 *boidPositions, glm::vec4 **boidVelPing, glm::vec4 **boidVelPong, int numBoids, float dt, GLuint *bPosVBO, GLuint *bVelVBO)
    {
        dim3 numBlocks((numBoids + NUM_THREADS - 1) / NUM_THREADS);

        // 1. hash every boid to a cell + seed the identity permutation
        gridHashKernel<<<numBlocks, NUM_THREADS>>>(numBoids, boidPositions, dGridHash, dBoidIndex);
        gpuErrchk(cudaGetLastError());

        // 2. sort boid indices by cell hash so each cell's members are contiguous
        thrust::sort_by_key(thrust::device,
                            thrust::device_pointer_cast(dGridHash),
                            thrust::device_pointer_cast(dGridHash + numBoids),
                            thrust::device_pointer_cast(dBoidIndex));

        // 3. gather positions/velocities into cell-sorted order
        reorderKernel<<<numBlocks, NUM_THREADS>>>(numBoids, dBoidIndex, boidPositions, *boidVelPing, dSortedPos, dSortedVel);
        gpuErrchk(cudaGetLastError());

        // 4. compute per-cell [start, end) ranges (empty cells stay -1)
        gpuErrchk(cudaMemset(dCellStart, 0xff, CUBE_INT * sizeof(int)));
        cellStartEndKernel<<<numBlocks, NUM_THREADS>>>(numBoids, dGridHash, dCellStart, dCellEnd);
        gpuErrchk(cudaGetLastError());

        // 5. flocking over the 3x3x3 cell neighbourhood
        gridVelocityUpdate<<<numBlocks, NUM_THREADS>>>(numBoids, dBoidIndex, dCellStart, dCellEnd, dSortedPos, dSortedVel, *boidVelPong);
        gpuErrchk(cudaGetLastError());

        // 6. integrate positions from the new velocities (original ordering)
        vanillaPositionUpdate<<<numBlocks, NUM_THREADS>>>(numBoids, dt, boidPositions, *boidVelPing, *boidVelPong);
        gpuErrchk(cudaGetLastError());

        gpuErrchk(cudaDeviceSynchronize());

        glm::vec4 *temp = *boidVelPing;
        *boidVelPing = *boidVelPong;
        *boidVelPong = temp;
    }
}
