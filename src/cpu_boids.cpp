#include "cpu_boids.h"
#include <cstdlib>
// #include <GL/glew.h>
#include <iostream>
#include <cmath>
#include <random>

#include <unordered_map>

#include <glm/glm.hpp>

#include <time.h>

namespace CPUSim
{
    // hash prototype
    int hashFunction(const glm::vec3 &pos);

    void initVanillaBoids(std::shared_ptr<STDBoids> &boids, size_t numBoids)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-BOUND, BOUND);

        boids->boidPositions.resize(numBoids);
        boids->boidVelocities.resize(numBoids);

        for (size_t i = 0; i < numBoids; ++i)
        {
            boids->boidPositions[i] = glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);

            // velocities between 0 and 1
            boids->boidVelocities[i] = glm::vec4(dis(gen), dis(gen), dis(gen), 0.0f);
        }
    }
    glm::vec3 limit(const glm::vec3 &vec, float max)
    {
        if (glm::length(vec) > max)
        {
            return glm::normalize(vec) * max;
        }
        return vec;
    }

    glm::vec3 computeNaiveBehavior(std::shared_ptr<STDBoids> &boids, size_t i)
    {
        glm::vec3 cohesion(0.0f);
        glm::vec3 separation(0.0f);
        glm::vec3 alignment(0.0f);
        glm::vec3 direction(0.0f);
        int cohesionNeighbors = 0;
        int alignmentNeighbors = 0;

        for (size_t j = 0; j < boids->boidPositions.size(); ++j)
        {
            if (i != j)
            {
                float distance = glm::distance(glm::vec3(boids->boidPositions[i]), glm::vec3(boids->boidPositions[j]));

                // Cohesion
                if (distance < COHERENCE_RADIUS)
                {
                    cohesion += glm::vec3(boids->boidPositions[j]);
                    cohesionNeighbors++;
                }

                // Separation
                if (distance < SEPARATION_RADIUS)
                {
                    separation -= (glm::vec3(boids->boidPositions[j]) - glm::vec3(boids->boidPositions[i]));
                }

                // Alignment
                if (distance < ALIGNMENT_RADIUS)
                {
                    alignment += glm::vec3(boids->boidVelocities[j]);
                    alignmentNeighbors++;
                }
            }
        }

        // Finalize cohesion
        if (cohesionNeighbors > 0)
        {
            cohesion /= static_cast<float>(cohesionNeighbors);
            direction = cohesion - glm::vec3(boids->boidPositions[i]);
            cohesion = direction * COHESION_FACTOR;
        }

        // Finalize alignment
        if (alignmentNeighbors > 0)
        {
            alignment /= static_cast<float>(alignmentNeighbors);
            alignment *= ALIGNMENT_FACTOR;
        }

        // Apply separation factor
        separation *= SEPARATION_FACTOR;

        // Combine all behaviors
        glm::vec3 combinedSteering = cohesion + separation + alignment;

        return combinedSteering;
    }
    void boundPosition(glm::vec4 &pos)
    {
        pos.x = pos.x > BOUND ? -BOUND : pos.x;
        pos.x = pos.x < -BOUND ? BOUND : pos.x;
        pos.y = pos.y > BOUND ? -BOUND : pos.y;
        pos.y = pos.y < -BOUND ? BOUND : pos.y;
        pos.z = pos.z > BOUND ? -BOUND : pos.z;
        pos.z = pos.z < -BOUND ? BOUND : pos.z;
    }

    void runVanillaSim(std::shared_ptr<STDBoids> &boids, double dt, GLuint &bPosVBO, GLuint &bVelVBO)
    {
        glm::vec3 dV;
        // std::cout << "running cpu sim" << std::endl;
        for (size_t i = 0; i < boids->boidPositions.size(); ++i)
        {
            dV = computeNaiveBehavior(boids, i);

            // Update velocity and limit it
            boids->boidVelocities[i] += glm::vec4(dV, 0.0f);

            // Update position
            boids->boidPositions[i] += boids->boidVelocities[i] * static_cast<float>(dt);

            boids->boidVelocities[i] = glm::vec4(limit(glm::vec3(boids->boidVelocities[i]), MAX_SPEED), 0.0f);

            // BOUND the position
            boundPosition(boids->boidPositions[i]);
        }

        std::vector<glm::vec4> boidPositions(NUM_BOIDS);
        for (size_t i = 0; i < boids->boidVelocities.size(); ++i)
        {
            boidPositions[i] = boids->boidPositions[i] * static_cast<float>(1.0f / SCALING);
        }
        // Update VBOs
        glBindBuffer(GL_ARRAY_BUFFER, bPosVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, NUM_BOIDS * sizeof(glm::vec4), boidPositions.data());

        glBindBuffer(GL_ARRAY_BUFFER, bVelVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, NUM_BOIDS * sizeof(glm::vec4), boids->boidVelocities.data());
    }

    // GRID SIM
    // FIXME make into kwarg

    const int CUBE_INT = (GRID_SIZE * GRID_SIZE * GRID_SIZE);

    int computeGridIndex(const glm::vec3 &pos)
    {
        // flatten 3D grid into 1D array
        int x = static_cast<int>((pos.x + BOUND) / (2 * BOUND) * GRID_SIZE) % GRID_SIZE;
        int y = static_cast<int>((pos.y + BOUND) / (2 * BOUND) * GRID_SIZE) % GRID_SIZE;
        int z = static_cast<int>((pos.z + BOUND) / (2 * BOUND) * GRID_SIZE) % GRID_SIZE;

        int index = x * GRID_SIZE * GRID_SIZE + y * GRID_SIZE + z;

        // std::cout << "index: " << index << std::endl;
        return index;
    }

    std::array<std::vector<size_t>, CUBE_INT> gridHashing(std::shared_ptr<STDBoids> &boids)
    {
        std::array<std::vector<size_t>, CUBE_INT> grid = {};

        for (size_t i = 0; i < NUM_BOIDS; ++i)
        {
            int index = computeGridIndex(glm::vec3(boids->boidPositions[i]));
            // std::cout << "index: " << index << std::endl;
            grid[index].push_back(i);
        }

        return grid;
    }

    // compute grid hashing prototype
    glm::vec3 computeGridBehavior(std::shared_ptr<STDBoids> &boids, size_t i, std::array<std::vector<size_t>, CUBE_INT> &grid)
    {
        glm::vec3 cohesion(0.0f);
        glm::vec3 separation(0.0f);
        glm::vec3 alignment(0.0f);
        glm::vec3 direction(0.0f);
        int cohesionNeighbors = 0;
        int alignmentNeighbors = 0;

        int gridIndex = computeGridIndex(glm::vec3(boids->boidPositions[i]));

        for (size_t j : grid[gridIndex])
        {
            if (i != j)
            {
                float distance = glm::distance(glm::vec3(boids->boidPositions[i]), glm::vec3(boids->boidPositions[j]));

                // Cohesion
                if (distance < COHERENCE_RADIUS)
                {
                    cohesion += glm::vec3(boids->boidPositions[j]);
                    cohesionNeighbors++;
                }

                // Separation
                if (distance < SEPARATION_RADIUS)
                {
                    separation -= (glm::vec3(boids->boidPositions[j]) - glm::vec3(boids->boidPositions[i]));
                }

                // Alignment
                if (distance < ALIGNMENT_RADIUS)
                {
                    alignment += glm::vec3(boids->boidVelocities[j]);
                    alignmentNeighbors++;
                }
            }
        }

        // Finalize cohesion
        if (cohesionNeighbors > 0)
        {
            cohesion /= static_cast<float>(cohesionNeighbors);
            direction = cohesion - glm::vec3(boids->boidPositions[i]);
            // direction = limit(direction, MAX_SPEED);
            cohesion = direction * COHESION_FACTOR;
        }

        // Finalize alignment
        if (alignmentNeighbors > 0)
        {
            alignment /= static_cast<float>(alignmentNeighbors);
            alignment *= ALIGNMENT_FACTOR;
        }

        // Apply separation factor
        separation *= SEPARATION_FACTOR;

        // Combine all behaviors
        glm::vec3 combinedSteering = cohesion + separation + alignment;

        return combinedSteering;
    }

    void runGridSim(std::shared_ptr<STDBoids> &boids, double dt, GLuint &bPosVBO, GLuint &bVelVBO)
    {
        glm::vec3 dV;
        std::array<std::vector<size_t>, GRID_SIZE * GRID_SIZE * GRID_SIZE> grid = gridHashing(boids);

        for (size_t i = 0; i < boids->boidPositions.size(); ++i)
        {
            dV = computeGridBehavior(boids, i, grid);

            // Update velocity
            boids->boidVelocities[i] += glm::vec4(dV, 0.0f);
            // limit velocity
            boids->boidVelocities[i] = glm::vec4(limit(glm::vec3(boids->boidVelocities[i]), MAX_SPEED), 0.0f);

            // Update position
            boids->boidPositions[i] += boids->boidVelocities[i] * static_cast<float>(dt);

            // BOUND the position
            boundPosition(boids->boidPositions[i]);
        }

        std::vector<glm::vec4> boidPos(NUM_BOIDS);
        for (size_t i = 0; i < boids->boidVelocities.size(); ++i)
        {
            boidPos[i] = boids->boidPositions[i] * static_cast<float>(1.0f / SCALING);
        }
        // Update VBOs
        glBindBuffer(GL_ARRAY_BUFFER, bPosVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, boids->boidPositions.size() * sizeof(glm::vec4), boidPos.data());

        glBindBuffer(GL_ARRAY_BUFFER, bVelVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, boids->boidVelocities.size() * sizeof(glm::vec4), boids->boidVelocities.data());
    }

    // SPATIAL HASHING

    int hashFunction(const glm::vec3 &pos)
    {
        // bucket the points into a hash table based on PERCEPTION_RADIUS
        // std::cout << "pos: " << pos.x << " " << pos.y << " " << pos.z << std::endl;
        int x = static_cast<int>((pos.x + BOUND) / (2 * BOUND) * GRID_SIZE) % GRID_SIZE;
        int y = static_cast<int>((pos.y + BOUND) / (2 * BOUND) * GRID_SIZE) % GRID_SIZE;
        int z = static_cast<int>((pos.z + BOUND) / (2 * BOUND) * GRID_SIZE) % GRID_SIZE;

        return x * 73856093 ^ y * 19349663 ^ z * 83492791;
    }

    std::unordered_map<int, std::vector<size_t>> spatialHashing(std::shared_ptr<STDBoids> &boids)
    {
        std::unordered_map<int, std::vector<size_t>> hashTable;

        // want similar boids to be in the same bucket, bucket size is 2 * PERCEPTION_RADIUS

        for (size_t i = 0; i < boids->boidPositions.size(); ++i)
        {
            int hash = hashFunction(glm::vec3(boids->boidPositions[i]));

            hashTable[hash].push_back(i);
        }

        return hashTable;
    }

    glm::vec3 computeSpatialHashingBehavior(std::shared_ptr<STDBoids> &boids, size_t i, std::unordered_map<int, std::vector<size_t>> &hashTable)
    {
        glm::vec3 cohesion(0.0f);
        glm::vec3 separation(0.0f);
        glm::vec3 alignment(0.0f);
        glm::vec3 direction(0.0f);
        int cohesionNeighbors = 0;
        int alignmentNeighbors = 0;

        int hash = hashFunction(glm::vec3(boids->boidPositions[i]));

        for (size_t j : hashTable[hash])
        {
            if (i != j)
            {
                float distance = glm::distance(glm::vec3(boids->boidPositions[i]), glm::vec3(boids->boidPositions[j]));

                // Cohesion
                if (distance < COHERENCE_RADIUS)
                {
                    cohesion += glm::vec3(boids->boidPositions[j]);
                    cohesionNeighbors++;
                }

                // Separation
                if (distance < SEPARATION_RADIUS)
                {
                    separation -= (glm::vec3(boids->boidPositions[j]) - glm::vec3(boids->boidPositions[i]));
                }

                // Alignment
                if (distance < ALIGNMENT_RADIUS)
                {
                    alignment += glm::vec3(boids->boidVelocities[j]);
                    alignmentNeighbors++;
                }
            }
        }

        // Finalize cohesion
        if (cohesionNeighbors > 0)
        {
            cohesion /= static_cast<float>(cohesionNeighbors);
            direction = cohesion - glm::vec3(boids->boidPositions[i]);
            // direction = limit(direction, MAX_SPEED);
            cohesion = direction * COHESION_FACTOR;
        }

        // Finalize alignment
        if (alignmentNeighbors > 0)
        {
            alignment /= static_cast<float>(alignmentNeighbors);
            alignment *= ALIGNMENT_FACTOR;
        }

        // Apply separation factor
        separation *= SEPARATION_FACTOR;

        // Combine all behaviors
        glm::vec3 combinedSteering = cohesion + separation + alignment;

        return combinedSteering;
    }

    void runSpatialHashingSim(std::shared_ptr<STDBoids> &boids, double dt, GLuint &bPosVBO, GLuint &bVelVBO)
    {
        glm::vec3 dV;
        std::unordered_map<int, std::vector<size_t>> hashTable = spatialHashing(boids);

        for (size_t i = 0; i < boids->boidPositions.size(); ++i)
        {

            dV = computeSpatialHashingBehavior(boids, i, hashTable);

            boids->boidVelocities[i] += glm::vec4(dV, 0.0f);
            boids->boidVelocities[i] = glm::vec4(limit(glm::vec3(boids->boidVelocities[i]), MAX_SPEED), 0.0f);

            boids->boidPositions[i] += boids->boidVelocities[i] * static_cast<float>(dt);

            boundPosition(boids->boidPositions[i]);
        }

        // scale velocity by SCALING to render
        std::vector<glm::vec4> boidPos(NUM_BOIDS);
        for (size_t i = 0; i < boids->boidVelocities.size(); ++i)
        {
            boidPos[i] = boids->boidPositions[i] * static_cast<float>(1.0f / SCALING);
        }
        // Update VBOs
        glBindBuffer(GL_ARRAY_BUFFER, bPosVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, boids->boidPositions.size() * sizeof(glm::vec4), boidPos.data());

        glBindBuffer(GL_ARRAY_BUFFER, bVelVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, boids->boidVelocities.size() * sizeof(glm::vec4), boids->boidVelocities.data());
    }
} 