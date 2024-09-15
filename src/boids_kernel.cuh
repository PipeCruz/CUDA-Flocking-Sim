#ifndef CUDA_BOIDS_KERNEL_H
#define CUDA_BOIDS_KERNEL_H

// cuda stuff

#include "config.h"
#include <cuda_runtime.h>

#include <glm/glm.hpp>

#include <cuda_gl_interop.h>

namespace GPUSim
{

    void initCuda(GLuint *bPosVBO, GLuint *bVelVBO);

    void initVanillaBoids(glm::vec4 **boidPositions, glm::vec4 **boidVelPing, glm::vec4 **boidVelPong, int numBoids);
    void runVanillaSim(glm::vec4 *boidPositions, glm::vec4 **boidVelPing, glm::vec4 **boidVelPong, int numBoids, float dt, GLuint *bPosVBO, GLuint *bVelVBO);

    void initGridBoids(glm::vec4 **boidPositions, glm::vec4 **boidVelPing, glm::vec4 **boidVelPong, int numBoids);
    void runGridSim(glm::vec4 *boidPositions, glm::vec4 **boidVelPing, glm::vec4 **boidVelPong, int numBoids, float dt, GLuint *bPosVBO, GLuint *bVelVBO);

    void copyPosToVBO(glm::vec4 *boidPositions, float *bPosVBO, int numBoids);
    void copyVelToVBO(glm::vec4 *boidVelPing, float *bVelVBO, int numBoids);
}
#endif