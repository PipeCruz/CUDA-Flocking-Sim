#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <vector>
#include <array>
#include <GL/glew.h>
#include <glm/glm.hpp>

#include <cuda_runtime.h>

const float ZOOM_FACTOR = 5.0f;

const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 800;

const float DT = 0.2f;
const bool USECPU = false;

// cpu: 1 = naive, 2 = spatial hashing, 3 = grid
// gpu: 1 = naive, 2 = grid
const size_t SIMTYPE = 2;

__device__  const size_t NUM_BOIDS = 10000;
__device__  const float SCALING = 100.0f;
__device__  const float COHESION_FACTOR = 0.01f;
__device__  const float SEPARATION_FACTOR = 0.1f;
__device__  const float ALIGNMENT_FACTOR = 0.1f;
__device__  const float MAX_SPEED = 1.0f;

const float COHERENCE_RADIUS = 5.0f;
const float SEPARATION_RADIUS = 1.0f;
const  float ALIGNMENT_RADIUS = 5.0f;

const float BOUND = 100.0f;

const int GRID_SIZE = 20;

const int NUM_THREADS = 128;

const int NUM_BLOCKS = (NUM_BOIDS + NUM_THREADS - 1) / NUM_THREADS;

const int NUM_BLOCKS_2D = (NUM_BLOCKS + NUM_THREADS - 1) / NUM_THREADS;

const int NUM_BLOCKS_3D = (NUM_BLOCKS_2D + NUM_THREADS - 1) / NUM_THREADS;

// for cpu demo
struct STDBoids
{
    std::vector<glm::vec4> boidVelocities;
    std::vector<glm::vec4> boidPositions;
};


struct GLData
{
    GLuint VAO;
    GLuint posVBO;
    GLuint velVBO;
    GLuint IBO;
    GLuint shaderProgram;
};

#endif