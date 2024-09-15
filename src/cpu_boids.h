#ifndef CPU_BOIDS_KERNEL_H
#define CPU_BOIDS_KERNEL_H

#include "config.h"

#include <memory>

namespace CPUSim
{

    void initVanillaBoids(std::shared_ptr<STDBoids> &boids, size_t numBoids);

    void runVanillaSim(std::shared_ptr<STDBoids> &boids, double dt, GLuint &bPosVBO, GLuint &bVelVBO);
    void runGridSim(std::shared_ptr<STDBoids> &boids, double dt, GLuint &bPosVBO, GLuint &bVelVBO);
    void runSpatialHashingSim(std::shared_ptr<STDBoids> &boids, double dt, GLuint &bPosVBO, GLuint &bVelVBO);

}

#endif