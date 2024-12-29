#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <getopt.h>
#include <cstdlib>
#include <cstdio>
#include <glm/gtc/type_ptr.hpp>
#include <memory>

#include "cpu_boids.h"

#include "boids_kernel.cuh"

#include <chrono>

#include "config.h"


namespace
{

    void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
    void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
    void updateCamera(GLuint &shaderProgram);
    void initGLVertexObjects(GLuint &vaoBuffer, GLuint &boidPositionBuffer, GLuint &boidVelocityBuffer, GLuint &boidIndexBuffer);
    void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos);
    void initShaders(GLuint &shaderProgram);
    void errorCallback(int error, const char *description);
    void runSim(GLData &glData, GLFWwindow *window);
    void runCudaSim(GLData *glData, GLFWwindow *window);

    glm::vec3 camOrigin = glm::vec3(0.0f, 0.0f, 0.0f),
              cameraUp = glm::vec3(0.0f, 0.0f, 1.0f);

    glm::mat4 projection;

    glm::vec3 lookAt;

    double lastX = 0.0, lastY = 0.0;

    bool leftMousePressed = false;

    float phi = (M_1_PI) / 2, theta = (M_1_PI) / 2;

    GLFWwindow *window;

    bool prepareAndRun(int argc, char *argv[], std::shared_ptr<GLData> glData)
    {
        if (!glfwInit())
        {
            fprintf(stderr, "Failed to initialize GLFW. Are you on a server and running this with X forwarding?\n");
            fprintf(stderr, "Make sure you connect to your ssh session with the -X flag (the -v flag for debugging can be useful also)\n");
            fprintf(stderr, "If you are running this on a remote server, and have mesa-utils installed, check with glxgears \n");
            return false;
        }

        glfwSetErrorCallback(errorCallback);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        // create window
        window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "cs179 final project", NULL, NULL);

        if (!window)
        {
            fprintf(stderr, "Failed to create window, check debug output\n");

            glfwTerminate();
            return false;
        }

        glfwMakeContextCurrent(window);
        glfwSetKeyCallback(window, keyCallback);
        glfwSetCursorPosCallback(window, cursorPositionCallback);
        glfwSetMouseButtonCallback(window, mouseButtonCallback);
        glfwSetWindowUserPointer(window, glData.get());

        // initialize glew
        glewExperimental = GL_TRUE;

        if (glewInit() != GLEW_OK)
        {
            fprintf(stderr, "Failed to initialize GLEW\n");
            return false;
        }

        // init GL, takes reference to the VAO, VBOs, and IBO
        initGLVertexObjects(glData->VAO, glData->posVBO, glData->velVBO, glData->IBO);

        // test initialization
        if (glData->VAO == 0 || glData->posVBO == 0 || glData->velVBO == 0 || glData->IBO == 0)
        {
            fprintf(stderr, "Failed to initialize GLVertexObjects\n");
            return false;
        }
        
        updateCamera(glData->shaderProgram);

        initShaders(glData->shaderProgram);

        updateCamera(glData->shaderProgram);


        glEnable(GL_DEPTH_TEST);

        if (USECPU)
        {
            runSim(*glData, window);
        }
        else
        {
            cudaGLRegisterBufferObject(glData.get()->posVBO);
            cudaGLRegisterBufferObject(glData.get()->velVBO);
            runCudaSim(glData.get(), window);
        }


        return true;
    }

    void initGLVertexObjects(GLuint &vaoBuffer, GLuint &boidPositionBuffer, GLuint &boidVelocityBuffer, GLuint &boidIndexBuffer)
    {

        std::unique_ptr<glm::vec4[]> boidPositionVelocityCalloc(new glm::vec4[NUM_BOIDS]);
        std::unique_ptr<GLuint[]> flockingIndices(new GLuint[NUM_BOIDS]);

        for (size_t i = 0; i < NUM_BOIDS; i++)
        {
            boidPositionVelocityCalloc[i] = glm::vec4(0.0f);
            flockingIndices[i] = i;
        }

        // Generate and bind VAO and VBOs
        glGenVertexArrays(1, &vaoBuffer);
        glGenBuffers(1, &boidPositionBuffer);
        glGenBuffers(1, &boidVelocityBuffer);
        glGenBuffers(1, &boidIndexBuffer);

        glBindVertexArray(vaoBuffer);

        glBindBuffer(GL_ARRAY_BUFFER, boidPositionBuffer);
        glBufferData(GL_ARRAY_BUFFER, NUM_BOIDS * sizeof(glm::vec4), boidPositionVelocityCalloc.get(), GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer((GLuint)0, 4, GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, boidVelocityBuffer);
        glBufferData(GL_ARRAY_BUFFER, NUM_BOIDS * sizeof(glm::vec4), boidPositionVelocityCalloc.get(), GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer((GLuint)1, 4, GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boidIndexBuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, NUM_BOIDS * sizeof(GLuint), flockingIndices.get(), GL_STATIC_DRAW);

        glBindVertexArray(0);
    }

    void initShaders(GLuint &shaderProgram)
    {

        const char *vertexShaderSource =
            "#version 330 core\n"
            "in vec4 pos;\n"
            "in vec4 Velocity;\n"
            "out vec4 vertFrag;\n"
            "void main() {\n"
            "    vertFrag = Velocity;\n"
            "    gl_Position = pos;\n"
            "}\n";

        const char *fragmentShaderSource =
            "#version 330 core\n"
            "in vec4 vFragColor;\n"
            "out vec4 fragColor;\n"
            "void main() {\n"
            "    fragColor.r = abs(vFragColor.r) + 0.3f;\n"
            "    fragColor.g = abs(vFragColor.g) + 0.3f;\n"
            "    fragColor.b = abs(vFragColor.b) + 0.3f;\n"
            "}\n";

        const char *boidGeometryShaderSource =
            "#version 330 core\n"
            "uniform mat4 u_projMatrix;\n"
            "layout(points) in;\n"
            "layout(points) out;\n"
            "layout(max_vertices = 1) out;\n"

            "in vec4 vertFrag[];\n"
            "out vec4 vFragColor;\n"
            "void main() {\n"
            "    vec3 pos = gl_in[0].gl_Position.xyz;\n"
            "    vFragColor = vertFrag[0];\n"
            "    gl_Position = u_projMatrix * vec4(pos, 1.0);\n"
            "    EmitVertex();\n"
            "    EndPrimitive();\n"
            "}\n";

        GLuint fVertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(fVertexShader, 1, &vertexShaderSource, NULL);
        glCompileShader(fVertexShader);

        GLint success;
        GLchar infoLog[512];
        glGetShaderiv(fVertexShader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(fVertexShader, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n"
                      << infoLog << std::endl;
        }

        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
        glCompileShader(fragmentShader);

        glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n"
                      << infoLog << std::endl;
        }

        GLuint geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(geometryShader, 1, &boidGeometryShaderSource, NULL);
        glCompileShader(geometryShader);

        glGetShaderiv(geometryShader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(geometryShader, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER::GEOMETRY::COMPILATION_FAILED\n"
                      << infoLog << std::endl;
        }

        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, fVertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glAttachShader(shaderProgram, geometryShader);
        glLinkProgram(shaderProgram);

        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if (!success)
        {
            glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n"
                      << infoLog << std::endl;
        }

        glDeleteShader(fVertexShader);
        glDeleteShader(fragmentShader);
        glDeleteShader(geometryShader);

        GLint loc = glGetUniformLocation(shaderProgram, "u_projMatrix");
        if (loc != -1)
        {
            glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(projection));
        }
        else
        {
            std::cout << "loc not found" << std::endl;
        }
    }

    void runSim(GLData &glData, GLFWwindow *window)
    {

        std::cout << "CPU Boids" << std::endl;
        std::cout << "NUM_BOIDS: " << NUM_BOIDS << std::endl;

        std::shared_ptr<STDBoids> boidsPointer;

        boidsPointer = std::make_shared<STDBoids>();

        CPUSim::initVanillaBoids(boidsPointer, NUM_BOIDS);

        auto start = std::chrono::high_resolution_clock::now();
        int frameCount = 0;

        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();

            if (SIMTYPE == 1)
            {
                CPUSim::runVanillaSim(boidsPointer, DT, glData.posVBO, glData.velVBO);
            }
            else if (SIMTYPE == 2)
            {
                CPUSim::runGridSim(boidsPointer, DT, glData.posVBO, glData.velVBO);
            }
            else if (SIMTYPE == 3)
            {
                CPUSim::runSpatialHashingSim(boidsPointer, DT, glData.posVBO, glData.velVBO);
            }
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glUseProgram(glData.shaderProgram);
            glBindVertexArray(glData.VAO);
            glDrawElements(GL_POINTS, NUM_BOIDS, GL_UNSIGNED_INT, 0);
            glPointSize(1.0f);

            glUseProgram(0);
            glBindVertexArray(0);

            glfwSwapBuffers(window);

            frameCount++;

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            if (elapsed.count() > 1.0)
            {
                std::cout << "cpu FPS: " << frameCount << std::endl;
                frameCount = 0;
                start = std::chrono::high_resolution_clock::now();
            }

        }

        glfwDestroyWindow(window);
        glfwTerminate();
    }
    void runCudaSim(GLData *glData, GLFWwindow *window)
    {
        std::cout << "GPU Boids" << std::endl;
        // print number of boids
        std::cout << "NUM_BOIDS: " << NUM_BOIDS << std::endl;
        glm::vec4 *boidPositions;
        glm::vec4 *boidVelPing;
        glm::vec4 *boidVelPong;

        GPUSim::initCuda(&glData->posVBO, &glData->velVBO);

        if (SIMTYPE == 1)
        {
            std::cout << "initGPUVanillaBoids" << std::endl;
            GPUSim::initVanillaBoids(&boidPositions, &boidVelPing, &boidVelPong, NUM_BOIDS);
        }
        else if (SIMTYPE == 2)
        {
            GPUSim::initGridBoids(&boidPositions, &boidVelPing, &boidVelPong, NUM_BOIDS);
        }

        auto start = std::chrono::high_resolution_clock::now();
        int frameCount = 0;

        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            float *rendboidPos = NULL;
            float *rendboidVel = NULL;

            cudaGLMapBufferObject((void **)&rendboidPos, glData->posVBO);
            cudaGLMapBufferObject((void **)&rendboidVel, glData->velVBO);

            if (SIMTYPE == 1)
            {
                GPUSim::runVanillaSim(boidPositions, &boidVelPing, &boidVelPong, NUM_BOIDS, DT, &glData->posVBO, &glData->velVBO);
            }
            else if (SIMTYPE == 2)
            {
                GPUSim::runGridSim(boidPositions, &boidVelPing, &boidVelPong, NUM_BOIDS, DT, &glData->posVBO, &glData->velVBO);
            }

            GPUSim::copyPosToVBO(boidPositions, rendboidPos, NUM_BOIDS);
            GPUSim::copyVelToVBO(boidVelPong, rendboidVel, NUM_BOIDS);

            cudaGLUnmapBufferObject(glData->posVBO);
            cudaGLUnmapBufferObject(glData->velVBO);

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glUseProgram(glData->shaderProgram);
            glBindVertexArray(glData->VAO);
            glDrawElements(GL_POINTS, NUM_BOIDS, GL_UNSIGNED_INT, 0);
            glPointSize(1.0f);

            glUseProgram(0);
            glBindVertexArray(0);

            glfwSwapBuffers(window);

            frameCount++;

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            if (elapsed.count() > 1.0)
            {
                std::cout << "gpu fps: " << frameCount << std::endl;
                frameCount = 0;
                start = std::chrono::high_resolution_clock::now();
            }

        }

        glfwDestroyWindow(window);
        glfwTerminate();

        cudaGLUnregisterBufferObject(glData->posVBO);
        cudaGLUnregisterBufferObject(glData->velVBO);

        cudaFree(boidPositions);
        cudaFree(boidVelPing);
        cudaFree(boidVelPong);
    }

    void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
    {
        leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    }
    void updateCamera(GLuint &shaderProgram)
    {

        lookAt = ZOOM_FACTOR * glm::vec3(sin(phi) * sin(theta),
                                         sin(theta) * cos(phi),
                                         cos(theta));

        lookAt += camOrigin;

        projection = glm::perspective(((float)(M_PI / 4)),
                                      (float)SCREEN_WIDTH / SCREEN_HEIGHT,
                                      1.0f,
                                      100.0f);

        glm::mat4 view = glm::lookAt(lookAt, camOrigin, cameraUp);

        projection = projection * view;
        glUseProgram(shaderProgram);

        GLint viewLoc = glGetUniformLocation(shaderProgram, "u_projMatrix");

        if (viewLoc != -1)
        {
            glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(projection));
        }
        else
        {
            std::cout << "viewLoc not found" << std::endl;
        }
    }

    void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
    {
        std::cout << "key: " << key << std::endl;
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, GL_TRUE);
            return;
        }

        updateCamera(static_cast<GLData *>(glfwGetWindowUserPointer(window))->shaderProgram);
    }

    void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos)
    {
        if (leftMousePressed)
        {
            float dX = (xpos - lastX) / SCREEN_WIDTH;
            float dY = (ypos - lastY) / SCREEN_HEIGHT;

            phi += dX;
            theta -= dY;

            theta = std::fmax(0.01f, std::fmin(theta, M_PI));
            updateCamera(static_cast<GLData *>(glfwGetWindowUserPointer(window))->shaderProgram);
        }

        lastX = xpos;
        lastY = ypos;
    }

    void errorCallback(int error, const char *description)
    {
        fprintf(stderr, "Error: %s\n", description);
    }

}

void printUsage()
{
    fprintf(stderr, "To change the simulation type, change the constants in config.h, explained in the readme\n");
}


int main(int argc, char *argv[])
{    

    printUsage();

    std::shared_ptr<GLData> glData = std::make_shared<GLData>();

    prepareAndRun(argc, argv, glData);

    return 0;
}
