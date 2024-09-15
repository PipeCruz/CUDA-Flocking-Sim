# Input Names
CUDA_FILES = boids_kernel.cu
CPP_FILES = cpu_boids.cpp
CPP_MAIN = main.cpp

# Directory names
BUILDDIR = build
SRCDIR = src
OBJDIR = $(BUILDDIR)/obj
BINDIR = $(BUILDDIR)/bin

# if it doesn't exist, create build directory
$(shell mkdir -p $(OBJDIR) $(BINDIR))

# ------------------------------------------------------------------------------

# CUDA path, compiler, and flags
CUDA_PATH = /usr/local/cuda
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
	NVCC_FLAGS := -m32
else
	NVCC_FLAGS := -m64
endif

# FIXME: figure out how to change linker flags depending on whether cpu or gpu demo? 

NVCC_FLAGS += -g -dc -Wno-deprecated-gpu-targets --std=c++11 \
	      --expt-relaxed-constexpr -D THRUST_IGNORE_DEPRECATED_CPP_DIALECT
NVCC_INCLUDE =
NVCC_CUDA_LIBS = 
NVCC_GENCODES = -gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61

# ------------------------------------------------------------------------------

# CUDA Linker and Flags
CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets

# ------------------------------------------------------------------------------

# C++ Compiler and Flags
GPP = g++
FLAGS = -g -Wall -D_REENTRANT -std=c++11 -pthread 
INCLUDE = -I$(CUDA_INC_PATH)
CUDA_LIBS = -L$(CUDA_LIB_PATH) -lglfw -lGL -lGLU -lGLEW -lcudart


# ------------------------------------------------------------------------------
# Object files
# ------------------------------------------------------------------------------

# CUDA Object Files
CUDA_OBJ = $(OBJDIR)/cuda.o
CUDA_OBJ_FILES = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(CUDA_FILES)))

# C++ Object Files
CPP_OBJ = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(CPP_FILES)))
MAIN_OBJ = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(CPP_MAIN)))

# List of all common objects needed to be linked into the final executable
COMMON_OBJ = $(CPP_OBJ) $(CUDA_OBJ) $(CUDA_OBJ_FILES)

# ------------------------------------------------------------------------------
# Make rules
# ------------------------------------------------------------------------------

# Top level rules
all: main

crun: clean run

run: main
	$(BINDIR)/main

main: $(MAIN_OBJ) $(COMMON_OBJ)
	$(GPP) $(FLAGS) -o $(BINDIR)/$@ $(INCLUDE) $^ $(CUDA_LIBS) 

$(MAIN_OBJ): $(addprefix $(SRCDIR)/, $(CPP_MAIN))
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $<

$(CPP_OBJ): $(OBJDIR)/%.o : $(SRCDIR)/%
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $<


# Compile CUDA Source Files
$(CUDA_OBJ_FILES): $(OBJDIR)/%.cu.o : $(SRCDIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o $@ $(NVCC_INCLUDE) $<

# Make linked device code
$(CUDA_OBJ): $(CUDA_OBJ_FILES)
	$(NVCC) $(CUDA_LINK_FLAGS) $(NVCC_GENCODES) -o $@ $(NVCC_INCLUDE) $<


# Clean everything 
clean:
	rm -f $(BINDIR)/* $(OBJDIR)/*.o $(SRCDIR)/*~ *~


.PHONY: clean all
