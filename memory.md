# GPU Glossary Memory - Complete Concept Database

## Overview
This document contains the complete memory of all GPU concepts, definitions, and relationships extracted from the Modal Labs GPU Glossary. It serves as the knowledge base for building the interactive educational web application.

---

## 1. Device Hardware Concepts

### 1.1 Processing Units & Cores

#### Core
- **Definition**: Primary compute units that make up the Streaming Multiprocessors (SMs)
- **Key Insight**: More like "pipes" than CPU cores - data goes in, transformed data comes out
- **Types**: CUDA Cores, Tensor Cores
- **Architecture Role**: Effect actual computations within SMs
- **Analogy**: Think of as specialized pipelines for different types of operations
- **Visual**: H100 SM diagram showing cores in green

#### CUDA Core
- **Definition**: GPU cores that execute scalar arithmetic instructions
- **Function**: Handle integer and floating-point operations
- **Scheduling**: Groups of cores (typically 32) receive same instruction simultaneously
- **Architecture Variations**: Different mixtures of 32-bit integer, 32-bit FP, 64-bit FP units
- **H100 Specs**: 128 FP32 CUDA Cores per SM (but only 64 INT32 or 64 FP64 units)
- **Key Difference**: Unlike CPU cores, not independently scheduled
- **Performance Note**: Groups can be as small as 1 thread but at performance cost

#### Tensor Core
- **Definition**: Specialized cores for matrix operations using single instructions
- **Purpose**: Accelerate neural network workloads with matrix multiplication
- **Operation**: Operates on entire matrices rather than scalars
- **H100 Specs**: 4 Tensor Cores per SM, support various precisions
- **Key Advantage**: Massive throughput for AI/ML operations
- **Programming**: Uses WMMA (Warp Matrix Multiply-Accumulate) intrinsics
- **Precision Support**: Multiple data types (FP16, INT8, etc.)

#### Special Function Unit (SFU)
- **Definition**: Accelerates transcendental mathematical operations
- **Functions**: exp, log, sin, cos, sqrt, reciprocal
- **Performance**: Much faster than computing via CUDA cores
- **Limitation**: Lower precision than CUDA core implementations
- **Usage**: Automatic compiler optimization for mathematical functions

### 1.2 Architecture Components

#### Streaming Multiprocessor (SM)
- **Definition**: Fundamental processing unit of GPU, designed for massive parallelism
- **CPU Analogy**: Closer to CPU core equivalent than individual GPU cores
- **Components**: Cores, register file, shared memory, warp schedulers
- **H100 Specs**: 132 SMs total, each handling up to 2048 concurrent threads
- **Threading**: Organizes threads into warps of 32 for SIMT execution
- **Key Feature**: Fast warp switching (single clock cycle vs. CPU microseconds)

#### Streaming Multiprocessor Architecture
- **Definition**: Versioned architectures defining SASS code compatibility
- **Purpose**: Determines which compiled code can run on which hardware
- **Evolution**: Each generation adds new capabilities and instruction sets
- **Compatibility**: Forward compatible within same architecture family
- **Examples**: Kepler, Maxwell, Pascal, Volta, Turing, Ampere, Hopper

#### CUDA Device Architecture
- **Definition**: Compute Unified Device Architecture - simplified GPU design
- **Historical Context**: Replaced complex graphics pipeline with unified compute
- **Key Innovation**: Made GPU programming accessible to general compute
- **Three Meanings**: Hardware architecture, programming model, software platform
- **Foundation**: Enables SIMT (Single Instruction, Multiple Thread) execution

#### Graphics Processing Cluster (GPC)
- **Definition**: Collection of Texture Processing Clusters plus raster engine
- **Role**: High-level organizational unit within GPU
- **Components**: Multiple TPCs, geometry processing, rasterization
- **Function**: Handles both compute and graphics workloads

#### Texture Processing Cluster (TPC)
- **Definition**: Generally synonymous with pair of Streaming Multiprocessors
- **Role**: Intermediate organizational level between GPC and SM
- **Function**: Groups SMs for efficient resource management

### 1.3 Memory & Storage

#### GPU RAM
- **Definition**: Large global memory store (DRAM) accessible by all SMs
- **Implementation**: Provides storage for global memory in programming model
- **Capacity**: Typically 16GB-80GB+ in modern GPUs
- **Access Pattern**: Optimized for coalesced memory access
- **Bandwidth**: Extremely high bandwidth (1TB/s+ in modern GPUs)

#### Register File
- **Definition**: Stores bits between manipulation by GPU cores
- **Organization**: Split into dynamically reallocatable 32-bit registers
- **Per-Thread**: Each thread has access to registers for private data
- **Performance**: Fastest memory tier, zero-latency access
- **Limitation**: Limited quantity, affects thread occupancy

#### L1 Data Cache
- **Definition**: Private SRAM memory of each SM, co-located with compute units
- **Shared Memory**: Programmable portion can be used as shared memory
- **Size**: Typically 64KB-192KB per SM
- **Configuration**: Configurable split between L1 cache and shared memory
- **Access**: Low-latency access for threads within same SM

### 1.4 Scheduling & Control

#### Warp Scheduler
- **Definition**: Decides which group of threads (warp) to execute next
- **Function**: Switches warps on per-clock-cycle basis
- **H100 Specs**: 4 warp schedulers per SM
- **Key Advantage**: Hides memory latency through rapid warp switching
- **Efficiency**: Enables high throughput by keeping cores busy

#### Load/Store Unit (LSU)
- **Definition**: Dispatches memory requests to various memory subsystems
- **Function**: Handles communication between compute units and memory hierarchy
- **Types**: Manages global, shared, and local memory operations
- **Coalescing**: Combines multiple memory requests for efficiency

---

## 2. Device Software Concepts

### 2.1 Programming Model Foundation

#### CUDA Programming Model
- **Definition**: Programming model with three key abstractions
- **Abstractions**: 
  1. Hierarchy of thread groups (thread → warp → block → grid)
  2. Hierarchy of memories (registers → shared → global)
  3. Barrier synchronization
- **Philosophy**: Scalable parallelism through hierarchical organization
- **Key Insight**: Maps naturally to GPU hardware architecture

#### Compute Capability
- **Definition**: Versioning system for PTX instruction compatibility
- **Format**: Major.Minor (e.g., 8.0 for H100)
- **Purpose**: Determines which PTX instructions are supported
- **Compatibility**: Forward compatible within major versions
- **Usage**: Compiler targets specific compute capability

#### Parallel Thread Execution (PTX)
- **Definition**: Intermediate representation and virtual machine instruction set
- **Role**: Target for CUDA C++ compiler, source for SASS compiler
- **Benefits**: Hardware independence, optimization opportunities
- **Format**: Assembly-like syntax with virtual registers
- **Evolution**: Stable API that evolves with new hardware capabilities

#### Streaming Assembler (SASS)
- **Definition**: Lowest-level assembly format for NVIDIA GPUs
- **Purpose**: Actual instructions executed by GPU hardware
- **Generation**: Compiled from PTX by GPU driver
- **Optimization**: Hardware-specific optimizations applied
- **Debugging**: Can be examined with cuobjdump/nvidisasm

### 2.2 Thread Hierarchy

#### Thread
- **Definition**: Lowest unit of GPU programming
- **Resources**: Own registers, instruction pointer, execution state
- **Execution**: Executes kernel code with unique thread ID
- **Memory**: Private registers, shared memory with block, global memory
- **Lifecycle**: Created when kernel launches, destroyed when kernel completes

#### Warp
- **Definition**: Group of 32 threads scheduled together in SIMT model
- **Execution**: All threads execute same instruction simultaneously
- **Branching**: Divergent branches reduce efficiency
- **Scheduling**: Atomic unit of scheduling by warp scheduler
- **Key Insight**: Fundamental unit of GPU parallelism

#### Thread Block
- **Definition**: Collection of threads scheduled onto same SM
- **Cooperation**: Threads can coordinate via shared memory and synchronization
- **Size**: Typically 32-1024 threads (must be multiple of 32)
- **Scheduling**: Entire block assigned to single SM
- **Memory**: Shared memory space accessible to all threads in block

#### Cooperative Thread Array (CTA)
- **Definition**: PTX/SASS implementation term for thread blocks
- **Equivalence**: CTA = Thread Block at lower level
- **Usage**: Appears in PTX code and hardware documentation
- **Function**: Same cooperation and synchronization capabilities

#### Thread Block Grid
- **Definition**: Highest level containing multiple thread blocks
- **Spanning**: Can span multiple SMs across entire GPU
- **Coordination**: Limited coordination between blocks (atomic operations)
- **Scalability**: Enables scaling across different GPU sizes
- **Execution**: Blocks execute independently and can be scheduled in any order

#### Kernel
- **Definition**: Unit of CUDA code launched once but executed by many threads
- **Execution**: Each thread executes same kernel code with different data
- **Launch**: Specified with grid and block dimensions
- **Types**: Regular kernels, cooperative kernels, dynamic parallelism
- **Lifecycle**: Launch → Execute → Synchronize → Complete

### 2.3 Memory Hierarchy

#### Registers
- **Definition**: Thread-private memory at lowest level of hierarchy
- **Storage**: Stored in register file or spilled to global memory
- **Speed**: Fastest memory tier, zero-latency access
- **Limitation**: Limited quantity affects thread occupancy
- **Allocation**: Automatic by compiler, manual optimization possible

#### Shared Memory
- **Definition**: Thread block-level memory in L1 cache
- **Management**: Programmer-managed and explicitly allocated
- **Speed**: Much faster than global memory
- **Cooperation**: Enables efficient data sharing within block
- **Synchronization**: Requires __syncthreads() for consistency

#### Global Memory
- **Definition**: Highest level accessible by all threads
- **Storage**: Implemented in GPU RAM (DRAM)
- **Capacity**: Large (GBs) but high latency
- **Access**: Optimized for coalesced access patterns
- **Persistence**: Survives across kernel launches

#### Memory Hierarchy
- **Definition**: Programmer-managed memory system matching thread group hierarchy
- **Mapping**: Thread groups → Memory levels (1:1 correspondence)
- **Strategy**: Move data closer to computation for better performance
- **Pattern**: Load global → shared → compute → writeback

---

## 3. Host Software Concepts

### 3.1 Core APIs & Runtime

#### CUDA Software Platform
- **Definition**: Complete collection of software for developing CUDA programs
- **Components**: Compiler, runtime, APIs, libraries, tools
- **Purpose**: Provides complete ecosystem for GPU programming
- **Integration**: Works with existing C++ development workflows

#### CUDA Driver API
- **Definition**: Low-level userspace component providing basic GPU functions
- **Functions**: Memory management (cuMalloc), kernel execution (cuLaunchKernel)
- **Verbosity**: More verbose but provides fine-grained control
- **Usage**: Advanced applications, other language bindings
- **Implementation**: libcuda.so on Linux

#### CUDA Runtime API
- **Definition**: Higher-level wrapper around Driver API with better ergonomics
- **Functions**: cudaMalloc, cudaMemcpy, kernel launch syntax
- **Convenience**: Easier to use, handles context management
- **Usage**: Most CUDA applications use Runtime API
- **Implementation**: libcudart.so on Linux

#### libcuda.so
- **Definition**: Binary shared object implementing CUDA Driver API on Linux
- **Role**: Interface between user applications and GPU driver
- **Functions**: Provides all Driver API functionality
- **Distribution**: Installed with NVIDIA GPU drivers

#### libcudart.so
- **Definition**: Binary shared object implementing CUDA Runtime API on Linux
- **Role**: Higher-level interface built on Driver API
- **Functions**: Provides all Runtime API functionality
- **Distribution**: Part of CUDA Toolkit

### 3.2 Development Tools

#### CUDA C++
- **Definition**: C++ extension implementing CUDA programming model
- **Features**: Kernels, memory management, synchronization
- **Syntax**: __global__ functions, <<<grid, block>>> launch syntax
- **Integration**: Seamless integration with standard C++
- **Compilation**: Requires nvcc compiler

#### NVIDIA CUDA Compiler Driver (nvcc)
- **Definition**: Toolchain compiling CUDA C++ to fat binaries
- **Process**: CUDA C++ → PTX → SASS
- **Output**: Fat binaries containing PTX and SASS for multiple architectures
- **Optimization**: Multiple optimization passes for performance
- **Integration**: Can be integrated with build systems

#### NVIDIA Runtime Compiler (nvrtc)
- **Definition**: Runtime compilation library for CUDA C to PTX
- **Purpose**: Compile CUDA C strings to PTX at runtime
- **Benefits**: Runtime specialization, dynamic kernel generation
- **Usage**: Advanced applications with runtime code generation

#### CUDA Binary Utilities
- **Definition**: Tools for examining and manipulating CUDA binaries
- **Tools**: cuobjdump, nvidisasm, fatbinary
- **Functions**: Disassemble PTX/SASS, examine fat binaries
- **Usage**: Debugging, performance analysis, reverse engineering

### 3.3 Profiling & Management

#### NVIDIA Nsight Systems
- **Definition**: Performance debugging tool with profiling, tracing, and analysis
- **Capabilities**: Timeline tracing, API calls, kernel execution
- **Visualization**: Graphical interface for performance analysis
- **Integration**: Works with CUDA applications and frameworks
- **Output**: Detailed performance reports and recommendations

#### CUDA Profiling Tools Interface (CUPTI)
- **Definition**: APIs for profiling CUDA execution with synchronized timestamps
- **Purpose**: Building custom profiling tools
- **Metrics**: Performance counters, API tracing, kernel metrics
- **Integration**: Used by Nsight Systems and other profilers

#### nvidia-smi
- **Definition**: Command line utility for querying and managing GPU state
- **Functions**: GPU utilization, memory usage, temperature, power
- **Monitoring**: Real-time monitoring of GPU resources
- **Management**: Basic GPU management operations
- **Usage**: System administration, debugging, monitoring

#### NVIDIA Management Library (NVML)
- **Definition**: Library for monitoring GPU state, power, temperature
- **Purpose**: Programmatic access to GPU information
- **Functions**: Similar to nvidia-smi but as library
- **Integration**: Used by monitoring systems and applications

#### libnvml.so
- **Definition**: Binary implementing NVML functions on Linux
- **Role**: Provides programmatic access to GPU management
- **Distribution**: Part of NVIDIA GPU drivers

### 3.4 System Integration

#### NVIDIA GPU Drivers
- **Definition**: Mediate interaction between host programs and GPU device
- **Role**: Kernel-level and userspace components
- **Functions**: Hardware abstraction, memory management, scheduling
- **Components**: nvidia.ko (kernel), libcuda.so (userspace)

#### nvidia.ko
- **Definition**: Linux kernel module executing in privileged mode
- **Purpose**: Direct hardware communication and resource management
- **Functions**: Memory allocation, context switching, interrupt handling
- **Security**: Runs with kernel privileges for hardware access

---

## 4. Educational Flow & Dependencies

### 4.1 Learning Progression
1. **Foundation**: CUDA Device Architecture → Programming Model
2. **Hardware**: SM → Cores → Memory Hierarchy
3. **Software**: Thread Hierarchy (Thread → Warp → Block → Grid)
4. **Memory**: Registers → Shared → Global
5. **Development**: CUDA C++ → nvcc → APIs
6. **Advanced**: Profiling, Optimization, Advanced APIs

### 4.2 Critical Relationships
- **Hardware-Software Mapping**: Thread blocks → SMs, Warps → Schedulers, Threads → Cores
- **Memory Hierarchy Alignment**: Thread levels ↔ Memory levels
- **Compilation Flow**: CUDA C++ → PTX → SASS → Execution
- **Performance Dependencies**: Occupancy, Coalescing, Register Pressure

### 4.3 Code Examples & Patterns
- **Matrix Multiplication**: Simple and optimized versions
- **Memory Coalescing**: Efficient access patterns
- **Shared Memory**: Blocking and tiling strategies
- **Synchronization**: __syncthreads() usage
- **Atomic Operations**: Cross-block coordination

### 4.4 Visual Concepts
- **SM Architecture**: H100 internal structure
- **Memory Hierarchy**: Visual representation of levels
- **Thread Hierarchy**: Grid → Block → Warp → Thread
- **Execution Model**: SIMT visualization
- **Performance Optimization**: Occupancy and throughput

---

## 5. Interactive Web App Considerations

### 5.1 Educational Structure
- **Progressive Disclosure**: Start simple, add complexity
- **Visual Learning**: Diagrams for abstract concepts
- **Interactive Elements**: Hover, click, expand/collapse
- **Cross-References**: Links between related concepts
- **Code Integration**: Runnable examples where possible

### 5.2 Navigation Strategy
- **Hierarchical**: Main categories → Subcategories → Concepts
- **Search**: Full-text search across all concepts
- **Filters**: By category, difficulty, hardware generation
- **Breadcrumbs**: Show current location in hierarchy
- **Related**: Suggest related concepts

### 5.3 Visual Design
- **Architecture Diagrams**: Interactive SM diagrams
- **Memory Hierarchy**: Visual representation with data flow
- **Timeline**: Evolution of GPU architectures
- **Comparison**: CPU vs GPU paradigms
- **Performance**: Visual performance characteristics

This comprehensive memory serves as the complete knowledge base for building the interactive educational GPU glossary web application.