import type { Concept } from '../../types';

export const hostSoftwareConcepts: Concept[] = [
  {
    id: 'cuda-software-platform',
    title: 'CUDA Software Platform',
    category: 'host-software',
    subcategory: 'apis-runtime',
    definition: 'Complete collection of software for developing CUDA programs, including compiler, runtime, APIs, libraries, and tools for GPU programming.',
    keyPoints: [
      'Complete ecosystem for GPU programming',
      'Includes compiler, runtime, APIs, libraries, tools',
      'Integrates with existing C++ development workflows',
      'Provides abstraction over GPU hardware complexity',
      'Enables high-performance computing on GPUs'
    ],
    technicalDetails: [
      {
        label: 'Components',
        value: 'Compiler (nvcc), Runtime, APIs, Libraries, Tools',
        description: 'Comprehensive development environment'
      },
      {
        label: 'Integration',
        value: 'Works with existing C++ workflows',
        description: 'Seamless integration with standard development'
      },
      {
        label: 'Purpose',
        value: 'Simplify GPU programming complexity',
        description: 'High-level abstractions over hardware'
      }
    ],
    relatedConcepts: ['cuda-c', 'nvcc', 'cuda-runtime-api', 'cuda-driver-api'],
    prerequisites: [],
    difficulty: 'beginner',
    tags: ['platform', 'ecosystem', 'development', 'cuda'],
    learningPath: ['gpu-fundamentals', 'cuda-development']
  },
  {
    id: 'cuda-driver-api',
    title: 'CUDA Driver API',
    category: 'host-software',
    subcategory: 'apis-runtime',
    definition: 'Low-level userspace component providing basic GPU functions with fine-grained control but requiring more verbose programming.',
    keyPoints: [
      'Low-level userspace GPU interface',
      'Fine-grained control over GPU operations',
      'More verbose but more powerful than Runtime API',
      'Used by advanced applications and language bindings',
      'Foundation for higher-level APIs'
    ],
    technicalDetails: [
      {
        label: 'Functions',
        value: 'cuMalloc, cuMemcpy, cuLaunchKernel, cuCtxCreate',
        description: 'Explicit memory and context management'
      },
      {
        label: 'Control Level',
        value: 'Fine-grained control over all operations',
        description: 'Manual management of contexts, modules, functions'
      },
      {
        label: 'Implementation',
        value: 'libcuda.so on Linux',
        description: 'Shared library providing driver interface'
      }
    ],
    codeExamples: [
      {
        language: 'c',
        code: `#include <cuda.h>

int main() {
    // Initialize CUDA Driver API
    cuInit(0);
    
    // Get device and create context
    CUdevice device;
    CUcontext context;
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);
    
    // Allocate memory
    CUdeviceptr d_data;
    size_t size = 1024 * sizeof(float);
    cuMemAlloc(&d_data, size);
    
    // Load module and get function
    CUmodule module;
    CUfunction kernel;
    cuModuleLoad(&module, "kernel.ptx");
    cuModuleGetFunction(&kernel, module, "vectorAdd");
    
    // Launch kernel
    void* args[] = { &d_data, &size };
    cuLaunchKernel(kernel,
                   256, 1, 1,    // grid dimensions
                   256, 1, 1,    // block dimensions
                   0,            // shared memory
                   NULL,         // stream
                   args,         // arguments
                   NULL);
    
    // Cleanup
    cuMemFree(d_data);
    cuCtxDestroy(context);
    return 0;
}`,
        description: 'CUDA Driver API usage with explicit context and memory management'
      }
    ],
    relatedConcepts: ['cuda-runtime-api', 'libcuda', 'cuda-software-platform'],
    prerequisites: ['cuda-software-platform'],
    difficulty: 'advanced',
    tags: ['driver-api', 'low-level', 'context', 'explicit'],
    learningPath: ['cuda-development']
  },
  {
    id: 'cuda-runtime-api',
    title: 'CUDA Runtime API',
    category: 'host-software',
    subcategory: 'apis-runtime',
    definition: 'Higher-level wrapper around Driver API providing better ergonomics and easier-to-use interface for most CUDA applications.',
    keyPoints: [
      'Higher-level wrapper around Driver API',
      'Better ergonomics and ease of use',
      'Automatic context management',
      'Most common API for CUDA applications',
      'Built on top of Driver API'
    ],
    technicalDetails: [
      {
        label: 'Functions',
        value: 'cudaMalloc, cudaMemcpy, kernel<<<grid,block>>>',
        description: 'Simplified memory management and kernel launch'
      },
      {
        label: 'Context Management',
        value: 'Automatic context creation and management',
        description: 'No explicit context handling required'
      },
      {
        label: 'Implementation',
        value: 'libcudart.so on Linux',
        description: 'Runtime library built on Driver API'
      }
    ],
    codeExamples: [
      {
        language: 'cuda',
        code: `#include <cuda_runtime.h>

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1024;
    size_t size = n * sizeof(float);
    
    // Allocate memory (automatic context management)
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Launch kernel with simplified syntax
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    // Copy result back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Cleanup (automatic context cleanup)
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}`,
        description: 'CUDA Runtime API with simplified memory management and kernel launch'
      }
    ],
    relatedConcepts: ['cuda-driver-api', 'libcudart', 'cuda-c'],
    prerequisites: ['cuda-software-platform'],
    difficulty: 'beginner',
    tags: ['runtime-api', 'high-level', 'ergonomic', 'automatic'],
    learningPath: ['gpu-fundamentals', 'cuda-programming']
  },
  {
    id: 'libcuda',
    title: 'libcuda.so',
    category: 'host-software',
    subcategory: 'apis-runtime',
    definition: 'Binary shared object implementing CUDA Driver API on Linux, providing the interface between user applications and GPU driver.',
    keyPoints: [
      'Binary implementation of CUDA Driver API',
      'Interface between applications and GPU driver',
      'Installed with NVIDIA GPU drivers',
      'Foundation for all CUDA functionality',
      'System-level library for GPU access'
    ],
    technicalDetails: [
      {
        label: 'Role',
        value: 'Driver API implementation and GPU interface',
        description: 'Bridges user space and kernel driver'
      },
      {
        label: 'Distribution',
        value: 'Installed with NVIDIA GPU drivers',
        description: 'Part of driver package, not CUDA toolkit'
      },
      {
        label: 'Functions',
        value: 'All Driver API functionality',
        description: 'Low-level GPU operations and management'
      }
    ],
    relatedConcepts: ['cuda-driver-api', 'nvidia-gpu-drivers', 'libcudart'],
    prerequisites: ['cuda-driver-api'],
    difficulty: 'advanced',
    tags: ['library', 'driver', 'system', 'binary'],
    learningPath: ['cuda-development']
  },
  {
    id: 'libcudart',
    title: 'libcudart.so',
    category: 'host-software',
    subcategory: 'apis-runtime',
    definition: 'Binary shared object implementing CUDA Runtime API on Linux, providing higher-level interface built on Driver API.',
    keyPoints: [
      'Binary implementation of CUDA Runtime API',
      'Higher-level interface built on Driver API',
      'Part of CUDA Toolkit distribution',
      'Automatic context and resource management',
      'Most commonly used CUDA library'
    ],
    technicalDetails: [
      {
        label: 'Implementation',
        value: 'Built on top of libcuda.so',
        description: 'Higher-level wrapper around Driver API'
      },
      {
        label: 'Distribution',
        value: 'Part of CUDA Toolkit',
        description: 'Separate from driver installation'
      },
      {
        label: 'Management',
        value: 'Automatic context and resource handling',
        description: 'Simplifies GPU programming'
      }
    ],
    relatedConcepts: ['cuda-runtime-api', 'libcuda', 'cuda-software-platform'],
    prerequisites: ['cuda-runtime-api'],
    difficulty: 'intermediate',
    tags: ['library', 'runtime', 'toolkit', 'wrapper'],
    learningPath: ['cuda-development']
  },
  {
    id: 'cuda-c',
    title: 'CUDA C++',
    category: 'host-software',
    subcategory: 'development-tools',
    definition: 'C++ extension implementing CUDA programming model with kernels, memory management, and synchronization primitives.',
    keyPoints: [
      'C++ extension for GPU programming',
      'Implements CUDA programming model',
      'Kernels, memory management, synchronization',
      'Seamless integration with standard C++',
      'Requires nvcc compiler for compilation'
    ],
    technicalDetails: [
      {
        label: 'Features',
        value: '__global__ functions, <<<grid,block>>> syntax',
        description: 'GPU-specific language extensions'
      },
      {
        label: 'Integration',
        value: 'Seamless with standard C++',
        description: 'Can mix CPU and GPU code naturally'
      },
      {
        label: 'Compilation',
        value: 'Requires nvcc compiler',
        description: 'Special compiler for device code'
      }
    ],
    codeExamples: [
      {
        language: 'cuda',
        code: `#include <cuda_runtime.h>
#include <iostream>

// Device function (runs on GPU)
__device__ float deviceFunction(float x) {
    return x * x + 1.0f;
}

// Kernel function (entry point for GPU)
__global__ void processArray(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Call device function
        output[idx] = deviceFunction(input[idx]);
        
        // Use CUDA built-ins
        if (threadIdx.x == 0) {
            printf("Block %d processing %d elements\\n", blockIdx.x, blockDim.x);
        }
    }
}

// Host function (runs on CPU)
int main() {
    const int n = 1024;
    size_t size = n * sizeof(float);
    
    // Host memory allocation (standard C++)
    float* h_input = new float[n];
    float* h_output = new float[n];
    
    // Initialize data
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(i);
    }
    
    // Device memory allocation (CUDA C++ extensions)
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Launch kernel (CUDA C++ syntax)
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    processArray<<<gridSize, blockSize>>>(d_input, d_output, n);
    
    // Synchronization
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
    
    return 0;
}`,
        description: 'Complete CUDA C++ program showing language extensions and integration'
      }
    ],
    relatedConcepts: ['nvcc', 'cuda-programming-model', 'kernel', 'cuda-runtime-api'],
    prerequisites: ['cuda-programming-model'],
    difficulty: 'beginner',
    tags: ['language', 'cpp', 'kernels', 'programming'],
    learningPath: ['cuda-programming', 'cuda-development']
  },
  {
    id: 'nvcc',
    title: 'NVIDIA CUDA Compiler Driver (nvcc)',
    category: 'host-software',
    subcategory: 'development-tools',
    definition: 'Toolchain that compiles CUDA C++ to fat binaries containing PTX and SASS for multiple architectures with optimization.',
    keyPoints: [
      'Compiles CUDA C++ to executable binaries',
      'Generates PTX intermediate and SASS machine code',
      'Creates fat binaries for multiple GPU architectures',
      'Applies multiple optimization passes',
      'Integrates with build systems and IDEs'
    ],
    technicalDetails: [
      {
        label: 'Compilation Flow',
        value: 'CUDA C++ → PTX → SASS → Fat Binary',
        description: 'Multi-stage compilation process'
      },
      {
        label: 'Output',
        value: 'Fat binaries with PTX and SASS for multiple targets',
        description: 'Single binary supports multiple GPU generations'
      },
      {
        label: 'Integration',
        value: 'Works with make, CMake, Visual Studio',
        description: 'Standard build system integration'
      }
    ],
    codeExamples: [
      {
        language: 'bash',
        code: `# Basic compilation
nvcc -o program program.cu

# Specify compute capability
nvcc -arch=sm_80 -o program program.cu

# Generate PTX for multiple architectures
nvcc -arch=sm_70 -arch=sm_80 -arch=sm_90 -o program program.cu

# Debug build with device debug info
nvcc -g -G -o program program.cu

# Optimized release build
nvcc -O3 -use_fast_math -o program program.cu

# Generate PTX assembly
nvcc -ptx program.cu

# Verbose compilation (show what nvcc is doing)
nvcc -v -o program program.cu

# Link with libraries
nvcc -lcublas -lcurand -o program program.cu

# Separate compilation (compile to object file)
nvcc -dc -o program.o program.cu
nvcc -dlink -o device_link.o program.o
g++ -o program program.o device_link.o -lcudart

# Show register usage and memory info
nvcc -Xptxas -v -o program program.cu`,
        description: 'Common nvcc compilation patterns and options'
      }
    ],
    relatedConcepts: ['cuda-c', 'parallel-thread-execution', 'streaming-assembler', 'compute-capability'],
    prerequisites: ['cuda-c'],
    difficulty: 'intermediate',
    tags: ['compiler', 'toolchain', 'optimization', 'build'],
    learningPath: ['cuda-development']
  },
  {
    id: 'nvrtc',
    title: 'NVIDIA Runtime Compiler (nvrtc)',
    category: 'host-software',
    subcategory: 'development-tools',
    definition: 'Runtime compilation library that compiles CUDA C strings to PTX at runtime, enabling dynamic kernel generation and specialization.',
    keyPoints: [
      'Runtime compilation of CUDA C to PTX',
      'Compiles CUDA C strings at application runtime',
      'Enables dynamic kernel generation and specialization',
      'Useful for JIT compilation and optimization',
      'Advanced applications with runtime code generation'
    ],
    technicalDetails: [
      {
        label: 'Input',
        value: 'CUDA C++ source code as strings',
        description: 'Source code provided at runtime'
      },
      {
        label: 'Output',
        value: 'PTX assembly code',
        description: 'Compiled PTX ready for execution'
      },
      {
        label: 'Use Cases',
        value: 'JIT compilation, runtime specialization',
        description: 'Dynamic optimization and code generation'
      }
    ],
    codeExamples: [
      {
        language: 'c',
        code: `#include <nvrtc.h>
#include <cuda.h>

int main() {
    // CUDA C++ source code as string
    const char* kernel_source = R"(
        extern "C" __global__ void vectorAdd(float* a, float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
    )";
    
    // Create NVRTC program
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, kernel_source, "vectorAdd.cu", 0, NULL, NULL);
    
    // Add compilation options
    const char* opts[] = {"--gpu-architecture=compute_80", "--use_fast_math"};
    
    // Compile the program
    nvrtcResult result = nvrtcCompileProgram(prog, 2, opts);
    
    if (result != NVRTC_SUCCESS) {
        // Handle compilation errors
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        char* log = new char[log_size];
        nvrtcGetProgramLog(prog, log);
        printf("Compilation error: %s\\n", log);
        delete[] log;
        return -1;
    }
    
    // Get compiled PTX
    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    char* ptx = new char[ptx_size];
    nvrtcGetPTX(prog, ptx);
    
    // Load and execute PTX using Driver API
    CUmodule module;
    CUfunction kernel;
    cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
    cuModuleGetFunction(&kernel, module, "vectorAdd");
    
    // Launch kernel...
    
    // Cleanup
    delete[] ptx;
    nvrtcDestroyProgram(&prog);
    return 0;
}`,
        description: 'Runtime compilation using NVRTC for dynamic kernel generation'
      }
    ],
    relatedConcepts: ['nvcc', 'parallel-thread-execution', 'cuda-driver-api'],
    prerequisites: ['nvcc', 'parallel-thread-execution'],
    difficulty: 'advanced',
    tags: ['runtime-compilation', 'jit', 'dynamic', 'specialization'],
    learningPath: ['cuda-development']
  },
  {
    id: 'cuda-binary-utilities',
    title: 'CUDA Binary Utilities',
    category: 'host-software',
    subcategory: 'development-tools',
    definition: 'Tools for examining and manipulating CUDA binaries, including cuobjdump and nvidisasm for disassembly and analysis.',
    keyPoints: [
      'Tools for examining CUDA binaries',
      'cuobjdump for binary inspection and disassembly',
      'nvidisasm for SASS disassembly',
      'fatbinary for fat binary manipulation',
      'Essential for debugging and optimization'
    ],
    technicalDetails: [
      {
        label: 'Tools',
        value: 'cuobjdump, nvidisasm, fatbinary',
        description: 'Binary analysis and manipulation utilities'
      },
      {
        label: 'Functions',
        value: 'Disassemble PTX/SASS, examine binaries',
        description: 'Inspect compiled code and binary structure'
      },
      {
        label: 'Use Cases',
        value: 'Debugging, optimization, reverse engineering',
        description: 'Understanding compiled code behavior'
      }
    ],
    codeExamples: [
      {
        language: 'bash',
        code: `# Examine CUDA binary sections
cuobjdump -all program

# Disassemble PTX code
cuobjdump -ptx program

# Disassemble SASS code
cuobjdump -sass program

# Show only SASS for specific architecture
cuobjdump -arch sm_80 -sass program

# Extract PTX to file
cuobjdump -xptx program.ptx program

# Show binary resources
cuobjdump -res program

# Detailed binary information
cuobjdump -elf program

# Using nvidisasm for SASS disassembly
nvidisasm -c program.cubin

# Disassemble specific function
nvidisasm -fun vectorAdd program.cubin

# Show control flow information
nvidisasm -cfg program.cubin

# Binary file analysis
file program
objdump -h program
readelf -S program

# Extract and examine fat binary
cuobjdump -xfatbin fatbin.fatbin program`,
        description: 'CUDA binary analysis and disassembly commands'
      }
    ],
    relatedConcepts: ['nvcc', 'parallel-thread-execution', 'streaming-assembler'],
    prerequisites: ['nvcc'],
    difficulty: 'advanced',
    tags: ['binary-analysis', 'disassembly', 'debugging', 'tools'],
    learningPath: ['cuda-development']
  },
  {
    id: 'nsight-systems',
    title: 'NVIDIA Nsight Systems',
    category: 'host-software',
    subcategory: 'profiling-management',
    definition: 'Performance debugging tool providing profiling, tracing, and analysis with graphical interface for CUDA application optimization.',
    keyPoints: [
      'Comprehensive performance debugging tool',
      'Timeline tracing of API calls and kernel execution',
      'Graphical interface for analysis',
      'System-wide profiling capabilities',
      'Integration with CUDA applications and frameworks'
    ],
    technicalDetails: [
      {
        label: 'Capabilities',
        value: 'Timeline tracing, API calls, kernel metrics',
        description: 'Comprehensive performance data collection'
      },
      {
        label: 'Visualization',
        value: 'Graphical timeline and analysis interface',
        description: 'Visual performance analysis tools'
      },
      {
        label: 'Integration',
        value: 'CUDA, OpenACC, OpenMP, framework support',
        description: 'Works with various parallel programming models'
      }
    ],
    codeExamples: [
      {
        language: 'bash',
        code: `# Basic profiling
nsys profile ./my_cuda_program

# Profile with specific options
nsys profile --stats=true --force-overwrite=true ./program

# Profile CUDA API and kernels only
nsys profile --trace=cuda,nvtx ./program

# Generate detailed report
nsys profile --stats=true --export=sqlite ./program
nsys stats --report summary report.sqlite

# Profile with NVTX markers
nsys profile --trace=cuda,nvtx,osrt ./program

# Command line analysis
nsys stats --report gputrace report.nsys-rep
nsys stats --report cudaapisum report.nsys-rep
nsys stats --report kernelsum report.nsys-rep

# Export for external analysis
nsys export --type=sqlite --output=data.sqlite report.nsys-rep`,
        description: 'Nsight Systems profiling commands and analysis'
      },
      {
        language: 'cuda',
        code: `#include <nvtx3/nvToolsExt.h>

__global__ void kernel1(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

__global__ void kernel2(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

int main() {
    // NVTX markers for better profiling
    nvtxRangePush("Initialization");
    
    float *d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));
    
    nvtxRangePop();
    
    // Profile kernel launches
    nvtxRangePush("Kernel1 Execution");
    kernel1<<<32, 32>>>(d_data, 1024);
    cudaDeviceSynchronize();
    nvtxRangePop();
    
    nvtxRangePush("Kernel2 Execution");
    kernel2<<<32, 32>>>(d_data, 1024);
    cudaDeviceSynchronize();
    nvtxRangePop();
    
    nvtxRangePush("Cleanup");
    cudaFree(d_data);
    nvtxRangePop();
    
    return 0;
}`,
        description: 'CUDA code with NVTX markers for better Nsight Systems profiling'
      }
    ],
    relatedConcepts: ['cupti', 'cuda-runtime-api', 'cuda-driver-api'],
    prerequisites: ['cuda-runtime-api'],
    difficulty: 'intermediate',
    tags: ['profiling', 'performance', 'debugging', 'analysis'],
    learningPath: ['cuda-development']
  },
  {
    id: 'cupti',
    title: 'CUDA Profiling Tools Interface (CUPTI)',
    category: 'host-software',
    subcategory: 'profiling-management',
    definition: 'APIs for profiling CUDA execution with synchronized timestamps, providing foundation for building custom profiling tools.',
    keyPoints: [
      'APIs for building custom profiling tools',
      'Synchronized timestamps for accurate measurement',
      'Performance counters and metrics collection',
      'API tracing and kernel execution monitoring',
      'Foundation for tools like Nsight Systems'
    ],
    technicalDetails: [
      {
        label: 'Purpose',
        value: 'Building custom profiling and analysis tools',
        description: 'API foundation for performance tools'
      },
      {
        label: 'Metrics',
        value: 'Performance counters, API tracing, kernel metrics',
        description: 'Comprehensive performance data access'
      },
      {
        label: 'Integration',
        value: 'Used by Nsight Systems and other profilers',
        description: 'Backend for performance analysis tools'
      }
    ],
    relatedConcepts: ['nsight-systems', 'cuda-runtime-api', 'cuda-driver-api'],
    prerequisites: ['cuda-runtime-api'],
    difficulty: 'expert',
    tags: ['profiling-api', 'metrics', 'performance', 'tools'],
    learningPath: ['cuda-development']
  },
  {
    id: 'nvidia-smi',
    title: 'nvidia-smi',
    category: 'host-software',
    subcategory: 'profiling-management',
    definition: 'Command line utility for querying and managing GPU state, including utilization, memory usage, temperature, and power monitoring.',
    keyPoints: [
      'Command line GPU management utility',
      'Monitor GPU utilization, memory, temperature, power',
      'Real-time system monitoring capabilities',
      'Basic GPU management operations',
      'Essential for system administration and debugging'
    ],
    technicalDetails: [
      {
        label: 'Functions',
        value: 'GPU monitoring, process listing, basic management',
        description: 'System administration and monitoring'
      },
      {
        label: 'Real-time',
        value: 'Continuous monitoring with watch mode',
        description: 'Live system state monitoring'
      },
      {
        label: 'Output',
        value: 'Human-readable and machine-parseable formats',
        description: 'Flexible output for scripts and monitoring'
      }
    ],
    codeExamples: [
      {
        language: 'bash',
        code: `# Basic GPU information
nvidia-smi

# Continuous monitoring (update every 2 seconds)
nvidia-smi -l 2

# Query specific information
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv

# List running processes
nvidia-smi pmon -i 0

# Detailed process information
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv

# Power and temperature monitoring
nvidia-smi --query-gpu=power.draw,temperature.gpu --format=csv

# Reset GPU (requires admin privileges)
nvidia-smi --gpu-reset -i 0

# Set persistence mode
nvidia-smi -pm 1

# Set power limit (if supported)
nvidia-smi -pl 250

# XML output for parsing
nvidia-smi -q -x

# Watch specific processes
watch -n 1 'nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv'

# Log GPU metrics to file
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,temperature.gpu --format=csv -l 1 -f gpu_log.csv`,
        description: 'Common nvidia-smi usage patterns for monitoring and management'
      }
    ],
    relatedConcepts: ['nvml', 'libnvml', 'nvidia-gpu-drivers'],
    prerequisites: ['nvidia-gpu-drivers'],
    difficulty: 'beginner',
    tags: ['monitoring', 'management', 'system-admin', 'cli'],
    learningPath: ['cuda-development']
  },
  {
    id: 'nvml',
    title: 'NVIDIA Management Library (NVML)',
    category: 'host-software',
    subcategory: 'profiling-management',
    definition: 'Library for programmatic monitoring of GPU state, power, temperature, and utilization, providing the functionality behind nvidia-smi.',
    keyPoints: [
      'Programmatic access to GPU monitoring',
      'Monitor state, power, temperature, utilization',
      'Foundation for nvidia-smi functionality',
      'Integration with monitoring systems',
      'Administrative and telemetry applications'
    ],
    technicalDetails: [
      {
        label: 'Purpose',
        value: 'Programmatic GPU monitoring and management',
        description: 'API for system monitoring applications'
      },
      {
        label: 'Functions',
        value: 'Similar to nvidia-smi but as library API',
        description: 'All nvidia-smi functionality via API calls'
      },
      {
        label: 'Integration',
        value: 'Used by monitoring systems and applications',
        description: 'Foundation for third-party monitoring tools'
      }
    ],
    relatedConcepts: ['nvidia-smi', 'libnvml', 'nvidia-gpu-drivers'],
    prerequisites: ['nvidia-smi'],
    difficulty: 'intermediate',
    tags: ['monitoring-api', 'management', 'telemetry', 'library'],
    learningPath: ['cuda-development']
  },
  {
    id: 'libnvml',
    title: 'libnvml.so',
    category: 'host-software',
    subcategory: 'profiling-management',
    definition: 'Binary shared object implementing NVML functions on Linux, providing programmatic access to GPU management and monitoring.',
    keyPoints: [
      'Binary implementation of NVML API',
      'Programmatic GPU management and monitoring',
      'Part of NVIDIA GPU driver distribution',
      'Used by nvidia-smi and other monitoring tools',
      'System-level library for GPU telemetry'
    ],
    technicalDetails: [
      {
        label: 'Implementation',
        value: 'Shared library implementing NVML API',
        description: 'Binary interface for GPU monitoring'
      },
      {
        label: 'Distribution',
        value: 'Part of NVIDIA GPU drivers',
        description: 'Installed with driver package'
      },
      {
        label: 'Usage',
        value: 'Backend for nvidia-smi and monitoring apps',
        description: 'Foundation for GPU monitoring tools'
      }
    ],
    relatedConcepts: ['nvml', 'nvidia-smi', 'nvidia-gpu-drivers'],
    prerequisites: ['nvml'],
    difficulty: 'advanced',
    tags: ['library', 'binary', 'monitoring', 'system'],
    learningPath: ['cuda-development']
  },
  {
    id: 'nvidia-gpu-drivers',
    title: 'NVIDIA GPU Drivers',
    category: 'host-software',
    subcategory: 'system-integration',
    definition: 'System software that mediates interaction between host programs and GPU device, including kernel and userspace components.',
    keyPoints: [
      'System software for GPU-host communication',
      'Kernel-level and userspace components',
      'Hardware abstraction and resource management',
      'Foundation for all GPU functionality',
      'Includes libcuda.so and kernel module'
    ],
    technicalDetails: [
      {
        label: 'Components',
        value: 'nvidia.ko (kernel) + libcuda.so (userspace)',
        description: 'Kernel module and userspace library'
      },
      {
        label: 'Functions',
        value: 'Hardware abstraction, memory management, scheduling',
        description: 'Complete GPU system integration'
      },
      {
        label: 'Role',
        value: 'Bridge between applications and hardware',
        description: 'Enables software to use GPU hardware'
      }
    ],
    relatedConcepts: ['nvidia-ko', 'libcuda', 'cuda-driver-api'],
    prerequisites: [],
    difficulty: 'intermediate',
    tags: ['drivers', 'system', 'hardware-abstraction', 'kernel'],
    learningPath: ['cuda-development']
  },
  {
    id: 'nvidia-ko',
    title: 'nvidia.ko',
    category: 'host-software',
    subcategory: 'system-integration',
    definition: 'Linux kernel module executing in privileged mode for direct hardware communication and low-level GPU resource management.',
    keyPoints: [
      'Linux kernel module for GPU hardware access',
      'Executes in privileged kernel mode',
      'Direct hardware communication and control',
      'Memory allocation and context switching',
      'Interrupt handling and resource management'
    ],
    technicalDetails: [
      {
        label: 'Privilege Level',
        value: 'Kernel mode with hardware access',
        description: 'Privileged access for direct hardware control'
      },
      {
        label: 'Functions',
        value: 'Memory allocation, context switching, interrupts',
        description: 'Low-level GPU resource management'
      },
      {
        label: 'Security',
        value: 'Runs with kernel privileges',
        description: 'Trusted system component for hardware access'
      }
    ],
    relatedConcepts: ['nvidia-gpu-drivers', 'libcuda', 'cuda-driver-api'],
    prerequisites: ['nvidia-gpu-drivers'],
    difficulty: 'expert',
    tags: ['kernel-module', 'hardware', 'privileged', 'system'],
    learningPath: ['cuda-development']
  }
];