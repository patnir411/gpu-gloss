import type { Concept } from '../../types';

export const deviceSoftwareConcepts: Concept[] = [
  {
    id: 'cuda-programming-model',
    title: 'CUDA Programming Model',
    category: 'device-software',
    subcategory: 'programming-model',
    definition: 'Programming model with three key abstractions: hierarchy of thread groups, hierarchy of memories, and barrier synchronization. Enables scalable parallelism through hierarchical organization.',
    keyPoints: [
      'Three key abstractions for parallel programming',
      'Hierarchy of thread groups (thread → warp → block → grid)',
      'Hierarchy of memories (registers → shared → global)',
      'Barrier synchronization for coordination',
      'Maps naturally to GPU hardware architecture'
    ],
    technicalDetails: [
      {
        label: 'Thread Hierarchy',
        value: 'Thread → Warp → Block → Grid',
        description: 'Nested levels of parallel execution organization'
      },
      {
        label: 'Memory Hierarchy',
        value: 'Registers → Shared → Global',
        description: 'Memory levels matching thread group hierarchy'
      },
      {
        label: 'Synchronization',
        value: 'Barrier synchronization within thread blocks',
        description: 'Coordination mechanism for parallel threads'
      }
    ],
    codeExamples: [
      {
        language: 'cuda',
        code: `// CUDA programming model example
__global__ void matrixMultiply(float* A, float* B, float* C, int N) {
    // Thread hierarchy: grid contains blocks, blocks contain threads
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Memory hierarchy: shared memory for block-level cooperation
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    for (int k = 0; k < N / TILE_SIZE; k++) {
        // Load from global to shared memory
        sA[threadIdx.y][threadIdx.x] = A[row * N + k * TILE_SIZE + threadIdx.x];
        sB[threadIdx.y][threadIdx.x] = B[(k * TILE_SIZE + threadIdx.y) * N + col];
        
        // Barrier synchronization
        __syncthreads();
        
        // Compute using shared memory
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}`,
        description: 'Matrix multiplication showing all three CUDA programming model abstractions'
      }
    ],
    relatedConcepts: ['thread', 'warp', 'thread-block', 'thread-block-grid', 'memory-hierarchy', 'streaming-multiprocessor'],
    prerequisites: [],
    difficulty: 'beginner',
    tags: ['programming-model', 'hierarchy', 'parallelism', 'synchronization'],
    learningPath: ['gpu-fundamentals', 'cuda-programming']
  },
  {
    id: 'compute-capability',
    title: 'Compute Capability',
    category: 'device-software',
    subcategory: 'programming-model',
    definition: 'Versioning system for PTX instruction compatibility with physical GPUs. Determines which PTX instructions are supported by the hardware.',
    keyPoints: [
      'Versioning system for PTX instruction compatibility',
      'Format: Major.Minor (e.g., 8.0 for H100)',
      'Determines supported PTX instruction set',
      'Forward compatible within major versions',
      'Used by compiler to target specific hardware'
    ],
    technicalDetails: [
      {
        label: 'Format',
        value: 'Major.Minor (e.g., 8.0, 7.5, 6.1)',
        description: 'Version numbering for hardware capabilities'
      },
      {
        label: 'Compatibility',
        value: 'Forward compatible within major versions',
        description: 'Newer hardware can run older PTX code'
      },
      {
        label: 'Usage',
        value: 'Compiler targeting and optimization',
        description: 'Determines which instructions can be generated'
      }
    ],
    relatedConcepts: ['parallel-thread-execution', 'nvcc', 'streaming-multiprocessor-architecture'],
    prerequisites: ['parallel-thread-execution'],
    difficulty: 'intermediate',
    tags: ['versioning', 'compatibility', 'ptx', 'hardware'],
    learningPath: ['cuda-development', 'gpu-architecture']
  },
  {
    id: 'parallel-thread-execution',
    title: 'Parallel Thread Execution (PTX)',
    category: 'device-software',
    subcategory: 'programming-model',
    definition: 'Intermediate representation and virtual machine instruction set that serves as compilation target for CUDA C++ and source for SASS compiler.',
    keyPoints: [
      'Intermediate representation between CUDA C++ and SASS',
      'Virtual machine instruction set architecture',
      'Hardware-independent compilation target',
      'Enables optimization opportunities',
      'Stable API that evolves with hardware'
    ],
    technicalDetails: [
      {
        label: 'Role',
        value: 'Compilation intermediate: CUDA C++ → PTX → SASS',
        description: 'Bridge between high-level code and hardware'
      },
      {
        label: 'Format',
        value: 'Assembly-like syntax with virtual registers',
        description: 'Human-readable assembly format'
      },
      {
        label: 'Benefits',
        value: 'Hardware independence and optimization',
        description: 'Allows forward compatibility and driver optimization'
      }
    ],
    codeExamples: [
      {
        language: 'ptx',
        code: `// PTX assembly example
.version 7.0
.target sm_80
.address_size 64

.visible .entry vectorAdd(
    .param .u64 vectorAdd_param_0,
    .param .u64 vectorAdd_param_1,
    .param .u64 vectorAdd_param_2,
    .param .u32 vectorAdd_param_3
) {
    .reg .pred %p<2>;
    .reg .f32 %f<4>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<11>;

    ld.param.u64 %rd1, [vectorAdd_param_0];
    ld.param.u64 %rd2, [vectorAdd_param_1];
    ld.param.u64 %rd3, [vectorAdd_param_2];
    ld.param.u32 %r1, [vectorAdd_param_3];
    
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mul.lo.s32 %r4, %r2, %r3;
    mov.u32 %r5, %tid.x;
    add.s32 %r4, %r4, %r5;
    
    setp.ge.s32 %p1, %r4, %r1;
    @%p1 bra BB0_2;
    
    cvt.s64.s32 %rd4, %r4;
    mul.wide.s32 %rd5, %r4, 4;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.f32 %f1, [%rd6];
    add.s64 %rd7, %rd2, %rd5;
    ld.global.f32 %f2, [%rd7];
    add.f32 %f3, %f1, %f2;
    add.s64 %rd8, %rd3, %rd5;
    st.global.f32 [%rd8], %f3;

BB0_2:
    ret;
}`,
        description: 'PTX assembly for vector addition kernel'
      }
    ],
    relatedConcepts: ['streaming-assembler', 'compute-capability', 'nvcc', 'cuda-c'],
    prerequisites: ['cuda-programming-model'],
    difficulty: 'advanced',
    tags: ['assembly', 'intermediate', 'compilation', 'virtual-machine'],
    learningPath: ['cuda-development', 'gpu-architecture']
  },
  {
    id: 'streaming-assembler',
    title: 'Streaming Assembler (SASS)',
    category: 'device-software',
    subcategory: 'programming-model',
    definition: 'Lowest-level assembly format for NVIDIA GPUs, containing the actual machine instructions executed by the hardware after compilation from PTX.',
    keyPoints: [
      'Lowest-level assembly format for NVIDIA GPUs',
      'Actual machine instructions executed by hardware',
      'Generated from PTX by GPU driver',
      'Hardware-specific optimizations applied',
      'Can be examined with cuobjdump/nvidisasm'
    ],
    technicalDetails: [
      {
        label: 'Generation',
        value: 'Compiled from PTX by GPU driver',
        description: 'Final compilation step to hardware instructions'
      },
      {
        label: 'Optimization',
        value: 'Hardware-specific optimizations',
        description: 'Driver applies target-specific optimizations'
      },
      {
        label: 'Examination',
        value: 'cuobjdump, nvidisasm tools',
        description: 'Tools for inspecting generated SASS code'
      }
    ],
    codeExamples: [
      {
        language: 'sass',
        code: `// SASS assembly example (H100)
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   S2R R0, SR_CTAID.X ;
        /*0020*/                   S2R R2, SR_TID.X ;
        /*0030*/                   IMAD R0, R0, c[0x0][0x0], R2 ;
        /*0040*/                   ISETP.GE.AND P0, PT, R0, c[0x0][0x160], PT ;
        /*0050*/              @P0  EXIT ;
        /*0060*/                   IMAD R2, R0, 0x4, c[0x0][0x140] ;
        /*0070*/                   IMAD R3, R0, 0x4, c[0x0][0x148] ;
        /*0080*/                   LDG.E R4, [R2] ;
        /*0090*/                   LDG.E R2, [R3] ;
        /*00a0*/                   FADD R4, R4, R2 ;
        /*00b0*/                   IMAD R2, R0, 0x4, c[0x0][0x150] ;
        /*00c0*/                   STG.E [R2], R4 ;
        /*00d0*/                   EXIT ;`,
        description: 'SASS assembly for vector addition (actual H100 instructions)'
      }
    ],
    relatedConcepts: ['parallel-thread-execution', 'cuda-binary-utilities', 'compute-capability'],
    prerequisites: ['parallel-thread-execution'],
    difficulty: 'expert',
    tags: ['assembly', 'machine-code', 'hardware', 'optimization'],
    learningPath: ['cuda-development', 'gpu-architecture']
  },
  {
    id: 'thread',
    title: 'Thread',
    category: 'device-software',
    subcategory: 'thread-hierarchy',
    definition: 'The lowest unit of GPU programming, with its own registers, instruction pointer, and execution state. Executes kernel code with a unique thread ID.',
    keyPoints: [
      'Lowest unit of GPU programming',
      'Has own registers, instruction pointer, execution state',
      'Executes kernel code with unique thread ID',
      'Access to private registers and shared memory with block',
      'Created when kernel launches, destroyed when complete'
    ],
    technicalDetails: [
      {
        label: 'Resources',
        value: 'Private registers, instruction pointer, execution state',
        description: 'Each thread has independent execution context'
      },
      {
        label: 'Memory Access',
        value: 'Registers (private), shared (block), global (all)',
        description: 'Hierarchical memory access based on scope'
      },
      {
        label: 'Identification',
        value: 'Unique thread ID within block and grid',
        description: 'threadIdx, blockIdx for data partitioning'
      }
    ],
    codeExamples: [
      {
        language: 'cuda',
        code: `__global__ void simpleKernel(float* data, int n) {
    // Each thread has unique ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Thread-private variables (in registers)
    float temp = 0.0f;
    int localVar = tid * 2;
    
    // Each thread processes different data element
    if (tid < n) {
        temp = data[tid] * 2.0f;
        data[tid] = temp + localVar;
    }
}

// Launch with multiple threads
int main() {
    int n = 1024;
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    
    // Each thread executes the kernel independently
    simpleKernel<<<gridSize, blockSize>>>(data, n);
}`,
        description: 'Example showing thread independence and unique IDs'
      }
    ],
    relatedConcepts: ['warp', 'thread-block', 'registers', 'kernel'],
    prerequisites: [],
    difficulty: 'beginner',
    tags: ['thread', 'parallelism', 'execution', 'registers'],
    learningPath: ['gpu-fundamentals', 'cuda-programming']
  },
  {
    id: 'warp',
    title: 'Warp',
    category: 'device-software',
    subcategory: 'thread-hierarchy',
    definition: 'Group of 32 threads scheduled together in SIMT (Single Instruction, Multiple Thread) model. The fundamental unit of GPU scheduling and execution.',
    keyPoints: [
      'Group of 32 threads scheduled together',
      'SIMT: all threads execute same instruction simultaneously',
      'Fundamental unit of GPU scheduling',
      'Divergent branches reduce efficiency',
      'Atomic unit for warp scheduler'
    ],
    technicalDetails: [
      {
        label: 'Size',
        value: '32 threads per warp',
        description: 'Fixed size across all NVIDIA GPU architectures'
      },
      {
        label: 'Execution Model',
        value: 'SIMT (Single Instruction, Multiple Thread)',
        description: 'All threads execute same instruction on different data'
      },
      {
        label: 'Scheduling',
        value: 'Atomic unit for warp scheduler',
        description: 'Entire warp scheduled together, not individual threads'
      }
    ],
    codeExamples: [
      {
        language: 'cuda',
        code: `__global__ void warpExample(int* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid / 32;  // Which warp this thread belongs to
    int laneId = tid % 32;  // Position within the warp (0-31)
    
    // All threads in warp execute same instruction
    if (tid < n) {
        data[tid] = warpId * 100 + laneId;
    }
    
    // Warp-level primitives
    int value = data[tid];
    
    // Warp shuffle - communicate within warp
    int doubled = __shfl_xor_sync(0xffffffff, value, 1);
    
    // Warp vote - collective decision within warp
    int mask = __ballot_sync(0xffffffff, value > 50);
    
    if (laneId == 0) {
        // Only first thread in warp executes
        printf("Warp %d has %d threads with value > 50\\n", 
               warpId, __popc(mask));
    }
}`,
        description: 'Example showing warp organization and warp-level operations'
      }
    ],
    relatedConcepts: ['thread', 'thread-block', 'warp-scheduler', 'streaming-multiprocessor'],
    prerequisites: ['thread'],
    difficulty: 'intermediate',
    tags: ['warp', 'simt', 'scheduling', 'collective'],
    learningPath: ['cuda-programming', 'gpu-architecture']
  },
  {
    id: 'thread-block',
    title: 'Thread Block',
    category: 'device-software',
    subcategory: 'thread-hierarchy',
    definition: 'Collection of threads scheduled onto the same Streaming Multiprocessor, enabling cooperation via shared memory and synchronization primitives.',
    keyPoints: [
      'Collection of threads on same SM',
      'Threads can cooperate via shared memory',
      'Barrier synchronization available within block',
      'Typically 32-1024 threads (multiple of 32)',
      'Entire block assigned to single SM'
    ],
    technicalDetails: [
      {
        label: 'Size',
        value: '32-1024 threads (must be multiple of 32)',
        description: 'Size determines resource usage and occupancy'
      },
      {
        label: 'Scheduling',
        value: 'Entire block assigned to single SM',
        description: 'All threads in block execute on same SM'
      },
      {
        label: 'Cooperation',
        value: 'Shared memory and __syncthreads()',
        description: 'Mechanisms for intra-block communication'
      }
    ],
    codeExamples: [
      {
        language: 'cuda',
        code: `__global__ void blockCooperationExample(float* input, float* output, int n) {
    // Shared memory visible to all threads in block
    __shared__ float shared_data[256];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (gid < n) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 0.0f;
    }
    
    // Synchronize all threads in block
    __syncthreads();
    
    // Parallel reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();  // Ensure all threads complete before next iteration
    }
    
    // First thread writes block result
    if (tid == 0 && blockIdx.x < (n + blockDim.x - 1) / blockDim.x) {
        output[blockIdx.x] = shared_data[0];
    }
}

// Launch configuration specifies block size
int main() {
    dim3 blockSize(256);  // 256 threads per block
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    
    blockCooperationExample<<<gridSize, blockSize>>>(input, output, n);
}`,
        description: 'Block-level cooperation using shared memory and synchronization'
      }
    ],
    relatedConcepts: ['thread', 'warp', 'shared-memory', 'thread-block-grid', 'streaming-multiprocessor'],
    prerequisites: ['thread', 'warp'],
    difficulty: 'intermediate',
    tags: ['block', 'cooperation', 'shared-memory', 'synchronization'],
    learningPath: ['cuda-programming']
  },
  {
    id: 'cooperative-thread-array',
    title: 'Cooperative Thread Array (CTA)',
    category: 'device-software',
    subcategory: 'thread-hierarchy',
    definition: 'PTX/SASS implementation term for thread blocks, representing the same cooperation and synchronization capabilities at the hardware level.',
    keyPoints: [
      'PTX/SASS term for thread blocks',
      'Hardware-level representation of block cooperation',
      'Same synchronization and memory sharing capabilities',
      'Appears in PTX code and hardware documentation',
      'Equivalent functionality to thread blocks'
    ],
    technicalDetails: [
      {
        label: 'Equivalence',
        value: 'CTA = Thread Block at lower level',
        description: 'Same concept at different abstraction levels'
      },
      {
        label: 'Usage',
        value: 'PTX assembly and hardware documentation',
        description: 'Low-level term for the same high-level concept'
      },
      {
        label: 'Capabilities',
        value: 'Cooperation and synchronization like thread blocks',
        description: 'All thread block features available'
      }
    ],
    relatedConcepts: ['thread-block', 'parallel-thread-execution', 'streaming-assembler'],
    prerequisites: ['thread-block', 'parallel-thread-execution'],
    difficulty: 'advanced',
    tags: ['cta', 'ptx', 'hardware', 'cooperation'],
    learningPath: ['cuda-development', 'gpu-architecture']
  },
  {
    id: 'thread-block-grid',
    title: 'Thread Block Grid',
    category: 'device-software',
    subcategory: 'thread-hierarchy',
    definition: 'Highest level of thread organization containing multiple thread blocks that can span multiple SMs across the entire GPU.',
    keyPoints: [
      'Highest level containing multiple thread blocks',
      'Can span multiple SMs across entire GPU',
      'Limited coordination between blocks (atomic operations)',
      'Enables scaling across different GPU sizes',
      'Blocks execute independently and in any order'
    ],
    technicalDetails: [
      {
        label: 'Spanning',
        value: 'Multiple SMs across entire GPU',
        description: 'Grid can utilize full GPU resources'
      },
      {
        label: 'Coordination',
        value: 'Limited - atomic operations only',
        description: 'No shared memory or barriers between blocks'
      },
      {
        label: 'Scalability',
        value: 'Independent block execution',
        description: 'Same grid can run on different GPU sizes'
      }
    ],
    codeExamples: [
      {
        language: 'cuda',
        code: `__global__ void gridExample(float* data, int* blockResults, int n) {
    // Grid spans multiple blocks across multiple SMs
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = blockId * blockDim.x + threadIdx.x;
    
    __shared__ float blockSum;
    
    // Initialize shared memory (once per block)
    if (threadIdx.x == 0) {
        blockSum = 0.0f;
    }
    __syncthreads();
    
    // Each block processes its portion of data
    float threadSum = 0.0f;
    for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
        threadSum += data[i];
    }
    
    // Atomic add within block (block-level cooperation)
    atomicAdd(&blockSum, threadSum);
    __syncthreads();
    
    // First thread writes block result (grid-level result)
    if (threadIdx.x == 0) {
        blockResults[blockId] = blockSum;
    }
}

// Launch configuration creates grid
int main() {
    dim3 blockSize(256);
    dim3 gridSize(32, 32);  // 2D grid of 1024 blocks total
    
    // Grid spans multiple SMs, blocks execute independently
    gridExample<<<gridSize, blockSize>>>(data, blockResults, n);
    
    // Blocks can complete in any order
    cudaDeviceSynchronize();
}`,
        description: 'Grid organization with independent block execution'
      }
    ],
    relatedConcepts: ['thread-block', 'kernel', 'streaming-multiprocessor'],
    prerequisites: ['thread-block'],
    difficulty: 'intermediate',
    tags: ['grid', 'scalability', 'independence', 'gpu-wide'],
    learningPath: ['cuda-programming']
  },
  {
    id: 'kernel',
    title: 'Kernel',
    category: 'device-software',
    subcategory: 'thread-hierarchy',
    definition: 'Unit of CUDA code launched once but executed by many threads. Each thread executes the same kernel code with different data.',
    keyPoints: [
      'Unit of CUDA code executed by many threads',
      'Launched once, executed by entire grid',
      'Each thread runs same code with different data',
      'Specified with grid and block dimensions',
      'Types: regular, cooperative, dynamic parallelism'
    ],
    technicalDetails: [
      {
        label: 'Execution',
        value: 'Same code, different data (SPMD)',
        description: 'Single Program, Multiple Data paradigm'
      },
      {
        label: 'Launch',
        value: 'Grid and block dimensions specified',
        description: 'Configuration determines thread organization'
      },
      {
        label: 'Types',
        value: 'Regular, cooperative, dynamic parallelism',
        description: 'Different kernel types for different needs'
      }
    ],
    codeExamples: [
      {
        language: 'cuda',
        code: `// Kernel definition - executed by many threads
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    // Each thread computes unique index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Same code, different data
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // Each thread processes different elements
    }
}

// Cooperative kernel - requires grid-wide synchronization
__global__ void cooperativeKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: process data
    if (idx < n) {
        data[idx] *= 2.0f;
    }
    
    // Grid-wide synchronization
    cooperative_groups::this_grid().sync();
    
    // Phase 2: all blocks synchronized
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

int main() {
    // Regular kernel launch
    vectorAdd<<<gridSize, blockSize>>>(a, b, c, n);
    
    // Cooperative kernel launch (requires special API)
    cudaLaunchCooperativeKernel((void*)cooperativeKernel, gridSize, blockSize, args);
}`,
        description: 'Different kernel types and launch methods'
      }
    ],
    relatedConcepts: ['thread', 'thread-block', 'thread-block-grid', 'cuda-programming-model'],
    prerequisites: ['thread', 'thread-block'],
    difficulty: 'beginner',
    tags: ['kernel', 'execution', 'spmd', 'launch'],
    learningPath: ['gpu-fundamentals', 'cuda-programming']
  },
  {
    id: 'registers',
    title: 'Registers',
    category: 'device-software',
    subcategory: 'memory-hierarchy',
    definition: 'Thread-private memory at the lowest level of the memory hierarchy, providing zero-latency access for each thread\'s private data.',
    keyPoints: [
      'Thread-private memory at lowest hierarchy level',
      'Zero-latency access for maximum performance',
      'Stored in register file or spilled to global memory',
      'Limited quantity affects thread occupancy',
      'Automatic allocation by compiler'
    ],
    technicalDetails: [
      {
        label: 'Performance',
        value: 'Zero-latency access',
        description: 'Fastest memory tier in GPU hierarchy'
      },
      {
        label: 'Storage',
        value: 'Register file or spilled to global memory',
        description: 'Hardware register file with spillover mechanism'
      },
      {
        label: 'Occupancy Impact',
        value: 'More registers per thread = fewer concurrent threads',
        description: 'Register usage affects SM occupancy'
      }
    ],
    codeExamples: [
      {
        language: 'cuda',
        code: `__global__ void registerExample(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // These variables are stored in registers (thread-private)
    float a = 1.0f;          // Register variable
    float b = 2.0f;          // Register variable
    float temp1, temp2;      // Register variables
    
    if (idx < n) {
        // Register operations (zero latency)
        temp1 = input[idx] * a;
        temp2 = temp1 + b;
        float result = temp2 * temp1;  // More register usage
        
        output[idx] = result;
    }
}

// Check register usage with compiler flags
// nvcc -Xptxas -v kernel.cu
// Output: ptxas info : Used 16 registers, 0 bytes shared memory

__global__ void highRegisterUsage(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Many register variables - may cause register spilling
    float r1, r2, r3, r4, r5, r6, r7, r8;
    float r9, r10, r11, r12, r13, r14, r15, r16;
    
    // Heavy register usage may reduce occupancy
    // or cause spilling to global memory
}`,
        description: 'Register usage and impact on performance and occupancy'
      }
    ],
    relatedConcepts: ['register-file', 'memory-hierarchy', 'thread', 'streaming-multiprocessor'],
    prerequisites: ['thread', 'memory-hierarchy'],
    difficulty: 'intermediate',
    tags: ['registers', 'memory', 'performance', 'occupancy'],
    learningPath: ['cuda-programming', 'gpu-architecture']
  },
  {
    id: 'shared-memory',
    title: 'Shared Memory',
    category: 'device-software',
    subcategory: 'memory-hierarchy',
    definition: 'Thread block-level memory in L1 cache that is programmer-managed, providing fast communication and data sharing within a thread block.',
    keyPoints: [
      'Thread block-level memory in L1 cache',
      'Programmer-managed and explicitly allocated',
      'Much faster than global memory access',
      'Enables efficient data sharing within block',
      'Requires __syncthreads() for consistency'
    ],
    technicalDetails: [
      {
        label: 'Speed',
        value: 'Much faster than global memory',
        description: 'On-chip SRAM with low latency'
      },
      {
        label: 'Management',
        value: 'Programmer-managed allocation',
        description: 'Explicit declaration and usage by programmer'
      },
      {
        label: 'Synchronization',
        value: 'Requires __syncthreads() barriers',
        description: 'Explicit synchronization needed for consistency'
      }
    ],
    codeExamples: [
      {
        language: 'cuda',
        code: `__global__ void sharedMemoryExample(float* input, float* output, int n) {
    // Declare shared memory for the block
    __shared__ float shared_data[256];
    __shared__ float partial_sums[32];  // For warp-level reductions
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (gid < n) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 0.0f;
    }
    
    // Synchronize to ensure all threads have loaded data
    __syncthreads();
    
    // Perform computation using shared memory
    float result = 0.0f;
    if (tid > 0 && tid < blockDim.x - 1) {
        // Stencil operation using neighboring elements
        result = 0.25f * (shared_data[tid-1] + 2*shared_data[tid] + shared_data[tid+1]);
    }
    
    // Store back to shared memory
    shared_data[tid] = result;
    __syncthreads();
    
    // Write result to global memory
    if (gid < n) {
        output[gid] = shared_data[tid];
    }
}

// Dynamic shared memory allocation
__global__ void dynamicSharedMem(float* data, int n) {
    extern __shared__ float dynamic_shared[];
    
    int tid = threadIdx.x;
    
    // Use dynamically allocated shared memory
    dynamic_shared[tid] = data[blockIdx.x * blockDim.x + tid];
    __syncthreads();
    
    // Process using shared memory...
}

// Launch with dynamic shared memory
// kernel<<<grid, block, sharedMemSize>>>(args);`,
        description: 'Shared memory usage patterns and synchronization'
      }
    ],
    relatedConcepts: ['l1-data-cache', 'thread-block', 'memory-hierarchy', 'global-memory'],
    prerequisites: ['thread-block', 'memory-hierarchy'],
    difficulty: 'intermediate',
    tags: ['shared-memory', 'cooperation', 'performance', 'synchronization'],
    learningPath: ['cuda-programming']
  },
  {
    id: 'global-memory',
    title: 'Global Memory',
    category: 'device-software',
    subcategory: 'memory-hierarchy',
    definition: 'Highest level of GPU memory hierarchy accessible by all threads, stored in GPU RAM with high capacity but high latency.',
    keyPoints: [
      'Highest level accessible by all threads',
      'Stored in GPU RAM (DRAM) with high capacity',
      'High latency but very high bandwidth',
      'Optimized for coalesced access patterns',
      'Survives across kernel launches'
    ],
    technicalDetails: [
      {
        label: 'Capacity',
        value: 'Large capacity (GBs)',
        description: 'Much larger than other memory levels'
      },
      {
        label: 'Latency',
        value: 'High latency (hundreds of cycles)',
        description: 'Much slower than registers or shared memory'
      },
      {
        label: 'Bandwidth',
        value: 'Very high bandwidth when coalesced',
        description: 'Excellent throughput for proper access patterns'
      }
    ],
    codeExamples: [
      {
        language: 'cuda',
        code: `__global__ void globalMemoryPatterns(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Coalesced access - optimal performance
    // Adjacent threads access adjacent memory locations
    if (idx < n) {
        float value = input[idx];        // Coalesced read
        output[idx] = value * 2.0f;      // Coalesced write
    }
}

__global__ void stridedAccess(float* input, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Strided access - may reduce performance
    int global_idx = idx * stride;
    if (global_idx < n) {
        output[idx] = input[global_idx];  // Non-coalesced if stride > 1
    }
}

__global__ void tiledGlobalAccess(float* matrix, float* result, int N) {
    __shared__ float tile[16][16];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load tile from global memory to shared memory
    if (row < N && col < N) {
        tile[threadIdx.y][threadIdx.x] = matrix[row * N + col];
    }
    __syncthreads();
    
    // Process using shared memory (avoid repeated global access)
    float sum = 0.0f;
    for (int i = 0; i < 16; i++) {
        sum += tile[threadIdx.y][i] * tile[i][threadIdx.x];
    }
    
    // Write result back to global memory
    if (row < N && col < N) {
        result[row * N + col] = sum;
    }
}`,
        description: 'Global memory access patterns and optimization strategies'
      }
    ],
    relatedConcepts: ['gpu-ram', 'memory-hierarchy', 'shared-memory', 'load-store-unit'],
    prerequisites: ['memory-hierarchy'],
    difficulty: 'beginner',
    tags: ['global-memory', 'coalescing', 'bandwidth', 'latency'],
    learningPath: ['cuda-programming']
  },
  {
    id: 'memory-hierarchy',
    title: 'Memory Hierarchy',
    category: 'device-software',
    subcategory: 'memory-hierarchy',
    definition: 'Programmer-managed memory system with levels matching thread group hierarchy, designed for moving data closer to computation.',
    keyPoints: [
      'Programmer-managed memory system',
      'Levels match thread group hierarchy',
      'Strategy: move data closer to computation',
      'Pattern: Load global → shared → compute → writeback',
      'Trade capacity for speed at each level'
    ],
    technicalDetails: [
      {
        label: 'Mapping',
        value: 'Thread groups ↔ Memory levels (1:1 correspondence)',
        description: 'Thread hierarchy matches memory hierarchy'
      },
      {
        label: 'Strategy',
        value: 'Move data closer to computation',
        description: 'Reduce memory access latency through hierarchy'
      },
      {
        label: 'Trade-offs',
        value: 'Capacity vs Speed at each level',
        description: 'Faster memory has smaller capacity'
      }
    ],
    visualReferences: [
      {
        type: 'diagram',
        src: '/diagrams/memory-hierarchy.svg',
        alt: 'GPU memory hierarchy with thread mapping',
        caption: 'Memory hierarchy levels and corresponding thread groups',
        interactive: true
      }
    ],
    codeExamples: [
      {
        language: 'cuda',
        code: `__global__ void memoryHierarchyExample(float* global_input, float* global_output, int n) {
    // Memory Hierarchy Levels:
    
    // 1. Registers (thread-private, fastest)
    int tid = threadIdx.x;
    float register_var = 1.0f;
    
    // 2. Shared Memory (block-level, fast)
    __shared__ float shared_buffer[256];
    
    // 3. Global Memory (grid-level, high capacity, high latency)
    int gid = blockIdx.x * blockDim.x + tid;
    
    // Optimal memory usage pattern:
    
    // Step 1: Load from global to shared (block cooperation)
    if (gid < n) {
        shared_buffer[tid] = global_input[gid];
    }
    __syncthreads();
    
    // Step 2: Load from shared to registers (thread-private)
    float local_data = shared_buffer[tid];
    
    // Step 3: Compute using registers (fastest)
    float result = local_data * register_var + 2.0f;
    
    // Step 4: Store back through hierarchy
    shared_buffer[tid] = result;
    __syncthreads();
    
    // Step 5: Write to global memory
    if (gid < n) {
        global_output[gid] = shared_buffer[tid];
    }
}

// Memory hierarchy mapping:
// Grid     ↔ Global Memory   (all threads)
// Block    ↔ Shared Memory   (threads in same block) 
// Thread   ↔ Registers       (individual thread)`,
        description: 'Optimal usage of all memory hierarchy levels'
      }
    ],
    relatedConcepts: ['registers', 'shared-memory', 'global-memory', 'cuda-programming-model'],
    prerequisites: ['cuda-programming-model'],
    difficulty: 'intermediate',
    tags: ['memory-hierarchy', 'optimization', 'performance', 'architecture'],
    learningPath: ['cuda-programming', 'gpu-architecture']
  }
];