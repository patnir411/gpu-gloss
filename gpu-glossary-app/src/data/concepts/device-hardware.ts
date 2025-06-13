import type { Concept } from '../../types';

export const deviceHardwareConcepts: Concept[] = [
  {
    id: 'core',
    title: 'GPU Core',
    category: 'device-hardware',
    subcategory: 'processing-units',
    definition: 'The primary compute units that make up the Streaming Multiprocessors (SMs). Unlike CPU cores, GPU cores are better thought of as specialized "pipes" for data transformation.',
    keyPoints: [
      'Primary compute units within Streaming Multiprocessors',
      'More like data transformation "pipes" than CPU cores',
      'Examples include CUDA Cores and Tensor Cores',
      'Associated with specific hardware instruction sets',
      'Provide different computational throughput affordances'
    ],
    technicalDetails: [
      {
        label: 'Types',
        value: 'CUDA Cores, Tensor Cores, Special Function Units',
        description: 'Different core types for different operations'
      },
      {
        label: 'Architecture Role',
        value: 'Execute actual computations within SMs',
        description: 'Handle the mathematical operations'
      }
    ],
    visualReferences: [
      {
        type: 'diagram',
        src: '/diagrams/h100-sm-architecture.svg',
        alt: 'H100 SM internal architecture showing cores in green',
        caption: 'H100 Streaming Multiprocessor showing CUDA and Tensor Cores',
        interactive: true
      }
    ],
    relatedConcepts: ['cuda-core', 'tensor-core', 'streaming-multiprocessor', 'special-function-unit'],
    prerequisites: [],
    difficulty: 'beginner',
    tags: ['hardware', 'computation', 'architecture'],
    learningPath: ['gpu-fundamentals', 'gpu-architecture']
  },
  {
    id: 'cuda-core',
    title: 'CUDA Core',
    category: 'device-hardware',
    subcategory: 'processing-units',
    definition: 'GPU cores that execute scalar arithmetic instructions, handling integer and floating-point operations. Groups of CUDA cores receive the same instruction simultaneously.',
    keyPoints: [
      'Execute scalar arithmetic instructions (int, float operations)',
      'Scheduled in groups, typically 32 cores (one warp)',
      'Not independently scheduled like CPU cores',
      'Different architectures have different unit mixtures',
      'Groups can be as small as 1 thread but with performance cost'
    ],
    technicalDetails: [
      {
        label: 'H100 Specs',
        value: '128 FP32 CUDA Cores per SM',
        description: 'But only 64 INT32 or 64 FP64 units due to shared resources'
      },
      {
        label: 'Scheduling',
        value: 'SIMT (Single Instruction, Multiple Thread)',
        description: 'All cores in group execute same instruction'
      },
      {
        label: 'Architecture Variations',
        value: 'Different mixtures of 32-bit int, 32-bit FP, 64-bit FP',
        description: 'Varies by GPU architecture generation'
      }
    ],
    codeExamples: [
      {
        language: 'cuda',
        code: `// Simple CUDA kernel using CUDA cores
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // Executed by CUDA cores
    }
}`,
        description: 'Basic vector addition using CUDA cores for scalar arithmetic'
      }
    ],
    relatedConcepts: ['core', 'tensor-core', 'warp', 'warp-scheduler', 'streaming-multiprocessor'],
    prerequisites: ['core', 'streaming-multiprocessor'],
    difficulty: 'beginner',
    tags: ['hardware', 'arithmetic', 'scalar', 'simt'],
    learningPath: ['gpu-fundamentals', 'cuda-programming', 'gpu-architecture']
  },
  {
    id: 'tensor-core',
    title: 'Tensor Core',
    category: 'device-hardware',
    subcategory: 'processing-units',
    definition: 'Specialized cores for matrix operations using single instructions, designed to accelerate neural network workloads with massive matrix multiplication throughput.',
    keyPoints: [
      'Operate on entire matrices with single instructions',
      'Designed specifically for AI/ML workloads',
      'Much higher throughput than CUDA cores for matrix operations',
      'Support multiple precision formats (FP16, INT8, etc.)',
      'Programmable via WMMA intrinsics'
    ],
    technicalDetails: [
      {
        label: 'H100 Specs',
        value: '4 Tensor Cores per SM',
        description: 'Fewer than CUDA cores but much more powerful for matrix ops'
      },
      {
        label: 'Operation Type',
        value: 'Matrix multiply-accumulate (C = A × B + C)',
        description: 'Fundamental operation for neural networks'
      },
      {
        label: 'Precision Support',
        value: 'FP64, FP32, FP16, BF16, INT8, INT4',
        description: 'Multiple data types for different ML workloads'
      }
    ],
    codeExamples: [
      {
        language: 'cuda',
        code: `// Using Tensor Cores with WMMA API
#include <mma.h>
using namespace nvcuda;

__global__ void tensorCoreGEMM() {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // Load matrices
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Perform matrix multiplication using Tensor Cores
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // Store result
    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}`,
        description: 'Matrix multiplication using Tensor Cores via WMMA intrinsics'
      }
    ],
    relatedConcepts: ['core', 'cuda-core', 'streaming-multiprocessor', 'cuda-programming-model'],
    prerequisites: ['core', 'cuda-core', 'streaming-multiprocessor'],
    difficulty: 'advanced',
    tags: ['hardware', 'matrix', 'ai', 'ml', 'wmma'],
    learningPath: ['gpu-architecture', 'cuda-development']
  },
  {
    id: 'special-function-unit',
    title: 'Special Function Unit (SFU)',
    category: 'device-hardware',
    subcategory: 'processing-units',
    definition: 'Specialized hardware units that accelerate transcendental mathematical operations like exp, log, sin, cos, sqrt, and reciprocal functions.',
    keyPoints: [
      'Accelerates transcendental functions (exp, log, sin, cos, sqrt)',
      'Much faster than computing via CUDA cores',
      'Lower precision than CUDA core implementations',
      'Automatically used by compiler optimization',
      'Trade precision for performance'
    ],
    technicalDetails: [
      {
        label: 'Functions',
        value: 'exp, log, sin, cos, sqrt, reciprocal',
        description: 'Common mathematical functions used in compute'
      },
      {
        label: 'Performance',
        value: 'Significantly faster than CUDA core computation',
        description: 'Dedicated hardware for these operations'
      },
      {
        label: 'Precision',
        value: 'Lower precision than CUDA cores',
        description: 'Trade-off between speed and accuracy'
      }
    ],
    relatedConcepts: ['cuda-core', 'streaming-multiprocessor'],
    prerequisites: ['cuda-core'],
    difficulty: 'intermediate',
    tags: ['hardware', 'math', 'transcendental', 'optimization'],
    learningPath: ['gpu-architecture', 'cuda-development']
  },
  {
    id: 'streaming-multiprocessor',
    title: 'Streaming Multiprocessor (SM)',
    category: 'device-hardware',
    subcategory: 'architecture-components',
    definition: 'The fundamental processing unit of a GPU, designed for massive parallelism. Each SM contains cores, register files, shared memory, and warp schedulers.',
    keyPoints: [
      'Fundamental processing unit of GPU architecture',
      'Closer to CPU core equivalent than individual GPU cores',
      'Contains cores, register file, shared memory, warp schedulers',
      'Handles up to 2048 concurrent threads per SM',
      'Enables fast warp switching (single clock cycle)'
    ],
    technicalDetails: [
      {
        label: 'H100 Configuration',
        value: '132 SMs total',
        description: 'Each SM can handle up to 2048 concurrent threads'
      },
      {
        label: 'Thread Organization',
        value: '64 warps of 32 threads each',
        description: 'Maximum occupancy per SM'
      },
      {
        label: 'Components',
        value: 'Cores, register file, shared memory, schedulers',
        description: 'All components needed for parallel execution'
      },
      {
        label: 'Total GPU Capacity',
        value: '250,000+ concurrent threads (H100)',
        description: 'Massive parallelism across all SMs'
      }
    ],
    visualReferences: [
      {
        type: 'diagram',
        src: '/diagrams/sm-architecture.svg',
        alt: 'Streaming Multiprocessor internal architecture',
        caption: 'SM components and organization',
        interactive: true
      }
    ],
    relatedConcepts: ['core', 'cuda-core', 'tensor-core', 'warp-scheduler', 'register-file', 'shared-memory'],
    prerequisites: [],
    difficulty: 'beginner',
    tags: ['hardware', 'architecture', 'parallelism', 'sm'],
    learningPath: ['gpu-fundamentals', 'cuda-programming', 'gpu-architecture']
  },
  {
    id: 'sm-architecture',
    title: 'Streaming Multiprocessor Architecture',
    category: 'device-hardware',
    subcategory: 'architecture-components',
    definition: 'Versioned architectures that define SASS code compatibility and capabilities. Each generation adds new instruction sets and features.',
    keyPoints: [
      'Versioned architectures defining hardware capabilities',
      'Determines SASS code compatibility',
      'Each generation adds new features and instructions',
      'Forward compatible within same architecture family',
      'Examples: Kepler, Maxwell, Pascal, Volta, Turing, Ampere, Hopper'
    ],
    technicalDetails: [
      {
        label: 'Architecture Examples',
        value: 'Kepler, Maxwell, Pascal, Volta, Turing, Ampere, Hopper',
        description: 'Evolution of SM architectures over time'
      },
      {
        label: 'Compatibility',
        value: 'Forward compatible within major versions',
        description: 'Newer hardware can run older code'
      },
      {
        label: 'Purpose',
        value: 'Defines instruction set and capabilities',
        description: 'What the hardware can execute'
      }
    ],
    relatedConcepts: ['streaming-multiprocessor', 'compute-capability', 'streaming-assembler'],
    prerequisites: ['streaming-multiprocessor'],
    difficulty: 'intermediate',
    tags: ['hardware', 'architecture', 'compatibility', 'versioning'],
    learningPath: ['gpu-architecture', 'cuda-development']
  },
  {
    id: 'cuda-device-architecture',
    title: 'CUDA Device Architecture',
    category: 'device-hardware',
    subcategory: 'architecture-components',
    definition: 'Compute Unified Device Architecture - the architectural design that simplified GPU programming by replacing complex graphics pipelines with unified compute units.',
    keyPoints: [
      'Compute Unified Device Architecture',
      'Simplified GPU from complex graphics pipeline to unified compute',
      'Made GPU programming accessible for general computation',
      'Enables SIMT (Single Instruction, Multiple Thread) execution',
      'Can refer to hardware architecture, programming model, or software platform'
    ],
    technicalDetails: [
      {
        label: 'Key Innovation',
        value: 'Unified compute architecture',
        description: 'Replaced specialized graphics pipeline stages'
      },
      {
        label: 'Execution Model',
        value: 'SIMT (Single Instruction, Multiple Thread)',
        description: 'Foundation of GPU parallel programming'
      },
      {
        label: 'Three Meanings',
        value: 'Hardware architecture, programming model, software platform',
        description: 'CUDA refers to multiple related concepts'
      }
    ],
    relatedConcepts: ['streaming-multiprocessor', 'cuda-programming-model', 'cuda-software-platform'],
    prerequisites: [],
    difficulty: 'beginner',
    tags: ['hardware', 'architecture', 'cuda', 'simt'],
    learningPath: ['gpu-fundamentals', 'cuda-programming']
  },
  {
    id: 'graphics-processing-cluster',
    title: 'Graphics Processing Cluster (GPC)',
    category: 'device-hardware',
    subcategory: 'architecture-components',
    definition: 'High-level organizational unit within the GPU that contains multiple Texture Processing Clusters plus raster engine, handling both compute and graphics workloads.',
    keyPoints: [
      'High-level organizational unit within GPU',
      'Contains multiple Texture Processing Clusters',
      'Includes raster engine for graphics processing',
      'Handles both compute and graphics workloads',
      'Intermediate level in GPU hierarchy'
    ],
    technicalDetails: [
      {
        label: 'Components',
        value: 'Multiple TPCs plus raster engine',
        description: 'Combines compute and graphics capabilities'
      },
      {
        label: 'Function',
        value: 'Both compute and graphics workloads',
        description: 'Unified processing for different workload types'
      }
    ],
    relatedConcepts: ['texture-processing-cluster', 'streaming-multiprocessor'],
    prerequisites: ['streaming-multiprocessor'],
    difficulty: 'intermediate',
    tags: ['hardware', 'architecture', 'organization'],
    learningPath: ['gpu-architecture']
  },
  {
    id: 'texture-processing-cluster',
    title: 'Texture Processing Cluster (TPC)',
    category: 'device-hardware',
    subcategory: 'architecture-components',
    definition: 'Intermediate organizational level generally synonymous with a pair of Streaming Multiprocessors, grouping SMs for efficient resource management.',
    keyPoints: [
      'Intermediate organizational level in GPU hierarchy',
      'Generally synonymous with pair of Streaming Multiprocessors',
      'Groups SMs for efficient resource management',
      'Part of Graphics Processing Cluster',
      'Organizational abstraction for hardware management'
    ],
    technicalDetails: [
      {
        label: 'Composition',
        value: 'Generally pair of Streaming Multiprocessors',
        description: 'Logical grouping of SMs'
      },
      {
        label: 'Purpose',
        value: 'Efficient resource management',
        description: 'Organizational unit for scheduling and management'
      }
    ],
    relatedConcepts: ['streaming-multiprocessor', 'graphics-processing-cluster'],
    prerequisites: ['streaming-multiprocessor'],
    difficulty: 'intermediate',
    tags: ['hardware', 'architecture', 'organization'],
    learningPath: ['gpu-architecture']
  },
  {
    id: 'gpu-ram',
    title: 'GPU RAM',
    category: 'device-hardware',
    subcategory: 'memory-storage',
    definition: 'Large global memory store (DRAM) accessible by all Streaming Multiprocessors, providing high-capacity storage for GPU computations with extremely high bandwidth.',
    keyPoints: [
      'Large global memory store using DRAM technology',
      'Accessible by all Streaming Multiprocessors',
      'Provides storage for global memory in programming model',
      'High capacity (16GB-80GB+ in modern GPUs)',
      'Extremely high bandwidth (1TB/s+ in modern GPUs)'
    ],
    technicalDetails: [
      {
        label: 'Capacity',
        value: '16GB-80GB+ in modern GPUs',
        description: 'Much larger than CPU caches but smaller than system RAM'
      },
      {
        label: 'Bandwidth',
        value: '1TB/s+ in modern GPUs',
        description: 'Extremely high memory bandwidth for parallel access'
      },
      {
        label: 'Access Pattern',
        value: 'Optimized for coalesced access',
        description: 'Best performance when threads access contiguous memory'
      }
    ],
    relatedConcepts: ['global-memory', 'memory-hierarchy', 'streaming-multiprocessor'],
    prerequisites: ['streaming-multiprocessor'],
    difficulty: 'beginner',
    tags: ['hardware', 'memory', 'dram', 'bandwidth'],
    learningPath: ['gpu-fundamentals', 'cuda-programming']
  },
  {
    id: 'register-file',
    title: 'Register File',
    category: 'device-hardware',
    subcategory: 'memory-storage',
    definition: 'Hardware that stores bits between manipulation by GPU cores, organized as dynamically reallocatable 32-bit registers providing zero-latency access for threads.',
    keyPoints: [
      'Stores data between core operations',
      'Organized as dynamically reallocatable 32-bit registers',
      'Provides zero-latency access for thread-private data',
      'Fastest memory tier in GPU hierarchy',
      'Limited quantity affects thread occupancy'
    ],
    technicalDetails: [
      {
        label: 'Organization',
        value: 'Dynamically reallocatable 32-bit registers',
        description: 'Flexible allocation based on kernel requirements'
      },
      {
        label: 'Performance',
        value: 'Zero-latency access',
        description: 'Fastest memory tier in GPU'
      },
      {
        label: 'Limitation',
        value: 'Limited quantity affects occupancy',
        description: 'More registers per thread = fewer concurrent threads'
      }
    ],
    relatedConcepts: ['registers', 'memory-hierarchy', 'streaming-multiprocessor'],
    prerequisites: ['streaming-multiprocessor'],
    difficulty: 'intermediate',
    tags: ['hardware', 'memory', 'registers', 'performance'],
    learningPath: ['cuda-programming', 'gpu-architecture']
  },
  {
    id: 'l1-data-cache',
    title: 'L1 Data Cache',
    category: 'device-hardware',
    subcategory: 'memory-storage',
    definition: 'Private SRAM memory of each Streaming Multiprocessor, co-located with compute units and configurable as L1 cache or shared memory.',
    keyPoints: [
      'Private SRAM memory for each SM',
      'Co-located with compute units for low latency',
      'Configurable split between L1 cache and shared memory',
      'Typically 64KB-192KB per SM',
      'Provides fast access for threads within same SM'
    ],
    technicalDetails: [
      {
        label: 'Size',
        value: '64KB-192KB per SM',
        description: 'Varies by GPU architecture'
      },
      {
        label: 'Configuration',
        value: 'Configurable L1 cache / shared memory split',
        description: 'Can be tuned for different workload needs'
      },
      {
        label: 'Access',
        value: 'Low-latency for threads within same SM',
        description: 'Much faster than global memory'
      }
    ],
    relatedConcepts: ['shared-memory', 'streaming-multiprocessor', 'memory-hierarchy'],
    prerequisites: ['streaming-multiprocessor', 'shared-memory'],
    difficulty: 'intermediate',
    tags: ['hardware', 'memory', 'cache', 'sram'],
    learningPath: ['cuda-programming', 'gpu-architecture']
  },
  {
    id: 'warp-scheduler',
    title: 'Warp Scheduler',
    category: 'device-hardware',
    subcategory: 'scheduling-control',
    definition: 'Hardware component that decides which group of 32 threads (warp) to execute next, enabling rapid warp switching to hide memory latency.',
    keyPoints: [
      'Decides which warp (group of 32 threads) to execute',
      'Switches warps on per-clock-cycle basis',
      'Hides memory latency through rapid context switching',
      'Multiple schedulers per SM (4 in H100)',
      'Enables high throughput by keeping cores busy'
    ],
    technicalDetails: [
      {
        label: 'H100 Configuration',
        value: '4 warp schedulers per SM',
        description: 'Multiple schedulers for higher throughput'
      },
      {
        label: 'Switching Speed',
        value: 'Single clock cycle',
        description: 'Extremely fast compared to CPU context switches (microseconds)'
      },
      {
        label: 'Purpose',
        value: 'Hide memory latency',
        description: 'Switch to ready warps while others wait for memory'
      }
    ],
    relatedConcepts: ['warp', 'streaming-multiprocessor', 'thread-block'],
    prerequisites: ['warp', 'streaming-multiprocessor'],
    difficulty: 'intermediate',
    tags: ['hardware', 'scheduling', 'latency-hiding', 'simt'],
    learningPath: ['cuda-programming', 'gpu-architecture']
  },
  {
    id: 'load-store-unit',
    title: 'Load/Store Unit (LSU)',
    category: 'device-hardware',
    subcategory: 'scheduling-control',
    definition: 'Hardware component that dispatches memory requests to various memory subsystems, handling communication between compute units and memory hierarchy.',
    keyPoints: [
      'Dispatches memory requests to memory subsystems',
      'Handles communication between compute and memory',
      'Manages global, shared, and local memory operations',
      'Performs memory coalescing for efficiency',
      'Critical for memory performance optimization'
    ],
    technicalDetails: [
      {
        label: 'Function',
        value: 'Dispatch memory requests to subsystems',
        description: 'Routes memory operations to appropriate memory levels'
      },
      {
        label: 'Memory Types',
        value: 'Global, shared, local memory operations',
        description: 'Handles all memory hierarchy levels'
      },
      {
        label: 'Optimization',
        value: 'Memory coalescing',
        description: 'Combines multiple requests for efficiency'
      }
    ],
    relatedConcepts: ['memory-hierarchy', 'global-memory', 'shared-memory', 'streaming-multiprocessor'],
    prerequisites: ['memory-hierarchy', 'streaming-multiprocessor'],
    difficulty: 'advanced',
    tags: ['hardware', 'memory', 'optimization', 'coalescing'],
    learningPath: ['gpu-architecture', 'cuda-development']
  }
];