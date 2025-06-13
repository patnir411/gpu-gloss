import type { LearningPath } from '../types';

export const learningPaths: LearningPath[] = [
  {
    id: 'gpu-fundamentals',
    name: 'GPU Fundamentals',
    description: 'Introduction to GPU computing concepts and basic architecture understanding',
    difficulty: 'beginner',
    estimatedTime: '2-3 hours',
    conceptSequence: [
      'cuda-device-architecture',
      'streaming-multiprocessor',
      'core',
      'cuda-core',
      'gpu-ram',
      'cuda-programming-model',
      'thread',
      'kernel',
      'memory-hierarchy',
      'global-memory',
      'cuda-runtime-api',
      'cuda-c'
    ],
    outcomes: [
      'Understand what makes GPUs different from CPUs',
      'Know the basic GPU hardware components',
      'Understand the CUDA programming model basics',
      'Write simple CUDA programs',
      'Understand memory hierarchy fundamentals'
    ]
  },
  {
    id: 'cuda-programming',
    name: 'CUDA Programming',
    description: 'Comprehensive CUDA programming with thread hierarchy, memory management, and optimization',
    difficulty: 'intermediate',
    estimatedTime: '4-6 hours',
    prerequisites: ['gpu-fundamentals'],
    conceptSequence: [
      'cuda-programming-model',
      'thread',
      'warp',
      'thread-block',
      'thread-block-grid',
      'kernel',
      'registers',
      'shared-memory',
      'global-memory',
      'memory-hierarchy',
      'warp-scheduler',
      'streaming-multiprocessor',
      'cooperative-thread-array',
      'parallel-thread-execution'
    ],
    outcomes: [
      'Master the thread hierarchy (thread → warp → block → grid)',
      'Effectively use all memory hierarchy levels',
      'Write optimized CUDA kernels',
      'Understand synchronization and cooperation',
      'Debug and profile CUDA applications'
    ]
  },
  {
    id: 'gpu-architecture',
    name: 'GPU Architecture Deep Dive',
    description: 'Deep understanding of GPU hardware architecture and performance optimization',
    difficulty: 'advanced',
    estimatedTime: '6-8 hours',
    prerequisites: ['cuda-programming'],
    conceptSequence: [
      'streaming-multiprocessor',
      'sm-architecture',
      'cuda-device-architecture',
      'cuda-core',
      'tensor-core',
      'special-function-unit',
      'warp-scheduler',
      'load-store-unit',
      'register-file',
      'l1-data-cache',
      'graphics-processing-cluster',
      'texture-processing-cluster',
      'compute-capability',
      'parallel-thread-execution',
      'streaming-assembler'
    ],
    outcomes: [
      'Deep understanding of SM internal architecture',
      'Know how different core types work',
      'Understand memory subsystem details',
      'Optimize code for specific architectures',
      'Read and understand PTX and SASS assembly'
    ]
  },
  {
    id: 'cuda-development',
    name: 'CUDA Development Tools',
    description: 'Master the complete CUDA development ecosystem and advanced tooling',
    difficulty: 'advanced',
    estimatedTime: '5-7 hours',
    prerequisites: ['cuda-programming'],
    conceptSequence: [
      'cuda-software-platform',
      'cuda-c',
      'nvcc',
      'cuda-driver-api',
      'cuda-runtime-api',
      'libcuda',
      'libcudart',
      'parallel-thread-execution',
      'streaming-assembler',
      'compute-capability',
      'nvrtc',
      'cuda-binary-utilities',
      'nsight-systems',
      'cupti',
      'nvidia-smi',
      'nvml',
      'nvidia-gpu-drivers',
      'nvidia-ko'
    ],
    outcomes: [
      'Use the complete CUDA toolchain effectively',
      'Profile and debug CUDA applications',
      'Understand compilation process deeply',
      'Use advanced development techniques',
      'Monitor and manage GPU systems',
      'Build production CUDA applications'
    ]
  }
];