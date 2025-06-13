import type { Category } from '../../types';

export const categories: Category[] = [
  {
    id: 'device-hardware',
    name: 'Device Hardware',
    description: 'Physical components of the GPU - the "device" in NVIDIA\'s terminology',
    icon: '🔧',
    color: '#76B900', // NVIDIA Green
    conceptCount: 15,
    subcategories: [
      {
        id: 'processing-units',
        name: 'Processing Units & Cores',
        description: 'Computational units that execute operations',
        conceptIds: ['core', 'cuda-core', 'tensor-core', 'special-function-unit']
      },
      {
        id: 'architecture-components',
        name: 'Architecture Components',
        description: 'Organizational units and architecture design',
        conceptIds: ['streaming-multiprocessor', 'sm-architecture', 'cuda-device-architecture', 'graphics-processing-cluster', 'texture-processing-cluster']
      },
      {
        id: 'memory-storage',
        name: 'Memory & Storage',
        description: 'Memory hierarchy and storage systems',
        conceptIds: ['gpu-ram', 'register-file', 'l1-data-cache']
      },
      {
        id: 'scheduling-control',
        name: 'Scheduling & Control',
        description: 'Execution control and scheduling mechanisms',
        conceptIds: ['warp-scheduler', 'load-store-unit']
      }
    ]
  },
  {
    id: 'device-software',
    name: 'Device Software',
    description: 'Software that runs on the GPU - the "device" programming model',
    icon: '⚙️',
    color: '#0073E6', // Blue
    conceptCount: 13,
    subcategories: [
      {
        id: 'programming-model',
        name: 'Programming Model Foundation',
        description: 'Core programming abstractions and execution model',
        conceptIds: ['cuda-programming-model', 'compute-capability', 'parallel-thread-execution', 'streaming-assembler']
      },
      {
        id: 'thread-hierarchy',
        name: 'Thread Hierarchy',
        description: 'Organization of parallel execution units',
        conceptIds: ['thread', 'warp', 'thread-block', 'cooperative-thread-array', 'thread-block-grid', 'kernel']
      },
      {
        id: 'memory-hierarchy',
        name: 'Memory Hierarchy',
        description: 'Programmer-accessible memory levels',
        conceptIds: ['registers', 'shared-memory', 'global-memory', 'memory-hierarchy']
      }
    ]
  },
  {
    id: 'host-software',
    name: 'Host Software',
    description: 'CPU-side software for GPU programming and management',
    icon: '💻',
    color: '#FF6B35', // Orange
    conceptCount: 15,
    subcategories: [
      {
        id: 'apis-runtime',
        name: 'APIs & Runtime',
        description: 'Programming interfaces and runtime libraries',
        conceptIds: ['cuda-software-platform', 'cuda-driver-api', 'cuda-runtime-api', 'libcuda', 'libcudart']
      },
      {
        id: 'development-tools',
        name: 'Development Tools',
        description: 'Compilers, utilities, and development environment',
        conceptIds: ['cuda-c', 'nvcc', 'nvrtc', 'cuda-binary-utilities']
      },
      {
        id: 'profiling-management',
        name: 'Profiling & Management',
        description: 'Performance analysis and system management tools',
        conceptIds: ['nsight-systems', 'cupti', 'nvidia-smi', 'nvml', 'libnvml']
      },
      {
        id: 'system-integration',
        name: 'System Integration',
        description: 'System-level drivers and kernel components',
        conceptIds: ['nvidia-gpu-drivers', 'nvidia-ko']
      }
    ]
  }
];