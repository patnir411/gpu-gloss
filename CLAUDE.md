# Claude Instructions for GPU Glossary Interactive Web App

## Task Summary
Create a cohesive, end-to-end educational interactive web-app book covering the entire GPU glossary from modal-labs-gpu-glossary.txt. Every concept must be covered piece by piece in an educational format.

## Key Requirements
- Read all detailed text information about GPUs from modal-labs-gpu-glossary.txt
- Create interactive web app book covering entire glossary
- Educational approach - every concept explained step by step
- Cohesive structure that flows logically through topics
- Store memory in memory.md file
- Store tasks/plan in plan.md file
- Remember all instructions in this CLAUDE.md file

## File Structure Observed
The glossary is organized into three main categories:
1. **Device Hardware** - Physical GPU components
2. **Device Software** - Software running on GPU
3. **Host Software** - CPU-side software for GPU programs

## Content Categories from Source
### Device Hardware
- Core, CUDA Core, Tensor Core
- GPU RAM, Register File, L1 Data Cache
- Streaming Multiprocessor & Architecture
- Graphics Processing Cluster, Texture Processing Cluster
- Load Store Unit, Special Function Unit
- Warp Scheduler

### Device Software  
- CUDA Programming Model, Compute Capability
- Thread, Warp, Thread Block, Grid
- Kernel, Cooperative Thread Array
- Memory Hierarchy, Global Memory, Shared Memory, Registers
- Parallel Thread Execution, Streaming Assembler

### Host Software
- CUDA C, CUDA Runtime/Driver APIs
- CUDA Software Platform, Binary Utilities
- NVIDIA GPU Drivers, nvidia-smi, NVML
- Profiling tools: CUPTI, Nsight Systems
- Libraries: libcuda, libcudart, libnvml
- Compilers: nvcc, nvrtc

## Implementation Approach
1. Create modern React-based interactive web app
2. Implement progressive learning structure
3. Include visual diagrams and interactive elements
4. Build comprehensive navigation system
5. Ensure mobile-responsive design
6. Add search and cross-reference functionality

## Technical Stack Considerations
- React with TypeScript for component structure
- Modern CSS/styled-components for styling
- Interactive visualizations for GPU architecture
- Progressive web app features
- Responsive design principles

## Progress Tracking
- Maintain progress.txt file documenting implementation steps
- Each step must be documented before execution
- Mark completion with checkmarks
- Include next step after each completion
- Do not finish until complete interactive web application is delivered