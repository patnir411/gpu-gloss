# GPU Glossary Interactive Web App

A comprehensive, interactive educational web application for learning GPU programming concepts, built with React and TypeScript.

## Features

- 📚 **42+ GPU Concepts**: Complete coverage of device hardware, device software, and host software
- 🎯 **4 Learning Paths**: Beginner to expert guided learning sequences
- 🎨 **Interactive Visualizations**: 
  - GPU Architecture diagrams with D3.js
  - Memory hierarchy visualization
  - Thread hierarchy exploration
- 🔍 **Smart Search**: Full-text search with fuzzy matching
- 🌙 **Dark/Light Theme**: Automatic theme switching
- 📱 **Mobile Responsive**: Optimized for all screen sizes
- ⚡ **Performance Optimized**: Code splitting and lazy loading

## Learning Paths

### 1. GPU Fundamentals (Beginner)
- CUDA Device Architecture
- Streaming Multiprocessors
- GPU Cores and Memory
- Basic Programming Model

### 2. CUDA Programming (Intermediate)
- Thread Hierarchy
- Memory Management
- Kernel Development
- Synchronization

### 3. GPU Architecture (Advanced)
- Hardware Deep Dive
- Performance Optimization
- PTX and SASS Assembly
- Architecture Internals

### 4. CUDA Development (Advanced)
- Development Tools
- Profiling and Debugging
- Advanced APIs
- Production Practices

## Technology Stack

- **Frontend**: React 18 + TypeScript
- **Styling**: Styled Components + CSS-in-JS
- **Visualizations**: D3.js + Framer Motion
- **Search**: Fuse.js
- **Build**: Vite
- **Code Highlighting**: React Syntax Highlighter

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

### Development

```bash
# Run development server
npm run dev

# Run type checking
npm run type-check

# Run linting
npm run lint

# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
src/
├── components/           # Reusable UI components
│   ├── common/          # Generic components
│   ├── navigation/      # Navigation components
│   └── interactive/     # Interactive visualizations
├── pages/               # Page components
├── data/                # GPU concepts and learning paths
├── contexts/            # React contexts (theme, state)
├── hooks/               # Custom React hooks
├── types/               # TypeScript type definitions
├── utils/               # Utility functions
└── styles/              # Global styles
```

## Data Sources

Based on the comprehensive Modal Labs GPU Glossary, covering:

- **Device Hardware**: GPU cores, memory systems, architecture
- **Device Software**: CUDA programming model, execution hierarchy
- **Host Software**: APIs, drivers, development tools

## Interactive Features

### GPU Architecture Diagram
- Clickable H100 GPU visualization
- Zoom and pan capabilities
- Component tooltips with detailed information

### Memory Hierarchy Visualization
- Interactive pyramid showing memory levels
- Performance metrics comparison
- Data flow animations

### Thread Hierarchy Visualization
- CUDA thread organization
- Multiple view modes (Grid, Block, Warp)
- Interactive exploration of thread relationships

## Performance

- **Bundle Size**: ~1.2MB compressed
- **Load Time**: <3 seconds on 3G
- **Lighthouse Score**: 90+
- **Mobile Optimized**: Responsive design

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## License

This project is licensed under the MIT License. GPU concept content is based on Modal Labs GPU Glossary (CC BY 4.0).

## Acknowledgments

- Modal Labs for the comprehensive GPU Glossary
- NVIDIA for GPU architecture documentation
- React and TypeScript communities
- D3.js for visualization capabilities
