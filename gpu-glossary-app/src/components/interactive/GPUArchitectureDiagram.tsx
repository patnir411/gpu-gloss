import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import * as d3 from 'd3';

interface ArchitectureComponent {
  id: string;
  name: string;
  type: 'gpu' | 'gpc' | 'tpc' | 'sm' | 'core' | 'memory';
  x: number;
  y: number;
  width: number;
  height: number;
  children?: string[];
  description: string;
  color: string;
}

const DiagramContainer = styled.div`
  width: 100%;
  height: 600px;
  background-color: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  position: relative;
  overflow: hidden;
`;

const SVGContainer = styled.svg`
  width: 100%;
  height: 100%;
  cursor: grab;
  
  &:active {
    cursor: grabbing;
  }
`;

const TooltipContainer = styled.div<{ $x: number; $y: number; $visible: boolean }>`
  position: absolute;
  left: ${({ $x }) => $x}px;
  top: ${({ $y }) => $y}px;
  background-color: ${({ theme }) => theme.colors.background};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.borderRadius.base};
  padding: ${({ theme }) => theme.spacing.base};
  box-shadow: 0 4px 12px ${({ theme }) => theme.colors.shadow};
  pointer-events: none;
  z-index: ${({ theme }) => theme.zIndex.tooltip};
  max-width: 250px;
  opacity: ${({ $visible }) => $visible ? 1 : 0};
  transition: opacity ${({ theme }) => theme.transitions.fast};
`;

const TooltipTitle = styled.h4`
  margin: 0 0 ${({ theme }) => theme.spacing.xs} 0;
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
`;

const TooltipDescription = styled.p`
  margin: 0;
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  color: ${({ theme }) => theme.colors.text.secondary};
  line-height: ${({ theme }) => theme.typography.lineHeight.normal};
`;

const ControlsContainer = styled.div`
  position: absolute;
  top: ${({ theme }) => theme.spacing.base};
  right: ${({ theme }) => theme.spacing.base};
  display: flex;
  gap: ${({ theme }) => theme.spacing.sm};
`;

const ControlButton = styled.button`
  padding: ${({ theme }) => theme.spacing.sm};
  background-color: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.borderRadius.base};
  color: ${({ theme }) => theme.colors.text.primary};
  cursor: pointer;
  transition: all ${({ theme }) => theme.transitions.fast};
  
  &:hover {
    background-color: ${({ theme }) => theme.colors.surfaceHover};
    border-color: ${({ theme }) => theme.colors.primary};
  }
`;

// H100 GPU Architecture Data
const h100Architecture: ArchitectureComponent[] = [
  {
    id: 'gpu',
    name: 'H100 GPU',
    type: 'gpu',
    x: 50,
    y: 50,
    width: 700,
    height: 500,
    description: 'NVIDIA H100 GPU with 132 Streaming Multiprocessors',
    color: '#76B900',
    children: ['gpc1', 'gpc2', 'gpc3', 'gpc4', 'gpc5', 'gpc6', 'gpc7', 'gpc8']
  },
  {
    id: 'gpc1',
    name: 'GPC 1',
    type: 'gpc',
    x: 70,
    y: 70,
    width: 160,
    height: 220,
    description: 'Graphics Processing Cluster containing multiple TPCs',
    color: '#0073E6',
    children: ['tpc1', 'tpc2']
  },
  {
    id: 'gpc2',
    name: 'GPC 2',
    type: 'gpc',
    x: 250,
    y: 70,
    width: 160,
    height: 220,
    description: 'Graphics Processing Cluster containing multiple TPCs',
    color: '#0073E6',
    children: ['tpc3', 'tpc4']
  },
  {
    id: 'tpc1',
    name: 'TPC 1',
    type: 'tpc',
    x: 80,
    y: 90,
    width: 140,
    height: 90,
    description: 'Texture Processing Cluster with Streaming Multiprocessors',
    color: '#FF6B35',
    children: ['sm1', 'sm2']
  },
  {
    id: 'tpc2',
    name: 'TPC 2',
    type: 'tpc',
    x: 80,
    y: 190,
    width: 140,
    height: 90,
    description: 'Texture Processing Cluster with Streaming Multiprocessors',
    color: '#FF6B35',
    children: ['sm3', 'sm4']
  },
  {
    id: 'sm1',
    name: 'SM 1',
    type: 'sm',
    x: 90,
    y: 100,
    width: 60,
    height: 70,
    description: '128 CUDA Cores, 4 Tensor Cores, 4 Warp Schedulers',
    color: '#28A745',
    children: ['cores1', 'tensor1', 'memory1']
  },
  {
    id: 'sm2',
    name: 'SM 2',
    type: 'sm',
    x: 160,
    y: 100,
    width: 60,
    height: 70,
    description: '128 CUDA Cores, 4 Tensor Cores, 4 Warp Schedulers',
    color: '#28A745',
    children: ['cores2', 'tensor2', 'memory2']
  },
  {
    id: 'cores1',
    name: 'CUDA Cores',
    type: 'core',
    x: 95,
    y: 105,
    width: 20,
    height: 25,
    description: '128 CUDA Cores for scalar arithmetic',
    color: '#FFC107'
  },
  {
    id: 'tensor1',
    name: 'Tensor Cores',
    type: 'core',
    x: 120,
    y: 105,
    width: 20,
    height: 25,
    description: '4 Tensor Cores for matrix operations',
    color: '#DC3545'
  },
  {
    id: 'memory1',
    name: 'Shared Memory',
    type: 'memory',
    x: 95,
    y: 135,
    width: 45,
    height: 15,
    description: '128KB shared memory / L1 cache',
    color: '#17A2B8'
  }
];

interface GPUArchitectureDiagramProps {
  width?: number;
  height?: number;
  interactive?: boolean;
}

export const GPUArchitectureDiagram: React.FC<GPUArchitectureDiagramProps> = ({
  width = 800,
  height = 600,
  interactive = true
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{
    visible: boolean;
    x: number;
    y: number;
    content: ArchitectureComponent | null;
  }>({
    visible: false,
    x: 0,
    y: 0,
    content: null
  });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (!svgRef.current || !interactive) return;

    const svg = d3.select(svgRef.current);
    
    // Clear previous content
    svg.selectAll('*').remove();
    
    // Create main group for zoom/pan
    const mainGroup = svg.append('g')
      .attr('transform', `translate(${pan.x}, ${pan.y}) scale(${zoom})`);

    // Draw components
    h100Architecture.forEach(component => {
      const group = mainGroup.append('g')
        .attr('class', `component-${component.type}`)
        .style('cursor', 'pointer');

      // Component rectangle
      group.append('rect')
        .attr('x', component.x)
        .attr('y', component.y)
        .attr('width', component.width)
        .attr('height', component.height)
        .attr('fill', component.color)
        .attr('fill-opacity', 0.7)
        .attr('stroke', component.color)
        .attr('stroke-width', 2)
        .attr('rx', 4);

      // Component label
      group.append('text')
        .attr('x', component.x + component.width / 2)
        .attr('y', component.y + component.height / 2)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', 'white')
        .attr('font-size', Math.min(component.width / 8, 12))
        .attr('font-weight', 'bold')
        .text(component.name);

      // Add hover effects
      group
        .on('mouseenter', function(event) {
          d3.select(this).select('rect')
            .transition()
            .duration(200)
            .attr('fill-opacity', 0.9)
            .attr('stroke-width', 3);

          const [mouseX, mouseY] = d3.pointer(event, svgRef.current);
          setTooltip({
            visible: true,
            x: mouseX + 10,
            y: mouseY - 10,
            content: component
          });
        })
        .on('mouseleave', function() {
          d3.select(this).select('rect')
            .transition()
            .duration(200)
            .attr('fill-opacity', 0.7)
            .attr('stroke-width', 2);

          setTooltip(prev => ({ ...prev, visible: false }));
        })
        .on('click', function() {
          // Center on clicked component
          const centerX = -(component.x + component.width / 2 - width / 2);
          const centerY = -(component.y + component.height / 2 - height / 2);
          setPan({ x: centerX, y: centerY });
        });
    });

    // Add zoom behavior
    const zoomBehavior = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 3])
      .on('zoom', (event) => {
        const { transform } = event;
        setZoom(transform.k);
        setPan({ x: transform.x, y: transform.y });
      });

    svg.call(zoomBehavior);

  }, [zoom, pan, interactive, width, height]);

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev * 1.2, 3));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev * 0.8, 0.5));
  };

  const handleReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  return (
    <DiagramContainer>
      <SVGContainer ref={svgRef} />
      
      {interactive && (
        <ControlsContainer>
          <ControlButton onClick={handleZoomIn} title="Zoom In">
            +
          </ControlButton>
          <ControlButton onClick={handleZoomOut} title="Zoom Out">
            -
          </ControlButton>
          <ControlButton onClick={handleReset} title="Reset View">
            ⌂
          </ControlButton>
        </ControlsContainer>
      )}

      <TooltipContainer
        $x={tooltip.x}
        $y={tooltip.y}
        $visible={tooltip.visible && !!tooltip.content}
      >
        {tooltip.content && (
          <>
            <TooltipTitle>{tooltip.content.name}</TooltipTitle>
            <TooltipDescription>{tooltip.content.description}</TooltipDescription>
          </>
        )}
      </TooltipContainer>
    </DiagramContainer>
  );
};