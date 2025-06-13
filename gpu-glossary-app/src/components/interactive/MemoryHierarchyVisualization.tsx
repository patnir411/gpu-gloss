import React, { useState } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';

interface MemoryLevel {
  id: string;
  name: string;
  size: string;
  latency: string;
  bandwidth: string;
  scope: string;
  description: string;
  color: string;
  relativeWidth: number;
  threadMapping: string;
}

const VisualizationContainer = styled.div`
  width: 100%;
  padding: ${({ theme }) => theme.spacing.xl};
  background-color: ${({ theme }) => theme.colors.surface};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  border: 1px solid ${({ theme }) => theme.colors.border};
`;

const Title = styled.h3`
  text-align: center;
  margin-bottom: ${({ theme }) => theme.spacing.xl};
  color: ${({ theme }) => theme.colors.text.primary};
  font-size: ${({ theme }) => theme.typography.fontSize.xl};
`;

const PyramidContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.base};
  margin-bottom: ${({ theme }) => theme.spacing.xl};
`;

const MemoryLevelContainer = styled(motion.div)<{ 
  $width: number; 
  $color: string; 
  $isActive: boolean 
}>`
  width: ${({ $width }) => $width}%;
  max-width: 100%;
  height: 80px;
  background: linear-gradient(135deg, ${({ $color }) => $color}dd, ${({ $color }) => $color}aa);
  border: 2px solid ${({ $color, $isActive }) => $isActive ? $color : `${$color}66`};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${({ theme }) => theme.spacing.base} ${({ theme }) => theme.spacing.lg};
  cursor: pointer;
  position: relative;
  overflow: hidden;
  
  &:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    transform: translateX(-100%);
    transition: transform 0.6s;
  }
  
  &:hover:before {
    transform: translateX(100%);
  }
`;

const LevelInfo = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.xs};
`;

const LevelName = styled.h4`
  margin: 0;
  color: white;
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  text-shadow: 0 1px 2px rgba(0,0,0,0.3);
`;

const LevelScope = styled.span`
  color: rgba(255,255,255,0.9);
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const LevelMetrics = styled.div`
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: ${({ theme }) => theme.spacing.xs};
`;

const Metric = styled.span`
  color: rgba(255,255,255,0.95);
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  background: rgba(0,0,0,0.2);
  padding: 2px ${({ theme }) => theme.spacing.xs};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
`;

const DetailPanel = styled(motion.div)`
  background-color: ${({ theme }) => theme.colors.background};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: ${({ theme }) => theme.spacing.xl};
  margin-top: ${({ theme }) => theme.spacing.lg};
`;

const DetailTitle = styled.h4`
  margin: 0 0 ${({ theme }) => theme.spacing.base} 0;
  color: ${({ theme }) => theme.colors.text.primary};
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
`;

const DetailDescription = styled.p`
  margin: 0 0 ${({ theme }) => theme.spacing.lg} 0;
  color: ${({ theme }) => theme.colors.text.secondary};
  line-height: ${({ theme }) => theme.typography.lineHeight.relaxed};
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: ${({ theme }) => theme.spacing.base};
`;

const MetricCard = styled.div`
  text-align: center;
  padding: ${({ theme }) => theme.spacing.base};
  background-color: ${({ theme }) => theme.colors.surface};
  border-radius: ${({ theme }) => theme.borderRadius.base};
  border: 1px solid ${({ theme }) => theme.colors.border};
`;

const MetricLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-bottom: ${({ theme }) => theme.spacing.xs};
`;

const MetricValue = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
`;

const DataFlowAnimation = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  margin-top: ${({ theme }) => theme.spacing.lg};
  gap: ${({ theme }) => theme.spacing.base};
`;

const FlowArrow = styled(motion.div)`
  font-size: ${({ theme }) => theme.typography.fontSize.xl};
  color: ${({ theme }) => theme.colors.primary};
`;

const memoryLevels: MemoryLevel[] = [
  {
    id: 'registers',
    name: 'Registers',
    size: '64KB',
    latency: '1 cycle',
    bandwidth: '~8TB/s',
    scope: 'Per Thread',
    description: 'Thread-private memory with zero latency. Fastest memory tier, automatically managed by compiler. Limited quantity affects thread occupancy.',
    color: '#DC3545',
    relativeWidth: 20,
    threadMapping: 'Individual Thread'
  },
  {
    id: 'shared',
    name: 'Shared Memory',
    size: '128KB',
    latency: '~20 cycles',
    bandwidth: '~4TB/s',
    scope: 'Per Thread Block',
    description: 'Programmer-managed fast memory shared by all threads in a block. Enables efficient cooperation and data sharing within thread blocks.',
    color: '#FFC107',
    relativeWidth: 40,
    threadMapping: 'Thread Block (up to 1024 threads)'
  },
  {
    id: 'l1',
    name: 'L1 Cache',
    size: '128KB',
    latency: '~80 cycles',
    bandwidth: '~1TB/s',
    scope: 'Per SM',
    description: 'Hardware-managed cache for global memory accesses. Automatically caches frequently accessed data from global memory.',
    color: '#17A2B8',
    relativeWidth: 60,
    threadMapping: 'Streaming Multiprocessor'
  },
  {
    id: 'l2',
    name: 'L2 Cache',
    size: '40MB',
    latency: '~200 cycles',
    bandwidth: '~500GB/s',
    scope: 'Per GPU',
    description: 'Larger cache shared across all SMs. Reduces traffic to global memory and improves overall memory bandwidth utilization.',
    color: '#28A745',
    relativeWidth: 80,
    threadMapping: 'Entire GPU'
  },
  {
    id: 'global',
    name: 'Global Memory',
    size: '80GB',
    latency: '~400 cycles',
    bandwidth: '3.35TB/s',
    scope: 'Per GPU',
    description: 'Large capacity memory accessible by all threads. High latency but massive bandwidth when accessed with proper coalescing patterns.',
    color: '#6F42C1',
    relativeWidth: 100,
    threadMapping: 'All Threads on GPU'
  }
];

export const MemoryHierarchyVisualization: React.FC = () => {
  const [selectedLevel, setSelectedLevel] = useState<MemoryLevel | null>(null);
  const [showDataFlow, setShowDataFlow] = useState(false);

  const handleLevelClick = (level: MemoryLevel) => {
    setSelectedLevel(selectedLevel?.id === level.id ? null : level);
  };

  const toggleDataFlow = () => {
    setShowDataFlow(!showDataFlow);
  };

  return (
    <VisualizationContainer>
      <Title>GPU Memory Hierarchy</Title>
      
      <PyramidContainer>
        {memoryLevels.map((level, index) => (
          <MemoryLevelContainer
            key={level.id}
            $width={level.relativeWidth}
            $color={level.color}
            $isActive={selectedLevel?.id === level.id}
            onClick={() => handleLevelClick(level)}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98 }}
          >
            <LevelInfo>
              <LevelName>{level.name}</LevelName>
              <LevelScope>{level.scope}</LevelScope>
            </LevelInfo>
            
            <LevelMetrics>
              <Metric>{level.size}</Metric>
              <Metric>{level.latency}</Metric>
            </LevelMetrics>
          </MemoryLevelContainer>
        ))}
      </PyramidContainer>

      <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
        <button
          onClick={toggleDataFlow}
          style={{
            padding: '0.5rem 1rem',
            backgroundColor: 'var(--primary)',
            color: 'white',
            border: 'none',
            borderRadius: '0.5rem',
            cursor: 'pointer'
          }}
        >
          {showDataFlow ? 'Hide' : 'Show'} Data Flow Animation
        </button>
      </div>

      <AnimatePresence>
        {showDataFlow && (
          <DataFlowAnimation>
            <span>CPU</span>
            <FlowArrow
              animate={{ x: [0, 10, 0] }}
              transition={{ repeat: Infinity, duration: 2 }}
            >
              →
            </FlowArrow>
            <span>Global Memory</span>
            <FlowArrow
              animate={{ x: [0, 10, 0] }}
              transition={{ repeat: Infinity, duration: 2, delay: 0.3 }}
            >
              →
            </FlowArrow>
            <span>L2 Cache</span>
            <FlowArrow
              animate={{ x: [0, 10, 0] }}
              transition={{ repeat: Infinity, duration: 2, delay: 0.6 }}
            >
              →
            </FlowArrow>
            <span>L1/Shared</span>
            <FlowArrow
              animate={{ x: [0, 10, 0] }}
              transition={{ repeat: Infinity, duration: 2, delay: 0.9 }}
            >
              →
            </FlowArrow>
            <span>Registers</span>
            <FlowArrow
              animate={{ x: [0, 10, 0] }}
              transition={{ repeat: Infinity, duration: 2, delay: 1.2 }}
            >
              →
            </FlowArrow>
            <span>CUDA Cores</span>
          </DataFlowAnimation>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {selectedLevel && (
          <DetailPanel
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            <DetailTitle>{selectedLevel.name}</DetailTitle>
            <DetailDescription>{selectedLevel.description}</DetailDescription>
            
            <MetricsGrid>
              <MetricCard>
                <MetricLabel>Size</MetricLabel>
                <MetricValue>{selectedLevel.size}</MetricValue>
              </MetricCard>
              <MetricCard>
                <MetricLabel>Latency</MetricLabel>
                <MetricValue>{selectedLevel.latency}</MetricValue>
              </MetricCard>
              <MetricCard>
                <MetricLabel>Bandwidth</MetricLabel>
                <MetricValue>{selectedLevel.bandwidth}</MetricValue>
              </MetricCard>
              <MetricCard>
                <MetricLabel>Thread Mapping</MetricLabel>
                <MetricValue style={{ fontSize: '0.8rem' }}>{selectedLevel.threadMapping}</MetricValue>
              </MetricCard>
            </MetricsGrid>
          </DetailPanel>
        )}
      </AnimatePresence>
    </VisualizationContainer>
  );
};