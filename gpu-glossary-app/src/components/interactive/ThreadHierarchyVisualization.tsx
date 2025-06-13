import React, { useState } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';

interface ThreadLevel {
  id: string;
  name: string;
  description: string;
  count: string;
  color: string;
  children?: ThreadLevel[];
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

const HierarchyContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.lg};
  align-items: center;
`;

const LevelContainer = styled(motion.div)<{ $color: string; $isExpanded: boolean }>`
  width: 100%;
  max-width: 800px;
  background: linear-gradient(135deg, ${({ $color }) => $color}22, ${({ $color }) => $color}11);
  border: 2px solid ${({ $color }) => $color};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: ${({ theme }) => theme.spacing.lg};
  cursor: pointer;
  transition: all ${({ theme }) => theme.transitions.normal};
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px ${({ theme }) => theme.colors.shadow};
  }
`;

const LevelHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${({ theme }) => theme.spacing.base};
`;

const LevelName = styled.h4<{ $color: string }>`
  margin: 0;
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ $color }) => $color};
`;

const LevelCount = styled.span<{ $color: string }>`
  background-color: ${({ $color }) => $color};
  color: white;
  padding: ${({ theme }) => theme.spacing.xs} ${({ theme }) => theme.spacing.sm};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const LevelDescription = styled.p`
  margin: 0;
  color: ${({ theme }) => theme.colors.text.secondary};
  line-height: ${({ theme }) => theme.typography.lineHeight.relaxed};
`;

const ChildrenContainer = styled(motion.div)`
  margin-top: ${({ theme }) => theme.spacing.lg};
  padding-left: ${({ theme }) => theme.spacing.xl};
  border-left: 2px dashed ${({ theme }) => theme.colors.border};
`;

const VisualizationGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(60px, 1fr));
  gap: ${({ theme }) => theme.spacing.sm};
  margin-top: ${({ theme }) => theme.spacing.lg};
  max-width: 600px;
`;

const ThreadBlock = styled(motion.div)<{ $color: string; $isActive: boolean }>`
  width: 60px;
  height: 60px;
  background-color: ${({ $color, $isActive }) => $isActive ? $color : `${$color}66`};
  border: 2px solid ${({ $color }) => $color};
  border-radius: ${({ theme }) => theme.borderRadius.base};
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: bold;
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  cursor: pointer;
  position: relative;
  
  &:before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(45deg, transparent, ${({ $color }) => $color}44, transparent);
    border-radius: ${({ theme }) => theme.borderRadius.base};
    z-index: -1;
    opacity: ${({ $isActive }) => $isActive ? 1 : 0};
    transition: opacity 0.3s;
  }
`;

const WarpContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(8, 1fr);
  gap: 2px;
  width: 100%;
  margin-top: ${({ theme }) => theme.spacing.base};
`;

const Thread = styled(motion.div)<{ $color: string; $isHighlighted: boolean }>`
  width: 16px;
  height: 16px;
  background-color: ${({ $color, $isHighlighted }) => $isHighlighted ? $color : `${$color}88`};
  border-radius: 2px;
  border: 1px solid ${({ $color }) => $color};
`;

const Controls = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.base};
  justify-content: center;
  margin-top: ${({ theme }) => theme.spacing.lg};
`;

const ControlButton = styled.button<{ $active: boolean }>`
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.base};
  background-color: ${({ $active, theme }) => $active ? theme.colors.primary : theme.colors.surface};
  color: ${({ $active, theme }) => $active ? theme.colors.text.inverse : theme.colors.text.primary};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.borderRadius.base};
  cursor: pointer;
  transition: all ${({ theme }) => theme.transitions.fast};
  
  &:hover {
    background-color: ${({ theme }) => theme.colors.primary};
    color: ${({ theme }) => theme.colors.text.inverse};
  }
`;

const threadHierarchy: ThreadLevel[] = [
  {
    id: 'grid',
    name: 'Grid',
    description: 'Highest level containing all thread blocks. Can span multiple SMs across the entire GPU. Blocks execute independently and can complete in any order.',
    count: '1 Grid',
    color: '#6F42C1',
    children: [
      {
        id: 'block',
        name: 'Thread Block',
        description: 'Collection of threads that execute on the same SM. Threads can cooperate via shared memory and synchronization barriers.',
        count: 'Multiple Blocks',
        color: '#28A745',
        children: [
          {
            id: 'warp',
            name: 'Warp',
            description: 'Group of 32 threads that execute the same instruction simultaneously (SIMT). Fundamental unit of scheduling.',
            count: '32 Threads',
            color: '#FFC107'
          }
        ]
      }
    ]
  }
];

type ViewMode = 'hierarchy' | 'grid' | 'block' | 'warp';

export const ThreadHierarchyVisualization: React.FC = () => {
  const [expandedLevels, setExpandedLevels] = useState<Set<string>>(new Set(['grid']));
  const [viewMode, setViewMode] = useState<ViewMode>('hierarchy');
  const [highlightedWarp, setHighlightedWarp] = useState<number | null>(null);

  const toggleLevel = (levelId: string) => {
    const newExpanded = new Set(expandedLevels);
    if (newExpanded.has(levelId)) {
      newExpanded.delete(levelId);
    } else {
      newExpanded.add(levelId);
    }
    setExpandedLevels(newExpanded);
  };

  const renderLevel = (level: ThreadLevel, depth = 0) => (
    <LevelContainer
      key={level.id}
      $color={level.color}
      $isExpanded={expandedLevels.has(level.id)}
      onClick={() => toggleLevel(level.id)}
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: depth * 0.1 }}
      style={{ marginLeft: `${depth * 20}px` }}
    >
      <LevelHeader>
        <LevelName $color={level.color}>{level.name}</LevelName>
        <LevelCount $color={level.color}>{level.count}</LevelCount>
      </LevelHeader>
      <LevelDescription>{level.description}</LevelDescription>
      
      <AnimatePresence>
        {expandedLevels.has(level.id) && level.children && (
          <ChildrenContainer
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            {level.children.map(child => renderLevel(child, depth + 1))}
          </ChildrenContainer>
        )}
      </AnimatePresence>
    </LevelContainer>
  );

  const renderGridVisualization = () => (
    <VisualizationGrid>
      {Array.from({ length: 16 }, (_, blockIndex) => (
        <ThreadBlock
          key={blockIndex}
          $color="#28A745"
          $isActive={true}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
        >
          B{blockIndex}
        </ThreadBlock>
      ))}
    </VisualizationGrid>
  );

  const renderBlockVisualization = () => (
    <div>
      <h4 style={{ textAlign: 'center', marginBottom: '1rem' }}>
        Thread Block (32 Warps × 32 Threads = 1024 Threads)
      </h4>
      <VisualizationGrid style={{ gridTemplateColumns: 'repeat(8, 1fr)', maxWidth: '400px' }}>
        {Array.from({ length: 32 }, (_, warpIndex) => (
          <ThreadBlock
            key={warpIndex}
            $color="#FFC107"
            $isActive={highlightedWarp === warpIndex}
            onClick={() => setHighlightedWarp(highlightedWarp === warpIndex ? null : warpIndex)}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
          >
            W{warpIndex}
          </ThreadBlock>
        ))}
      </VisualizationGrid>
    </div>
  );

  const renderWarpVisualization = () => (
    <div>
      <h4 style={{ textAlign: 'center', marginBottom: '1rem' }}>
        Warp (32 Threads executing SIMT)
      </h4>
      <WarpContainer style={{ maxWidth: '400px', margin: '0 auto' }}>
        {Array.from({ length: 32 }, (_, threadIndex) => (
          <Thread
            key={threadIndex}
            $color="#DC3545"
            $isHighlighted={true}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: threadIndex * 0.05 }}
            whileHover={{ scale: 1.2 }}
            title={`Thread ${threadIndex}`}
          />
        ))}
      </WarpContainer>
      <p style={{ textAlign: 'center', marginTop: '1rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
        All 32 threads execute the same instruction simultaneously
      </p>
    </div>
  );

  return (
    <VisualizationContainer>
      <Title>CUDA Thread Hierarchy</Title>
      
      <Controls>
        <ControlButton 
          $active={viewMode === 'hierarchy'} 
          onClick={() => setViewMode('hierarchy')}
        >
          Hierarchy
        </ControlButton>
        <ControlButton 
          $active={viewMode === 'grid'} 
          onClick={() => setViewMode('grid')}
        >
          Grid View
        </ControlButton>
        <ControlButton 
          $active={viewMode === 'block'} 
          onClick={() => setViewMode('block')}
        >
          Block View
        </ControlButton>
        <ControlButton 
          $active={viewMode === 'warp'} 
          onClick={() => setViewMode('warp')}
        >
          Warp View
        </ControlButton>
      </Controls>

      <AnimatePresence mode="wait">
        {viewMode === 'hierarchy' && (
          <motion.div
            key="hierarchy"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <HierarchyContainer>
              {threadHierarchy.map(level => renderLevel(level))}
            </HierarchyContainer>
          </motion.div>
        )}
        
        {viewMode === 'grid' && (
          <motion.div
            key="grid"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}
          >
            <h4>Grid of Thread Blocks</h4>
            {renderGridVisualization()}
            <p style={{ textAlign: 'center', marginTop: '1rem', fontSize: '0.9rem' }}>
              Each block can execute on any available SM
            </p>
          </motion.div>
        )}
        
        {viewMode === 'block' && (
          <motion.div
            key="block"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}
          >
            {renderBlockVisualization()}
            <p style={{ textAlign: 'center', marginTop: '1rem', fontSize: '0.9rem' }}>
              Click a warp to highlight it. Warps execute independently.
            </p>
          </motion.div>
        )}
        
        {viewMode === 'warp' && (
          <motion.div
            key="warp"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}
          >
            {renderWarpVisualization()}
          </motion.div>
        )}
      </AnimatePresence>
    </VisualizationContainer>
  );
};