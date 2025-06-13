import React from 'react';
import styled from 'styled-components';
import { learningPaths } from '../../data/learning-paths';
import { categories } from '../../data/categories/categories';
import { allConcepts } from '../../data/concepts';
import { Card } from '../../components/common/Card';
import { Link } from 'react-router-dom';

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: ${({ theme }) => theme.spacing.xl};
`;

const Title = styled.h1`
  font-size: ${({ theme }) => theme.typography.fontSize['3xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing.xl};
`;

const Section = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing['2xl']};
`;

const SectionTitle = styled.h2`
  font-size: ${({ theme }) => theme.typography.fontSize['2xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing.lg};
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: ${({ theme }) => theme.spacing.lg};
`;

const StatsCard = styled(Card)`
  padding: ${({ theme }) => theme.spacing.lg};
  text-align: center;
`;

const StatNumber = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize['4xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.primary};
  margin-bottom: ${({ theme }) => theme.spacing.sm};
`;

const StatLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const ProgressCard = styled(Card)`
  padding: ${({ theme }) => theme.spacing.lg};
`;

const ProgressHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${({ theme }) => theme.spacing.base};
`;

const ProgressTitle = styled.h3`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin: 0;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 12px;
  background-color: ${({ theme }) => theme.colors.surface};
  border-radius: ${({ theme }) => theme.borderRadius.full};
  margin: ${({ theme }) => theme.spacing.base} 0;
  overflow: hidden;
`;

const ProgressFill = styled.div<{ $progress: number; $color?: string }>`
  width: ${({ $progress }) => $progress}%;
  height: 100%;
  background-color: ${({ $color, theme }) => $color || theme.colors.primary};
  transition: width ${({ theme }) => theme.transitions.normal};
`;

const ProgressText = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const DifficultyBadge = styled.span<{ $difficulty: string }>`
  padding: 2px 8px;
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  color: white;
  
  background-color: ${({ $difficulty, theme }) => {
    switch ($difficulty) {
      case 'beginner':
        return theme.colors.success;
      case 'intermediate':
        return theme.colors.warning;
      case 'advanced':
        return theme.colors.secondary;
      case 'expert':
        return theme.colors.error;
      default:
        return theme.colors.text.secondary;
    }
  }};
`;

const ActionLinks = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.base};
  margin-top: ${({ theme }) => theme.spacing.base};
`;

export const ProgressPage: React.FC = () => {
  // Mock progress data - in a real app, this would come from user state/localStorage
  const mockProgress = {
    completedConcepts: new Set(['core', 'cuda-core', 'gpu-ram', 'cuda-programming-model']),
    learningPaths: {
      'gpu-fundamentals': 4,
      'cuda-programming': 2,
      'gpu-architecture': 0,
      'cuda-development': 0,
    },
    categories: {
      'device-hardware': 4,
      'device-software': 2,
      'host-software': 0,
    }
  };

  const totalConcepts = allConcepts.length;
  const completedCount = mockProgress.completedConcepts.size;
  const overallProgress = (completedCount / totalConcepts) * 100;

  return (
    <Container>
      <Title>Learning Progress</Title>

      {/* Overall Stats */}
      <Section>
        <Grid>
          <StatsCard>
            <StatNumber>{completedCount}</StatNumber>
            <StatLabel>Concepts Learned</StatLabel>
          </StatsCard>
          <StatsCard>
            <StatNumber>{totalConcepts}</StatNumber>
            <StatLabel>Total Concepts</StatLabel>
          </StatsCard>
          <StatsCard>
            <StatNumber>{Math.round(overallProgress)}%</StatNumber>
            <StatLabel>Overall Progress</StatLabel>
          </StatsCard>
          <StatsCard>
            <StatNumber>
              {Object.values(mockProgress.learningPaths).filter(p => p > 0).length}
            </StatNumber>
            <StatLabel>Paths Started</StatLabel>
          </StatsCard>
        </Grid>
      </Section>

      {/* Learning Paths Progress */}
      <Section>
        <SectionTitle>Learning Paths</SectionTitle>
        <Grid>
          {learningPaths.map(path => {
            const completed = mockProgress.learningPaths[path.id as keyof typeof mockProgress.learningPaths] || 0;
            const total = path.conceptSequence.length;
            const progress = (completed / total) * 100;

            return (
              <ProgressCard key={path.id}>
                <ProgressHeader>
                  <ProgressTitle>{path.name}</ProgressTitle>
                  <DifficultyBadge $difficulty={path.difficulty}>
                    {path.difficulty}
                  </DifficultyBadge>
                </ProgressHeader>
                <ProgressBar>
                  <ProgressFill $progress={progress} />
                </ProgressBar>
                <ProgressText>
                  {completed} of {total} concepts completed
                </ProgressText>
                <ActionLinks>
                  <Link to={`/path/${path.id}`}>
                    {completed > 0 ? 'Continue' : 'Start'} Learning Path
                  </Link>
                </ActionLinks>
              </ProgressCard>
            );
          })}
        </Grid>
      </Section>

      {/* Category Progress */}
      <Section>
        <SectionTitle>Categories</SectionTitle>
        <Grid>
          {categories.map(category => {
            const categoryCompleted = mockProgress.categories[category.id as keyof typeof mockProgress.categories] || 0;
            const total = category.conceptCount;
            const progress = (categoryCompleted / total) * 100;

            return (
              <ProgressCard key={category.id}>
                <ProgressHeader>
                  <ProgressTitle>
                    {category.icon} {category.name}
                  </ProgressTitle>
                </ProgressHeader>
                <ProgressBar>
                  <ProgressFill 
                    $progress={progress} 
                    $color={category.color}
                  />
                </ProgressBar>
                <ProgressText>
                  {categoryCompleted} of {total} concepts completed
                </ProgressText>
                <ActionLinks>
                  <Link to={`/category/${category.id}`}>
                    Explore Category
                  </Link>
                </ActionLinks>
              </ProgressCard>
            );
          })}
        </Grid>
      </Section>

      {/* Recent Activity */}
      <Section>
        <SectionTitle>Recent Activity</SectionTitle>
        <ProgressCard>
          <div style={{ color: '#6B7280', textAlign: 'center', padding: '24px' }}>
            <p>🎯 Start learning to see your recent activity here!</p>
            <p>Your completed concepts and learning milestones will appear in this section.</p>
          </div>
        </ProgressCard>
      </Section>
    </Container>
  );
};