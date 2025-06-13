import React from 'react';
import { Link } from 'react-router-dom';
import styled from 'styled-components';
import { PageContainer } from '../../components/common/Layout';
import { Card } from '../../components/common/Card';
import { Button } from '../../components/common/Button';
import { categories } from '../../data/categories/categories';
import { learningPaths } from '../../data/learning-paths';
import { allConcepts } from '../../data/concepts';

const HeroSection = styled.div`
  text-align: center;
  padding: ${({ theme }) => theme.spacing['3xl']} 0;
  background: linear-gradient(135deg, 
    ${({ theme }) => theme.colors.primary}15, 
    ${({ theme }) => theme.colors.secondary}15
  );
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  margin-bottom: ${({ theme }) => theme.spacing['3xl']};
`;

const HeroTitle = styled.h1`
  font-size: ${({ theme }) => theme.typography.fontSize['4xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing.lg};
  
  @media (min-width: ${({ theme }) => theme.breakpoints.md}) {
    font-size: 3.5rem;
  }
`;

const HeroSubtitle = styled.p`
  font-size: ${({ theme }) => theme.typography.fontSize.xl};
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-bottom: ${({ theme }) => theme.spacing['2xl']};
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
`;

const HeroActions = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.base};
  justify-content: center;
  flex-wrap: wrap;
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: ${({ theme }) => theme.spacing.lg};
  margin-bottom: ${({ theme }) => theme.spacing['3xl']};
`;

const StatCard = styled(Card)`
  text-align: center;
`;

const StatNumber = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize['3xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.primary};
  margin-bottom: ${({ theme }) => theme.spacing.sm};
`;

const StatLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
`;


const Section = styled.section`
  margin-bottom: ${({ theme }) => theme.spacing['3xl']};
`;

const SectionTitle = styled.h2`
  font-size: ${({ theme }) => theme.typography.fontSize['2xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing.lg};
`;

const CategoryGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: ${({ theme }) => theme.spacing.lg};
`;

const CategoryCard = styled(Card)`
  cursor: pointer;
  transition: transform ${({ theme }) => theme.transitions.normal};
  
  &:hover {
    transform: translateY(-4px);
  }
`;

const CategoryIcon = styled.div`
  font-size: 2rem;
  margin-bottom: ${({ theme }) => theme.spacing.base};
`;

const CategoryTitle = styled.h3`
  font-size: ${({ theme }) => theme.typography.fontSize.xl};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing.sm};
`;

const CategoryDescription = styled.p`
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-bottom: ${({ theme }) => theme.spacing.lg};
`;

const ConceptCount = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.primary};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const PathGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: ${({ theme }) => theme.spacing.lg};
`;

const PathCard = styled(Card)`
  cursor: pointer;
  transition: transform ${({ theme }) => theme.transitions.normal};
  
  &:hover {
    transform: translateY(-2px);
  }
`;

const PathHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${({ theme }) => theme.spacing.base};
`;

const PathTitle = styled.h3`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin: 0;
`;

const DifficultyBadge = styled.span<{ $difficulty: string }>`
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  padding: 4px 8px;
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  color: white;
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  
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

const PathDescription = styled.p`
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-bottom: ${({ theme }) => theme.spacing.base};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
`;

const PathMeta = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  color: ${({ theme }) => theme.colors.text.secondary};
`;

export const HomePage: React.FC = () => {
  const totalConcepts = allConcepts.length;
  const totalCategories = categories.length;
  const totalPaths = learningPaths.length;

  return (
    <PageContainer>
      <HeroSection>
        <HeroTitle>GPU Programming Glossary</HeroTitle>
        <HeroSubtitle>
          Master GPU programming from hardware to software with our comprehensive, 
          interactive educational guide covering 42+ essential concepts.
        </HeroSubtitle>
        <HeroActions>
          <Button variant="primary" size="lg">
            <Link to="/path/gpu-fundamentals" style={{ color: 'inherit', textDecoration: 'none' }}>
              Start Learning
            </Link>
          </Button>
          <Button variant="outline" size="lg">
            <Link to="/search" style={{ color: 'inherit', textDecoration: 'none' }}>
              Explore Concepts
            </Link>
          </Button>
        </HeroActions>
      </HeroSection>

      <StatsGrid>
        <StatCard padding="md">
          <StatNumber>{totalConcepts}</StatNumber>
          <StatLabel>GPU Concepts</StatLabel>
        </StatCard>
        <StatCard padding="md">
          <StatNumber>{totalCategories}</StatNumber>
          <StatLabel>Categories</StatLabel>
        </StatCard>
        <StatCard padding="md">
          <StatNumber>{totalPaths}</StatNumber>
          <StatLabel>Learning Paths</StatLabel>
        </StatCard>
        <StatCard padding="md">
          <StatNumber>100%</StatNumber>
          <StatLabel>Interactive</StatLabel>
        </StatCard>
      </StatsGrid>

      <Section>
        <SectionTitle>Learning Paths</SectionTitle>
        <PathGrid>
          {learningPaths.map(path => (
            <PathCard 
              key={path.id} 
              variant="elevated" 
              padding="lg"
              onClick={() => window.location.href = `/path/${path.id}`}
            >
              <PathHeader>
                <PathTitle>{path.name}</PathTitle>
                <DifficultyBadge $difficulty={path.difficulty}>
                  {path.difficulty}
                </DifficultyBadge>
              </PathHeader>
              <PathDescription>{path.description}</PathDescription>
              <PathMeta>
                <span>{path.conceptSequence.length} concepts</span>
                <span>{path.estimatedTime}</span>
              </PathMeta>
            </PathCard>
          ))}
        </PathGrid>
      </Section>

      <Section>
        <SectionTitle>Concept Categories</SectionTitle>
        <CategoryGrid>
          {categories.map(category => (
            <CategoryCard 
              key={category.id} 
              variant="elevated" 
              padding="lg"
              onClick={() => window.location.href = `/category/${category.id}`}
            >
              <CategoryIcon>{category.icon}</CategoryIcon>
              <CategoryTitle>{category.name}</CategoryTitle>
              <CategoryDescription>{category.description}</CategoryDescription>
              <ConceptCount>{category.conceptCount} concepts</ConceptCount>
            </CategoryCard>
          ))}
        </CategoryGrid>
      </Section>

      <Section>
        <Card variant="filled" padding="lg">
          <SectionTitle>Quick Start Guide</SectionTitle>
          <div style={{ display: 'grid', gap: '1rem' }}>
            <div>
              <strong>New to GPU Programming?</strong>
              <p>Start with the <Link to="/path/gpu-fundamentals">GPU Fundamentals</Link> path to understand the basics.</p>
            </div>
            <div>
              <strong>Know the Basics?</strong>
              <p>Jump into <Link to="/path/cuda-programming">CUDA Programming</Link> to learn parallel programming.</p>
            </div>
            <div>
              <strong>Want Deep Understanding?</strong>
              <p>Explore <Link to="/path/gpu-architecture">GPU Architecture</Link> for hardware internals.</p>
            </div>
            <div>
              <strong>Building Applications?</strong>
              <p>Check out <Link to="/path/cuda-development">CUDA Development</Link> for tools and best practices.</p>
            </div>
          </div>
        </Card>
      </Section>
    </PageContainer>
  );
};