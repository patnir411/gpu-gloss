import React, { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import styled from 'styled-components';
import { learningPaths } from '../../data/learning-paths';
import { allConcepts } from '../../data/concepts';
import { Card } from '../../components/common/Card';
import { Button } from '../../components/common/Button';

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: ${({ theme }) => theme.spacing.xl};
`;

const Header = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing.xl};
`;

const Title = styled.h1`
  font-size: ${({ theme }) => theme.typography.fontSize['3xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing.base};
`;

const Description = styled.p`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-bottom: ${({ theme }) => theme.spacing.lg};
`;

const PathInfo = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.lg};
  flex-wrap: wrap;
  margin-bottom: ${({ theme }) => theme.spacing.lg};
`;

const InfoItem = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const DifficultyBadge = styled.span<{ $difficulty: string }>`
  padding: 4px 12px;
  border-radius: ${({ theme }) => theme.borderRadius.full};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
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

const ConceptsList = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.base};
`;

const ConceptItem = styled(Card)<{ $isCompleted: boolean; $isCurrent: boolean }>`
  padding: ${({ theme }) => theme.spacing.lg};
  border: 2px solid ${({ $isCurrent, theme }) => 
    $isCurrent ? theme.colors.primary : 'transparent'
  };
  opacity: ${({ $isCompleted }) => $isCompleted ? 0.7 : 1};
  
  ${({ $isCompleted }) => $isCompleted && `
    text-decoration: line-through;
  `}
`;

const ConceptHeader = styled.div`
  display: flex;
  justify-content: between;
  align-items: flex-start;
  gap: ${({ theme }) => theme.spacing.base};
  margin-bottom: ${({ theme }) => theme.spacing.base};
`;

const ConceptTitle = styled.h3`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin: 0;
  flex: 1;
`;

const ConceptDescription = styled.p`
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  color: ${({ theme }) => theme.colors.text.secondary};
  margin: 0 0 ${({ theme }) => theme.spacing.base} 0;
`;

const ConceptActions = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.base};
  align-items: center;
`;

const CheckMark = styled.span<{ $isCompleted: boolean }>`
  font-size: 20px;
  color: ${({ $isCompleted, theme }) => 
    $isCompleted ? theme.colors.success : theme.colors.text.secondary
  };
`;

const OutcomesList = styled.ul`
  list-style: none;
  padding: 0;
  margin: ${({ theme }) => theme.spacing.lg} 0;
`;

const OutcomeItem = styled.li`
  padding: ${({ theme }) => theme.spacing.sm} 0;
  color: ${({ theme }) => theme.colors.text.secondary};
  
  &::before {
    content: '✓';
    color: ${({ theme }) => theme.colors.success};
    margin-right: ${({ theme }) => theme.spacing.sm};
  }
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 8px;
  background-color: ${({ theme }) => theme.colors.surface};
  border-radius: ${({ theme }) => theme.borderRadius.full};
  margin: ${({ theme }) => theme.spacing.lg} 0;
  overflow: hidden;
`;

const ProgressFill = styled.div<{ $progress: number }>`
  width: ${({ $progress }) => $progress}%;
  height: 100%;
  background-color: ${({ theme }) => theme.colors.primary};
  transition: width ${({ theme }) => theme.transitions.normal};
`;

export const LearningPathPage: React.FC = () => {
  const { pathId } = useParams<{ pathId: string }>();
  const [completedConcepts, setCompletedConcepts] = useState<Set<string>>(new Set());
  const [currentConceptIndex] = useState(0);

  const learningPath = learningPaths.find(path => path.id === pathId);

  if (!learningPath) {
    return (
      <Container>
        <Title>Learning Path Not Found</Title>
        <Description>The requested learning path does not exist.</Description>
        <Link to="/">
          <Button variant="primary">Back to Home</Button>
        </Link>
      </Container>
    );
  }

  const pathConcepts = learningPath.conceptSequence.map(conceptId => 
    allConcepts.find(concept => concept.id === conceptId)
  ).filter(Boolean);

  const progressPercentage = (completedConcepts.size / pathConcepts.length) * 100;

  const toggleConceptCompletion = (conceptId: string) => {
    const newCompleted = new Set(completedConcepts);
    if (newCompleted.has(conceptId)) {
      newCompleted.delete(conceptId);
    } else {
      newCompleted.add(conceptId);
    }
    setCompletedConcepts(newCompleted);
  };

  // Commented out unused functions
  // const startLearning = () => {
  //   setCurrentConceptIndex(0);
  // };

  // const nextConcept = () => {
  //   if (currentConceptIndex < pathConcepts.length - 1) {
  //     setCurrentConceptIndex(currentConceptIndex + 1);
  //   }
  // };

  return (
    <Container>
      <Header>
        <Title>{learningPath.name}</Title>
        <Description>{learningPath.description}</Description>
        
        <PathInfo>
          <InfoItem>
            <span>📊</span>
            <DifficultyBadge $difficulty={learningPath.difficulty}>
              {learningPath.difficulty}
            </DifficultyBadge>
          </InfoItem>
          <InfoItem>
            <span>⏱️</span>
            <span>{learningPath.estimatedTime}</span>
          </InfoItem>
          <InfoItem>
            <span>📚</span>
            <span>{pathConcepts.length} concepts</span>
          </InfoItem>
          <InfoItem>
            <span>✅</span>
            <span>{completedConcepts.size} completed</span>
          </InfoItem>
        </PathInfo>

        <ProgressBar>
          <ProgressFill $progress={progressPercentage} />
        </ProgressBar>

        {learningPath.prerequisites && (
          <InfoItem style={{ marginBottom: '16px' }}>
            <span>📋</span>
            <span>Prerequisites: {learningPath.prerequisites.join(', ')}</span>
          </InfoItem>
        )}
      </Header>

      {learningPath.outcomes && (
        <div style={{ marginBottom: '24px' }}>
          <Card>
            <h3 style={{ marginTop: 0 }}>Learning Outcomes</h3>
            <OutcomesList>
              {learningPath.outcomes.map((outcome, index) => (
                <OutcomeItem key={index}>{outcome}</OutcomeItem>
              ))}
            </OutcomesList>
          </Card>
        </div>
      )}

      <ConceptsList>
        {pathConcepts.map((concept, index) => {
          if (!concept) return null;
          
          const isCompleted = completedConcepts.has(concept.id);
          const isCurrent = index === currentConceptIndex;

          return (
            <ConceptItem
              key={concept.id}
              $isCompleted={isCompleted}
              $isCurrent={isCurrent}
            >
              <ConceptHeader>
                <ConceptTitle>{concept.title}</ConceptTitle>
                <ConceptActions>
                  <CheckMark
                    $isCompleted={isCompleted}
                    onClick={() => toggleConceptCompletion(concept.id)}
                    style={{ cursor: 'pointer' }}
                  >
                    {isCompleted ? '✅' : '⭕'}
                  </CheckMark>
                  <Link to={`/concept/${concept.id}`}>
                    <Button size="sm">Learn</Button>
                  </Link>
                </ConceptActions>
              </ConceptHeader>
              <ConceptDescription>{concept.definition}</ConceptDescription>
              {isCurrent && (
                <p style={{ color: '#0073E6', fontWeight: 'bold', margin: 0 }}>
                  📍 Current concept in your learning path
                </p>
              )}
            </ConceptItem>
          );
        })}
      </ConceptsList>
    </Container>
  );
};