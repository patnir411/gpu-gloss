import React from 'react';
import { useParams, Link } from 'react-router-dom';
import styled from 'styled-components';
import { PageContainer, PageHeader, PageTitle, PageDescription } from '../../components/common/Layout';
import { Card } from '../../components/common/Card';
import { Button } from '../../components/common/Button';
import { conceptsById } from '../../data/concepts';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useTheme } from '../../contexts/ThemeContext';
import { GPUArchitectureDiagram } from '../../components/interactive/GPUArchitectureDiagram';
import { MemoryHierarchyVisualization } from '../../components/interactive/MemoryHierarchyVisualization';
import { ThreadHierarchyVisualization } from '../../components/interactive/ThreadHierarchyVisualization';

const ConceptContainer = styled.div`
  max-width: 800px;
  margin: 0 auto;
`;

const ConceptMeta = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.base};
  margin-bottom: ${({ theme }) => theme.spacing.lg};
  flex-wrap: wrap;
`;

const CategoryBadge = styled.span`
  background-color: ${({ theme }) => theme.colors.primary};
  color: ${({ theme }) => theme.colors.text.inverse};
  padding: ${({ theme }) => theme.spacing.xs} ${({ theme }) => theme.spacing.sm};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const DifficultyBadge = styled.span<{ $difficulty: string }>`
  padding: ${({ theme }) => theme.spacing.xs} ${({ theme }) => theme.spacing.sm};
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

const TagContainer = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.xs};
  flex-wrap: wrap;
`;

const Tag = styled.span`
  background-color: ${({ theme }) => theme.colors.surface};
  color: ${({ theme }) => theme.colors.text.secondary};
  padding: ${({ theme }) => theme.spacing.xs} ${({ theme }) => theme.spacing.sm};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  border: 1px solid ${({ theme }) => theme.colors.border};
`;

const ContentSection = styled.section`
  margin-bottom: ${({ theme }) => theme.spacing['2xl']};
`;

const SectionTitle = styled.h2`
  font-size: ${({ theme }) => theme.typography.fontSize.xl};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing.base};
`;

const KeyPointsList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const KeyPoint = styled.li`
  display: flex;
  align-items: flex-start;
  gap: ${({ theme }) => theme.spacing.sm};
  margin-bottom: ${({ theme }) => theme.spacing.sm};
  
  &:before {
    content: '✓';
    color: ${({ theme }) => theme.colors.success};
    font-weight: bold;
    flex-shrink: 0;
    margin-top: 2px;
  }
`;

const TechnicalDetailsGrid = styled.div`
  display: grid;
  gap: ${({ theme }) => theme.spacing.base};
`;

const TechnicalDetail = styled.div`
  padding: ${({ theme }) => theme.spacing.base};
  background-color: ${({ theme }) => theme.colors.surface};
  border-radius: ${({ theme }) => theme.borderRadius.base};
  border: 1px solid ${({ theme }) => theme.colors.border};
`;

const DetailLabel = styled.div`
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing.xs};
`;

const DetailValue = styled.div`
  color: ${({ theme }) => theme.colors.primary};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  margin-bottom: ${({ theme }) => theme.spacing.xs};
`;

const DetailDescription = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const CodeSection = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing.lg};
`;

const CodeTitle = styled.h4`
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing.sm};
`;

const CodeDescription = styled.p`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-bottom: ${({ theme }) => theme.spacing.base};
`;

const RelatedConceptsList = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${({ theme }) => theme.spacing.sm};
`;

const RelatedConceptLink = styled(Link)`
  display: inline-block;
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.base};
  background-color: ${({ theme }) => theme.colors.surface};
  color: ${({ theme }) => theme.colors.primary};
  text-decoration: none;
  border-radius: ${({ theme }) => theme.borderRadius.base};
  border: 1px solid ${({ theme }) => theme.colors.border};
  transition: all ${({ theme }) => theme.transitions.fast};
  
  &:hover {
    background-color: ${({ theme }) => theme.colors.primary};
    color: ${({ theme }) => theme.colors.text.inverse};
    border-color: ${({ theme }) => theme.colors.primary};
  }
`;

const NavigationButtons = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.base};
  justify-content: space-between;
  margin-top: ${({ theme }) => theme.spacing['2xl']};
  padding-top: ${({ theme }) => theme.spacing.lg};
  border-top: 1px solid ${({ theme }) => theme.colors.border};
`;

const NotFoundMessage = styled.div`
  text-align: center;
  padding: ${({ theme }) => theme.spacing['3xl']};
`;

export const ConceptPage: React.FC = () => {
  const { conceptId } = useParams<{ conceptId: string }>();
  const { themeType } = useTheme();
  
  if (!conceptId || !conceptsById[conceptId]) {
    return (
      <PageContainer>
        <NotFoundMessage>
          <h1>Concept Not Found</h1>
          <p>The concept "{conceptId}" could not be found.</p>
          <Button variant="primary">
            <Link to="/" style={{ color: 'inherit', textDecoration: 'none' }}>
              Back to Home
            </Link>
          </Button>
        </NotFoundMessage>
      </PageContainer>
    );
  }

  const concept = conceptsById[conceptId];

  // Determine which visualization to show based on concept
  const getVisualization = () => {
    if (['streaming-multiprocessor', 'cuda-device-architecture', 'core', 'cuda-core', 'tensor-core'].includes(conceptId)) {
      return (
        <ContentSection>
          <SectionTitle>Interactive Architecture Diagram</SectionTitle>
          <GPUArchitectureDiagram />
        </ContentSection>
      );
    }
    
    if (['memory-hierarchy', 'registers', 'shared-memory', 'global-memory'].includes(conceptId)) {
      return (
        <ContentSection>
          <SectionTitle>Interactive Memory Hierarchy</SectionTitle>
          <MemoryHierarchyVisualization />
        </ContentSection>
      );
    }
    
    if (['thread', 'warp', 'thread-block', 'thread-block-grid', 'cuda-programming-model'].includes(conceptId)) {
      return (
        <ContentSection>
          <SectionTitle>Interactive Thread Hierarchy</SectionTitle>
          <ThreadHierarchyVisualization />
        </ContentSection>
      );
    }
    
    return null;
  };

  return (
    <PageContainer>
      <ConceptContainer>
        <PageHeader>
          <ConceptMeta>
            <CategoryBadge>{concept.category.replace('-', ' ')}</CategoryBadge>
            <DifficultyBadge $difficulty={concept.difficulty}>
              {concept.difficulty}
            </DifficultyBadge>
            <TagContainer>
              {concept.tags.map(tag => (
                <Tag key={tag}>{tag}</Tag>
              ))}
            </TagContainer>
          </ConceptMeta>
          
          <PageTitle>{concept.title}</PageTitle>
          <PageDescription>{concept.definition}</PageDescription>
        </PageHeader>

        <ContentSection>
          <SectionTitle>Key Points</SectionTitle>
          <Card padding="lg">
            <KeyPointsList>
              {concept.keyPoints.map((point, index) => (
                <KeyPoint key={index}>{point}</KeyPoint>
              ))}
            </KeyPointsList>
          </Card>
        </ContentSection>

        {concept.technicalDetails && concept.technicalDetails.length > 0 && (
          <ContentSection>
            <SectionTitle>Technical Details</SectionTitle>
            <TechnicalDetailsGrid>
              {concept.technicalDetails.map((detail, index) => (
                <TechnicalDetail key={index}>
                  <DetailLabel>{detail.label}</DetailLabel>
                  <DetailValue>{detail.value}</DetailValue>
                  {detail.description && (
                    <DetailDescription>{detail.description}</DetailDescription>
                  )}
                </TechnicalDetail>
              ))}
            </TechnicalDetailsGrid>
          </ContentSection>
        )}

        {getVisualization()}

        {concept.codeExamples && concept.codeExamples.length > 0 && (
          <ContentSection>
            <SectionTitle>Code Examples</SectionTitle>
            {concept.codeExamples.map((example, index) => (
              <CodeSection key={index}>
                <Card padding="none">
                  <div style={{ padding: '1rem 1rem 0 1rem' }}>
                    <CodeTitle>{example.filename || `Example ${index + 1}`}</CodeTitle>
                    <CodeDescription>{example.description}</CodeDescription>
                  </div>
                  <SyntaxHighlighter
                    language={example.language}
                    style={themeType === 'dark' ? oneDark : oneLight}
                    showLineNumbers
                    customStyle={{
                      margin: 0,
                      borderRadius: '0 0 8px 8px',
                    }}
                  >
                    {example.code}
                  </SyntaxHighlighter>
                </Card>
              </CodeSection>
            ))}
          </ContentSection>
        )}

        {concept.relatedConcepts.length > 0 && (
          <ContentSection>
            <SectionTitle>Related Concepts</SectionTitle>
            <RelatedConceptsList>
              {concept.relatedConcepts.map(relatedId => {
                const relatedConcept = conceptsById[relatedId];
                return relatedConcept ? (
                  <RelatedConceptLink key={relatedId} to={`/concept/${relatedId}`}>
                    {relatedConcept.title}
                  </RelatedConceptLink>
                ) : null;
              })}
            </RelatedConceptsList>
          </ContentSection>
        )}

        <NavigationButtons>
          <Button variant="outline">
            <Link to={`/category/${concept.category}`} style={{ color: 'inherit', textDecoration: 'none' }}>
              ← Back to {concept.category.replace('-', ' ')}
            </Link>
          </Button>
          <Button variant="ghost">
            <Link to="/search" style={{ color: 'inherit', textDecoration: 'none' }}>
              Explore More Concepts
            </Link>
          </Button>
        </NavigationButtons>
      </ConceptContainer>
    </PageContainer>
  );
};