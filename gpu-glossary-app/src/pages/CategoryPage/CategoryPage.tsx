import React from 'react';
import { useParams, Link } from 'react-router-dom';
import styled from 'styled-components';
import { PageContainer, PageHeader, PageTitle, PageDescription } from '../../components/common/Layout';
import { Card } from '../../components/common/Card';
import { Button } from '../../components/common/Button';
import { categories } from '../../data/categories/categories';
import { allConcepts } from '../../data/concepts';

const CategoryContainer = styled.div`
  max-width: 1000px;
  margin: 0 auto;
`;

const ConceptGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: ${({ theme }) => theme.spacing.lg};
  margin-bottom: ${({ theme }) => theme.spacing['2xl']};
`;

const ConceptCard = styled(Card)`
  cursor: pointer;
  transition: transform ${({ theme }) => theme.transitions.normal};
  
  &:hover {
    transform: translateY(-2px);
  }
`;

const ConceptTitle = styled.h3`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing.sm};
`;

const ConceptDefinition = styled.p`
  color: ${({ theme }) => theme.colors.text.secondary};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  line-height: ${({ theme }) => theme.typography.lineHeight.relaxed};
  margin-bottom: ${({ theme }) => theme.spacing.base};
`;

const ConceptMeta = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: auto;
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
  margin-top: ${({ theme }) => theme.spacing.sm};
`;

const Tag = styled.span`
  background-color: ${({ theme }) => theme.colors.surface};
  color: ${({ theme }) => theme.colors.text.secondary};
  padding: 2px ${({ theme }) => theme.spacing.xs};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  border: 1px solid ${({ theme }) => theme.colors.border};
`;

const SubcategorySection = styled.section`
  margin-bottom: ${({ theme }) => theme.spacing['3xl']};
`;

const SubcategoryTitle = styled.h2`
  font-size: ${({ theme }) => theme.typography.fontSize['2xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing.lg};
`;

const CategoryStats = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: ${({ theme }) => theme.spacing.base};
  margin-bottom: ${({ theme }) => theme.spacing['2xl']};
`;

const StatCard = styled(Card)`
  text-align: center;
`;

const StatNumber = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize['2xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.primary};
  margin-bottom: ${({ theme }) => theme.spacing.xs};
`;

const StatLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const NotFoundMessage = styled.div`
  text-align: center;
  padding: ${({ theme }) => theme.spacing['3xl']};
`;

export const CategoryPage: React.FC = () => {
  const { categoryId } = useParams<{ categoryId: string }>();
  
  if (!categoryId) {
    return (
      <PageContainer>
        <NotFoundMessage>
          <h1>Category Not Found</h1>
          <p>No category specified.</p>
          <Button variant="primary">
            <Link to="/" style={{ color: 'inherit', textDecoration: 'none' }}>
              Back to Home
            </Link>
          </Button>
        </NotFoundMessage>
      </PageContainer>
    );
  }

  const category = categories.find(cat => cat.id === categoryId);
  
  if (!category) {
    return (
      <PageContainer>
        <NotFoundMessage>
          <h1>Category Not Found</h1>
          <p>The category "{categoryId}" could not be found.</p>
          <Button variant="primary">
            <Link to="/" style={{ color: 'inherit', textDecoration: 'none' }}>
              Back to Home
            </Link>
          </Button>
        </NotFoundMessage>
      </PageContainer>
    );
  }

  const categoryConcepts = allConcepts.filter(concept => concept.category === categoryId);
  const difficultyStats = categoryConcepts.reduce((acc, concept) => {
    acc[concept.difficulty] = (acc[concept.difficulty] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <PageContainer>
      <CategoryContainer>
        <PageHeader>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
            <span style={{ fontSize: '2rem' }}>{category.icon}</span>
            <PageTitle>{category.name}</PageTitle>
          </div>
          <PageDescription>{category.description}</PageDescription>
        </PageHeader>

        <CategoryStats>
          <StatCard padding="md">
            <StatNumber>{categoryConcepts.length}</StatNumber>
            <StatLabel>Total Concepts</StatLabel>
          </StatCard>
          <StatCard padding="md">
            <StatNumber>{category.subcategories.length}</StatNumber>
            <StatLabel>Subcategories</StatLabel>
          </StatCard>
          <StatCard padding="md">
            <StatNumber>{difficultyStats.beginner || 0}</StatNumber>
            <StatLabel>Beginner</StatLabel>
          </StatCard>
          <StatCard padding="md">
            <StatNumber>{difficultyStats.intermediate || 0}</StatNumber>
            <StatLabel>Intermediate</StatLabel>
          </StatCard>
          <StatCard padding="md">
            <StatNumber>{difficultyStats.advanced || 0}</StatNumber>
            <StatLabel>Advanced</StatLabel>
          </StatCard>
        </CategoryStats>

        {category.subcategories.map(subcategory => {
          const subcategoryConcepts = categoryConcepts.filter(concept => 
            subcategory.conceptIds.includes(concept.id)
          );
          
          if (subcategoryConcepts.length === 0) return null;

          return (
            <SubcategorySection key={subcategory.id}>
              <SubcategoryTitle>{subcategory.name}</SubcategoryTitle>
              <p style={{ marginBottom: '2rem', color: 'var(--text-secondary)' }}>
                {subcategory.description}
              </p>
              <ConceptGrid>
                {subcategoryConcepts.map(concept => (
                  <ConceptCard
                    key={concept.id}
                    variant="elevated"
                    padding="lg"
                    onClick={() => window.location.href = `/concept/${concept.id}`}
                  >
                    <ConceptTitle>{concept.title}</ConceptTitle>
                    <ConceptDefinition>
                      {concept.definition.length > 120 
                        ? concept.definition.substring(0, 120) + '...'
                        : concept.definition
                      }
                    </ConceptDefinition>
                    <ConceptMeta>
                      <DifficultyBadge $difficulty={concept.difficulty}>
                        {concept.difficulty}
                      </DifficultyBadge>
                    </ConceptMeta>
                    <TagContainer>
                      {concept.tags.slice(0, 3).map(tag => (
                        <Tag key={tag}>{tag}</Tag>
                      ))}
                      {concept.tags.length > 3 && (
                        <Tag>+{concept.tags.length - 3}</Tag>
                      )}
                    </TagContainer>
                  </ConceptCard>
                ))}
              </ConceptGrid>
            </SubcategorySection>
          );
        })}

        {/* Show concepts that don't belong to any subcategory */}
        {(() => {
          const allSubcategoryConceptIds = category.subcategories.flatMap(sub => sub.conceptIds);
          const uncategorizedConcepts = categoryConcepts.filter(concept => 
            !allSubcategoryConceptIds.includes(concept.id)
          );
          
          if (uncategorizedConcepts.length === 0) return null;

          return (
            <SubcategorySection>
              <SubcategoryTitle>Other Concepts</SubcategoryTitle>
              <ConceptGrid>
                {uncategorizedConcepts.map(concept => (
                  <ConceptCard
                    key={concept.id}
                    variant="elevated"
                    padding="lg"
                    onClick={() => window.location.href = `/concept/${concept.id}`}
                  >
                    <ConceptTitle>{concept.title}</ConceptTitle>
                    <ConceptDefinition>
                      {concept.definition.length > 120 
                        ? concept.definition.substring(0, 120) + '...'
                        : concept.definition
                      }
                    </ConceptDefinition>
                    <ConceptMeta>
                      <DifficultyBadge $difficulty={concept.difficulty}>
                        {concept.difficulty}
                      </DifficultyBadge>
                    </ConceptMeta>
                    <TagContainer>
                      {concept.tags.slice(0, 3).map(tag => (
                        <Tag key={tag}>{tag}</Tag>
                      ))}
                      {concept.tags.length > 3 && (
                        <Tag>+{concept.tags.length - 3}</Tag>
                      )}
                    </TagContainer>
                  </ConceptCard>
                ))}
              </ConceptGrid>
            </SubcategorySection>
          );
        })()}

        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '3rem', paddingTop: '2rem', borderTop: '1px solid var(--border)' }}>
          <Button variant="outline">
            <Link to="/" style={{ color: 'inherit', textDecoration: 'none' }}>
              ← Back to Home
            </Link>
          </Button>
          <Button variant="ghost">
            <Link to="/search" style={{ color: 'inherit', textDecoration: 'none' }}>
              Search All Concepts
            </Link>
          </Button>
        </div>
      </CategoryContainer>
    </PageContainer>
  );
};