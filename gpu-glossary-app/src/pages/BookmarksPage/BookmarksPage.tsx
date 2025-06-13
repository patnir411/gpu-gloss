import React, { useState } from 'react';
import styled from 'styled-components';
import { Link } from 'react-router-dom';
import { allConcepts } from '../../data/concepts';
import { Card } from '../../components/common/Card';
import { Button } from '../../components/common/Button';

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

const EmptyState = styled.div`
  text-align: center;
  padding: ${({ theme }) => theme.spacing['3xl']};
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const EmptyIcon = styled.div`
  font-size: 64px;
  margin-bottom: ${({ theme }) => theme.spacing.lg};
`;

const EmptyTitle = styled.h2`
  font-size: ${({ theme }) => theme.typography.fontSize.xl};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing.base};
`;

const EmptyDescription = styled.p`
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-bottom: ${({ theme }) => theme.spacing.lg};
`;

const BookmarksList = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.base};
`;

const BookmarkCard = styled(Card)`
  padding: ${({ theme }) => theme.spacing.lg};
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: ${({ theme }) => theme.spacing.base};
`;

const BookmarkContent = styled.div`
  flex: 1;
`;

const BookmarkTitle = styled.h3`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin: 0 0 ${({ theme }) => theme.spacing.sm} 0;
`;

const BookmarkDescription = styled.p`
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  color: ${({ theme }) => theme.colors.text.secondary};
  margin: 0 0 ${({ theme }) => theme.spacing.base} 0;
`;

const BookmarkMeta = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.base};
  align-items: center;
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const CategoryBadge = styled.span<{ $color: string }>`
  padding: 2px 8px;
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  color: white;
  background-color: ${({ $color }) => $color};
`;

const BookmarkActions = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.sm};
`;

const FilterSection = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.base};
  margin-bottom: ${({ theme }) => theme.spacing.xl};
  flex-wrap: wrap;
`;

const FilterButton = styled(Button)<{ $isActive: boolean }>`
  ${({ $isActive, theme }) => $isActive && `
    background-color: ${theme.colors.primary};
    color: ${theme.colors.text.inverse};
  `}
`;

const Stats = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.lg};
  margin-bottom: ${({ theme }) => theme.spacing.lg};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
`;

export const BookmarksPage: React.FC = () => {
  // Mock bookmarks data - in a real app, this would come from user state/localStorage
  const [bookmarks] = useState<string[]>([
    'cuda-programming-model',
    'streaming-multiprocessor',
    'shared-memory',
    'warp',
    'tensor-core'
  ]);
  
  const [filter, setFilter] = useState<string>('all');

  const bookmarkedConcepts = allConcepts.filter(concept => 
    bookmarks.includes(concept.id)
  );

  const filteredConcepts = filter === 'all' 
    ? bookmarkedConcepts 
    : bookmarkedConcepts.filter(concept => concept.category === filter);

  const getCategoryColor = (categoryId: string) => {
    switch (categoryId) {
      case 'device-hardware':
        return '#76B900';
      case 'device-software':
        return '#0073E6';
      case 'host-software':
        return '#FF6B35';
      default:
        return '#6B7280';
    }
  };

  const getCategoryName = (categoryId: string) => {
    switch (categoryId) {
      case 'device-hardware':
        return 'Device Hardware';
      case 'device-software':
        return 'Device Software';
      case 'host-software':
        return 'Host Software';
      default:
        return 'Unknown';
    }
  };

  const categories = ['all', 'device-hardware', 'device-software', 'host-software'];
  const categoryStats = {
    all: bookmarkedConcepts.length,
    'device-hardware': bookmarkedConcepts.filter(c => c.category === 'device-hardware').length,
    'device-software': bookmarkedConcepts.filter(c => c.category === 'device-software').length,
    'host-software': bookmarkedConcepts.filter(c => c.category === 'host-software').length,
  };

  if (bookmarks.length === 0) {
    return (
      <Container>
        <Title>Bookmarks</Title>
        <EmptyState>
          <EmptyIcon>🔖</EmptyIcon>
          <EmptyTitle>No bookmarks yet</EmptyTitle>
          <EmptyDescription>
            Bookmark concepts as you learn to create your personal reference collection.
            <br />
            Look for the bookmark icon on concept pages to save them here.
          </EmptyDescription>
          <Link to="/">
            <Button variant="primary">Start Exploring</Button>
          </Link>
        </EmptyState>
      </Container>
    );
  }

  return (
    <Container>
      <Title>Bookmarks</Title>
      
      <Stats>
        <span>📚 {bookmarks.length} concepts bookmarked</span>
        <span>🗂️ {categories.length - 1} categories</span>
      </Stats>

      <FilterSection>
        {categories.map(category => (
          <FilterButton
            key={category}
            variant="outline"
            size="sm"
            $isActive={filter === category}
            onClick={() => setFilter(category)}
          >
            {category === 'all' ? 'All' : getCategoryName(category)} 
            ({categoryStats[category as keyof typeof categoryStats]})
          </FilterButton>
        ))}
      </FilterSection>

      <BookmarksList>
        {filteredConcepts.map(concept => (
          <BookmarkCard key={concept.id}>
            <BookmarkContent>
              <BookmarkTitle>{concept.title}</BookmarkTitle>
              <BookmarkDescription>{concept.definition}</BookmarkDescription>
              <BookmarkMeta>
                <CategoryBadge $color={getCategoryColor(concept.category)}>
                  {getCategoryName(concept.category)}
                </CategoryBadge>
                <span>•</span>
                <span>{concept.keyPoints?.length || 0} key points</span>
                <span>•</span>
                <span>{concept.relatedConcepts?.length || 0} related concepts</span>
              </BookmarkMeta>
            </BookmarkContent>
            <BookmarkActions>
              <Link to={`/concept/${concept.id}`}>
                <Button size="sm">View</Button>
              </Link>
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  // In a real app, this would remove from bookmarks
                  console.log('Remove bookmark:', concept.id);
                }}
              >
                Remove
              </Button>
            </BookmarkActions>
          </BookmarkCard>
        ))}
      </BookmarksList>
    </Container>
  );
};