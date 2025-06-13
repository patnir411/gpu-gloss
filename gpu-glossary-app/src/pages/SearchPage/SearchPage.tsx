import React, { useState, useEffect, useMemo } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import styled from 'styled-components';
import Fuse from 'fuse.js';
import { PageContainer, PageHeader, PageTitle, PageDescription } from '../../components/common/Layout';
import { Card } from '../../components/common/Card';
import { Button } from '../../components/common/Button';
import { allConcepts } from '../../data/concepts';
import { categories } from '../../data/categories/categories';

const SearchContainer = styled.div`
  max-width: 800px;
  margin: 0 auto;
`;

const SearchInputContainer = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing['2xl']};
`;

const SearchInput = styled.input`
  width: 100%;
  padding: ${({ theme }) => theme.spacing.base} ${({ theme }) => theme.spacing.lg};
  border: 2px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  background-color: ${({ theme }) => theme.colors.surface};
  color: ${({ theme }) => theme.colors.text.primary};
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  transition: border-color ${({ theme }) => theme.transitions.fast};
  
  &:focus {
    outline: none;
    border-color: ${({ theme }) => theme.colors.primary};
    box-shadow: 0 0 0 3px ${({ theme }) => theme.colors.primary}33;
  }
  
  &::placeholder {
    color: ${({ theme }) => theme.colors.text.secondary};
  }
`;

const FiltersContainer = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.base};
  margin-bottom: ${({ theme }) => theme.spacing.lg};
  flex-wrap: wrap;
`;

const FilterSelect = styled.select`
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.base};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.borderRadius.base};
  background-color: ${({ theme }) => theme.colors.surface};
  color: ${({ theme }) => theme.colors.text.primary};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  
  &:focus {
    outline: none;
    border-color: ${({ theme }) => theme.colors.primary};
  }
`;

const ResultsContainer = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing['2xl']};
`;

const ResultsHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${({ theme }) => theme.spacing.lg};
`;

const ResultsCount = styled.div`
  color: ${({ theme }) => theme.colors.text.secondary};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
`;

const ResultsList = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.base};
`;

const ResultCard = styled(Card)`
  cursor: pointer;
  transition: transform ${({ theme }) => theme.transitions.fast};
  
  &:hover {
    transform: translateY(-1px);
  }
`;

const ResultTitle = styled.h3`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing.sm};
`;

const ResultDefinition = styled.p`
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-bottom: ${({ theme }) => theme.spacing.base};
  line-height: ${({ theme }) => theme.typography.lineHeight.relaxed};
`;

const ResultMeta = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: auto;
`;

const ResultBadges = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.sm};
  align-items: center;
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

const ScoreBadge = styled.span`
  background-color: ${({ theme }) => theme.colors.surface};
  color: ${({ theme }) => theme.colors.text.secondary};
  padding: ${({ theme }) => theme.spacing.xs} ${({ theme }) => theme.spacing.sm};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  border: 1px solid ${({ theme }) => theme.colors.border};
`;

const EmptyState = styled.div`
  text-align: center;
  padding: ${({ theme }) => theme.spacing['3xl']};
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const PopularSearches = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing['2xl']};
`;

const PopularSearchesList = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.sm};
  flex-wrap: wrap;
`;

const PopularSearchTag = styled.button`
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.base};
  background-color: ${({ theme }) => theme.colors.surface};
  color: ${({ theme }) => theme.colors.primary};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.borderRadius.base};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  cursor: pointer;
  transition: all ${({ theme }) => theme.transitions.fast};
  
  &:hover {
    background-color: ${({ theme }) => theme.colors.primary};
    color: ${({ theme }) => theme.colors.text.inverse};
    border-color: ${({ theme }) => theme.colors.primary};
  }
`;

// Fuse.js configuration
const fuseOptions = {
  keys: [
    { name: 'title', weight: 0.3 },
    { name: 'definition', weight: 0.2 },
    { name: 'keyPoints', weight: 0.2 },
    { name: 'tags', weight: 0.15 },
    { name: 'category', weight: 0.1 },
    { name: 'subcategory', weight: 0.05 },
  ],
  threshold: 0.4,
  includeScore: true,
  minMatchCharLength: 2,
};

const popularSearches = [
  'CUDA', 'GPU', 'Thread', 'Memory', 'Core', 'Warp', 'Kernel', 'SM', 
  'Architecture', 'Programming', 'Parallel', 'nvcc', 'PTX', 'SASS'
];

export const SearchPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [searchQuery, setSearchQuery] = useState(searchParams.get('q') || '');
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [difficultyFilter, setDifficultyFilter] = useState('all');

  const fuse = useMemo(() => new Fuse(allConcepts, fuseOptions), []);

  const searchResults = useMemo(() => {
    if (!searchQuery.trim()) return [];

    let results = fuse.search(searchQuery);

    // Apply filters
    if (categoryFilter !== 'all') {
      results = results.filter(result => result.item.category === categoryFilter);
    }

    if (difficultyFilter !== 'all') {
      results = results.filter(result => result.item.difficulty === difficultyFilter);
    }

    return results;
  }, [searchQuery, categoryFilter, difficultyFilter, fuse]);

  useEffect(() => {
    const query = searchParams.get('q');
    if (query) {
      setSearchQuery(query);
    }
  }, [searchParams]);

  const handleSearchChange = (value: string) => {
    setSearchQuery(value);
    if (value.trim()) {
      setSearchParams({ q: value });
    } else {
      setSearchParams({});
    }
  };

  const handlePopularSearchClick = (term: string) => {
    handleSearchChange(term);
  };

  return (
    <PageContainer>
      <SearchContainer>
        <PageHeader>
          <PageTitle>Search GPU Concepts</PageTitle>
          <PageDescription>
            Search through {allConcepts.length} GPU programming concepts and find what you need to know.
          </PageDescription>
        </PageHeader>

        <SearchInputContainer>
          <SearchInput
            type="text"
            placeholder="Search for GPU concepts, terms, or topics..."
            value={searchQuery}
            onChange={(e) => handleSearchChange(e.target.value)}
            autoFocus
          />
        </SearchInputContainer>

        <FiltersContainer>
          <FilterSelect
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
          >
            <option value="all">All Categories</option>
            {categories.map(category => (
              <option key={category.id} value={category.id}>
                {category.name}
              </option>
            ))}
          </FilterSelect>

          <FilterSelect
            value={difficultyFilter}
            onChange={(e) => setDifficultyFilter(e.target.value)}
          >
            <option value="all">All Difficulties</option>
            <option value="beginner">Beginner</option>
            <option value="intermediate">Intermediate</option>
            <option value="advanced">Advanced</option>
            <option value="expert">Expert</option>
          </FilterSelect>
        </FiltersContainer>

        {!searchQuery.trim() && (
          <PopularSearches>
            <h3 style={{ marginBottom: '1rem' }}>Popular Searches</h3>
            <PopularSearchesList>
              {popularSearches.map(term => (
                <PopularSearchTag
                  key={term}
                  onClick={() => handlePopularSearchClick(term)}
                >
                  {term}
                </PopularSearchTag>
              ))}
            </PopularSearchesList>
          </PopularSearches>
        )}

        <ResultsContainer>
          {searchQuery.trim() && (
            <ResultsHeader>
              <ResultsCount>
                {searchResults.length} result{searchResults.length !== 1 ? 's' : ''} 
                {searchQuery && ` for "${searchQuery}"`}
              </ResultsCount>
            </ResultsHeader>
          )}

          {searchQuery.trim() && searchResults.length === 0 && (
            <EmptyState>
              <h3>No results found</h3>
              <p>Try different keywords or check your spelling.</p>
              <div style={{ marginTop: '2rem' }}>
                <Button variant="outline" onClick={() => handleSearchChange('')}>
                  Clear Search
                </Button>
              </div>
            </EmptyState>
          )}

          <ResultsList>
            {searchResults.map(({ item: concept, score }) => (
              <ResultCard
                key={concept.id}
                variant="elevated"
                padding="lg"
                onClick={() => window.location.href = `/concept/${concept.id}`}
              >
                <ResultTitle>{concept.title}</ResultTitle>
                <ResultDefinition>
                  {concept.definition.length > 200
                    ? concept.definition.substring(0, 200) + '...'
                    : concept.definition
                  }
                </ResultDefinition>
                <ResultMeta>
                  <ResultBadges>
                    <CategoryBadge>
                      {concept.category.replace('-', ' ')}
                    </CategoryBadge>
                    <DifficultyBadge $difficulty={concept.difficulty}>
                      {concept.difficulty}
                    </DifficultyBadge>
                    <ScoreBadge>
                      {Math.round((1 - (score || 0)) * 100)}% match
                    </ScoreBadge>
                  </ResultBadges>
                </ResultMeta>
              </ResultCard>
            ))}
          </ResultsList>
        </ResultsContainer>

        {!searchQuery.trim() && (
          <div style={{ textAlign: 'center', marginTop: '3rem' }}>
            <Button variant="outline">
              <Link to="/" style={{ color: 'inherit', textDecoration: 'none' }}>
                ← Back to Home
              </Link>
            </Button>
          </div>
        )}
      </SearchContainer>
    </PageContainer>
  );
};