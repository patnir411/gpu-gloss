import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import styled from 'styled-components';
import { categories } from '../../data/categories/categories';
import { learningPaths } from '../../data/learning-paths';
import { allConcepts } from '../../data/concepts';

const SidebarContainer = styled.div`
  padding: ${({ theme }) => theme.spacing.lg};
  height: 100%;
  overflow-y: auto;
`;

const SidebarSection = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing['2xl']};
`;

const SectionHeader = styled.h3`
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing.base};
  padding-bottom: ${({ theme }) => theme.spacing.sm};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
`;

const NavList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const NavItem = styled.li`
  margin-bottom: ${({ theme }) => theme.spacing.xs};
`;

const NavLink = styled(Link)<{ $isActive?: boolean }>`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.sm};
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.base};
  border-radius: ${({ theme }) => theme.borderRadius.base};
  text-decoration: none;
  color: ${({ theme }) => theme.colors.text.primary};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  transition: all ${({ theme }) => theme.transitions.fast};
  
  ${({ $isActive, theme }) => $isActive && `
    background-color: ${theme.colors.primary};
    color: ${theme.colors.text.inverse};
  `}
  
  &:hover {
    background-color: ${({ $isActive, theme }) => 
      $isActive ? theme.colors.primary : theme.colors.surfaceHover
    };
    text-decoration: none;
    color: ${({ $isActive, theme }) => 
      $isActive ? theme.colors.text.inverse : theme.colors.primary
    };
  }
`;

const CategoryIcon = styled.span`
  font-size: 18px;
  width: 20px;
  text-align: center;
`;

const ConceptCount = styled.span`
  margin-left: auto;
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  color: ${({ theme }) => theme.colors.text.secondary};
  background-color: ${({ theme }) => theme.colors.surface};
  padding: 2px 6px;
  border-radius: ${({ theme }) => theme.borderRadius.sm};
`;

const ExpandableSection = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing.base};
`;

const ExpandButton = styled.button<{ $isExpanded: boolean }>`
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.base};
  border: none;
  background: none;
  color: ${({ theme }) => theme.colors.text.primary};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  border-radius: ${({ theme }) => theme.borderRadius.base};
  cursor: pointer;
  transition: all ${({ theme }) => theme.transitions.fast};
  
  &:hover {
    background-color: ${({ theme }) => theme.colors.surfaceHover};
  }
  
  &::after {
    content: '${({ $isExpanded }) => $isExpanded ? '▼' : '▶'}';
    font-size: 12px;
    transition: transform ${({ theme }) => theme.transitions.fast};
  }
`;

const SubNavList = styled.ul<{ $isExpanded: boolean }>`
  list-style: none;
  padding: 0;
  margin: 0;
  padding-left: ${({ theme }) => theme.spacing.lg};
  max-height: ${({ $isExpanded }) => $isExpanded ? 'none' : '0'};
  overflow: hidden;
  transition: all ${({ theme }) => theme.transitions.normal};
`;

const SubNavItem = styled.li`
  margin-bottom: ${({ theme }) => theme.spacing.xs};
`;

const SubNavLink = styled(Link)<{ $isActive?: boolean }>`
  display: block;
  padding: ${({ theme }) => theme.spacing.xs} ${({ theme }) => theme.spacing.sm};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  text-decoration: none;
  color: ${({ theme }) => theme.colors.text.secondary};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  transition: all ${({ theme }) => theme.transitions.fast};
  
  ${({ $isActive, theme }) => $isActive && `
    background-color: ${theme.colors.primary};
    color: ${theme.colors.text.inverse};
  `}
  
  &:hover {
    background-color: ${({ $isActive, theme }) => 
      $isActive ? theme.colors.primary : theme.colors.surfaceHover
    };
    color: ${({ $isActive, theme }) => 
      $isActive ? theme.colors.text.inverse : theme.colors.primary
    };
    text-decoration: none;
  }
`;

const DifficultyBadge = styled.span<{ $difficulty: string }>`
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  padding: 2px 6px;
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

export const Sidebar: React.FC = () => {
  const location = useLocation();
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    'device-hardware': true,
    'device-software': false,
    'host-software': false,
    'learning-paths': false,
  });

  const toggleSection = (sectionId: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }));
  };

  const isActive = (path: string) => location.pathname === path;

  return (
    <SidebarContainer>
      {/* Learning Paths */}
      <SidebarSection>
        <ExpandableSection>
          <ExpandButton
            $isExpanded={expandedSections['learning-paths']}
            onClick={() => toggleSection('learning-paths')}
          >
            📚 Learning Paths
          </ExpandButton>
          <SubNavList $isExpanded={expandedSections['learning-paths']}>
            {learningPaths.map(path => (
              <SubNavItem key={path.id}>
                <SubNavLink 
                  to={`/path/${path.id}`}
                  $isActive={isActive(`/path/${path.id}`)}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span>{path.name}</span>
                    <DifficultyBadge $difficulty={path.difficulty}>
                      {path.difficulty}
                    </DifficultyBadge>
                  </div>
                </SubNavLink>
              </SubNavItem>
            ))}
          </SubNavList>
        </ExpandableSection>
      </SidebarSection>

      {/* Categories */}
      <SidebarSection>
        <SectionHeader>Categories</SectionHeader>
        <NavList>
          {categories.map(category => (
            <NavItem key={category.id}>
              <ExpandableSection>
                <ExpandButton
                  $isExpanded={expandedSections[category.id]}
                  onClick={() => toggleSection(category.id)}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <CategoryIcon>{category.icon}</CategoryIcon>
                    <span>{category.name}</span>
                    <ConceptCount>{category.conceptCount}</ConceptCount>
                  </div>
                </ExpandButton>
                <SubNavList $isExpanded={expandedSections[category.id]}>
                  {category.subcategories.map(subcategory => (
                    <SubNavItem key={subcategory.id}>
                      <SubNavLink 
                        to={`/category/${category.id}/${subcategory.id}`}
                        $isActive={isActive(`/category/${category.id}/${subcategory.id}`)}
                      >
                        {subcategory.name} ({subcategory.conceptIds.length})
                      </SubNavLink>
                    </SubNavItem>
                  ))}
                  {/* Individual concepts */}
                  {allConcepts
                    .filter(concept => concept.category === category.id)
                    .map(concept => (
                      <SubNavItem key={concept.id}>
                        <SubNavLink 
                          to={`/concept/${concept.id}`}
                          $isActive={isActive(`/concept/${concept.id}`)}
                        >
                          {concept.title}
                        </SubNavLink>
                      </SubNavItem>
                    ))
                  }
                </SubNavList>
              </ExpandableSection>
            </NavItem>
          ))}
        </NavList>
      </SidebarSection>

      {/* Quick Links */}
      <SidebarSection>
        <SectionHeader>Quick Links</SectionHeader>
        <NavList>
          <NavItem>
            <NavLink to="/search" $isActive={isActive('/search')}>
              🔍 Search All Concepts
            </NavLink>
          </NavItem>
          <NavItem>
            <NavLink to="/progress" $isActive={isActive('/progress')}>
              📈 Learning Progress
            </NavLink>
          </NavItem>
          <NavItem>
            <NavLink to="/bookmarks" $isActive={isActive('/bookmarks')}>
              🔖 Bookmarks
            </NavLink>
          </NavItem>
        </NavList>
      </SidebarSection>
    </SidebarContainer>
  );
};