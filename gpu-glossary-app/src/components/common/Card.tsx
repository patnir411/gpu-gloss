import React from 'react';
import styled, { css } from 'styled-components';
import type { Theme } from '../../contexts/ThemeContext';

export interface CardProps {
  children: React.ReactNode;
  variant?: 'default' | 'elevated' | 'outlined' | 'filled';
  padding?: 'none' | 'sm' | 'md' | 'lg';
  interactive?: boolean;
  className?: string;
  onClick?: () => void;
}

const CardWrapper = styled.div<{
  $variant: CardProps['variant'];
  $padding: CardProps['padding'];
  $interactive: boolean;
  theme: Theme;
}>`
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  transition: all ${({ theme }) => theme.transitions.normal};
  position: relative;
  overflow: hidden;

  /* Padding Variants */
  ${({ $padding, theme }) => {
    switch ($padding) {
      case 'none':
        return css`padding: 0;`;
      case 'sm':
        return css`padding: ${theme.spacing.base};`;
      case 'lg':
        return css`padding: ${theme.spacing['2xl']};`;
      default:
        return css`padding: ${theme.spacing.xl};`;
    }
  }}

  /* Interactive States */
  ${({ $interactive }) => $interactive && css`
    cursor: pointer;
    
    &:focus {
      outline: 2px solid ${({ theme }) => theme.colors.primary};
      outline-offset: 2px;
    }
  `}

  /* Style Variants */
  ${({ $variant, $interactive, theme }) => {
    switch ($variant) {
      case 'elevated':
        return css`
          background-color: ${theme.colors.surface};
          border: 1px solid ${theme.colors.border};
          box-shadow: 0 4px 6px -1px ${theme.colors.shadow},
                      0 2px 4px -1px ${theme.colors.shadow};

          ${$interactive && css`
            &:hover {
              transform: translateY(-2px);
              box-shadow: 0 10px 15px -3px ${theme.colors.shadow},
                          0 4px 6px -2px ${theme.colors.shadow};
            }

            &:active {
              transform: translateY(-1px);
              box-shadow: 0 4px 6px -1px ${theme.colors.shadow},
                          0 2px 4px -1px ${theme.colors.shadow};
            }
          `}
        `;
      
      case 'outlined':
        return css`
          background-color: transparent;
          border: 2px solid ${theme.colors.border};

          ${$interactive && css`
            &:hover {
              border-color: ${theme.colors.primary};
              background-color: ${theme.colors.surfaceHover};
            }

            &:active {
              background-color: ${theme.colors.surface};
            }
          `}
        `;
      
      case 'filled':
        return css`
          background-color: ${theme.colors.surface};
          border: 1px solid ${theme.colors.border};

          ${$interactive && css`
            &:hover {
              background-color: ${theme.colors.surfaceHover};
              border-color: ${theme.colors.primary};
            }

            &:active {
              background-color: ${theme.colors.surface};
            }
          `}
        `;
      
      default:
        return css`
          background-color: ${theme.colors.surface};
          border: 1px solid ${theme.colors.border};

          ${$interactive && css`
            &:hover {
              border-color: ${theme.colors.primary};
              box-shadow: 0 2px 4px ${theme.colors.shadow};
            }

            &:active {
              box-shadow: 0 1px 2px ${theme.colors.shadow};
            }
          `}
        `;
    }
  }}
`;

const CardHeader = styled.div<{ theme: Theme }>`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: ${({ theme }) => theme.spacing.lg};
  padding-bottom: ${({ theme }) => theme.spacing.base};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};

  &:last-child {
    margin-bottom: 0;
    padding-bottom: 0;
    border-bottom: none;
  }
`;

const CardTitle = styled.h3<{ theme: Theme }>`
  margin: 0;
  font-size: ${({ theme }) => theme.typography.fontSize.xl};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
`;

const CardSubtitle = styled.p<{ theme: Theme }>`
  margin: 0;
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-top: ${({ theme }) => theme.spacing.xs};
`;

const CardContent = styled.div<{ theme: Theme }>`
  flex: 1;
`;

const CardFooter = styled.div<{ theme: Theme }>`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: ${({ theme }) => theme.spacing.lg};
  padding-top: ${({ theme }) => theme.spacing.base};
  border-top: 1px solid ${({ theme }) => theme.colors.border};

  &:first-child {
    margin-top: 0;
    padding-top: 0;
    border-top: none;
  }
`;

const CardActions = styled.div<{ theme: Theme }>`
  display: flex;
  gap: ${({ theme }) => theme.spacing.sm};
  margin-top: ${({ theme }) => theme.spacing.base};
`;

interface CardComponent extends React.FC<CardProps> {
  Header: typeof CardHeader;
  Title: typeof CardTitle;
  Subtitle: typeof CardSubtitle;
  Content: typeof CardContent;
  Footer: typeof CardFooter;
  Actions: typeof CardActions;
}

export const Card: CardComponent = ({
  children,
  variant = 'default',
  padding = 'md',
  interactive = false,
  className,
  onClick,
}) => {
  return (
    <CardWrapper
      $variant={variant}
      $padding={padding}
      $interactive={interactive}
      className={className}
      onClick={onClick}
      tabIndex={interactive ? 0 : undefined}
      role={interactive ? 'button' : undefined}
    >
      {children}
    </CardWrapper>
  );
};

// Export sub-components for composition
Card.Header = CardHeader;
Card.Title = CardTitle;
Card.Subtitle = CardSubtitle;
Card.Content = CardContent;
Card.Footer = CardFooter;
Card.Actions = CardActions;