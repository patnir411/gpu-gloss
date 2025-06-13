import React from 'react';
import styled, { css } from 'styled-components';
import type { Theme } from '../../contexts/ThemeContext';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  fullWidth?: boolean;
}

const ButtonWrapper = styled.button<{
  $variant: ButtonProps['variant'];
  $size: ButtonProps['size'];
  $fullWidth: boolean;
  theme: Theme;
}>`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing.sm};
  font-family: ${({ theme }) => theme.typography.fontFamily.body};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  text-decoration: none;
  border-radius: ${({ theme }) => theme.borderRadius.base};
  transition: all ${({ theme }) => theme.transitions.fast};
  cursor: pointer;
  border: 1px solid transparent;
  
  &:focus {
    outline: 2px solid ${({ theme }) => theme.colors.primary};
    outline-offset: 2px;
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    pointer-events: none;
  }

  ${({ $fullWidth }) => $fullWidth && css`
    width: 100%;
  `}

  /* Size Variants */
  ${({ $size, theme }) => {
    switch ($size) {
      case 'sm':
        return css`
          padding: ${theme.spacing.sm} ${theme.spacing.base};
          font-size: ${theme.typography.fontSize.sm};
          min-height: 2rem;
        `;
      case 'lg':
        return css`
          padding: ${theme.spacing.base} ${theme.spacing.xl};
          font-size: ${theme.typography.fontSize.lg};
          min-height: 3rem;
        `;
      default:
        return css`
          padding: ${theme.spacing.sm} ${theme.spacing.lg};
          font-size: ${theme.typography.fontSize.base};
          min-height: 2.5rem;
        `;
    }
  }}

  /* Color Variants */
  ${({ $variant, theme }) => {
    switch ($variant) {
      case 'primary':
        return css`
          background-color: ${theme.colors.primary};
          color: ${theme.colors.text.inverse};
          border-color: ${theme.colors.primary};

          &:hover:not(:disabled) {
            background-color: ${theme.colors.secondary};
            border-color: ${theme.colors.secondary};
            transform: translateY(-1px);
            box-shadow: 0 4px 12px ${theme.colors.shadow};
          }

          &:active {
            transform: translateY(0);
            box-shadow: 0 2px 4px ${theme.colors.shadow};
          }
        `;
      
      case 'secondary':
        return css`
          background-color: ${theme.colors.secondary};
          color: ${theme.colors.text.inverse};
          border-color: ${theme.colors.secondary};

          &:hover:not(:disabled) {
            background-color: ${theme.colors.primary};
            border-color: ${theme.colors.primary};
            transform: translateY(-1px);
            box-shadow: 0 4px 12px ${theme.colors.shadow};
          }

          &:active {
            transform: translateY(0);
            box-shadow: 0 2px 4px ${theme.colors.shadow};
          }
        `;
      
      case 'outline':
        return css`
          background-color: transparent;
          color: ${theme.colors.primary};
          border-color: ${theme.colors.primary};

          &:hover:not(:disabled) {
            background-color: ${theme.colors.primary};
            color: ${theme.colors.text.inverse};
            transform: translateY(-1px);
            box-shadow: 0 4px 12px ${theme.colors.shadow};
          }

          &:active {
            transform: translateY(0);
            box-shadow: 0 2px 4px ${theme.colors.shadow};
          }
        `;
      
      case 'ghost':
        return css`
          background-color: transparent;
          color: ${theme.colors.text.primary};
          border-color: transparent;

          &:hover:not(:disabled) {
            background-color: ${theme.colors.surfaceHover};
            color: ${theme.colors.primary};
          }

          &:active {
            background-color: ${theme.colors.surface};
          }
        `;
      
      case 'danger':
        return css`
          background-color: ${theme.colors.error};
          color: ${theme.colors.text.inverse};
          border-color: ${theme.colors.error};

          &:hover:not(:disabled) {
            background-color: #c82333;
            border-color: #c82333;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px ${theme.colors.shadow};
          }

          &:active {
            transform: translateY(0);
            box-shadow: 0 2px 4px ${theme.colors.shadow};
          }
        `;
      
      default:
        return css`
          background-color: ${theme.colors.surface};
          color: ${theme.colors.text.primary};
          border-color: ${theme.colors.border};

          &:hover:not(:disabled) {
            background-color: ${theme.colors.surfaceHover};
            transform: translateY(-1px);
            box-shadow: 0 4px 12px ${theme.colors.shadow};
          }

          &:active {
            transform: translateY(0);
            box-shadow: 0 2px 4px ${theme.colors.shadow};
          }
        `;
    }
  }}
`;

const LoadingSpinner = styled.div<{ theme: Theme }>`
  width: 1rem;
  height: 1rem;
  border: 2px solid transparent;
  border-top: 2px solid currentColor;
  border-radius: 50%;
  animation: spin 1s linear infinite;

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
`;

const IconWrapper = styled.span`
  display: flex;
  align-items: center;
  justify-content: center;
`;

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  isLoading = false,
  leftIcon,
  rightIcon,
  fullWidth = false,
  disabled,
  ...props
}) => {
  return (
    <ButtonWrapper
      $variant={variant}
      $size={size}
      $fullWidth={fullWidth}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? (
        <LoadingSpinner />
      ) : (
        <>
          {leftIcon && <IconWrapper>{leftIcon}</IconWrapper>}
          {children}
          {rightIcon && <IconWrapper>{rightIcon}</IconWrapper>}
        </>
      )}
    </ButtonWrapper>
  );
};