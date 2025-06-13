import React from 'react';
import styled, { keyframes } from 'styled-components';

const spin = keyframes`
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
`;

const pulse = keyframes`
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
`;

const LoadingContainer = styled.div<{ $size: 'sm' | 'md' | 'lg' }>`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing.base};
  padding: ${({ theme, $size }) => {
    switch ($size) {
      case 'sm': return theme.spacing.base;
      case 'lg': return theme.spacing['3xl'];
      default: return theme.spacing.xl;
    }
  }};
`;

const Spinner = styled.div<{ $size: 'sm' | 'md' | 'lg' }>`
  width: ${({ $size }) => {
    switch ($size) {
      case 'sm': return '20px';
      case 'lg': return '60px';
      default: return '40px';
    }
  }};
  height: ${({ $size }) => {
    switch ($size) {
      case 'sm': return '20px';
      case 'lg': return '60px';
      default: return '40px';
    }
  }};
  border: ${({ $size }) => $size === 'sm' ? '2px' : '3px'} solid ${({ theme }) => theme.colors.border};
  border-top: ${({ $size }) => $size === 'sm' ? '2px' : '3px'} solid ${({ theme }) => theme.colors.primary};
  border-radius: 50%;
  animation: ${spin} 1s linear infinite;
`;

const LoadingText = styled.div<{ $size: 'sm' | 'md' | 'lg' }>`
  color: ${({ theme }) => theme.colors.text.secondary};
  font-size: ${({ theme, $size }) => {
    switch ($size) {
      case 'sm': return theme.typography.fontSize.sm;
      case 'lg': return theme.typography.fontSize.lg;
      default: return theme.typography.fontSize.base;
    }
  }};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  animation: ${pulse} 2s ease-in-out infinite;
`;

const DotsContainer = styled.div`
  display: flex;
  gap: 4px;
`;

const Dot = styled.div<{ $delay: number }>`
  width: 8px;
  height: 8px;
  background-color: ${({ theme }) => theme.colors.primary};
  border-radius: 50%;
  animation: ${pulse} 1.4s ease-in-out infinite;
  animation-delay: ${({ $delay }) => $delay}s;
`;

const SkeletonContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.base};
  width: 100%;
  max-width: 400px;
`;

const SkeletonLine = styled.div<{ $width?: string }>`
  height: 16px;
  background: linear-gradient(
    90deg,
    ${({ theme }) => theme.colors.surface} 25%,
    ${({ theme }) => theme.colors.surfaceHover} 50%,
    ${({ theme }) => theme.colors.surface} 75%
  );
  background-size: 200% 100%;
  animation: ${keyframes`
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  `} 2s ease-in-out infinite;
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  width: ${({ $width }) => $width || '100%'};
`;

interface LoadingProps {
  size?: 'sm' | 'md' | 'lg';
  text?: string;
  variant?: 'spinner' | 'dots' | 'skeleton';
  className?: string;
}

export const Loading: React.FC<LoadingProps> = ({
  size = 'md',
  text,
  variant = 'spinner',
  className
}) => {
  if (variant === 'dots') {
    return (
      <LoadingContainer $size={size} className={className}>
        <DotsContainer>
          <Dot $delay={0} />
          <Dot $delay={0.2} />
          <Dot $delay={0.4} />
        </DotsContainer>
        {text && <LoadingText $size={size}>{text}</LoadingText>}
      </LoadingContainer>
    );
  }

  if (variant === 'skeleton') {
    return (
      <LoadingContainer $size={size} className={className}>
        <SkeletonContainer>
          <SkeletonLine $width="80%" />
          <SkeletonLine $width="60%" />
          <SkeletonLine $width="90%" />
          <SkeletonLine $width="40%" />
        </SkeletonContainer>
      </LoadingContainer>
    );
  }

  return (
    <LoadingContainer $size={size} className={className}>
      <Spinner $size={size} />
      {text && <LoadingText $size={size}>{text}</LoadingText>}
    </LoadingContainer>
  );
};