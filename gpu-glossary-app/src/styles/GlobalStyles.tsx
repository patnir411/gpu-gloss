import { createGlobalStyle } from 'styled-components';

export const GlobalStyles = createGlobalStyle`
  /* CSS Reset and Base Styles */
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  html {
    font-size: 16px;
    scroll-behavior: smooth;
  }

  body {
    font-family: ${({ theme }) => theme.typography.fontFamily.body};
    font-size: ${({ theme }) => theme.typography.fontSize.base};
    line-height: ${({ theme }) => theme.typography.lineHeight.normal};
    color: ${({ theme }) => theme.colors.text.primary};
    background-color: ${({ theme }) => theme.colors.background};
    transition: color ${({ theme }) => theme.transitions.normal}, 
                background-color ${({ theme }) => theme.transitions.normal};
    overflow-x: hidden;
  }

  /* Typography */
  h1, h2, h3, h4, h5, h6 {
    font-family: ${({ theme }) => theme.typography.fontFamily.heading};
    font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
    line-height: ${({ theme }) => theme.typography.lineHeight.tight};
    margin-bottom: ${({ theme }) => theme.spacing.base};
    color: ${({ theme }) => theme.colors.text.primary};
  }

  h1 {
    font-size: ${({ theme }) => theme.typography.fontSize['4xl']};
    font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  }

  h2 {
    font-size: ${({ theme }) => theme.typography.fontSize['3xl']};
  }

  h3 {
    font-size: ${({ theme }) => theme.typography.fontSize['2xl']};
  }

  h4 {
    font-size: ${({ theme }) => theme.typography.fontSize.xl};
  }

  h5 {
    font-size: ${({ theme }) => theme.typography.fontSize.lg};
  }

  h6 {
    font-size: ${({ theme }) => theme.typography.fontSize.base};
  }

  p {
    margin-bottom: ${({ theme }) => theme.spacing.base};
    line-height: ${({ theme }) => theme.typography.lineHeight.relaxed};
  }

  /* Links */
  a {
    color: ${({ theme }) => theme.colors.primary};
    text-decoration: none;
    transition: color ${({ theme }) => theme.transitions.fast};

    &:hover {
      color: ${({ theme }) => theme.colors.secondary};
      text-decoration: underline;
    }

    &:focus {
      outline: 2px solid ${({ theme }) => theme.colors.primary};
      outline-offset: 2px;
      border-radius: ${({ theme }) => theme.borderRadius.sm};
    }
  }

  /* Code Styling */
  code {
    font-family: ${({ theme }) => theme.typography.fontFamily.code};
    font-size: 0.875em;
    background-color: ${({ theme }) => theme.colors.surface};
    padding: 0.125rem 0.25rem;
    border-radius: ${({ theme }) => theme.borderRadius.sm};
    border: 1px solid ${({ theme }) => theme.colors.border};
  }

  pre {
    font-family: ${({ theme }) => theme.typography.fontFamily.code};
    background-color: ${({ theme }) => theme.colors.surface};
    padding: ${({ theme }) => theme.spacing.base};
    border-radius: ${({ theme }) => theme.borderRadius.base};
    border: 1px solid ${({ theme }) => theme.colors.border};
    overflow-x: auto;
    margin-bottom: ${({ theme }) => theme.spacing.base};

    code {
      background: none;
      padding: 0;
      border: none;
      font-size: ${({ theme }) => theme.typography.fontSize.sm};
    }
  }

  /* Lists */
  ul, ol {
    margin-bottom: ${({ theme }) => theme.spacing.base};
    padding-left: ${({ theme }) => theme.spacing.lg};
  }

  li {
    margin-bottom: ${({ theme }) => theme.spacing.sm};
  }

  /* Form Elements */
  input, textarea, select, button {
    font-family: inherit;
    font-size: inherit;
  }

  button {
    cursor: pointer;
    border: none;
    background: none;
    padding: 0;
    
    &:focus {
      outline: 2px solid ${({ theme }) => theme.colors.primary};
      outline-offset: 2px;
    }

    &:disabled {
      cursor: not-allowed;
      opacity: 0.6;
    }
  }

  input, textarea, select {
    &:focus {
      outline: 2px solid ${({ theme }) => theme.colors.primary};
      outline-offset: 0;
    }
  }

  /* Scrollbar Styling */
  ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }

  ::-webkit-scrollbar-track {
    background: ${({ theme }) => theme.colors.surface};
  }

  ::-webkit-scrollbar-thumb {
    background: ${({ theme }) => theme.colors.border};
    border-radius: ${({ theme }) => theme.borderRadius.full};
    
    &:hover {
      background: ${({ theme }) => theme.colors.text.secondary};
    }
  }

  /* Selection */
  ::selection {
    background: ${({ theme }) => theme.colors.primary};
    color: ${({ theme }) => theme.colors.text.inverse};
  }

  /* Print Styles */
  @media print {
    * {
      background: white !important;
      color: black !important;
      box-shadow: none !important;
      text-shadow: none !important;
    }
    
    a, a:visited {
      text-decoration: underline;
    }
    
    pre, blockquote {
      border: 1px solid #999;
      page-break-inside: avoid;
    }
    
    thead {
      display: table-header-group;
    }
    
    tr, img {
      page-break-inside: avoid;
    }
    
    img {
      max-width: 100% !important;
    }
    
    p, h2, h3 {
      orphans: 3;
      widows: 3;
    }
    
    h2, h3 {
      page-break-after: avoid;
    }
  }

  /* Utility Classes */
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  .no-scroll {
    overflow: hidden;
  }

  /* Animation Classes */
  .fade-in {
    animation: fadeIn ${({ theme }) => theme.transitions.normal};
  }

  .slide-up {
    animation: slideUp ${({ theme }) => theme.transitions.normal};
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }

  @keyframes slideUp {
    from {
      transform: translateY(20px);
      opacity: 0;
    }
    to {
      transform: translateY(0);
      opacity: 1;
    }
  }

  /* Focus Management */
  .js-focus-visible :focus:not(.focus-visible) {
    outline: none;
  }

  /* High Contrast Mode Support */
  @media (prefers-contrast: high) {
    * {
      border-color: ButtonText !important;
    }
  }

  /* Reduced Motion Support */
  @media (prefers-reduced-motion: reduce) {
    *, 
    *::before, 
    *::after {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
      scroll-behavior: auto !important;
    }
  }
`;