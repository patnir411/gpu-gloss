import React, { createContext, useContext, useState, useEffect } from 'react';
import { ThemeProvider as StyledThemeProvider } from 'styled-components';
import type { ThemeType } from '../types';

// Theme definitions
const lightTheme = {
  name: 'light' as ThemeType,
  colors: {
    primary: '#76B900', // NVIDIA Green
    secondary: '#0073E6', // Blue
    accent: '#FF6B35', // Orange
    background: '#FFFFFF',
    surface: '#F8F9FA',
    surfaceHover: '#E9ECEF',
    text: {
      primary: '#212529',
      secondary: '#6C757D',
      inverse: '#FFFFFF',
    },
    border: '#DEE2E6',
    shadow: 'rgba(0, 0, 0, 0.1)',
    success: '#28A745',
    warning: '#FFC107',
    error: '#DC3545',
    info: '#17A2B8',
  },
  typography: {
    fontFamily: {
      body: '"Inter", "Segoe UI", "Roboto", sans-serif',
      heading: '"Inter", "Segoe UI", "Roboto", sans-serif',
      code: '"JetBrains Mono", "Fira Code", "Consolas", monospace',
    },
    fontSize: {
      xs: '0.75rem',   // 12px
      sm: '0.875rem',  // 14px
      base: '1rem',    // 16px
      lg: '1.125rem',  // 18px
      xl: '1.25rem',   // 20px
      '2xl': '1.5rem', // 24px
      '3xl': '1.875rem', // 30px
      '4xl': '2.25rem',  // 36px
    },
    fontWeight: {
      normal: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },
    lineHeight: {
      tight: 1.25,
      normal: 1.5,
      relaxed: 1.75,
    },
  },
  spacing: {
    xs: '0.25rem',   // 4px
    sm: '0.5rem',    // 8px
    base: '1rem',    // 16px
    lg: '1.5rem',    // 24px
    xl: '2rem',      // 32px
    '2xl': '3rem',   // 48px
    '3xl': '4rem',   // 64px
  },
  borderRadius: {
    sm: '0.25rem',
    base: '0.5rem',
    lg: '0.75rem',
    xl: '1rem',
    full: '9999px',
  },
  breakpoints: {
    sm: '640px',
    md: '768px',
    lg: '1024px',
    xl: '1280px',
    '2xl': '1536px',
  },
  zIndex: {
    dropdown: 1000,
    sticky: 1020,
    fixed: 1030,
    modal: 1040,
    popover: 1050,
    tooltip: 1060,
  },
  transitions: {
    fast: '150ms ease-in-out',
    normal: '300ms ease-in-out',
    slow: '500ms ease-in-out',
  },
};

const darkTheme = {
  ...lightTheme,
  name: 'dark' as ThemeType,
  colors: {
    ...lightTheme.colors,
    background: '#0D1117',
    surface: '#161B22',
    surfaceHover: '#21262D',
    text: {
      primary: '#F0F6FC',
      secondary: '#8B949E',
      inverse: '#0D1117',
    },
    border: '#30363D',
    shadow: 'rgba(0, 0, 0, 0.3)',
  },
};

export type Theme = typeof lightTheme;

interface ThemeContextType {
  theme: Theme;
  themeType: ThemeType;
  toggleTheme: () => void;
  setTheme: (theme: ThemeType) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

interface ThemeProviderProps {
  children: React.ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [themeType, setThemeType] = useState<ThemeType>(() => {
    // Check localStorage for saved theme preference
    const saved = localStorage.getItem('gpu-glossary-theme');
    if (saved === 'light' || saved === 'dark') {
      return saved;
    }
    
    // Check system preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    
    return 'dark'; // Default to dark theme for developer-friendly experience
  });

  const theme = themeType === 'dark' ? darkTheme : lightTheme;

  const toggleTheme = () => {
    setThemeType(prev => prev === 'dark' ? 'light' : 'dark');
  };

  const setTheme = (newTheme: ThemeType) => {
    setThemeType(newTheme);
  };

  // Save theme preference to localStorage
  useEffect(() => {
    localStorage.setItem('gpu-glossary-theme', themeType);
  }, [themeType]);

  // Listen for system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = (e: MediaQueryListEvent) => {
      // Only update if user hasn't set a preference
      const saved = localStorage.getItem('gpu-glossary-theme');
      if (!saved) {
        setThemeType(e.matches ? 'dark' : 'light');
      }
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  const contextValue: ThemeContextType = {
    theme,
    themeType,
    toggleTheme,
    setTheme,
  };

  return (
    <ThemeContext.Provider value={contextValue}>
      <StyledThemeProvider theme={theme}>
        {children}
      </StyledThemeProvider>
    </ThemeContext.Provider>
  );
};