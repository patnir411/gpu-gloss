import React from 'react';
import styled from 'styled-components';
import type { Theme } from '../../contexts/ThemeContext';

interface LayoutProps {
  children: React.ReactNode;
  sidebar?: React.ReactNode;
  header?: React.ReactNode;
  footer?: React.ReactNode;
  sidebarWidth?: string;
  className?: string;
}

const LayoutContainer = styled.div<{ theme: Theme }>`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: ${({ theme }) => theme.colors.background};
`;

const Header = styled.header<{ theme: Theme }>`
  position: sticky;
  top: 0;
  z-index: ${({ theme }) => theme.zIndex.sticky};
  background-color: ${({ theme }) => theme.colors.surface};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
  backdrop-filter: blur(8px);
  background-color: ${({ theme }) => 
    theme.name === 'dark' 
      ? 'rgba(22, 27, 34, 0.8)' 
      : 'rgba(248, 249, 250, 0.8)'
  };
`;

const MainContainer = styled.div<{ theme: Theme }>`
  flex: 1;
  display: flex;
  min-height: 0; /* Important for flex children to shrink */
`;

const Sidebar = styled.aside<{ 
  $width: string;
  theme: Theme;
}>`
  width: ${({ $width }) => $width};
  min-width: ${({ $width }) => $width};
  background-color: ${({ theme }) => theme.colors.surface};
  border-right: 1px solid ${({ theme }) => theme.colors.border};
  overflow-y: auto;
  position: sticky;
  top: 0;
  height: 100vh;
  
  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    position: fixed;
    top: 0;
    left: 0;
    height: 100vh;
    z-index: ${({ theme }) => theme.zIndex.modal};
    transform: translateX(-100%);
    transition: transform ${({ theme }) => theme.transitions.normal};
    
    &.sidebar-open {
      transform: translateX(0);
    }
  }
`;

const SidebarOverlay = styled.div<{ theme: Theme }>`
  display: none;
  
  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    display: block;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    z-index: ${({ theme }) => theme.zIndex.modal - 1};
    opacity: 0;
    visibility: hidden;
    transition: opacity ${({ theme }) => theme.transitions.normal},
                visibility ${({ theme }) => theme.transitions.normal};
    
    &.overlay-open {
      opacity: 1;
      visibility: visible;
    }
  }
`;

const ContentArea = styled.main<{ 
  $hasSidebar: boolean;
  theme: Theme;
}>`
  flex: 1;
  min-width: 0; /* Prevent flex item from overflowing */
  overflow-x: auto;
  
  ${({ $hasSidebar, theme }) => $hasSidebar && `
    @media (max-width: ${theme.breakpoints.lg}) {
      width: 100%;
    }
  `}
`;

const Footer = styled.footer<{ theme: Theme }>`
  background-color: ${({ theme }) => theme.colors.surface};
  border-top: 1px solid ${({ theme }) => theme.colors.border};
  margin-top: auto;
`;

const Container = styled.div<{ theme: Theme }>`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 ${({ theme }) => theme.spacing.base};
  
  @media (min-width: ${({ theme }) => theme.breakpoints.sm}) {
    padding: 0 ${({ theme }) => theme.spacing.lg};
  }
  
  @media (min-width: ${({ theme }) => theme.breakpoints.lg}) {
    padding: 0 ${({ theme }) => theme.spacing.xl};
  }
`;

export const Layout: React.FC<LayoutProps> = ({
  children,
  sidebar,
  header,
  footer,
  sidebarWidth = '280px',
  className,
}) => {
  const [sidebarOpen, setSidebarOpen] = React.useState(false);

  const closeSidebar = () => {
    setSidebarOpen(false);
  };

  // Close sidebar on escape key
  React.useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        closeSidebar();
      }
    };

    if (sidebarOpen) {
      document.addEventListener('keydown', handleEscape);
      return () => document.removeEventListener('keydown', handleEscape);
    }
  }, [sidebarOpen]);

  // Prevent body scroll when sidebar is open on mobile
  React.useEffect(() => {
    if (sidebarOpen) {
      document.body.classList.add('no-scroll');
    } else {
      document.body.classList.remove('no-scroll');
    }

    return () => {
      document.body.classList.remove('no-scroll');
    };
  }, [sidebarOpen]);

  return (
    <LayoutContainer className={className}>
      {header && (
        <Header>
          <Container>{header}</Container>
        </Header>
      )}
      
      <MainContainer>
        {sidebar && (
          <>
            <SidebarOverlay 
              className={sidebarOpen ? 'overlay-open' : ''} 
              onClick={closeSidebar}
            />
            <Sidebar 
              $width={sidebarWidth}
              className={sidebarOpen ? 'sidebar-open' : ''}
            >
              {sidebar}
            </Sidebar>
          </>
        )}
        
        <ContentArea $hasSidebar={!!sidebar}>
          {children}
        </ContentArea>
      </MainContainer>
      
      {footer && (
        <Footer>
          <Container>{footer}</Container>
        </Footer>
      )}
    </LayoutContainer>
  );
};

// Export utility components
export const PageContainer: React.FC<{ children: React.ReactNode; className?: string }> = ({ 
  children, 
  className 
}) => (
  <Container className={className}>
    {children}
  </Container>
);

export const PageHeader = styled.div<{ theme: Theme }>`
  padding: ${({ theme }) => theme.spacing['2xl']} 0;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
  margin-bottom: ${({ theme }) => theme.spacing['2xl']};
`;

export const PageTitle = styled.h1<{ theme: Theme }>`
  margin: 0 0 ${({ theme }) => theme.spacing.base} 0;
  font-size: ${({ theme }) => theme.typography.fontSize['4xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.text.primary};
`;

export const PageDescription = styled.p<{ theme: Theme }>`
  margin: 0;
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  color: ${({ theme }) => theme.colors.text.secondary};
  line-height: ${({ theme }) => theme.typography.lineHeight.relaxed};
`;

// Context for layout state
const LayoutContext = React.createContext<{
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  closeSidebar: () => void;
}>({
  sidebarOpen: false,
  toggleSidebar: () => {},
  closeSidebar: () => {},
});

export const useLayout = () => {
  return React.useContext(LayoutContext);
};