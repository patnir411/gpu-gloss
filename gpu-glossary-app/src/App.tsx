import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext';
import { GlobalStyles } from './styles/GlobalStyles';
import { Layout } from './components/common/Layout';
import { Header } from './components/navigation/Header';
import { Sidebar } from './components/navigation/Sidebar';
import { HomePage } from './pages/HomePage/HomePage';
import { ConceptPage } from './pages/ConceptPage/ConceptPage';
import { CategoryPage } from './pages/CategoryPage/CategoryPage';
import { SearchPage } from './pages/SearchPage/SearchPage';
import { LearningPathPage } from './pages/LearningPathPage/LearningPathPage';
import { ProgressPage } from './pages/ProgressPage/ProgressPage';
import { BookmarksPage } from './pages/BookmarksPage/BookmarksPage';

const App: React.FC = () => {
  return (
    <ThemeProvider>
      <GlobalStyles />
      <Router>
        <Layout
          header={<Header />}
          sidebar={<Sidebar />}
          sidebarWidth="320px"
        >
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/category/:categoryId" element={<CategoryPage />} />
            <Route path="/category/:categoryId/:subcategoryId" element={<CategoryPage />} />
            <Route path="/concept/:conceptId" element={<ConceptPage />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="/path/:pathId" element={<LearningPathPage />} />
            <Route path="/progress" element={<ProgressPage />} />
            <Route path="/bookmarks" element={<BookmarksPage />} />
            <Route path="*" element={<div>404 - Page Not Found</div>} />
          </Routes>
        </Layout>
      </Router>
    </ThemeProvider>
  );
};

export default App;
