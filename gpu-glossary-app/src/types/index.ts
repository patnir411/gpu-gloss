// Core data types for the GPU Glossary application

export interface Concept {
  id: string;
  title: string;
  category: CategoryType;
  subcategory?: string;
  definition: string;
  keyPoints: string[];
  technicalDetails?: TechnicalDetail[];
  codeExamples?: CodeExample[];
  visualReferences?: VisualReference[];
  relatedConcepts: string[]; // Array of concept IDs
  prerequisites?: string[]; // Array of concept IDs
  difficulty: DifficultyLevel;
  tags: string[];
  learningPath: LearningPathType[];
}

export interface TechnicalDetail {
  label: string;
  value: string;
  description?: string;
}

export interface CodeExample {
  language: string;
  code: string;
  description: string;
  filename?: string;
}

export interface VisualReference {
  type: 'diagram' | 'image' | 'animation';
  src: string;
  alt: string;
  caption: string;
  interactive?: boolean;
}

export interface Category {
  id: string;
  name: string;
  description: string;
  icon: string;
  color: string;
  subcategories: Subcategory[];
  conceptCount: number;
}

export interface Subcategory {
  id: string;
  name: string;
  description: string;
  conceptIds: string[];
}

export interface LearningPath {
  id: string;
  name: string;
  description: string;
  difficulty: DifficultyLevel;
  estimatedTime: string;
  conceptSequence: string[]; // Array of concept IDs in order
  prerequisites?: string[];
  outcomes: string[];
}

export interface CrossReference {
  fromConceptId: string;
  toConceptId: string;
  relationship: RelationshipType;
  description?: string;
}

export interface UserProgress {
  conceptsRead: Set<string>;
  conceptsUnderstood: Set<string>;
  currentPath?: string;
  bookmarks: Set<string>;
  notes: Record<string, string>; // conceptId -> note
  completedPaths: Set<string>;
}

// Enums and Union Types
export type CategoryType = 'device-hardware' | 'device-software' | 'host-software';

export type DifficultyLevel = 'beginner' | 'intermediate' | 'advanced' | 'expert';

export type LearningPathType = 
  | 'gpu-fundamentals' 
  | 'cuda-programming' 
  | 'gpu-architecture' 
  | 'cuda-development';

export type RelationshipType = 
  | 'prerequisite' 
  | 'implements' 
  | 'uses' 
  | 'extends' 
  | 'related' 
  | 'opposite' 
  | 'part-of';

export type ThemeType = 'dark' | 'light';

// Application State Types
export interface AppState {
  currentConcept?: string;
  currentCategory?: string;
  currentPath?: string;
  searchQuery: string;
  filters: FilterState;
  userProgress: UserProgress;
  theme: ThemeType;
  sidebarOpen: boolean;
}

export interface FilterState {
  categories: CategoryType[];
  difficulty: DifficultyLevel[];
  tags: string[];
  hasCodeExamples: boolean;
  hasVisuals: boolean;
}

// Search Types
export interface SearchResult {
  concept: Concept;
  score: number;
  matches: SearchMatch[];
}

export interface SearchMatch {
  field: string;
  value: string;
  indices: number[][];
}

// Navigation Types
export interface NavigationItem {
  id: string;
  label: string;
  path: string;
  icon?: string;
  children?: NavigationItem[];
}

// Interactive Visualization Types
export interface VisualizationData {
  nodes: VisualizationNode[];
  links: VisualizationLink[];
  metadata: Record<string, any>;
}

export interface VisualizationNode {
  id: string;
  label: string;
  type: string;
  x?: number;
  y?: number;
  properties: Record<string, any>;
}

export interface VisualizationLink {
  source: string;
  target: string;
  type: string;
  properties: Record<string, any>;
}