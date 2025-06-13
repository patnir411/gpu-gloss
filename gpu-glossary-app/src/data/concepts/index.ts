import type { Concept } from '../../types';
import { deviceHardwareConcepts } from './device-hardware';
import { deviceSoftwareConcepts } from './device-software';
import { hostSoftwareConcepts } from './host-software';

// Export all concepts as a single array
export const allConcepts: Concept[] = [
  ...deviceHardwareConcepts,
  ...deviceSoftwareConcepts,
  ...hostSoftwareConcepts,
];

// Export concepts by category
export const conceptsByCategory = {
  'device-hardware': deviceHardwareConcepts,
  'device-software': deviceSoftwareConcepts,
  'host-software': hostSoftwareConcepts,
};

// Create concept lookup map for quick access
export const conceptsById = allConcepts.reduce((acc, concept) => {
  acc[concept.id] = concept;
  return acc;
}, {} as Record<string, Concept>);

// Export individual concept arrays
export { deviceHardwareConcepts, deviceSoftwareConcepts, hostSoftwareConcepts };