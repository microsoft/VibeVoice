// Quota-related constants used across the frontend
// These mirror client-side logic in Dashboard (per-request max iterations -> per-document and total caps)
export const DEFAULT_MAX_ITERATIONS_PER_REQUEST = 10
export const PER_ITERATION_MULTIPLIER = 5 // per-doc quota = max_iterations_per_request * PER_ITERATION_MULTIPLIER
export const TOTAL_MULTIPLIER = 3 // total quota = per-doc quota * TOTAL_MULTIPLIER

/**
 * Compute per-document maximum stored audio items.
 *
 * Formula: perDocMax = max_iterations_per_request * PER_ITERATION_MULTIPLIER
 * Example default: 10 (default max iterations) * 5 = 50 per-doc items
 */
export function computePerDocMax(iterations: number) {
  return iterations * PER_ITERATION_MULTIPLIER
}

/**
 * Compute total maximum stored audio items across all documents.
 *
 * Formula: totalMax = perDocMax * TOTAL_MULTIPLIER
 * Example default: perDocMax 50 * 3 = 150 total items
 */
export function computeTotalMax(iterations: number) {
  return computePerDocMax(iterations) * TOTAL_MULTIPLIER
}
