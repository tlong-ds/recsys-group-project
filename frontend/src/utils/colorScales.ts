/**
 * Color-scale helpers for embedding & attention visualization.
 *
 * Uses d3-scale to create colour maps that mirror the paper's
 * warm-gradient palettes (yellow → orange → pink → red).
 */

import { scaleSequential, scaleLinear } from 'd3-scale';
import { interpolateRgb } from 'd3-interpolate';

/* ── Embedding palettes (per-node, matching the paper) ─────────────── */

/** Paper uses a distinct warm gradient per node */
export const NODE_PALETTES: [string, string][] = [
  ['#facc15', '#fef9c3'], // v1 – gold → pale yellow
  ['#ef4444', '#fecaca'], // v2 – red → light red
  ['#f97316', '#fed7aa'], // v3 – orange → light peach
  ['#ec4899', '#fbcfe8'], // v4 – pink → light pink
  ['#8b5cf6', '#ddd6fe'], // v5 – violet → light violet
  ['#06b6d4', '#cffafe'], // v6 – cyan → light cyan
  ['#10b981', '#d1fae5'], // v7 – emerald → light emerald
];

/** Get a sequential colour scale for a given node index (0-based) */
export function embeddingScale(nodeIdx: number, dimCount: number = 6) {
  const [a, b] = NODE_PALETTES[nodeIdx % NODE_PALETTES.length];
  return scaleSequential(interpolateRgb(a, b)).domain([0, dimCount - 1]);
}

/* ── Attention colour mapping ──────────────────────────────────────── */

/** Map α ∈ [0,1] → opacity / saturation */
export function attentionOpacity(alpha: number, max: number = 1): number {
  return 0.2 + 0.8 * (alpha / (max || 1));
}

/** Blue-ish scale for s_g vector */
export const sgScale = scaleSequential(
  interpolateRgb('#818cf8', '#c7d2fe'),
).domain([0, 5]);

/** Red-ish scale for s_l vector */
export const slScale = scaleSequential(
  interpolateRgb('#ef4444', '#fecaca'),
).domain([0, 5]);

/** Purple scale for s_h vector */
export const shScale = scaleSequential(
  interpolateRgb('#a855f7', '#e9d5ff'),
).domain([0, 5]);

/* ── Prediction-score colour ───────────────────────────────────────── */

export const predScoreColor = scaleLinear<string>()
  .domain([0, 0.5, 1])
  .range(['#64748b', '#f59e0b', '#10b981']);

/* ── Node accent (red circle for session items, grey for others) ──── */
export const NODE_ACCENT = '#ef4444';
export const NODE_NEIGHBOR = '#64748b';
