/**
 * Client-side session graph construction.
 *
 * Mirrors the backend `_graph_from_sequence()` logic in
 * `src/recsys/models/srgnn.py` so the visualizer shows
 * exactly the same graph the model would process.
 */

import type { GraphNode, GraphEdge, SessionGraph } from '../types';

/**
 * Build a directed session graph from an item sequence.
 *
 * Example:
 *   sequence = [104, 209, 315, 209, 452]
 *   → nodes: 104, 209, 315, 452  (deduplicated)
 *   → edges: 104→209, 209→315, 315→209, 209→452
 */
export function buildSessionGraph(sequence: number[]): SessionGraph {
  if (sequence.length === 0) {
    return { nodes: [], edges: [], aliasInputs: [], adjIn: [], adjOut: [] };
  }

  // ── Step 1: Build unique nodes + alias mapping ──────────────────────
  const nodeMap = new Map<number, number>(); // itemId → localIdx
  const nodes: GraphNode[] = [];
  const aliasInputs: number[] = [];
  const freq = new Map<number, number>();

  for (const itemId of sequence) {
    freq.set(itemId, (freq.get(itemId) || 0) + 1);

    if (!nodeMap.has(itemId)) {
      const idx = nodes.length;
      nodeMap.set(itemId, idx);
      nodes.push({
        itemId,
        localIdx: idx,
        label: `v${idx + 1}(${itemId})`,
        x: 0,
        y: 0,
        frequency: 0,
      });
    }
    aliasInputs.push(nodeMap.get(itemId)!);
  }

  // Set frequencies
  for (const node of nodes) {
    node.frequency = freq.get(node.itemId) || 1;
  }

  // ── Step 2: Build edges from consecutive transitions ────────────────
  const edgeCountMap = new Map<string, number>();
  const edgeSrcDst: [number, number][] = [];

  for (let i = 0; i < aliasInputs.length - 1; i++) {
    const src = aliasInputs[i];
    const dst = aliasInputs[i + 1];
    const key = `${src},${dst}`;
    edgeCountMap.set(key, (edgeCountMap.get(key) || 0) + 1);
  }

  const edges: GraphEdge[] = [];
  for (const [key, weight] of edgeCountMap) {
    const [src, dst] = key.split(',').map(Number);
    edges.push({
      srcIdx: src,
      dstIdx: dst,
      weight,
      isSelfLoop: src === dst,
    });
    edgeSrcDst.push([src, dst]);
  }

  // ── Step 3: Build normalized adjacency matrices ─────────────────────
  const n = nodes.length;
  const adjIn  = Array.from({ length: n }, () => Array(n).fill(0) as number[]);
  const adjOut = Array.from({ length: n }, () => Array(n).fill(0) as number[]);

  for (const [src, dst] of edgeSrcDst) {
    const w = edgeCountMap.get(`${src},${dst}`) || 1;
    adjOut[src][dst] += w;
    adjIn[dst][src]  += w;
  }

  // Row-normalize
  for (let i = 0; i < n; i++) {
    const outSum = adjOut[i].reduce((a, b) => a + b, 0);
    const inSum  = adjIn[i].reduce((a, b) => a + b, 0);
    if (outSum > 0) adjOut[i] = adjOut[i].map(v => v / outSum);
    if (inSum > 0)  adjIn[i]  = adjIn[i].map(v => v / inSum);
  }

  // ── Step 4: Compute initial positions (circular layout) ─────────────
  const cx = 300, cy = 200, radius = Math.min(120, 40 + n * 20);
  for (let i = 0; i < n; i++) {
    const angle = (2 * Math.PI * i) / n - Math.PI / 2;
    nodes[i].x = cx + radius * Math.cos(angle);
    nodes[i].y = cy + radius * Math.sin(angle);
  }

  return { nodes, edges, aliasInputs, adjIn, adjOut };
}
