/**
 * Shared types for the SR-GNN Visualizer pipeline.
 *
 * Data flows top-down:
 *   ItemSequence → SessionGraph
 */

// ── Graph types ──────────────────────────────────────────────────────────

export interface GraphNode {
  /** Original item ID (from the user sequence) */
  itemId: number;
  /** Local index in the deduplicated node array */
  localIdx: number;
  /** Visual label (e.g. "v₁(104)") */
  label: string;
  /** Positions for the force layout (mutable by the renderer) */
  x: number;
  y: number;
  /** Number of times this item appears in the original sequence */
  frequency: number;
}

export interface GraphEdge {
  /** Local index of source node */
  srcIdx: number;
  /** Local index of destination node */
  dstIdx: number;
  /** Edge weight (count of transitions between these two nodes) */
  weight: number;
  /** Whether this is a self-loop (src === dst) */
  isSelfLoop: boolean;
}

export interface SessionGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** Alias mapping: sequence position → local node index */
  aliasInputs: number[];
  /** Normalized incoming adjacency matrix (n × n) */
  adjIn: number[][];
  /** Normalized outgoing adjacency matrix (n × n) */
  adjOut: number[][];
}

// ── Pipeline state ───────────────────────────────────────────────────────

export type ModelArchitecture = 'SR-GNN' | 'TAGNN' | 'GGNN';

export type ActiveSection = 'pipeline' | 'playground' | 'dashboard';

export interface PipelineState {
  /** Selected architecture */
  selectedModel: ModelArchitecture;
  /** Raw item sequence entered by the user */
  sequence: number[];
  /** Constructed session graph */
  graph: SessionGraph | null;
  /** Recommended item IDs from API */
  apiRecommendations: number[];
  /** Whether the pipeline is computing */
  loading: boolean;
  /** API error message */
  error: string | null;
}

// ── Metrics types ────────────────────────────────────────────────────────

export interface ModelMetrics {
  profile: string;
  dataVersion: string;
  hrAtK: number;
  mrrAtK: number;
}

export interface Product {
  id: number;
  categoryId: number;
  name: string | null;
  price: number | null;
  originalPrice?: number;
  discount?: string;
  imageUrl?: string;
  isNew?: boolean;
}
