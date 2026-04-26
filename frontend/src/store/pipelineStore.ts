/**
 * Zustand store for the SR-GNN Visualizer.
 *
 * Derives graph from session sequence client-side,
 * AND fetches real recommendations from the API.
 */

import { create } from 'zustand';
import { buildSessionGraph } from '../utils/graphBuilder';
import { fetchRecommendations, fetchWithAuth, fetchProducts as fetchProductsApi, API_BASE } from '../utils/api';
import { SessionManager } from '../utils/session';
import type {
  SessionGraph,
  ModelArchitecture,
  ActiveSection,
  Product,
} from '../types';

export interface PipelineStore {
  /* ── Source state ─────────────────────────────────────────────── */
  sequence: number[];
  selectedModel: ModelArchitecture;
  bestModel: ModelArchitecture | null;
  activeSection: ActiveSection;
  products: Product[];
  currentPage: number;
  totalPages: number;
  selectedCategory: number | null;
  isFetchingProducts: boolean;

  /* ── Derived state ───────────────────────────────────────────── */
  graph: SessionGraph | null;
/* ── API state ───────────────────────────────────────────────── */
apiRecommendations: number[];
recommendedProductsMetadata: Product[];
loading: boolean;
error: string | null;

/* ── Actions ─────────────────────────────────────────────────── */
setSequence: (seq: number[]) => void;
setModel: (m: ModelArchitecture) => void;
setActiveSection: (s: ActiveSection) => void;
setSelectedCategory: (catId: number | null) => void;
setPage: (page: number) => Promise<void>;
runRecommendation: () => Promise<void>;
fetchBestModel: () => Promise<void>;
fetchProducts: (page?: number) => Promise<void>;
logView: (itemId: number) => Promise<void>;
}

function deriveAll(sequence: number[]) {
if (sequence.length === 0) {
  return { graph: null };
}
const graph = buildSessionGraph(sequence);
return { graph };
}

export const usePipelineStore = create<PipelineStore>((set, get) => ({
sequence: [],
selectedModel: 'SR-GNN',
bestModel: null,
activeSection: 'playground',
products: [],
currentPage: 1,
totalPages: 1,
selectedCategory: null,
isFetchingProducts: false,
graph: null,
apiRecommendations: [],
recommendedProductsMetadata: [],
loading: false,
error: null,
  setSequence: (seq) => {
    const derived = deriveAll(seq);
    SessionManager.updateInteraction();
    set({ sequence: seq, ...derived, apiRecommendations: [], recommendedProductsMetadata: [], error: null });
  },

setModel: (m) => {
  SessionManager.updateInteraction();
  set({ selectedModel: m });
},

setActiveSection: (s) => set({ activeSection: s }),

setSelectedCategory: (catId) => {
  set({ selectedCategory: catId, products: [], currentPage: 1, totalPages: 1 });
  get().fetchProducts(1);
},

setPage: async (page: number) => {
  set({ currentPage: page });
  await get().fetchProducts(page);
},

runRecommendation: async () => {  const { sequence } = get();
  if (sequence.length === 0) return;

  set({ loading: true, error: null });
  try {
    const sessionId = SessionManager.getOrCreateSessionId();
    const res = await fetchRecommendations(sequence, 10, sessionId);
    set({ 
      apiRecommendations: res.recommendations, 
      recommendedProductsMetadata: (res.recommended_products || []) as Product[],
      loading: false 
    });
  } catch (err) {    const msg = err instanceof Error ? err.message : 'Failed to fetch recommendations';
    set({ error: msg, loading: false });
  }
},

  fetchBestModel: async () => {
    try {
      const response = await fetchWithAuth(`${API_BASE}/evaluations`);
      if (!response.ok) return;

      const data = await response.json();
      const metrics = data.metrics || [];
      if (metrics.length === 0) return;

      // Find best model by HR@K
      const best = metrics.reduce((prev: any, current: any) => 
        (prev.hrAtK > current.hrAtK) ? prev : current
      );

      const profile = best.profile.toUpperCase();
      let architecture: ModelArchitecture = 'SR-GNN';

      if (profile.includes('SR-GNN') || profile.includes('SRGNN')) architecture = 'SR-GNN';
      else if (profile.includes('TAGNN')) architecture = 'TAGNN';
      else if (profile.includes('GGNN')) architecture = 'GGNN';

      set({ selectedModel: architecture, bestModel: architecture });
    } catch (err) {
      console.error('Failed to auto-select best model:', err);
    }
  },

  fetchProducts: async (page) => {
    const { currentPage, isFetchingProducts, selectedCategory } = get();
    if (isFetchingProducts) return;

    const targetPage = page ?? currentPage;
    set({ isFetchingProducts: true });
    try {
      const data = await fetchProductsApi(targetPage, 20, selectedCategory ?? undefined);

      const newItems = Array.isArray(data?.items) ? (data.items as Product[]) : [];

      set({ 
        products: newItems,
        currentPage: data.current_page,
        totalPages: data.total_pages,
        isFetchingProducts: false
      });
    } catch (err) {
      console.error('Failed to fetch products:', err);
      set({ isFetchingProducts: false });
    }
  },

  logView: async (itemId: number) => {
    try {
      const sessionId = SessionManager.getOrCreateSessionId();
      await fetchWithAuth(`${API_BASE}/views`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          sessionId: sessionId, 
          itemId: itemId 
        }),
      });
    } catch (err) {
      console.error('Failed to log view:', err);
    }
  },
}));
