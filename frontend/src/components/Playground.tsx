/**
 * Playground — Section 2: Live recommendation interface.
 *
 * 2×2 grid: Input + Graph + Adjacency + Results
 */

import { useMemo } from 'react';
import type { SessionGraph } from '../types';
import { usePipelineStore } from '../store/pipelineStore';
import { RecommendationList } from './playground/RecommendationList';
import { CategoryBar } from './playground/CategoryBar';
import { CatalogGrid } from './playground/CatalogGrid';
import { Pagination } from './playground/Pagination';

interface Props {
  sequence: number[];
  graph: SessionGraph | null;
  apiRecommendations: number[];
  loading: boolean;
  error: string | null;
  onReorder: (seq: number[]) => void;
}

export function Playground({
  sequence, apiRecommendations,
  loading, error, onReorder,
}: Props) {
  const { 
    products, 
    logView, 
    isFetchingProducts,
    selectedCategory 
  } = usePipelineStore();

  const categories = useMemo(() => {
    // In production, categories should come from a separate endpoint or meta
    // For now, we extract from whatever products we have loaded
    const cats = new Set(products.map(p => p.categoryId));
    return Array.from(cats);
  }, [products]);

  const handleItemClick = (itemId: number) => {
    // Only add item to sequence (no toggle/cancel)
    onReorder([...sequence, itemId]);
    logView(itemId);
  };

  return (
    <section id="section-playground" className="section">
      <div className="section__header">
        <div className="section__badge">🛍️ E-Commerce Demo</div>
        <h2 className="section__title">Product Catalog & Playground</h2>
        <p className="section__subtitle">
          Select items to build a session and get real-time recommendations.
        </p>
      </div>

      <div className="playground-grid-v2">
        {/* Left Column: Item Navigator */}
        <div className="playground-column-main">
          <div className="glass-card">
            <div className="glass-card__header">
              <div className="glass-card__icon" style={{ background: 'var(--accent-soft)' }}>📦</div>
              <div className="glass-card__title">Product Catalog</div>
            </div>
            <div className="glass-card__body">
              <CategoryBar categories={categories} />
              <div className="catalog-section" style={{ marginTop: '24px' }}>
                <div className="section-label">
                  {selectedCategory === null ? 'All Products' : `Category ${selectedCategory}`}
                </div>
                <CatalogGrid 
                  products={products}
                  onItemClick={handleItemClick}
                  selectedItems={sequence}
                  isLoading={isFetchingProducts}
                />
                <Pagination />
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Recommendations */}
        <div className="playground-column-sidebar">
          <div className="glass-card" style={{ height: '100%' }}>
            <div className="glass-card__header">
              <div className="glass-card__icon" style={{ background: 'var(--emerald-soft)' }}>🎯</div>
              <div className="glass-card__title">Recommended for You</div>
            </div>
            <div className="glass-card__body">
              <RecommendationList
                apiRecommendations={apiRecommendations}
                loading={loading}
                error={error}
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
