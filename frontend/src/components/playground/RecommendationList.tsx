/**
 * RecommendationList — Displays prediction results as item cards using raw IDs.
 *
 * Shows real API recommendations.
 */

import { motion, AnimatePresence } from 'framer-motion';
import { usePipelineStore } from '../../store/pipelineStore';

interface Props {
  apiRecommendations: number[];
  loading: boolean;
  error: string | null;
}

export function RecommendationList({ apiRecommendations, loading, error }: Props) {
  const recommendedProductsMetadata = usePipelineStore((state) => state.recommendedProductsMetadata);

  const getProductById = (id: number) => recommendedProductsMetadata.find(p => p.id === id);

  if (loading) {
    return (
      <div className="state-box">
        <div className="spinner" />
        <div className="state-box__text">Predicting next items…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="state-box">
        <div className="state-box__icon">⚠️</div>
        <div className="state-box__text" style={{ color: 'var(--rose)' }}>{error}</div>
      </div>
    );
  }

  const hasApi = apiRecommendations.length > 0;
  
  if (!hasApi) {
    return (
      <div className="state-box">
        <div className="state-box__icon">🛍️</div>
        <div className="state-box__text">Build a session to see predictions</div>
      </div>
    );
  }

  return (
    <div className="recommendations-container">
      <div style={{ fontSize: '0.68rem', color: 'var(--emerald)', marginBottom: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        ✓ Predicted Items
      </div>
      <div className="recommendation-stack" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <AnimatePresence mode="popLayout">
          {apiRecommendations.map((id, i) => {
            const product = getProductById(id);
            return (
              <motion.div
                key={`${id}-${i}`}
                layout
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{
                  layout: { type: 'spring', stiffness: 300, damping: 25 },
                  opacity: { duration: 0.2, delay: i * 0.04 },
                }}
                className="item-card"
                style={{ 
                  display: 'flex',
                  flexDirection: 'row', 
                  justifyContent: 'flex-start', 
                  padding: '12px 16px', 
                  gap: '16px', 
                  textAlign: 'left',
                  background: 'var(--bg-surface)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)',
                  alignItems: 'center',
                  boxShadow: 'var(--shadow-sm)'
                }}
              >
                <div className="product-card__rank-badge" style={{ position: 'static', flexShrink: 0 }}>{i + 1}</div>
                <div className="catalog-card__icon" style={{ fontSize: '1.5rem' }}>📦</div>
                <div className="catalog-card__info" style={{ alignItems: 'flex-start', textAlign: 'left' }}>
                  <span className="catalog-card__id" style={{ fontSize: '0.85rem', fontWeight: 700 }}>Product {id}</span>
                  {product && (
                    <span className="catalog-card__category" style={{ fontSize: '0.65rem', color: 'var(--text-4)', textTransform: 'uppercase', fontWeight: 600 }}>
                      Category {product.categoryId}
                    </span>
                  )}
                  <div className="catalog-card__meta">
                    {product?.price !== undefined && product?.price !== null && (
                      <span className="catalog-card__price" style={{ fontSize: '0.7rem', color: 'var(--text-3)', fontWeight: 600 }}>
                        Price: ${product.price.toFixed(2)}
                      </span>
                    )}
                  </div>
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
}
