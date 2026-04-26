import { motion, AnimatePresence } from 'framer-motion';
import type { Product } from '../../types';

interface Props {
  products: Product[];
  onItemClick: (itemId: number) => void;
  selectedItems: number[];
  isLoading?: boolean;
}

export function CatalogGrid({ products, onItemClick, selectedItems, isLoading }: Props) {
  if (isLoading && products.length === 0) {
    return (
      <div className="state-box">
        <div className="spinner" />
        <div className="state-box__text">Loading catalog...</div>
      </div>
    );
  }

  if (products.length === 0) {
    return (
      <div className="state-box">
        <div className="state-box__icon">📭</div>
        <div className="state-box__text">No products found in this category.</div>
      </div>
    );
  }

  return (
    <div className="catalog-grid">
      <AnimatePresence mode="popLayout">
        {products.map((product) => {
          const isSelected = selectedItems.includes(product.id);
          return (
            <motion.div
              key={product.id}
              layout
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              whileHover={{ y: -4 }}
              className={`catalog-card ${isSelected ? 'catalog-card--selected' : ''}`}
              onClick={() => onItemClick(product.id)}
            >
              <div className="catalog-card__icon">📦</div>
              <div className="catalog-card__info">
                <span className="catalog-card__id">Product {product.id}</span>
                <div className="catalog-card__meta">
                  {product.price !== undefined && product.price !== null && (
                    <span className="catalog-card__price">Price: ${product.price.toFixed(2)}</span>
                  )}
                </div>
              </div>
              {isSelected && <div className="catalog-card__check">✓</div>}
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}

