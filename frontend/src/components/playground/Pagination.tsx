import { usePipelineStore } from '../../store/pipelineStore';

export function Pagination() {
  const { currentPage, totalPages, setPage, isFetchingProducts } = usePipelineStore();

  if (totalPages <= 1) return null;

  const handlePrev = () => {
    if (currentPage > 1) setPage(currentPage - 1);
  };

  const handleNext = () => {
    if (currentPage < totalPages) setPage(currentPage + 1);
  };

  // Generate page numbers to show (e.g., 1, 2, 3 ... last)
  const getPageNumbers = () => {
    const pages = [];
    const maxVisible = 5;
    
    if (totalPages <= maxVisible) {
      for (let i = 1; i <= totalPages; i++) pages.push(i);
    } else {
      pages.push(1);
      if (currentPage > 3) pages.push('...');
      
      const start = Math.max(2, currentPage - 1);
      const end = Math.min(totalPages - 1, currentPage + 1);
      
      for (let i = start; i <= end; i++) {
        if (!pages.includes(i)) pages.push(i);
      }
      
      if (currentPage < totalPages - 2) pages.push('...');
      if (!pages.includes(totalPages)) pages.push(totalPages);
    }
    return pages;
  };

  return (
    <div className="pagination-container" style={{ 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center', 
      gap: '8px',
      marginTop: '32px',
      padding: '16px 0',
      borderTop: '1px solid var(--border)'
    }}>
      <button 
        className="navbar__link" 
        onClick={handlePrev} 
        disabled={currentPage === 1 || isFetchingProducts}
        style={{ opacity: (currentPage === 1 || isFetchingProducts) ? 0.5 : 1 }}
      >
        Prev
      </button>

      <div style={{ display: 'flex', gap: '4px' }}>
        {getPageNumbers().map((p, i) => (
          typeof p === 'number' ? (
            <button
              key={i}
              className={`navbar__link ${currentPage === p ? 'navbar__link--active' : ''}`}
              onClick={() => setPage(p)}
              disabled={isFetchingProducts}
              style={{ minWidth: '36px', justifyContent: 'center' }}
            >
              {p}
            </button>
          ) : (
            <span key={i} style={{ padding: '0 8px', color: 'var(--text-4)' }}>{p}</span>
          )
        ))}
      </div>

      <button 
        className="navbar__link" 
        onClick={handleNext} 
        disabled={currentPage === totalPages || isFetchingProducts}
        style={{ opacity: (currentPage === totalPages || isFetchingProducts) ? 0.5 : 1 }}
      >
        Next
      </button>
    </div>
  );
}
