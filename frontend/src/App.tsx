/**
 * SR-GNN RecSys — Application Shell (v3)
 *
 * Three-section single-page app:
 *   1. Recommendation Playground (E-commerce Demo)
 *   2. Pipeline Architecture Diagram
 *   3. Model Comparison Dashboard
 */

import { useCallback, useEffect } from 'react';
import { usePipelineStore } from './store/pipelineStore';
import { Navbar } from './components/Navbar';
import { PipelineDiagram } from './components/PipelineDiagram';
import { Playground } from './components/Playground';
import { Dashboard } from './components/Dashboard';
import { SessionManager } from './utils/session';
import './index.css';

function App() {
  const {
    sequence,
    selectedModel,
    activeSection,
    graph,
    apiRecommendations,
    loading,
    error,
    setSequence,
    setModel,
    setActiveSection,
    runRecommendation,
    fetchBestModel,
    fetchProducts,
  } = usePipelineStore();

  useEffect(() => {
    // 1. Session management: check expiration
    if (SessionManager.isExpired()) {
      setSequence([]);
    }

    // 2. Auto-select best performing model on mount
    fetchBestModel();

    // 3. Fetch product catalog from DB
    fetchProducts();

    // 4. Periodic expiration check (every minute)
    const interval = setInterval(() => {
      if (SessionManager.isExpired()) {
        setSequence([]);
      }
    }, 60 * 1000);

    return () => clearInterval(interval);
  }, [fetchBestModel, fetchProducts, setSequence]);

  // 3. Reactive Recommendations: Auto-fetch when sequence changes
  useEffect(() => {
    if (sequence.length > 0) {
      const timer = setTimeout(() => {
        runRecommendation();
      }, 300); // 300ms debounce
      return () => clearTimeout(timer);
    }
  }, [sequence, runRecommendation]);

  const handleReorder = useCallback(
    (seq: number[]) => {
      setSequence(seq);
    },
    [setSequence],
  );

  return (
    <div className="app">
      <Navbar
        activeSection={activeSection}
        onSectionChange={setActiveSection}
      />

      <main className="app-content">
        {activeSection === 'playground' && (
          <Playground
            sequence={sequence}
            graph={graph}
            apiRecommendations={apiRecommendations}
            loading={loading}
            error={error}
            onReorder={handleReorder}
          />
        )}

        {activeSection === 'pipeline' && (
          <PipelineDiagram 
            selectedModel={selectedModel} 
            onModelChange={setModel}
          />
        )}

        {activeSection === 'dashboard' && (
          <Dashboard />
        )}
      </main>

      <footer className="app-footer">
        SR-GNN / TAGNN / GGNN · Session-Based Recommendation with Graph Neural Networks
      </footer>
    </div>
  );
}

export default App;
