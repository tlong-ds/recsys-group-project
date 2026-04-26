/**
 * PipelineDiagram — Section: Interactive SVG pipeline visualization with architecture toggles.
 */

import { usePipelineStore } from '../store/pipelineStore';
import type { ModelArchitecture } from '../types';

import srgnnImg from '../assets/srgnn.png';
import tagnnImg from '../assets/tagnn.png';
import ggnnImg from '../assets/ggnn.png';

const MODELS: ModelArchitecture[] = ['SR-GNN', 'TAGNN', 'GGNN'];

const EXPLANATIONS = {
  'SR-GNN': "SR-GNN (Session-based Recommendation with Graph Neural Networks) models user sessions as directed graphs to capture complex item transitions. It uses a Graph Neural Network to learn item embeddings based on graph connectivity. Finally, an attention mechanism combines the user's global session preference with their current interest (the last clicked item) to predict the next item.",
  'TAGNN': "TAGNN (Target Attentive Graph Neural Networks) extends the graph-based approach by introducing a target-aware attention mechanism. Instead of compressing a session into a single static vector, TAGNN dynamically generates the session embedding based on the specific target item being evaluated. This allows the model to adaptively activate different facets of a user's interest depending on what is being predicted.",
  'GGNN': "GGNN (Gated Graph Neural Networks) adapt standard graph message passing by using recurrent updates. For each item node in the graph, incoming and outgoing transitions are aggregated and passed through a Gated Recurrent Unit (GRU). This iterative update process over several time steps allows the network to effectively capture complex, multi-hop dependencies within the session sequence."
};

const REFERENCES = {
  'SR-GNN': 'Shu Wu, Yuyuan Tang, et al. "Session-based Recommendation with Graph Neural Networks." AAAI 2019.',
  'TAGNN': 'Yu Zheng, Chang Liu, et al. "TAGNN: Target Attentive Graph Neural Networks for Session-based Recommendation." SIGIR 2020.',
  'GGNN': 'Yujia Li, Daniel Tarlow, et al. "Gated Graph Sequence Neural Networks." ICLR 2016.'
};

interface Props {
  selectedModel: ModelArchitecture;
  onModelChange: (m: ModelArchitecture) => void;
}

export function PipelineDiagram({ selectedModel, onModelChange }: Props) {
  const bestModel = usePipelineStore((state) => state.bestModel);

  const modelLabel = selectedModel === 'TAGNN' ? 'Target Attentive' :
                     selectedModel === 'GGNN' ? 'Gated' : 'Session-based';

  const imageSrc = selectedModel === 'TAGNN' ? tagnnImg :
                   selectedModel === 'GGNN' ? ggnnImg : srgnnImg;

  const explanation = EXPLANATIONS[selectedModel] || EXPLANATIONS['SR-GNN'];
  const reference = REFERENCES[selectedModel] || REFERENCES['SR-GNN'];

  return (
    <section id="section-pipeline" className="section">
      <div className="section__header">
        <div className="section__badge">📐 Architecture</div>
        
        {/* Header content: Uses a grid to keep title centered while allowing chips on the right */}
        <div className="pipeline-header__grid">
          {/* Left Column (Spacer) */}
          <div className="header-spacer" aria-hidden="true" style={{ minWidth: '0' }}></div>

          {/* Center Column (Title & Subtitle) */}
          <div style={{ textAlign: 'center' }}>
            <h1 className="section__title" style={{ margin: 0 }}>{selectedModel} Pipeline</h1>
            <p className="section__subtitle" style={{ margin: '8px auto 0' }}>
              {modelLabel} Graph Neural Network — forward propagation architecture.
            </p>
          </div>

          {/* Right Column (Model Selector) */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', minWidth: '0' }}>
            <div className="navbar__model-group" style={{ marginBottom: '8px' }}>
              {MODELS.map((m) => (
                <button
                  key={m}
                  className={`model-chip ${selectedModel === m ? 'model-chip--active' : ''}`}
                  onClick={() => onModelChange(m)}
                  style={{ position: 'relative' }}
                >
                  {m}
                  {bestModel === m && (
                    <span 
                      style={{ 
                        position: 'absolute', 
                        top: -8, 
                        right: -4, 
                        background: 'var(--emerald)', 
                        color: 'white', 
                        fontSize: '0.5rem', 
                        padding: '1px 4px', 
                        borderRadius: 4,
                        fontWeight: 800,
                      }}
                    >
                      BEST
                    </span>
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="pipeline-diagram" style={{ display: 'flex', flexDirection: 'row', alignItems: 'flex-start', padding: '20px', gap: '40px', flexWrap: 'wrap' }}>
        
        {/* Left Column: Image */}
        <div style={{ flex: '1 1 60%', minWidth: '400px' }}>
          <img 
            src={imageSrc} 
            alt={`${selectedModel} Architecture`} 
            style={{ width: '100%', height: 'auto', objectFit: 'contain', borderRadius: '12px', border: '1px solid var(--border)', background: '#fff', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }} 
          />
        </div>
        
        {/* Right Column: Description & References */}
        <div style={{ flex: '1 1 35%', minWidth: '300px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div style={{ background: 'var(--bg-surface)', padding: '24px', borderRadius: '8px', border: '1px solid var(--border)', textAlign: 'left' }}>
            <h3 style={{ fontSize: '1.1rem', marginBottom: '12px', color: 'var(--text-1)' }}>Architecture Overview</h3>
            <p style={{ color: 'var(--text-3)', fontSize: '0.95rem', lineHeight: '1.6' }}>
              {explanation}
            </p>
          </div>

          <div style={{ background: 'var(--bg-surface)', padding: '20px 24px', borderRadius: '8px', border: '1px dashed var(--border)', textAlign: 'left' }}>
            <h3 style={{ fontSize: '1.0rem', marginBottom: '8px', color: 'var(--text-2)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>📄</span> Academic Reference
            </h3>
            <p style={{ color: 'var(--text-3)', fontSize: '0.85rem', lineHeight: '1.5', fontStyle: 'italic' }}>
              {reference}
            </p>
          </div>
        </div>

      </div>
    </section>
  );
}
