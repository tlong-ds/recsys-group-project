import { useState, useEffect } from 'react';
import { fetchWithAuth, API_BASE } from '../utils/api';
import type { ModelMetrics } from '../types';

export function Dashboard() {
  const [metrics, setMetrics] = useState<ModelMetrics[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchMetrics() {
      try {
        const response = await fetchWithAuth(`${API_BASE}/evaluations`);
        if (!response.ok) {
          throw new Error('Failed to fetch evaluations');
        }
        const data = await response.json();
        setMetrics(data.metrics || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error occurred');
      } finally {
        setLoading(false);
      }
    }
    fetchMetrics();
  }, []);

  const bestHr = metrics.length > 0 ? Math.max(...metrics.map((m) => m.hrAtK)) : 0;
  const bestMrr = metrics.length > 0 ? Math.max(...metrics.map((m) => m.mrrAtK)) : 0;
  const dataVersions = [...new Set(metrics.map((m) => m.dataVersion))];

  return (
    <section id="section-dashboard" className="section">
      <div className="section__header">
        <div className="section__badge">📈 Analytics</div>
        <h2 className="section__title">Model Comparison</h2>
        <p className="section__subtitle">
          Evaluation metrics (HR@K, MRR@K) across model architectures and data versions.
        </p>
      </div>

      {loading ? (
        <div className="state-box" style={{ marginTop: '20px' }}>
          <div className="spinner"></div>
          <div className="state-box__text">Loading evaluation metrics...</div>
        </div>
      ) : error ? (
        <div className="state-box error" style={{ marginTop: '20px' }}>
          <div className="state-box__icon">⚠️</div>
          <div className="state-box__text">{error}</div>
        </div>
      ) : metrics.length === 0 ? (
        <div className="state-box" style={{ marginTop: '20px' }}>
          <div className="state-box__icon">📊</div>
          <div className="state-box__text">No evaluation metrics found. Run experiments to generate data.</div>
        </div>
      ) : (
        <div className="dashboard-grid">
          {/* Metrics Table */}
          <div className="glass-card" style={{ gridColumn: '1 / -1' }}>
            <div className="glass-card__header">
              <div className="glass-card__icon" style={{ background: 'var(--accent-soft)' }}>📋</div>
              <div className="glass-card__title">Evaluation Results</div>
            </div>
            <div className="glass-card__body">
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Data Version</th>
                    <th>HR@K</th>
                    <th>MRR@K</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.map((m, i) => (
                    <tr key={i}>
                      <td style={{ fontWeight: 600 }}>{m.profile}</td>
                      <td>
                        <span className="stat-badge">{m.dataVersion}</span>
                      </td>
                      <td>
                        <span className={`metric-value ${m.hrAtK === bestHr && bestHr > 0 ? 'metric-value--best' : ''}`}>
                          {m.hrAtK.toFixed(3)}
                        </span>
                      </td>
                      <td>
                        <span className={`metric-value ${m.mrrAtK === bestMrr && bestMrr > 0 ? 'metric-value--best' : ''}`}>
                          {m.mrrAtK.toFixed(3)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* HR@K Chart */}
          <div className="glass-card">
            <div className="glass-card__header">
              <div className="glass-card__icon" style={{ background: 'var(--emerald-soft)' }}>📊</div>
              <div className="glass-card__title">HR@K by Model (Latest Version)</div>
            </div>
            <div className="glass-card__body">
              <div className="bar-chart">
                {metrics.filter((m) => m.dataVersion === dataVersions[0]).map((m, i) => (
                  <div key={i} className="bar-row">
                    <div className="bar-row__label">{m.profile}</div>
                    <div className="bar-row__track">
                      <div
                        className="bar-row__fill"
                        style={{
                          width: `${(m.hrAtK / bestHr) * 100}%`,
                          background: m.hrAtK === bestHr && bestHr > 0
                            ? 'linear-gradient(90deg, #10b981, #34d399)'
                            : 'linear-gradient(90deg, #6366f1, #818cf8)',
                        }}
                      >
                        {m.hrAtK.toFixed(3)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* MRR@K Chart */}
          <div className="glass-card">
            <div className="glass-card__header">
              <div className="glass-card__icon" style={{ background: 'rgba(168,85,247,0.12)' }}>📊</div>
              <div className="glass-card__title">MRR@K by Model (Latest Version)</div>
            </div>
            <div className="glass-card__body">
              <div className="bar-chart">
                {metrics.filter((m) => m.dataVersion === dataVersions[0]).map((m, i) => (
                  <div key={i} className="bar-row">
                    <div className="bar-row__label">{m.profile}</div>
                    <div className="bar-row__track">
                      <div
                        className="bar-row__fill"
                        style={{
                          width: `${(m.mrrAtK / bestMrr) * 100}%`,
                          background: m.mrrAtK === bestMrr && bestMrr > 0
                            ? 'linear-gradient(90deg, #10b981, #34d399)'
                            : 'linear-gradient(90deg, #a855f7, #c084fc)',
                        }}
                      >
                        {m.mrrAtK.toFixed(3)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
