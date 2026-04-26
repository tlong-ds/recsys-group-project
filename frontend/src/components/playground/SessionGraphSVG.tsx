/**
 * SessionGraphSVG — Renders the session graph as a pure SVG.
 *
 * Circular layout, directed edges, node labels.
 */

import type { SessionGraph } from '../../types';

const NODE_COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];

interface Props {
  graph: SessionGraph;
}

export function SessionGraphSVG({ graph }: Props) {
  const { nodes, edges } = graph;
  if (nodes.length === 0) {
    return (
      <div className="state-box">
        <div className="state-box__icon">🔗</div>
        <div className="state-box__text">Enter a sequence to see the graph</div>
      </div>
    );
  }

  const W = 320;
  const H = 280;
  const cx = W / 2;
  const cy = H / 2;
  const radius = Math.min(90, 30 + nodes.length * 15);
  const nodeR = 22;

  // Circular layout
  const positions = nodes.map((_, i) => {
    const angle = (2 * Math.PI * i) / nodes.length - Math.PI / 2;
    return { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
  });

  // Shorten edge to stop at node border
  const shortenEdge = (x1: number, y1: number, x2: number, y2: number, r: number) => {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    return {
      x1: x1 + (dx / len) * r,
      y1: y1 + (dy / len) * r,
      x2: x2 - (dx / len) * (r + 6),
      y2: y2 - (dy / len) * (r + 6),
    };
  };

  return (
    <div>
      <div className="graph-svg-container">
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H}>
          <defs>
            <marker id="graph-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
              <polygon points="0 0, 8 3, 0 6" fill="#94a3b8" />
            </marker>
          </defs>

          {/* Edges */}
          {edges.filter((e) => !e.isSelfLoop).map((e, i) => {
            const from = positions[e.srcIdx];
            const to = positions[e.dstIdx];
            const s = shortenEdge(from.x, from.y, to.x, to.y, nodeR);
            return (
              <line key={`e-${i}`}
                x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2}
                stroke="#475569" strokeWidth={1 + e.weight * 0.5}
                markerEnd="url(#graph-arrow)" opacity="0.6"
              />
            );
          })}

          {/* Nodes */}
          {nodes.map((n, i) => {
            const pos = positions[i];
            const color = NODE_COLORS[i % NODE_COLORS.length];
            return (
              <g key={i} className="graph-node">
                <circle cx={pos.x} cy={pos.y} r={nodeR}
                  fill="var(--bg-surface)" stroke={color} strokeWidth="2" />
                <text x={pos.x} y={pos.y + 1} textAnchor="middle" dominantBaseline="middle"
                  fontSize="12" fontWeight="600" fill={color}>
                  {n.itemId}
                </text>
                <text x={pos.x} y={pos.y + nodeR + 14} textAnchor="middle"
                  fontSize="9" fill="var(--text-4)">
                  v{i + 1}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="graph-stats-row">
        <span className="stat-badge">|V| = {nodes.length}</span>
        <span className="stat-badge">|E| = {edges.filter((e) => !e.isSelfLoop).length}</span>
      </div>
    </div>
  );
}
