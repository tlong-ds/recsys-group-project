/**
 * AdjacencyHeatmap — Renders A_in and A_out as interactive heatmaps.
 */

import React from 'react';
import type { SessionGraph } from '../../types';

interface Props {
  graph: SessionGraph;
}

function Matrix({ title, data, nodes }: { title: string; data: number[][]; nodes: { itemId: number }[] }) {
  const n = data.length;
  if (n === 0) return null;

  const cellSize = Math.min(36, Math.floor(220 / n));

  return (
    <div className="adj-matrix">
      <div className="adj-matrix__title">{title}</div>
      <div
        className="adj-matrix__grid"
        style={{
          gridTemplateColumns: `28px repeat(${n}, ${cellSize}px)`,
          gridTemplateRows: `20px repeat(${n}, ${cellSize}px)`,
        }}
      >
        {/* Header corner */}
        <div />
        {/* Column labels */}
        {nodes.slice(0, n).map((nd, j) => (
          <div key={`col-${j}`} className="adj-matrix__label">{nd.itemId}</div>
        ))}

        {/* Rows */}
        {data.map((row, i) => (
          <React.Fragment key={`row-group-${i}`}>
            <div className="adj-matrix__label">{nodes[i]?.itemId}</div>
            {row.map((val, j) => {
              const intensity = Math.min(val, 1);
              const bg = intensity > 0
                ? `rgba(99, 102, 241, ${0.15 + intensity * 0.75})`
                : 'rgba(148, 163, 184, 0.05)';
              return (
                <div
                  key={`${i}-${j}`}
                  className="adj-matrix__cell"
                  style={{ background: bg }}
                  title={`[${nodes[i]?.itemId}, ${nodes[j]?.itemId}] = ${val.toFixed(2)}`}
                />
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

export function AdjacencyHeatmap({ graph }: Props) {
  if (graph.nodes.length === 0) {
    return (
      <div className="state-box">
        <div className="state-box__icon">📊</div>
        <div className="state-box__text">Adjacency matrices appear after building the graph</div>
      </div>
    );
  }

  return (
    <div className="adj-container">
      <Matrix title="A_in (incoming)" data={graph.adjIn} nodes={graph.nodes} />
      <Matrix title="A_out (outgoing)" data={graph.adjOut} nodes={graph.nodes} />
    </div>
  );
}
