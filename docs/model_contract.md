# Model Contract

`SRGNNRecommender` is the package-level model interface used by training and
serving.

Required capabilities:

- fit on session prefix / target examples
- score candidate next items
- return top-k recommendations
- save and load a portable artifact

Current implementation detail:

- the wrapper is backed by a transition-graph baseline so the repo remains
  runnable before a full neural SR-GNN implementation is plugged in
