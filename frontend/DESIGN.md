# Frontend Design: RecSys Visualizer

This document outlines the design for an educational frontend that illustrates the end-to-end workflow of Session-based Recommendation with Graph Neural Networks (SR-GNN, TAGNN, GGNN).

## 1. Objective
Provide an interactive, visual playground where users can:
- Input custom item sequences.
- Observe how sequences are transformed into session graphs.
- Visualize the message-passing and attention mechanisms of various GNN architectures.
- Understand how the final "top-k" recommendations are calculated.

---

## 2. The Visual Pipeline

The frontend is structured as a linear pipeline reflecting the mathematical flow of the models:

### Step 1: Input Session Sequence
- **Visual**: A horizontal "chip" array of item IDs.
- **Goal**: Show the raw temporal order of user interactions.
- **Interactive**: User can add/remove items to see how the graph topology changes in real-time.

### Step 2: Session-as-Graph (Graph Construction)
- **Visual**: A directed graph layout (using Force-Directed Simulation).
- **Core Concepts**:
    - **Nodes**: Unique items in the session.
    - **Edges**: Transitions between items (e.g., $v_1 \to v_2$).
    - **Adjacency Matrices**: Display the $A_{in}$ and $A_{out}$ matrices that the GNN uses for computation.
- **Educational Note**: Illustrates how SR-GNN breaks free from linear constraints by modeling multi-directional transitions.

### Step 3: GNN Message Passing (Propagation)
- **Visual**: Animation of "pulses" moving along edges.
- **Architecture Specifics**:
    - **SR-GNN**: Shows the Gated Update (GRU-like) process where node $i$ aggregates info from neighbors.
    - **GGNN**: Focuses on the gated propagation steps.
- **Goal**: Explain how "item context" is captured by looking at neighbors.

### Step 4: Attention Mechanism (Readout)
- **Visual**: A "Spotlight" or heat-map on nodes based on their importance relative to the last item ($v_n$).
- **Concept**:
    - **Last Item ($s_l$)**: Represented as the "current interest".
    - **Global Interest ($s_g$)**: A weighted sum of all nodes in the graph.
    - **Attention Weights ($\alpha$)**: Visualized as varying node brightness or connecting lines to the "Readout" node.
- **TAGNN Difference**: Illustrates the *Target Attention* where importance is recalculated based on the candidate items.

### Step 5: Prediction & Top-K
- **Visual**: A ranked list of product cards with "Similarity Scores".
- **Concept**: The dot product between the Session Embedding and all Item Embeddings in the catalog.
- **Goal**: Show the transition from a latent vector to concrete product recommendations.

---

## 3. Technical Architecture

### Stack
- **Framework**: React + TypeScript (Vite)
- **Styling**: Tailwind CSS / Custom CSS Modules for precise control over diagram aesthetics.
- **Visualization**:
    - **D3.js / React Force Graph**: For the Session Graph layout.
    - **Framer Motion**: For smooth transitions between pipeline steps.
- **State Management**: React Context or Zustand for keeping the sequence, graph, and predictions in sync.

### Backend Integration (Optional but Recommended)
While current visualizations are simulated, the "Deep Dive" mode should fetch actual intermediate states from the FastAPI backend:
- `POST /recommend/trace`: A proposed endpoint that returns not just IDs, but:
    - Adjacency matrices.
    - Node embeddings after $k$ steps.
    - Attention weights for the specific input.

---

## 4. UI/UX Principles
1. **Explain the "Why"**: Tooltips on every component explaining the mathematical step (e.g., "This node is bright because the attention mechanism marked it as highly relevant to the last click").
2. **Model Comparison**: A toggle to switch between SR-GNN, TAGNN, and GGNN to see how the "Readout" or "Propagation" logic changes visually.
3. **Responsive Flow**: A vertical "Waterfall" or horizontal "Stepper" layout that guides the user from data to prediction.

---

## 5. Implementation Roadmap
- [x] Basic React/Vite scaffolding.
- [x] Input sequence to Graph logic.
- [ ] D3.js integration for dynamic session graphs.
- [ ] Attention weight visualization components.
- [ ] Integration with `src/recsys/serving/api.py` for real-time inference tracing.
