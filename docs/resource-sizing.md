# Resource Sizing Rationale

This document explains the CPU, memory, and replica configuration for the `recsys-api` deployment.

## Cluster Context

- **Node provisioner**: Karpenter with `WhenUnderutilized` / `WhenEmptyOrUnderutilized` consolidation
- **Capacity type**: Spot-only (`c`, `m`, `r` instance families, gen 3+)
- **Architecture**: amd64 (default pool) + arm64 (spot-compute pool)

## Why Spot Instances

SR-GNN inference is **stateless** — a spot interruption only affects in-flight requests, not persistent state. The following mechanisms protect availability:

| Mechanism | Protection |
|---|---|
| `minAvailable: 1` PDB | At least 1 pod survives voluntary disruptions |
| `maxUnavailable: 0` rolling update | Zero-downtime deploys |
| `minReplicas: 2` HPA | Minimum 2 pods across availability zones |
| `topologySpreadConstraints` | Pods spread across AZs |
| `podAntiAffinity` | Pods prefer different nodes |

Combined, these ensure that a single spot interruption never causes total downtime.

## CPU: 250m request / 1 core limit

- **Request (250m)**: SR-GNN forward pass is lightweight — single session, small graph (~20 nodes). 250m lets Karpenter pack **2 pods on a `c5.large` (2 vCPU)** with room for system overhead.
- **Limit (1 core)**: Caps burst CPU to prevent starving other pods on the same node. PyTorch may briefly spike during model loading.

## Memory: 512Mi request / 2Gi limit

- **Request (512Mi)**: Covers PyTorch runtime (~200MB), model weights (~50-100MB), asyncpg connection pool (~50MB), and application overhead.
- **Limit (2Gi)**: Safety ceiling. Catches memory leaks (e.g., unbounded session caching) before the OOM killer targets the pod.

## HPA Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `minReplicas` | 2 | Availability during spot interruptions |
| `maxReplicas` | 10 | Cost cap; sufficient for expected traffic |
| CPU target | 70% | Scale up before saturation |
| Memory target | 80% | Detect leaks before OOM |

## Model Cache PVC

- **Size**: 20Gi (`ReadWriteMany` via EFS)
- **Purpose**: Shared model artifact cache across pods and init containers, avoiding redundant downloads from MLflow registry on each pod start.

## Review Triggers

Revisit these values when:

1. **Phase 5 observability** is in place — check p95/p99 latency and memory high-watermark in Grafana
2. Model architecture changes significantly (e.g., larger embedding tables)
3. Traffic patterns change (sustained high concurrency vs. bursty)
4. Instance family pricing shifts (check Spot Advisor for `c`/`m`/`r` interruption rates)

## Cost Optimization Tips

- **Tight requests → smaller nodes → cheaper spot prices**: Karpenter selects the cheapest instance that fits the pod requests. Don't inflate requests "just in case."
- **Consolidation**: `WhenUnderutilized` policy automatically terminates underused nodes and repacks pods.
- **arm64 eligibility**: The spot-compute pool allows arm64 nodes, which are typically 20% cheaper. Ensure the Docker image supports multi-arch if targeting these.
