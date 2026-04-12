# Problem Definition

Build a session-based recommender for anonymous or weakly identified traffic.
The product task is next-item prediction: given the ordered events inside the
current session, rank the next likely item.

Primary offline objectives:

- improve `HR@K`
- improve `MRR@K`
- keep `NDCG@K` stable across evaluation windows

Non-goals for this scaffold:

- feature-complete online feature store
- distributed training
- real-time drift automation
