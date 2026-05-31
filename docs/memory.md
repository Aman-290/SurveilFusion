# Incident Memory

SurveilFusion should remember what happened before, not just alert in the moment.

## Implemented Now

- Local incident summary by kind, camera, severity, and acknowledgement state.
- Local search over incident titles, summaries, camera ids, kinds, detections, and metadata.
- Ranking combines term matches, severity, and recency.
- Similar-event lookup for a specific incident.
- Dependency-free fallback that works immediately after clone.

## API

- `GET /api/memory/summary`
- `GET /api/memory/search?q=fire%20front%20door`
- `GET /api/events/{id}/similar`

## Future Qdrant Path

The Docker stack already includes Qdrant so the retrieval layer can grow into true vector memory:

1. Generate embeddings from event title, summary, camera id, labels, and transcript snippets.
2. Store embeddings in Qdrant with event id, camera id, kind, severity, and timestamps.
3. Search with hybrid retrieval: keyword filters plus vector similarity.
4. Feed top matches into the incident agent for context-aware recommendations.

Cloud embedding providers should remain opt-in. Local embedding models should be the default path for private deployments.
