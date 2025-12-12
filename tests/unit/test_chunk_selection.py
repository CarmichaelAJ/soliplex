import pytest
from haiku.rag.store.models.chunk import Chunk

from soliplex.chunk_selection import (
    NeighborAwareSelectionConfig,
    apply_neighbor_aware_selection,
)


class _StubEmbedder:
    def __init__(self, vectors):
        self._vectors = vectors

    async def embed(self, texts):
        return self._vectors[: len(texts)]


def _chunk(content: str, order: int) -> Chunk:
    return Chunk(
        content=content,
        metadata={},
        document_id=f"doc-{order}",
        order=order,
    )


def test_neighbor_config_from_mapping_aliases():
    cfg = NeighborAwareSelectionConfig.from_mapping(
        {
            "method": "neighbor_aware_utility_selection",
            "parameters": {
                "chunk_budget": 5,
                "lambda": 0.7,
                "initial_pool_factor": 4.5,
            },
        }
    )
    assert cfg.chunk_budget == 5
    assert cfg.redundancy_penalty == 0.7
    assert cfg.initial_pool_multiplier == 4.5


@pytest.mark.anyio
async def test_apply_neighbor_aware_selection_basic():
    cfg = NeighborAwareSelectionConfig(
        chunk_budget=2,
        gamma=0.0,
        redundancy_penalty=0.0,
        noise_penalty_c=0.0,
        a=1.0,
        b=0.0,
        gain_function="log",
    )
    hits = [
        (_chunk("alpha", 0), 0.9),
        (_chunk("beta", 1), 0.7),
        (_chunk("gamma", 2), 0.2),
    ]
    embedder = _StubEmbedder(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    selected = await apply_neighbor_aware_selection(
        hits=hits,
        embedder=embedder,
        config=cfg,
        fallback_limit=2,
    )

    assert len(selected) == 2
    assert [chunk.content for chunk, _ in selected] == ["alpha", "beta"]
