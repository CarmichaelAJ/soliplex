from __future__ import annotations

import dataclasses
import math
from typing import List, Sequence, Tuple, cast

from haiku.rag.embeddings.base import EmbedderBase
from haiku.rag.store.models.chunk import Chunk


def _sigmoid(value: float) -> float:
    """Stable sigmoid to convert scores into pseudo-probabilities."""
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def _logistic_with_neighbors(
    scores: Sequence[float],
    cfg: NeighborAwareSelectionConfig,
) -> List[float]:
    probabilities: List[float] = []
    for idx, score in enumerate(scores):
        left = scores[idx - 1] if idx > 0 else 0.0
        right = scores[idx + 1] if idx + 1 < len(scores) else 0.0
        adjusted = cfg.a * score + cfg.b + cfg.gamma * (left + right)
        probabilities.append(_sigmoid(adjusted))
    return probabilities


def _vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in values))


def _cosine_similarity(
    a: Sequence[float],
    b: Sequence[float],
    norm_a: float,
    norm_b: float,
) -> float:
    denom = norm_a * norm_b
    if denom <= 0:
        return 0.0
    dot_product = sum(x * y for x, y in zip(a, b))
    similarity = dot_product / denom
    return max(-1.0, min(1.0, similarity))


def _gain(probability: float, mode: str) -> float:
    clamped = max(probability, 1e-12)
    if mode == "sqrt":
        return math.sqrt(clamped)
    return math.log1p(clamped)


@dataclasses.dataclass(slots=True)
class NeighborAwareSelectionConfig:
    """Configuration for the neighbor-aware utility optimizer."""

    method: str = "neighbor_aware_utility_selection"
    chunk_budget: int | None = None
    gamma: float = 0.4
    redundancy_penalty: float = 0.25
    noise_penalty_c: float = 0.01
    a: float = 4.0
    b: float = -2.0
    gain_function: str = "log"
    initial_pool_multiplier: float = 3.0

    def __post_init__(self) -> None:
        if self.method != "neighbor_aware_utility_selection":
            raise ValueError(f"Unsupported method '{self.method}'.")
        if self.chunk_budget is not None and self.chunk_budget <= 0:
            raise ValueError("chunk_budget must be a positive integer.")
        if self.redundancy_penalty < 0:
            raise ValueError("redundancy_penalty must be non-negative.")
        if self.noise_penalty_c < 0:
            raise ValueError("noise_penalty_c must be non-negative.")
        if self.gain_function not in {"log", "sqrt"}:
            raise ValueError("gain_function must be 'log' or 'sqrt'.")
        if self.initial_pool_multiplier < 1.0:
            self.initial_pool_multiplier = 1.0

    def target_budget(self, fallback_limit: int) -> int:
        budget = self.chunk_budget if self.chunk_budget is not None else fallback_limit
        return max(1, budget)

    def requested_candidates(self, fallback_limit: int) -> int:
        budget = self.target_budget(fallback_limit)
        scaled = int(math.ceil(budget * self.initial_pool_multiplier))
        return max(fallback_limit, budget, scaled)

    @classmethod
    def from_mapping(cls, mapping: dict) -> NeighborAwareSelectionConfig:
        params = mapping.get("parameters", mapping)
        defaults = cls()
        return cls(
            method=mapping.get("method", defaults.method),
            chunk_budget=params.get("chunk_budget", defaults.chunk_budget),
            gamma=params.get("gamma", defaults.gamma),
            redundancy_penalty=params.get(
                "lambda", params.get("redundancy_penalty", defaults.redundancy_penalty)
            ),
            noise_penalty_c=params.get("noise_penalty_c", defaults.noise_penalty_c),
            a=params.get("a", defaults.a),
            b=params.get("b", defaults.b),
            gain_function=params.get("gain_function", defaults.gain_function),
            initial_pool_multiplier=params.get(
                "initial_pool_multiplier",
                params.get("initial_pool_factor", defaults.initial_pool_multiplier),
            ),
        )


async def apply_neighbor_aware_selection(
    hits: List[Tuple[Chunk, float]],
    embedder: EmbedderBase,
    config: NeighborAwareSelectionConfig,
    fallback_limit: int,
) -> List[Tuple[Chunk, float]]:
    if not hits:
        return []

    candidate_count = min(len(hits), config.requested_candidates(fallback_limit))
    candidates = hits[:candidate_count]
    scores = [float(score) for _, score in candidates]
    probabilities = _logistic_with_neighbors(scores, config)

    contents = [chunk.content for chunk, _ in candidates]
    embeddings_raw = await embedder.embed(contents)

    embeddings: List[List[float]]
    if contents and (
        isinstance(embeddings_raw, list)
        and embeddings_raw
        and isinstance(embeddings_raw[0], (int, float))
    ):
        embeddings = [cast(List[float], embeddings_raw)]
    else:
        embeddings = cast(List[List[float]], embeddings_raw)

    if len(embeddings) != len(candidates):
        raise ValueError("Embedder returned unexpected number of vectors.")

    norms = [_vector_norm(vec) for vec in embeddings]
    budget = min(config.target_budget(fallback_limit), len(candidates))

    selected: List[Tuple[Chunk, float]] = []
    selected_indices: set[int] = set()
    selected_embeddings: List[List[float]] = []
    selected_norms: List[float] = []

    while len(selected) < budget:
        best_idx = None
        best_gain = float("-inf")

        for idx, (_, _) in enumerate(candidates):
            if idx in selected_indices:
                continue
            base_gain = _gain(probabilities[idx], config.gain_function)
            redundancy = 0.0
            for emb, norm in zip(selected_embeddings, selected_norms):
                redundancy += _cosine_similarity(
                    embeddings[idx], emb, norms[idx], norm
                )
            marginal_gain = (
                base_gain
                - config.redundancy_penalty * redundancy
                - config.noise_penalty_c
            )
            if marginal_gain > best_gain:
                best_gain = marginal_gain
                best_idx = idx

        if best_idx is None or best_gain <= 0:
            break

        selected_indices.add(best_idx)
        selected_embeddings.append(embeddings[best_idx])
        selected_norms.append(norms[best_idx])
        selected.append((candidates[best_idx][0], probabilities[best_idx]))

    if not selected:
        fallback = min(budget, len(candidates))
        return candidates[:fallback]

    selected.sort(key=lambda item: item[1], reverse=True)
    return selected
