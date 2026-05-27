"""Shared Gemini embedding helpers for Proxy-Pointer index builders."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence


def normalize_embedding_response(result, expected_count: int):
    """Return a list of embedding vectors from Gemini single or batch shapes."""
    if not isinstance(result, dict):
        raise ValueError(f"Unexpected embedding response type: {type(result).__name__}")

    embeddings = result.get("embeddings", result.get("embedding"))
    if embeddings is None:
        raise ValueError(f"Embedding response missing embedding data: {result!r}")

    if expected_count == 1 and embeddings and isinstance(embeddings[0], (int, float)):
        return [embeddings]
    if not isinstance(embeddings, Sequence):
        raise ValueError(f"Embedding response is not a sequence: {result!r}")

    vectors = list(embeddings)
    if len(vectors) != expected_count:
        raise ValueError(f"Expected {expected_count} embedding(s), received {len(vectors)}")
    return vectors


def is_rate_limit_error(exc: Exception) -> bool:
    """Return True when an SDK exception looks like quota or rate limiting."""
    status_code = (
        getattr(getattr(exc, "response", None), "status_code", None)
        or getattr(exc, "status_code", None)
    )
    if status_code == 429:
        return True

    error_str = str(exc).lower()
    error_code = str(getattr(exc, "code", "")).lower()
    return (
        "429" in error_str
        or "resourceexhausted" in error_str
        or "resource exhausted" in error_str
        or "quota exceeded" in error_str
        or "quota_exceeded" in error_code
        or "too_many_requests" in error_code
        or "rate limit" in error_str
    )


def embed_content_with_retry(
    genai_module,
    *,
    model: str,
    content,
    output_dimensionality: int,
    max_retries: int = 5,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
):
    """Call Gemini embeddings with exponential backoff for rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return genai_module.embed_content(
                model=model,
                content=content,
                output_dimensionality=output_dimensionality,
            )
        except Exception as exc:
            if not is_rate_limit_error(exc) or attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logging.warning("Rate limit hit during embedding. Retrying in %ss...", delay)
            sleep(delay)

    raise RuntimeError("Embedding request failed without returning or raising")


def embed_texts_batched(
    genai_module,
    texts: Sequence[str],
    *,
    model: str,
    output_dimensionality: int,
    batch_size: int,
    batch_delay: float,
    max_retries: int = 5,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
):
    """Embed texts in bounded batches, preserving input order."""
    if not texts:
        return []

    safe_batch_size = max(1, int(batch_size))
    all_embeddings = []

    for start in range(0, len(texts), safe_batch_size):
        batch = list(texts[start:start + safe_batch_size])
        result = embed_content_with_retry(
            genai_module,
            model=model,
            content=batch,
            output_dimensionality=output_dimensionality,
            max_retries=max_retries,
            base_delay=base_delay,
            sleep=sleep,
        )
        all_embeddings.extend(normalize_embedding_response(result, len(batch)))

        has_more = start + safe_batch_size < len(texts)
        if has_more and batch_delay > 0:
            sleep(batch_delay)

    return all_embeddings
