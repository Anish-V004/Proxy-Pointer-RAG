import pytest

from pprag.gemini_embeddings import (
    embed_texts_batched,
    normalize_embedding_response,
)


class FakeGenAI:
    def __init__(self, fail_first=False):
        self.calls = []
        self.fail_first = fail_first

    def embed_content(self, *, model, content, output_dimensionality):
        self.calls.append({
            "model": model,
            "content": content,
            "output_dimensionality": output_dimensionality,
        })
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("429 Resource exhausted")
        return {
            "embedding": [
                [float(len(str(item))), float(output_dimensionality)]
                for item in content
            ]
        }


def test_embed_texts_batched_splits_requests_and_preserves_order():
    fake_genai = FakeGenAI()
    sleeps = []

    vectors = embed_texts_batched(
        fake_genai,
        ["alpha", "beta", "gamma", "delta", "epsilon"],
        model="models/test-embedding",
        output_dimensionality=2,
        batch_size=2,
        batch_delay=0.25,
        sleep=sleeps.append,
    )

    assert [call["content"] for call in fake_genai.calls] == [
        ["alpha", "beta"],
        ["gamma", "delta"],
        ["epsilon"],
    ]
    assert vectors == [
        [5.0, 2.0],
        [4.0, 2.0],
        [5.0, 2.0],
        [5.0, 2.0],
        [7.0, 2.0],
    ]
    assert sleeps == [0.25, 0.25]


def test_embed_texts_batched_retries_rate_limit_before_batch_delay():
    fake_genai = FakeGenAI(fail_first=True)
    sleeps = []

    vectors = embed_texts_batched(
        fake_genai,
        ["alpha", "beta", "gamma"],
        model="models/test-embedding",
        output_dimensionality=2,
        batch_size=2,
        batch_delay=0.5,
        base_delay=3,
        sleep=sleeps.append,
    )

    assert [call["content"] for call in fake_genai.calls] == [
        ["alpha", "beta"],
        ["alpha", "beta"],
        ["gamma"],
    ]
    assert vectors == [[5.0, 2.0], [4.0, 2.0], [5.0, 2.0]]
    assert sleeps == [3, 0.5]


def test_normalize_embedding_response_rejects_count_mismatch():
    with pytest.raises(ValueError, match="Expected 2 embedding"):
        normalize_embedding_response({"embedding": [[1.0, 2.0]]}, 2)
