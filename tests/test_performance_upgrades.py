import builtins
import time
from pathlib import Path

from pprag_doc_comparator.comparison import section_comparator
from pprag_doc_comparator.comparison.section_comparator import compare_section_matches
from pprag_doc_comparator.comparison.section_selector import (
    _read_md_lines_cached,
    load_full_section_text,
)
from pprag_multimodal.agent.mm_rag_bot import MultimodalProxyPointerRAG


def test_load_full_section_text_reuses_cached_markdown_lines(monkeypatch, tmp_path):
    md_path = tmp_path / "doc.md"
    md_path.write_text("# Heading\nline 1\nline 2\nline 3\n", encoding="utf-8")

    _read_md_lines_cached.cache_clear()
    open_count = 0
    real_open = builtins.open

    def counting_open(*args, **kwargs):
        nonlocal open_count
        if Path(args[0]) == md_path:
            open_count += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)

    assert load_full_section_text("doc_hash", 0, 2, md_path=str(md_path)) == "# Heading\nline 1"
    assert load_full_section_text("doc_hash", 1, 4, md_path=str(md_path)) == "line 1\nline 2\nline 3"
    assert open_count == 1


def test_compare_section_matches_preserves_order_with_parallel_speedup(monkeypatch):
    def fake_compare(comparison_prompt, doc1_section, doc2_section, doc1_name="Doc 1", doc2_name="Doc 2", doc_type="document"):
        time.sleep(0.05)
        return {"rating": "GREEN", "doc2_title": doc2_section["title"]}

    monkeypatch.setattr(section_comparator, "compare_sections", fake_compare)

    matches = [{"title": f"Match {i}"} for i in range(3)]

    start = time.perf_counter()
    serial = compare_section_matches("prompt", {}, matches, max_workers=1)
    serial_seconds = time.perf_counter() - start

    start = time.perf_counter()
    parallel = compare_section_matches("prompt", {}, matches, max_workers=3)
    parallel_seconds = time.perf_counter() - start

    assert [item["doc2_title"] for item in parallel] == ["Match 0", "Match 1", "Match 2"]
    assert serial == parallel
    assert parallel_seconds < serial_seconds * 0.75


def test_multimodal_bot_reuses_markdown_and_tree_caches(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "data"
    paper_dir = dataset_dir / "Paper"
    trees_dir = tmp_path / "trees"
    paper_dir.mkdir(parents=True)
    trees_dir.mkdir()

    md_path = paper_dir / "Paper.md"
    tree_path = trees_dir / "Paper_structure.json"
    md_path.write_text("# Paper\nsection text\n", encoding="utf-8")
    tree_path.write_text(
        '{"structure": [{"title": "Paper", "node_id": "0001", "line_num": 1}]}',
        encoding="utf-8",
    )

    bot = MultimodalProxyPointerRAG.__new__(MultimodalProxyPointerRAG)
    bot.dataset_dir = str(dataset_dir)
    bot.trees_dir = str(trees_dir)
    bot._md_path_cache = {}
    bot._md_lines_cache = {}
    bot._tree_node_cache = {}

    open_counts = {md_path.resolve(): 0, tree_path.resolve(): 0}
    real_open = builtins.open

    def counting_open(*args, **kwargs):
        path = Path(args[0]).resolve()
        if path in open_counts:
            open_counts[path] += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)

    assert bot._get_md_lines("Paper") == ["# Paper\n", "section text\n"]
    assert bot._get_md_lines("Paper") == ["# Paper\n", "section text\n"]
    assert "0001" in bot._get_tree_node_map("Paper")
    assert "0001" in bot._get_tree_node_map("Paper")
    assert open_counts == {md_path.resolve(): 1, tree_path.resolve(): 1}
