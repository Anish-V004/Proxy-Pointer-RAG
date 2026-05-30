# Indexing Pipeline — Functional Changes

> Changes since commit `6f636f2 – Performance optimization`  
> Files modified: `build_pp_index.py`, `build_md_index.py`, `build_doc_index.py`, `gemini_embeddings.py`

---

## 1. Resumable Indexing (Batch-Level Checkpointing)

**Problem:** If the indexing process was interrupted (crash, network cut, manual stop), the entire run had to restart from zero. All embedding API calls and time were wasted.

**Fix:** Introduced `indexing_state.json` inside each pipeline's `INDEX_DIR`. Every time a batch of chunks is embedded and saved to FAISS, the state file is updated immediately. On the next run, already-completed chunks are skipped.

**State schema per document:**
```json
{
  "DocName_<hash>": {
    "status": "in_progress | completed",
    "completed_chunks": 120,
    "total_chunks": 200
  }
}
```

**Applies to:** Text-Only, MultiModal, Doc-Comparator pipelines.

---

## 2. Atomic State File Writes (Crash-Safe Saves)

**Problem:** If the process crashed exactly while writing `indexing_state.json`, the file could be left half-written and corrupted, breaking all future runs.

**Fix:** State is always written to a `.tmp` file first, then atomically replaced via `os.replace()`. If the crash happens mid-write, the `.tmp` is discarded and the last good state file remains intact.

```
Write → indexing_state.json.tmp
Then  → os.replace(.tmp → indexing_state.json)  ← atomic on all OSes
```

**Applies to:** Text-Only, MultiModal, Doc-Comparator pipelines.

---

## 3. FAISS ↔ State Sync on Startup

**Problem:** A crash between `vector_db.save_local()` and `save_indexing_state()` leaves FAISS ahead of the state file. On the next run, the state would report fewer `completed_chunks` than actually exist in FAISS, causing those chunks to be re-embedded — creating **duplicates**.

**Fix:** On every startup, for each document marked `in_progress`, the actual chunk count in FAISS is compared against what the state file recorded. If FAISS has more, the state is advanced to match before any indexing begins.

```python
actual_count = _count_indexed_chunks(vector_db, doc_id)
if actual_count > saved_count:
    doc_state["completed_chunks"] = actual_count
    # mark completed if all chunks are present
```

**Applies to:** Text-Only, MultiModal, Doc-Comparator pipelines.

---

## 4. Content-Hash Based Document IDs

**Problem:** Documents were identified only by filename. If a document was updated (same filename, new content), the indexer had no way to detect the change — it would either skip re-indexing or create duplicate chunks alongside the old ones.

**Fix:** `doc_id` is now derived from **file content** using SHA-256:
```
doc_id = "<basename>_<sha256[:12]>"
e.g.  "AMD_2022_10K_a3f9c12b84e1"
```

- Same file content → same hash → same `doc_id` → correctly skipped on re-run
- Modified file → new hash → new `doc_id` → triggers full re-index
- The basename prefix is retained so logs remain human-readable

**Note:** `build_doc_index.py` (Doc-Comparator) already used this pattern. Applied to Text-Only and MultiModal pipelines in this change.

**Applies to:** Text-Only, MultiModal pipelines (Doc-Comparator already had this).

---

## 5. Automatic Stale Chunk Cleanup on Document Modification

**Problem:** Even with content-hash IDs, when a document was modified and re-indexed, the old chunks (old `doc_id`) remained in FAISS indefinitely alongside the new ones. This wastes vector space and causes outdated content to appear in query results.

**Fix:** Before ingesting any document, the FAISS docstore is scanned for chunks whose `doc_id` starts with `<doc_name>_` but does **not** match the current hash. Any such chunks are **deleted** from FAISS, the matching state entries are pruned, and the index is saved — before the fresh ingestion begins.

```
Detected:  "AMD_2022_10K_a3f9c12b84e1" in FAISS
Current:   "AMD_2022_10K_f7d2e891bc03"
Action:    Delete all chunks with old doc_id → re-index with new one
```

**Applies to:** Text-Only, MultiModal, Doc-Comparator pipelines.

---

## 6. Per-Document Progress Bars with `[idx/total]` Logging

**Problem:** During long indexing runs (multiple large documents), there was no indication of overall progress — only per-batch output.

**Fix:** Added `[idx/total]` prefixes to all document-level log lines and a `tqdm` progress bar per document showing chunk-level progress (starting from `initial=completed_chunks` so resumed runs show the correct position).

```
[1/4] Processing: AMD_2022_10K_a3f9...
Embedding AMD_2022_10K_a3f9...:  60%|██████    | 120/200 [chunk]
[2/4] [SKIP] BOEING_2022_10K_...: Already completely indexed in FAISS.
```

**Applies to:** Text-Only, MultiModal, Doc-Comparator pipelines.

---

## 7. Smarter Rate-Limit Retry Delay (gemini_embeddings.py)

**Problem:** On rate-limit errors, the retry logic was not robust enough. Standard short exponential backoffs failed to clear rolling rate limits, while ignoring the API's recommended retry delay led to either under-waiting or over-waiting.

**Fix:** Refined the logic to handle two scenarios:
1. **Suggested Delay from API**: If a recommended retry delay is parsed from the error message (using `"Please retry in X.XXs"` or `"retry_delay { seconds: X }"` patterns), the client waits **exactly** that long (without any extra buffer) up to a maximum of **3 attempts**.
2. **No Suggested Delay (Custom Backoff)**: If no retry delay is provided, the client performs exponential backoff starting at `base_delay` (defaulting to `2s`), doubling each time (`2s`, `4s`, `8s`, `16s`, `32s`), and capping the backoff at **`60s`**. If it still fails, it waits exactly once more for **another `60s`** (totalling 120s of capped delays), allowing a total of **8 attempts** for the backoff sequence.

**Applies to:** `gemini_embeddings.py` → used by Text-Only and Doc-Comparator `embed_content_with_retry`.

---

## 8. Test Support for Hashed Document IDs (tests/test_text_rag_integration.py)

**Problem:** Changing document IDs to content-hash based IDs (`fixture_<hash>`) broke the text RAG integration test assertion which expected the exact string `"fixture"`.

**Fix:** Updated the RAG retrieval test assertion from a strict equality check to a `.startswith("fixture")` check, ensuring test compatibility with the new hash-appended document IDs.

**Applies to:** `test_text_rag_integration.py`.

---

## 9. LLM Generation Retry & Safety Handling (pp_rag_bot.py, mm_rag_bot.py)

**Problem:** Re-ranking and synthesis LLM queries were failing directly on Gemini Free Tier due to the strict 15 RPM (requests per minute) rate limit. Additionally, when Gemini returned an empty/blocked response (finish reason `STOP` but no parts), calling `response.text` raised an `Invalid operation` exception.

**Fix:** 
1. Wrapped RAG model `_generate_content` calls in a custom retry handler using both explicit API retry delays and a capped exponential fallback (similar to the embedding retry logic).
2. Added a validation guard that detects empty/partless candidate responses (finish reason 1 with empty parts) and raises a transient exception to trigger a retry.
3. Added a `hasattr(response, "candidates")` check to maintain full compatibility with mock response objects in testing.

**Applies to:** `pp_rag_bot.py` and `mm_rag_bot.py`.

---

## Summary Table

| # | Fix | Files Affected |
|---|-----|---------------|
| 1 | Resumable batch-level checkpointing | All 3 indexers |
| 2 | Atomic state file writes (crash-safe) | All 3 indexers |
| 3 | FAISS ↔ state sync on startup | All 3 indexers |
| 4 | Content-hash based document IDs | Text-Only, MultiModal |
| 5 | Stale chunk cleanup on doc modification | All 3 indexers |
| 6 | Progress bars + `[idx/total]` logging | All 3 indexers |
| 7 | Smarter rate-limit retry delay | `gemini_embeddings.py` |
| 8 | Test support for hashed document IDs | `test_text_rag_integration.py` |
| 9 | LLM generation retry & safety handling | `pp_rag_bot.py`, `mm_rag_bot.py` |

---

## Migration Note

Existing FAISS indexes were built with **filename-only** `doc_id`s (e.g. `"AMD_2022_10K"`).
The new format uses `"AMD_2022_10K_<hash>"`. On first run after this change, the old chunks
won't be recognized as stale (they don't start with the new prefix pattern) and will coexist.

**Recommended action:** Run with `--fresh` once per pipeline to rebuild cleanly:
```bash
python -m pprag_text_only.indexing.build_pp_index --fresh
python -m pprag_multimodal.indexing.build_md_index --fresh
python -m pprag_doc_comparator.indexing.build_doc_index
```
