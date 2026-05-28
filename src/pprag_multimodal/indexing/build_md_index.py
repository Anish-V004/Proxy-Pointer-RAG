"""
Proxy-Pointer MultiModal: Build MD Index Pipeline

Combined pipeline that:
  Step 0: Builds skeleton trees for Markdown articles (pure-Python, no external deps)
  Step 1: LLM-based noise filtering (removes TOC, abbreviations, etc.)
  Step 2: Chunks and embeds document sections (1536-dim Gemini embeddings)
  Step 3: Builds/updates FAISS vector index
"""
import os
import sys
import json
import logging
import argparse
import time
import hashlib
from collections.abc import Sequence as SequenceABC

from pprag_multimodal.config import (
    DATASET_DIR, TREES_DIR, INDEX_DIR,
    EMBEDDING_MODEL, EMBEDDING_DIMS, NOISE_FILTER_MODEL,
    EMBEDDING_BATCH_SIZE, EMBEDDING_BATCH_DELAY, MIN_SECTION_LENGTH,
)
from pprag_multimodal.indexing.md_tree_builder import (
    build_skeleton_trees, get_md_path_for_doc
)
from pprag.faiss_security import require_trusted_faiss_deserialization

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(message)s')


def _normalize_embeddings(result, expected_count):
    if not isinstance(result, dict):
        raise ValueError(f"Unexpected embedding response type: {type(result).__name__}")
    embeddings = result.get("embeddings", result.get("embedding"))
    if embeddings is None:
        raise ValueError(f"Embedding response missing embedding data: {result!r}")
    if expected_count == 1 and embeddings and isinstance(embeddings[0], (int, float)):
        return [embeddings]
    if not isinstance(embeddings, SequenceABC):
        raise ValueError(f"Embedding response is not a sequence: {result!r}")
    vectors = list(embeddings)
    if len(vectors) != expected_count:
        raise ValueError(f"Expected {expected_count} embedding(s), received {len(vectors)}")
    return vectors


def _is_rate_limit_error(exc):
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "quota" in text or "rate" in text or "resource exhausted" in text


def _extract_retry_delay(exc):
    """Attempt to parse a retry delay from a Google API exception string."""
    exc_str = str(exc)
    import re
    # 1. Check for "Please retry in X.XXs"
    match = re.search(r"please retry in\s+(\d+(?:\.\d+)?)\s*s", exc_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    # 2. Check for "retry_delay { seconds: X }"
    match_sec = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", exc_str, re.IGNORECASE)
    if match_sec:
        try:
            return float(match_sec.group(1))
        except ValueError:
            pass
    return None


def _strip_json_fence(text):
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.split("\n")
    if lines and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _node_line_index(node, md_lines, doc_id, node_id):
    try:
        line_num = int(node["line_num"])
    except (KeyError, TypeError, ValueError):
        logging.warning("  [SKIP] %s/%s: invalid line_num", doc_id, node_id)
        return None
    if line_num < 1:
        logging.warning("  [SKIP] %s/%s: line_num must be >= 1", doc_id, node_id)
        return None
    return max(0, min(len(md_lines), line_num - 1))


def _existing_doc_ids(vector_db):
    docs = getattr(getattr(vector_db, "docstore", None), "_dict", None)
    if docs is None:
        logging.warning("FAISS docstore does not expose iterable documents; incremental skip disabled.")
        return set()
    existing = set()
    for doc in docs.values():
        metadata = getattr(doc, "metadata", {})
        if isinstance(metadata, dict) and "doc_id" in metadata:
            existing.add(metadata["doc_id"])
    return existing


def get_doc_id(md_file_path: str) -> str:
    """Stable content-based document ID: '<basename>_<sha256[:12]>'.

    The hash is computed from raw file bytes, so it is identical across runs
    as long as the file is unchanged. Any edit to the file produces a new
    hash, which invalidates the state entry and triggers a full re-index.
    """
    with open(md_file_path, "rb") as f:
        content_hash = hashlib.sha256(f.read()).hexdigest()[:12]
    base_name = os.path.splitext(os.path.basename(md_file_path))[0]
    return f"{base_name}_{content_hash}"


def _remove_stale_doc_chunks(vector_db, doc_name: str, current_doc_id: str) -> int:
    """Delete FAISS chunks that belong to an older version of doc_name.

    A chunk is stale when its metadata['doc_id'] starts with '<doc_name>_'
    but does not match current_doc_id (same filename, different content hash).
    Returns the number of stale chunk vectors removed.
    """
    if vector_db is None:
        return 0
    docstore_dict = getattr(getattr(vector_db, "docstore", None), "_dict", None)
    if docstore_dict is None:
        return 0

    prefix = f"{doc_name}_"
    stale_ids = [
        docstore_id
        for docstore_id, doc in docstore_dict.items()
        if getattr(doc, "metadata", {}).get("doc_id", "").startswith(prefix)
        and getattr(doc, "metadata", {}).get("doc_id") != current_doc_id
    ]

    if stale_ids:
        vector_db.delete(stale_ids)

    return len(stale_ids)


# ── Custom Embedding Wrapper ────────────────────────────────────────────
class GeminiEmbeddings(Embeddings):
    """LangChain-compatible wrapper for Gemini embeddings with rate-limit protection."""

    def __init__(self, model=EMBEDDING_MODEL, dimensionality=EMBEDDING_DIMS):
        self.model = model
        self.dimensionality = dimensionality

    def embed_documents(self, texts):
        all_embeddings = []
        batch_size = max(1, EMBEDDING_BATCH_SIZE)

        from tqdm import tqdm
        pbar = tqdm(total=len(texts), desc="Embedding chunks", unit="chunk")

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            for attempt in range(3):
                try:
                    result = genai.embed_content(
                        model=self.model,
                        content=batch,
                        output_dimensionality=self.dimensionality
                    )
                    all_embeddings.extend(_normalize_embeddings(result, len(batch)))
                    break
                except Exception as e:
                    if not _is_rate_limit_error(e):
                        logging.exception("Embedding request failed")
                        pbar.close()
                        raise

                    if attempt < 2:
                        retry_delay = _extract_retry_delay(e)
                        if retry_delay is not None:
                            wait = retry_delay + 5.0
                            logging.info(
                                f"  Rate limited. Found retry delay of {retry_delay}s. "
                                f"Waiting {wait}s (delay + 5s) (attempt {attempt+1}/3): {e}"
                            )
                        else:
                            # No retry delay in error — use a conservative flat default.
                            wait = 60.0
                            logging.info(
                                f"  Rate limited (no retry delay in error). "
                                f"Waiting {wait}s (attempt {attempt+1}/3)."
                            )
                        time.sleep(wait)
                    else:
                        logging.exception("Embedding request failed")
                        pbar.close()
                        raise
            pbar.update(len(batch))
            if i + batch_size < len(texts):
                time.sleep(max(0.0, EMBEDDING_BATCH_DELAY))

        pbar.close()
        return all_embeddings

    def embed_query(self, text):
        result = genai.embed_content(
            model=self.model,
            content=text,
            output_dimensionality=self.dimensionality
        )
        return _normalize_embeddings(result, 1)[0]


# ── Noise Filter ────────────────────────────────────────────────────────
def get_noise_node_ids(doc_name, structure):
    """Send the tree structure to an LLM to identify non-technical noise nodes."""
    # Slim down tree for smaller token usage
    def _slim_tree(nodes):
        slim = []
        for n in nodes:
            entry = {"title": n["title"], "node_id": n["node_id"]}
            if "nodes" in n:
                entry["nodes"] = _slim_tree(n["nodes"])
            slim.append(entry)
        return slim

    slim_structure = _slim_tree(structure)
    tree_json = json.dumps(slim_structure, indent=2, ensure_ascii=False)

    prompt = f"""You are a document-structure analyst. I will give you the
structural tree of a document called "{doc_name}" as JSON.

Your task: Identify every node whose title matches one of these noise categories:
  1. Table of contents (e.g. Contents, Index)
  2. Abbreviations or glossary (e.g. Acronyms, Glossary)
  3. Acknowledgments (e.g. Credits, Acknowledgements)
  4. Foreword/Preface
  5. References / Bibliography
  6. Executive Summary (if generic)

Only flag nodes that clearly fall into one of the above 6 categories.
Do NOT flag technical sections (Methodology, Experiments, Results).

── DOCUMENT TREE ──
{tree_json}

── RESPONSE FORMAT ──
Return ONLY a valid JSON object:
{{
  "noise_nodes": [
    {{"node_id": "XXXX", "title": "...", "category": "which of the 6 above"}}
  ]
}}
No markdown fencing, no extra text.
"""

    model = genai.GenerativeModel(NOISE_FILTER_MODEL)
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=2048,
            )
        )
        text = _strip_json_fence(response.text)
        result = json.loads(text)
    except Exception as e:
        raw = getattr(locals().get("response", None), "text", "")
        logging.exception("  [Error running noise filter for %s]; skipping noise filtering. Raw: %.500r", doc_name, raw)
        return set()

    noise_ids = set()
    for entry in result.get("noise_nodes", []):
        nid = entry.get("node_id")
        if nid:
            noise_ids.add(nid)
            logging.info(f"  [NOISE] {nid} {entry.get('title', '')} — {entry.get('category', '')}")

    return noise_ids


def _count_indexed_chunks(vector_db, doc_id: str) -> int:
    if vector_db is None:
        return 0
    docs = getattr(getattr(vector_db, "docstore", None), "_dict", None)
    if docs is None:
        return 0
    count = 0
    for doc in docs.values():
        if getattr(doc, "metadata", {}).get("doc_id") == doc_id:
            count += 1
    return count


def load_indexing_state(index_dir: str, fresh: bool) -> dict:
    state_path = os.path.join(index_dir, "indexing_state.json")
    if fresh or not os.path.exists(state_path):
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_indexing_state(index_dir: str, state: dict):
    state_path = os.path.join(index_dir, "indexing_state.json")
    tmp_path = os.path.join(index_dir, "indexing_state.json.tmp")
    try:
        os.makedirs(index_dir, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, state_path)
    except Exception as e:
        logging.warning(f"Failed to save indexing state: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# ── Main Build Pipeline ────────────────────────────────────────────────
def build_md_index(incremental=True):
    # Step 0: Build skeleton trees
    logging.info("\n" + "=" * 60)
    logging.info("STEP 0: Building skeleton trees for Markdown articles...")
    logging.info("=" * 60)
    build_skeleton_trees(DATASET_DIR, TREES_DIR)

    # Step 1: Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    # Step 2: Initialize Gemini Embeddings
    embeddings = GeminiEmbeddings()
    logging.info(f"Embedding model: {embeddings.model} @ {embeddings.dimensionality} dims")

    existing_docs = set()
    vector_db = None
    save_path = str(INDEX_DIR)

    if incremental and os.path.exists(save_path):
        try:
            # LangChain FAISS persists metadata via pickle. Only load trusted
            # indexes generated locally by this application; re-ingest raw data
            # instead of using indexes from untrusted sources.
            require_trusted_faiss_deserialization(save_path, "PP_TRUST_FAISS_INDEX")
            vector_db = FAISS.load_local(
                save_path, embeddings, allow_dangerous_deserialization=True
            )
            existing_docs = _existing_doc_ids(vector_db)
            logging.info(f"Loaded existing index with {len(existing_docs)} completely indexed document(s).")
        except Exception as e:
            logging.warning(f"Could not load existing index: {e}. Building fresh.")
            vector_db = None

    state = load_indexing_state(save_path, fresh=not incremental)

    # Sync state completed_chunks with actual FAISS chunk count if needed.
    # If a crash interrupted saving indexing_state.json after FAISS was already
    # written, the state file may lag behind reality — advance it so we don't
    # re-embed chunks already in the index.
    if vector_db is not None and state:
        for doc_id, doc_state in list(state.items()):
            if doc_state.get("status") == "in_progress":
                actual_count = _count_indexed_chunks(vector_db, doc_id)
                saved_count = doc_state.get("completed_chunks", 0)
                if actual_count > saved_count:
                    logging.info(
                        f"State sync: {doc_id} has {actual_count} chunks in FAISS but "
                        f"state showed {saved_count}. Syncing state to {actual_count}."
                    )
                    doc_state["completed_chunks"] = actual_count
                    if actual_count >= doc_state.get("total_chunks", 0):
                        doc_state["status"] = "completed"
        save_indexing_state(save_path, state)

    new_chunks_added = 0
    tree_files = sorted([
        f for f in os.listdir(TREES_DIR)
        if f.endswith("_structure.json")
    ])
    logging.info(f"Found {len(tree_files)} tree(s)")

    total_files = len(tree_files)
    for idx, file in enumerate(tree_files, 1):
        tree_path = os.path.join(TREES_DIR, file)
        with open(tree_path, "r", encoding="utf-8") as f:
            tree_data = json.load(f)

        doc_name = tree_data.get("doc_name", file.replace("_structure.json", ""))

        # Resolve md_path first so we can compute a content-hash based doc_id.
        md_path = get_md_path_for_doc(DATASET_DIR, doc_name)
        if not md_path:
            logging.error(f"  [{idx}/{total_files}] [Error] Could not find markdown for {doc_name}")
            continue

        # Content-hash based ID: changes whenever the file is modified,
        # which invalidates the state entry and forces a fresh re-index.
        doc_id = get_doc_id(md_path)

        # If the file was modified, its hash has changed. Remove any FAISS chunks
        # from the previous version before starting a fresh ingestion.
        stale_removed = _remove_stale_doc_chunks(vector_db, doc_name, doc_id)
        if stale_removed > 0:
            logging.info(
                f"  [STALE] Removed {stale_removed} outdated chunk(s) for '{doc_name}' "
                f"(file was modified). Re-indexing from scratch."
            )
            # Prune old hash-based state entries for this doc_name
            for old_key in [k for k in state if k.startswith(f"{doc_name}_") and k != doc_id]:
                del state[old_key]
            os.makedirs(save_path, exist_ok=True)
            vector_db.save_local(save_path)
            save_indexing_state(save_path, state)

        doc_state = state.get(doc_id, {})
        if doc_state.get("status") == "completed" and doc_id in existing_docs:
            logging.info(f"[{idx}/{total_files}] [SKIP] {doc_id}: Already indexed.")
            continue

        logging.info(f"[{idx}/{total_files}] Processing: {doc_id} ({doc_name})...")

        with open(md_path, "r", encoding="utf-8") as f:
            md_lines = f.readlines()

        structure = tree_data.get("structure")
        if not isinstance(structure, list):
            logging.error("  [Error] Invalid or missing structure for %s", doc_id)
            continue

        noise_node_ids = get_noise_node_ids(doc_id, structure)

        doc_chunks = []

        def process_node(node_list, parent_end=None, breadcrumb=""):
            if parent_end is None:
                parent_end = len(md_lines)

            for i, node in enumerate(node_list):
                node_id = node.get("node_id")
                title = node.get("title", "")

                if node_id in noise_node_ids:
                    continue

                current_crumb = f"{breadcrumb} > {title}" if breadcrumb else title
                start_idx = _node_line_index(node, md_lines, doc_id, node_id)
                if start_idx is None:
                    continue

                if i + 1 < len(node_list):
                    next_idx = _node_line_index(node_list[i + 1], md_lines, doc_id, node_id)
                    end_idx = parent_end if next_idx is None else next_idx
                else:
                    end_idx = parent_end
                end_idx = max(start_idx, min(len(md_lines), end_idx))

                node_end = end_idx
                if "nodes" in node and node["nodes"]:
                    first_child_line = _node_line_index(node["nodes"][0], md_lines, doc_id, node_id)
                    if first_child_line is not None:
                        end_idx = min(end_idx, first_child_line)

                section_text = "".join(md_lines[start_idx:end_idx]).strip()

                if len(section_text) >= MIN_SECTION_LENGTH:
                    chunks = text_splitter.split_text(section_text)
                    for chunk in chunks:
                        enriched_content = f"[{current_crumb}]\n{chunk}"
                        doc_chunks.append(Document(
                            page_content=enriched_content,
                            metadata={
                                "doc_id": doc_id,
                                "node_id": node_id,
                                "title": title,
                                "breadcrumb": current_crumb,
                                "start_line": start_idx,
                                "end_line": node_end,
                            }
                        ))

                if "nodes" in node and node["nodes"]:
                    process_node(node["nodes"], node_end, current_crumb)

        process_node(structure)

        # Determine remaining chunks using the state file
        completed_chunks = doc_state.get("completed_chunks", 0)
        # If total_chunks is different from current chunks, reset progress to be safe
        if doc_state.get("total_chunks") != len(doc_chunks):
            completed_chunks = 0

        remaining_chunks = doc_chunks[completed_chunks:]

        if remaining_chunks:
            from tqdm import tqdm
            pbar = tqdm(
                initial=completed_chunks,
                total=len(doc_chunks),
                desc=f"Embedding {doc_id}",
                unit="chunk",
            )

            batch_size = EMBEDDING_BATCH_SIZE
            for i in range(0, len(remaining_chunks), batch_size):
                chunk_batch = remaining_chunks[i : i + batch_size]

                if vector_db is not None:
                    vector_db.add_documents(chunk_batch)
                else:
                    vector_db = FAISS.from_documents(chunk_batch, embeddings)

                os.makedirs(save_path, exist_ok=True)
                vector_db.save_local(save_path)

                completed_chunks += len(chunk_batch)
                state[doc_id] = {
                    "status": "in_progress" if completed_chunks < len(doc_chunks) else "completed",
                    "completed_chunks": completed_chunks,
                    "total_chunks": len(doc_chunks),
                }
                save_indexing_state(save_path, state)
                pbar.update(len(chunk_batch))

                has_more = i + batch_size < len(remaining_chunks)
                if has_more and EMBEDDING_BATCH_DELAY > 0:
                    time.sleep(EMBEDDING_BATCH_DELAY)

            pbar.close()
            new_chunks_added += len(remaining_chunks)
        else:
            logging.info(f"No new chunks to embed for {doc_id}.")

    if new_chunks_added == 0:
        logging.warning("No new chunks generated.")
    else:
        logging.info(f"Successfully processed and saved a total of {new_chunks_added} new chunks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Multimodal FAISS index")
    parser.add_argument("--fresh", action="store_true", help="Rebuild from scratch")
    args = parser.parse_args()
    build_md_index(incremental=not args.fresh)
