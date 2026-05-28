"""
DocComparator: Build Shared FAISS Index

Combined pipeline that:
  Step 0: Builds skeleton trees for new .md files
  Step 1: LLM-based noise filtering (removes TOC, abbreviations, etc.)
  Step 2: Chunks and embeds document sections (1536-dim Gemini embeddings)
  Step 3: Builds/updates shared FAISS vector index with doc_id metadata

Each chunk carries doc_id in metadata for query-time filtering.

Usage:
    from pprag_doc_comparator.indexing.build_doc_index import build_comparator_index
"""
import os
import sys
import time
import json
import hashlib
import logging
from pathlib import Path

from pprag_doc_comparator.config import (
    DOCUMENTS_DIR, TREES_DIR, INDEX_DIR,
    EMBEDDING_MODEL, EMBEDDING_DIMS, LLM_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_BATCH_SIZE, EMBEDDING_BATCH_DELAY,
)
from pprag_doc_comparator.indexing.build_skeleton_trees import build_skeleton_trees
from pprag.faiss_security import require_trusted_faiss_deserialization
from pprag.gemini_embeddings import (
    embed_content_with_retry,
    embed_texts_batched,
    normalize_embedding_response,
)

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ── Document Identity ──────────────────────────────────────────────────
def get_doc_id(file_path: str) -> str:
    """Generate a stable ID from file content using content hash."""
    with open(file_path, "rb") as f:
        content_hash = hashlib.sha256(f.read()).hexdigest()[:12]
    base_name = Path(file_path).stem
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


# ── Custom Embedding Wrapper ──────────────────────────────────────────
class GeminiEmbeddings(Embeddings):
    """LangChain-compatible wrapper for Gemini embeddings at configurable dims."""

    def __init__(self, model=EMBEDDING_MODEL, dimensionality=EMBEDDING_DIMS):
        self.model = model
        self.dimensionality = dimensionality

    def embed_documents(self, texts):
        """Embed a list of documents."""
        return embed_texts_batched(
            genai,
            texts,
            model=self.model,
            output_dimensionality=self.dimensionality,
            batch_size=EMBEDDING_BATCH_SIZE,
            batch_delay=EMBEDDING_BATCH_DELAY,
        )

    def embed_query(self, text):
        """Embed a single query."""
        result = embed_content_with_retry(
            genai,
            model=self.model,
            content=text,
            output_dimensionality=self.dimensionality,
        )
        return normalize_embedding_response(result, 1)[0]


# ── Noise Filter ────────────────────────────────────────────────────────
def get_noise_node_ids(doc_name, structure):
    """Send the tree to an LLM and return a set of noise node_ids."""
    tree_json = json.dumps(structure, indent=2, ensure_ascii=False)

    prompt = f"""You are a document-structure analyst. I will give you the
structural tree of a document called "{doc_name}" as JSON.

Your task: Identify every node whose title matches one of these
noise categories:
  1. Table of contents (e.g. "TABLE OF CONTENTS", "Contents", "Summary of Contents", "Index of Sections")
  2. Abbreviations or acronym lists (e.g. Abbreviations, Abbreviations (continued), Glossary)
  3. Acknowledgments (e.g. Acknowledgements, Note of Thanks, Credits)
  4. Foreword (e.g. Preface, Introductory Remarks)
  5. Executive Summary (e.g. Overview Summary, Key Highlights)
  6. References (e.g. Bibliography, Works Cited, Sources)
  7. Definitions (e.g. Definitions, Defined Terms, Accounting Terms)
  8. Bare sub-clause labels — nodes whose title is ONLY a parenthetical label with no descriptive text (e.g. "(a)", "(b)", "(c)", "(i)", "(ii)", "(iv)", "(x)"). These are inline sub-clauses, not real sections.

Only flag nodes that clearly fall into one of the above 8 categories.
Do NOT flag anything else.

── DOCUMENT TREE ──
{tree_json}

── RESPONSE FORMAT ──
Return ONLY a valid JSON object:
{{
  "noise_nodes": [
    {{"node_id": "XXXX", "title": "...", "category": "which of the 8 above"}}
  ]
}}

No markdown fencing, no extra text.
"""

    model = genai.GenerativeModel(LLM_MODEL)

    max_retries = 5
    base_delay = 2.0
    response = None
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=2048,
                )
            )
            break
        except Exception as e:
            if "429" in str(e) or "Resource exhausted" in str(e):
                if attempt == max_retries - 1:
                    raise e
                delay = base_delay * (2 ** attempt)
                logging.warning(f"Rate limit hit during noise filtering. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise e

    # Robustly extract JSON or fallback to regex
    try:
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        result = json.loads(text)
        noise_ids = set()
        for entry in result.get("noise_nodes", []):
            nid = entry.get("node_id")
            if nid:
                noise_ids.add(nid)
        return noise_ids

    except Exception as e:
        logging.warning(f"  [WARNING] Failed to parse noise filter JSON: {e}. Falling back to regex extraction.")
        # Fallback: extract any node_id that the LLM tried to output
        import re
        noise_ids = set()

        # Use getattr to safely access text in case response is blocked
        raw_text = ""
        try:
            raw_text = response.text
        except ValueError:
            pass

        if raw_text:
            matches = re.findall(r'"node_id"\s*:\s*"([^"]+)"', raw_text)
            for nid in matches:
                noise_ids.add(nid)

        if noise_ids:
            return noise_ids

        logging.warning("  [ERROR] Regex fallback also found no node_ids. Proceeding without filter.")
        return set()


# ── Index a Single Document ────────────────────────────────────────────
def index_single_document(md_path, tree_path, progress_callback=None):
    """
    Build chunks for a single document. Returns list of Document objects.
    Each chunk has doc_id in metadata for query-time filtering.
    """
    with open(tree_path, "r", encoding="utf-8") as f:
        tree_data = json.load(f)

    doc_id = get_doc_id(md_path)

    with open(md_path, "r", encoding="utf-8") as f:
        md_lines = f.readlines()

    if not tree_data.get("structure"):
        logging.warning(f"  [SKIP] {doc_id}: No structure found (headerless document).")
        return [], doc_id

    logging.info(f"Processing: {doc_id}...")

    # LLM-based noise filter (best effort)
    try:
        noise_node_ids = get_noise_node_ids(
            tree_data.get("doc_name", Path(md_path).stem),
            tree_data["structure"]
        )
    except Exception as exc:
        logging.warning("Noise filtering unavailable for %s: %s. Continuing without filter.", doc_id, exc)
        noise_node_ids = set()
    logging.info(f"  Noise nodes excluded: {len(noise_node_ids)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []

    def process_node(node_list, parent_end=None, breadcrumb=""):
        if parent_end is None:
            parent_end = len(md_lines)

        for i, node in enumerate(node_list):
            node_id = node.get("node_id")
            title = node.get("title", "")

            if node_id in noise_node_ids:
                continue

            current_crumb = (
                f"{breadcrumb} > {title}" if breadcrumb else title
            )
            start_idx = node["line_num"] - 1

            if i + 1 < len(node_list):
                end_idx = node_list[i + 1]["line_num"] - 1
            else:
                end_idx = parent_end

            node_end = end_idx
            if "nodes" in node and node["nodes"]:
                first_child_line = node["nodes"][0]["line_num"] - 1
                end_idx = min(end_idx, first_child_line)

            section_text = "".join(md_lines[start_idx:end_idx]).strip()

            # Parent nodes (chapter headings with children) often contain
            # only a short preamble. Require more substance to index them.
            has_children = "nodes" in node and node["nodes"]
            min_chars = 300 if has_children else 100

            if len(section_text) >= min_chars:
                chunks = text_splitter.split_text(section_text)
                for chunk in chunks:
                    enriched_content = f"[{current_crumb}]\n{chunk}"
                    doc = Document(
                        page_content=enriched_content,
                        metadata={
                            "doc_id": doc_id,
                            "node_id": node_id,
                            "title": title,
                            "breadcrumb": current_crumb,
                            "start_line": start_idx,
                            "end_line": node_end,
                        },
                    )
                    all_chunks.append(doc)

            if "nodes" in node and node["nodes"]:
                process_node(node["nodes"], node_end, current_crumb)

    process_node(tree_data["structure"])
    logging.info(f"  Generated {len(all_chunks)} chunks for {doc_id}")
    return all_chunks, doc_id


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
def build_comparator_index(md_paths=None, incremental=True, progress_callback=None):
    """
    Builds a single shared FAISS index across documents.
    Each chunk's metadata includes doc_id — filter at query time.

    Args:
        md_paths: List of .md file paths to index. If None, indexes all in DOCUMENTS_DIR.
        incremental: If True, appends to existing index.
        progress_callback: Optional callback(message) for UI progress updates.

    Returns:
        (vector_db, doc_ids) — the FAISS index and list of doc_ids indexed.
    """
    data_dir = str(DOCUMENTS_DIR)
    trees_dir = str(TREES_DIR)
    save_path = str(INDEX_DIR)

    # Step 0: Build skeleton trees
    if progress_callback:
        progress_callback("Building document structure trees...")
    build_skeleton_trees(data_dir, trees_dir)

    # Initialize embeddings
    embeddings = GeminiEmbeddings()
    logging.info(f"Embedding model: {embeddings.model} @ {embeddings.dimensionality} dims")

    # Load existing index if incremental
    existing_docs = set()
    vector_db = None
    if incremental and os.path.exists(save_path):
        try:
            # LangChain FAISS stores metadata in a pickle-backed docstore. Only
            # load trusted indexes generated locally by this application.
            require_trusted_faiss_deserialization(save_path, "DC_TRUST_FAISS_INDEX")
            vector_db = FAISS.load_local(
                save_path, embeddings, allow_dangerous_deserialization=True
            )
            docs = getattr(getattr(vector_db, "docstore", None), "_dict", None)
            if docs is None:
                logging.warning("FAISS docstore internals unavailable; incremental indexing disabled.")
                vector_db = None
            else:
                for doc in docs.values():
                    metadata = getattr(doc, "metadata", {})
                    if isinstance(metadata, dict) and "doc_id" in metadata:
                        existing_docs.add(metadata["doc_id"])
                logging.info(f"Loaded existing index with {len(existing_docs)} document(s).")
        except Exception as e:
            logging.warning(f"Could not load existing index: {e}. Building fresh.")
            vector_db = None

    state = load_indexing_state(save_path, fresh=not incremental)

    # Sync state completed_chunks with actual FAISS chunk count if needed.
    # If a crash interrupted saving indexing_state.json after FAISS was already
    # written, the state file may lag behind reality — advance it so we don't
    # re-embed chunks already in the index.
    if vector_db is not None and state:
        for doc_id_key, doc_state in list(state.items()):
            if doc_state.get("status") == "in_progress":
                actual_count = _count_indexed_chunks(vector_db, doc_id_key)
                saved_count = doc_state.get("completed_chunks", 0)
                if actual_count > saved_count:
                    logging.info(
                        f"State sync: {doc_id_key} has {actual_count} chunks in FAISS but "
                        f"state showed {saved_count}. Syncing state to {actual_count}."
                    )
                    doc_state["completed_chunks"] = actual_count
                    if actual_count >= doc_state.get("total_chunks", 0):
                        doc_state["status"] = "completed"
        save_indexing_state(save_path, state)

    # Determine which files to index
    if md_paths is None:
        md_paths = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith(".md")
        ]

    indexed_doc_ids = []
    new_chunks_added = 0

    total_files = len(md_paths)
    for idx, md_path in enumerate(md_paths, 1):
        base_name = Path(md_path).stem
        tree_path = os.path.join(trees_dir, f"{base_name}_structure.json")

        if not os.path.exists(tree_path):
            logging.warning(f"[{idx}/{total_files}] [SKIP] No tree for {base_name}")
            continue

        doc_id = get_doc_id(md_path)

        # If the file was modified, its hash has changed. Remove any FAISS chunks
        # from the previous version before starting a fresh ingestion.
        stale_removed = _remove_stale_doc_chunks(vector_db, base_name, doc_id)
        if stale_removed > 0:
            logging.info(
                f"  [STALE] Removed {stale_removed} outdated chunk(s) for '{base_name}' "
                f"(file was modified). Re-indexing from scratch."
            )
            # Prune old hash-based state entries for this doc_name
            for old_key in [k for k in state if k.startswith(f"{base_name}_") and k != doc_id]:
                del state[old_key]
            os.makedirs(save_path, exist_ok=True)
            vector_db.save_local(save_path)
            save_indexing_state(save_path, state)

        doc_state = state.get(doc_id, {})
        if doc_state.get("status") == "completed" and doc_id in existing_docs:
            logging.info(f"[{idx}/{total_files}] [CACHED] {doc_id}: Already indexed.")
            indexed_doc_ids.append(doc_id)
            continue

        if progress_callback:
            progress_callback(f"[{idx}/{total_files}] Indexing: {base_name}...")

        logging.info(f"[{idx}/{total_files}] Processing: {base_name}...")
        chunks, doc_id = index_single_document(md_path, tree_path, progress_callback)
        indexed_doc_ids.append(doc_id)

        # Determine remaining chunks using the state file
        completed_chunks = doc_state.get("completed_chunks", 0)
        # If total_chunks is different from current chunks, reset progress to be safe
        if doc_state.get("total_chunks") != len(chunks):
            completed_chunks = 0

        remaining_chunks = chunks[completed_chunks:]

        if remaining_chunks:
            if progress_callback:
                progress_callback(f"Adding {len(remaining_chunks)} chunks to vector index...")

            from tqdm import tqdm
            pbar = tqdm(
                initial=completed_chunks,
                total=len(chunks),
                desc=f"Embedding {base_name}",
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
                    "status": "in_progress" if completed_chunks < len(chunks) else "completed",
                    "completed_chunks": completed_chunks,
                    "total_chunks": len(chunks),
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

    if new_chunks_added == 0 and vector_db is None:
        logging.warning("No chunks generated and no existing index.")

    return vector_db, indexed_doc_ids
