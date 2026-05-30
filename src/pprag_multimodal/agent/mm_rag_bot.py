"""
Multimodal Proxy-Pointer: Structural RAG Bot with Image Selection

Interactive RAG bot that:
  1. Vector search (k=200) for broad recall
  2. Deduplicates by (doc_id, node_id)
  3. LLM re-ranker selects top 5 by hierarchical path + semantic snippets
  4. Loads full section text from Markdown via line-slicing
  5. Extracts image anchors from skeleton tree JSON
  6. LLM synthesizer generates grounded answers + selects relevant images
  7. Optional vision filter validates image relevance

Usage:
    python -m src.agent.mm_rag_bot
"""
import os
import re
import json
import logging
from collections.abc import Sequence as SequenceABC
from pathlib import Path
# pyrefly: ignore [missing-import]
from PIL import Image


import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from pprag_multimodal.config import DATASET_DIR, TREES_DIR, INDEX_DIR, EMBEDDING_MODEL, EMBEDDING_DIMS, SYNTH_MODEL, VISION_FILTER
from pprag.gemini_embeddings import is_rate_limit_error, extract_retry_delay
import time
from pprag.faiss_security import require_trusted_faiss_deserialization
from pprag_multimodal.indexing.md_tree_builder import get_md_path_for_doc


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


def _parse_non_negative_int(value, default=0):
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _safe_join_under(base_dir, relative_name):
    if not relative_name:
        return None
    norm_name = os.path.normpath(str(relative_name))
    if os.path.isabs(norm_name) or norm_name.startswith(".."):
        return None
    base_real = os.path.realpath(base_dir)
    full_real = os.path.realpath(os.path.join(base_real, norm_name))
    try:
        if os.path.commonpath([base_real, full_real]) != base_real:
            return None
    except ValueError:
        return None
    return full_real


def _load_trusted_faiss_index(index_path, embeddings):
    """Load locally generated FAISS metadata; never pass untrusted indexes here."""
    require_trusted_faiss_deserialization(index_path, "PP_TRUST_FAISS_INDEX")
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ── Custom Embedding Wrapper ────────────────────────────────────────────
class GeminiEmbeddings(Embeddings):
    """LangChain-compatible wrapper for Gemini embeddings at configurable dims."""

    def __init__(self, model=EMBEDDING_MODEL, dimensionality=EMBEDDING_DIMS):
        self.model = model
        self.dimensionality = dimensionality

    def embed_documents(self, texts):
        res = genai.embed_content(model=self.model, content=texts, output_dimensionality=self.dimensionality)
        return _normalize_embeddings(res, len(texts))

    def embed_query(self, text):
        res = genai.embed_content(model=self.model, content=text, output_dimensionality=self.dimensionality)
        return _normalize_embeddings(res, 1)[0]


class MultimodalProxyPointerRAG:
    def __init__(self, index_path=None, trees_dir=None, dataset_dir=None):
        self.dataset_dir = str(dataset_dir or DATASET_DIR)
        self.trees_dir = str(trees_dir or TREES_DIR)
        index_path = str(index_path or INDEX_DIR)

        # 1. Load Gemini Embeddings
        print(f"Loading {EMBEDDING_MODEL} @ {EMBEDDING_DIMS} dims...")
        self.embeddings = GeminiEmbeddings()

        # 2. Load FAISS Index
        print(f"Loading index from {index_path}...")
        self.vector_db = _load_trusted_faiss_index(index_path, self.embeddings)

        # 3. Initialize models
        self.llm = genai.GenerativeModel(SYNTH_MODEL)
        self.llm_timeout = float(os.getenv("PPRAG_LLM_TIMEOUT", "120"))
        self._md_path_cache = {}
        self._md_lines_cache = {}
        self._tree_node_cache = {}

    def _generate_content(self, *args, **kwargs):
        """Call self.llm.generate_content with retry logic on 429 or empty responses."""
        attempt = 0
        max_retries = 3
        base_delay = 2.0

        while True:
            try:
                try:
                    response = self.llm.generate_content(
                        *args, request_options={"timeout": self.llm_timeout}, **kwargs
                    )
                except TypeError:
                    response = self.llm.generate_content(*args, **kwargs)

                # Check if the response contains valid text parts
                # If there are candidates but no parts, raise a transient error to trigger retry
                if hasattr(response, "candidates") and (
                    not response.candidates
                    or not response.candidates[0].content
                    or not response.candidates[0].content.parts
                ):
                    raise RuntimeError("Transient generation failure: Gemini response has no valid content parts.")

                return response

            except Exception as exc:
                is_transient = "Transient generation failure" in str(exc)
                if not is_transient and not is_rate_limit_error(exc):
                    raise

                retry_delay = extract_retry_delay(exc)
                if retry_delay is not None:
                    if attempt >= max_retries - 1:
                        raise
                    delay = retry_delay
                    logging.warning(
                        "Rate limit hit during LLM generation. Found retry delay of %ss from error. Retrying in %ss (attempt %s/%s)...",
                        retry_delay,
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                else:
                    # Exponential backoff sequence: base_delay -> double -> cap at 60 -> another 60
                    backoff_delays = []
                    current_delay = base_delay
                    while current_delay < 60.0:
                        backoff_delays.append(current_delay)
                        current_delay *= 2.0
                    backoff_delays.append(60.0)
                    backoff_delays.append(60.0)

                    if attempt >= len(backoff_delays):
                        raise

                    delay = backoff_delays[attempt]
                    logging.warning(
                        "Rate limit or generation failure hit during LLM generation. "
                        "Waiting %ss before retry (attempt %s/%s)...",
                        delay,
                        attempt + 1,
                        len(backoff_delays) + 1,
                    )

                attempt += 1
                time.sleep(delay)

    # ── RAG Pipeline Features ────────────────────────────────────────────────
    VISION_FILTER = VISION_FILTER  # Controlled from config.py

    @staticmethod
    def _index_tree_nodes(node_list):
        node_map = {}
        for node in node_list:
            node_id = node.get("node_id")
            if node_id:
                node_map[node_id] = node
            if node.get("nodes"):
                node_map.update(MultimodalProxyPointerRAG._index_tree_nodes(node["nodes"]))
        return node_map

    def _get_md_path(self, doc_id):
        if doc_id not in self._md_path_cache:
            self._md_path_cache[doc_id] = get_md_path_for_doc(self.dataset_dir, doc_id)
        return self._md_path_cache[doc_id]

    def _get_md_lines(self, doc_id):
        if doc_id in self._md_lines_cache:
            return self._md_lines_cache[doc_id]

        md_path = self._get_md_path(doc_id)
        if not md_path:
            self._md_lines_cache[doc_id] = None
            return None

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except (OSError, UnicodeDecodeError) as exc:
            logging.warning("Unable to load markdown context for %s: %s", doc_id, exc)
            lines = None
        self._md_lines_cache[doc_id] = lines
        return lines

    def _get_tree_node_map(self, doc_id):
        if doc_id in self._tree_node_cache:
            return self._tree_node_cache[doc_id]

        tree_path = os.path.join(self.trees_dir, f"{doc_id}_structure.json")
        try:
            with open(tree_path, "r", encoding="utf-8") as f:
                tree_data = json.load(f)
            node_map = self._index_tree_nodes(tree_data.get("structure", []))
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Unable to load tree context for %s: %s", doc_id, exc)
            node_map = {}
        self._tree_node_cache[doc_id] = node_map
        return node_map

    def retrieve_unique_nodes(self, query, k_search=200, k_final=5):
        """Stage 1: Broad vector recall → Stage 2: LLM structural re-ranking."""

        # Stage 1: Broad Recall
        docs = self.vector_db.similarity_search(query, k=k_search)

        candidates = []
        seen_nodes = set()   # (doc_id, node_id)
        for doc in docs:
            node_id = doc.metadata.get("node_id")
            doc_id = doc.metadata.get("doc_id", "UNK")
            dedup_key = (doc_id, node_id)
            if dedup_key not in seen_nodes:
                seen_nodes.add(dedup_key)
                info = {
                    "node_id": node_id,
                    "doc_id": doc_id,
                    "breadcrumb": doc.metadata.get("breadcrumb", "Unknown Path"),
                    "snippet": doc.page_content[:150].replace("\n", " "),
                    "full_metadata": doc.metadata
                }
                candidates.append(info)

        # Stage 2: LLM Re-Ranker (Anchor-Aware + Semantic Snippets)
        candidate_subset = candidates[:50]
        index_map = {str(i): c for i, c in enumerate(candidate_subset)}
        candidates_text = ""
        for i, c in enumerate(candidate_subset):
            candidates_text += f"{i}. [{c['doc_id']} > {c['breadcrumb']}] | Preview: {c['snippet']}...\n"

        prompt = f"""You are a structural & semantic re-ranker for technical research papers.
Your goal is to find the Top {k_final} most relevant sections based on their HIERARCHICAL PATH and the content snippets provided.

User Query: "{query}"

CANDIDATES (INDEX | Path | Snippet):
{candidates_text}

RANKING RULES:
0. DOCUMENT MATCHING: If the User Query explicitly names a specific document (e.g. 'Paper 1', 'Company A'), absolutely exclude or heavily penalize candidates from other documents.
1. ANCHOR AWARENESS: If the query mentions a specific anchor like 'Figure 5' or 'Table I', prioritize sections that physically contain that reference.
2. TECHNICAL SPECIFICITY: Prioritize technical deep-dives (e.g. 'Methodology', 'Experiments') over generic introductions.
3. CONTEXTUAL RELEVANCE: Match the query's technical terms to the content snippets.
4. Each INDEX must appear ONLY ONCE.
5. Output ONLY a comma-separated list of up to {k_final} unique numeric indices that are actually relevant. If fewer are relevant, output fewer. No text.

Output Example: 4, 12, 0
"""
        try:
            response = self._generate_content(prompt).text.strip()
            clean_text = re.sub(r"[^0-9, ]", "", response)
            ranked_ids = [rid.strip() for rid in clean_text.split(",") if rid.strip()]

            final_pointers = []
            seen = set()
            for rid in ranked_ids:
                if rid in index_map and rid not in seen:
                    final_pointers.append(index_map[rid])
                    seen.add(rid)
                if len(final_pointers) >= k_final:
                    break

            if final_pointers:
                return final_pointers
        except Exception as e:
            print(f"WARNING: LLM Ranker Failed ({e}). Falling back to top {k_final}.")

        return candidate_subset[:k_final]

    def chat(self, query):
        """Orchestrate Retrieval, Multimodal Synthesis, and UI Parsing."""
        pointers = self.retrieve_unique_nodes(query)

        context_blocks = []
        found_images = []

        for p in pointers:
            # 1. Load full markdown text
            lines = self._get_md_lines(p['doc_id'])
            if lines:
                start = _parse_non_negative_int(p['full_metadata'].get("start_line", 0))
                end = _parse_non_negative_int(p['full_metadata'].get("end_line", start))
                safe_start = max(0, min(len(lines), start))
                safe_end = max(safe_start, min(len(lines), end))
                section_text = "".join(lines[safe_start:safe_end]) or p['snippet']
            else:
                section_text = f"(Full text missing) {p['snippet']}"

            context_blocks.append(f"### REFERENCE: {p['doc_id']} > {p['breadcrumb']}\n{section_text}")

            # 2. Extract specific image anchors from tree JSON
            target_node = self._get_tree_node_map(p['doc_id']).get(p['node_id'])
            if target_node and target_node.get("figures"):
                doc_folder = os.path.join(self.dataset_dir, p['doc_id'])
                for fig in target_node["figures"]:
                    safe_img_path = _safe_join_under(doc_folder, fig.get("filename"))
                    if safe_img_path is None:
                        logging.warning("Skipping unsafe figure path for %s: %r", p['doc_id'], fig.get("filename"))
                        continue
                    try:
                        rel_path = os.path.join(p['doc_id'], os.path.relpath(safe_img_path, doc_folder)).replace("\\", "/")
                    except ValueError:
                        rel_path = os.path.join(p['doc_id'], os.path.basename(safe_img_path)).replace("\\", "/")
                    full_img_path = safe_img_path.replace("\\", "/")

                    clean_doc_id = str(p['doc_id']).strip()
                    clean_fig_label = str(fig.get('label', 'Figure')).strip()

                    found_images.append({
                        "label": f"{clean_doc_id} - {clean_fig_label}",
                        "relative_path": rel_path,
                        "full_path": full_img_path,
                        "exists": os.path.exists(full_img_path)
                    })

        context_str = "\n".join(context_blocks)
        synth_prompt = f"""You are an advanced Multimodal RAG Assistant.

Query: "{query}"

Context:
{context_str}

INSTRUCTIONAL RULES:
1. Answer the query concisely using ONLY the provided context.
2. If the query asks about a specific document (e.g. Paper 1), ONLY use context and images from that specific document, ignoring the others.
3. If the context contains a table or figure anchor that answers the query, explicitly mention its ID (e.g. Figure 5).
4. Do NOT reference internal node IDs (e.g. 'node: 0045') or breadcrumb segments in the body of the answer.
5. IMAGE SELECTION: If a figure or table mentioned in the context is relevant to the asked query, list its doc-qualified relative path (`doc_id/filename`) and a SHORT caption (including Figure/Table number) in brackets: [SHOW: doc_id/filename | short caption]. Provide ONE image per bracket. Limit to TOP 6 most relevant.

Output format:
[Answer Text]

[SHOW: paper_id/figure1.png | Figure 1: Short caption text]
"""
        generation_config = genai.GenerationConfig(temperature=0.0)
        response = self._generate_content(synth_prompt, generation_config=generation_config)
        answer_text = response.text

        # 4. Extract filenames and labels, apply Vision Filter
        requested_images = []
        img_matches = re.findall(r"\[SHOW:\s*([^\|\]]+)(?:\|\s*([^\]]+))?\]", answer_text, re.I)
        requested_filenames = []
        llm_labels = {}
        for fname, label in img_matches:
            requested_key = fname.strip().replace("\\", "/").lower()
            clean_fname = os.path.basename(requested_key)
            requested_filenames.append((requested_key, clean_fname))
            if label:
                llm_labels[requested_key] = label.strip()

        for requested_key, clean_fname in requested_filenames:
            for meta in found_images:
                meta_rel = str(meta.get("relative_path", "")).strip().replace("\\", "/").lower()
                meta_fname = os.path.basename(meta["full_path"]).strip().lower()
                if ("/" in requested_key and requested_key == meta_rel) or (
                    "/" not in requested_key and clean_fname == meta_fname
                ):
                    if meta not in requested_images:
                        # Use LLM description if available, otherwise keep original
                        if requested_key in llm_labels:
                            original_label = meta.get("label", "")
                            if isinstance(original_label, str) and " - " in original_label:
                                doc_prefix = original_label.rsplit(" - ", 1)[0]
                            else:
                                doc_prefix = str(original_label or meta_fname)
                            meta["label"] = f"{doc_prefix} - {llm_labels[requested_key]}"
                        requested_images.append(meta)
                    break

        # --- STAGE 4: Final Vision Filter (Optional) ---
        final_verified_images = []
        if requested_images and self.VISION_FILTER:
            try:
                # Filter only existing images
                valid_images = [img for img in requested_images if img["exists"]]
                if valid_images:
                    # Prepare Batch Vision Prompt
                    vision_prompt = f"""You are a MultiModal Verification Agent.
User Query: "{query}"
Proposed Answer: "{answer_text}"

Below are {len(valid_images)} images the text-model selected.
For each image, determine if it is relevant evidence for the query and answer.
Return a simple comma-separated list of ONLY the indices (e.g., 0, 2) of the images that are RELEVANT.
If none are relevant, return 'NONE'.
"""
                    # Attach all images to the prompt
                    content_list = [vision_prompt]
                    for img in valid_images:
                        with Image.open(img["full_path"]) as im:
                            content_list.append(im.copy())

                    v_res = self._generate_content(content_list)
                    res_text = v_res.text.upper()

                    if "NONE" not in res_text:
                        # Extract indices
                        indices = re.findall(r"\d+", res_text)
                        for idx_str in indices:
                            idx = int(idx_str)
                            if 0 <= idx < len(valid_images):
                                final_verified_images.append(valid_images[idx])
                    else:
                        final_verified_images = [] # None verified
                else:
                    final_verified_images = []
            except Exception as e:
                logging.warning("Vision verification failed; suppressing unverified images: %s", e)
                final_verified_images = []
        else:
            # Skip verification pass for benchmark speed
            final_verified_images = requested_images

        # Deduplicate paths for clean citation
        unique_paths = []
        seen_paths = set()
        for p in pointers:
            path_str = f"{p['doc_id']} > {p['breadcrumb']}"
            if path_str not in seen_paths:
                unique_paths.append(path_str)
                seen_paths.add(path_str)

        return {
            "text": answer_text,
            "images": final_verified_images,
            "paths": unique_paths
        }
