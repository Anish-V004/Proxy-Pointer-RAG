"""
DocComparator: Centralized Configuration

All paths are relative to the project root. Override via .env or environment variables.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai

# ── Project Root ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent

# ── Environment ─────────────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ── Paths ───────────────────────────────────────────────────────────────
UPLOADS_DIR   = Path(os.getenv("DC_UPLOADS_DIR",   PROJECT_ROOT / "data" / "uploads"))
DOCUMENTS_DIR = Path(os.getenv("DC_DOCUMENTS_DIR", PROJECT_ROOT / "data" / "documents"))
TREES_DIR     = Path(os.getenv("DC_TREES_DIR",     PROJECT_ROOT / "data" / "trees"))
INDEX_DIR     = Path(os.getenv("DC_INDEX_DIR",      PROJECT_ROOT / "data" / "index"))

# ── LlamaParse ──────────────────────────────────────────────────────────
LLAMA_PARSE_TIER = os.getenv("LLAMA_PARSE_TIER", "cost_effective")

# ── Models ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL    = "models/gemini-embedding-001"
EMBEDDING_DIMS     = 1536
LLM_MODEL          = "gemini-3-flash-preview"

# ── Chunking ────────────────────────────────────────────────────────────
CHUNK_SIZE    = 2000
CHUNK_OVERLAP = 200

# ── Comparison Limits ──────────────────────────────────────────────────
MAX_DOC1_SECTIONS = 10
MAX_DOC2_MATCHES  = 3
