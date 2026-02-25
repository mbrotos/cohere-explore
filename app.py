"""
Research Paper Q&A — Adam Sorrenti
===================================
Streamlit app powered by Cohere's retrieval stack:
  Embed (embed-v4.0) → Rerank (rerank-v4.0-pro) → Chat RAG (command-a-03-2025)

Modes
-----
1. **Paper Q&A** – ask anything about the research papers.
2. **Cross-Paper Synthesis** – discover themes that connect multiple papers.

Run:  uv run streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from textwrap import dedent

import cohere
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# ── Config ───────────────────────────────────────────────────────────────────

DOCS_DIR = Path(__file__).parent / "docs"
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

EMBED_MODEL = "embed-v4.0"
RERANK_MODEL = "rerank-v4.0-pro"
CHAT_MODEL = "command-a-03-2025"

CHUNK_SIZE = 1500  # ~375 tokens per chunk — fewer chunks = fewer embed calls
CHUNK_OVERLAP = 200
TOP_K_RETRIEVE = 20  # initial dense retrieval
TOP_N_RERANK = 8  # after rerank


# ── Paper metadata ───────────────────────────────────────────────────────────

PAPERS: dict[str, dict] = {
    "2405.20059v1": {
        "title": "Spectral Mapping of Singing Voices: U-Net-Assisted Vocal Segmentation",
        "short": "SoundSeg",
        "year": 2024,
        "url": "https://arxiv.org/abs/2405.20059",
    },
    "2408.02654v1": {
        "title": "On Using Quasirandom Sequences in Machine Learning for Model Weight Initialization",
        "short": "QRNG Init",
        "year": 2024,
        "url": "https://arxiv.org/abs/2408.02654",
    },
    "2506.23985v1": {
        "title": "Lock Prediction for Zero-Downtime Database Encryption",
        "short": "Lock Prediction",
        "year": 2025,
        "url": "https://arxiv.org/abs/2506.23985",
    },
    "2602.07698v1": {
        "title": "On Sequence-to-Sequence Models for Automated Log Parsing",
        "short": "Log Parsing",
        "year": 2026,
        "url": "https://arxiv.org/abs/2602.07698",
    },
    "Adam_Sorrenti_resume_min": {
        "title": "Adam Sorrenti – Resume",
        "short": "Resume",
        "year": 2026,
        "url": "https://linkedin.com/in/adam-sorrenti",
    },
}


# ── API Key ──────────────────────────────────────────────────────────────────


def _get_api_key() -> str:
    """Load API key from Streamlit secrets (cloud) or .env (local)."""
    # 1. Streamlit Community Cloud secrets
    try:
        key = st.secrets.get("CO_API_KEY")
        if key:
            return key
    except Exception:
        pass
    # 2. Local .env file
    load_dotenv()
    key = os.getenv("CO_API_KEY", "")
    if key:
        return key
    # 3. Not found
    st.error(
        "**Missing `CO_API_KEY`.**\n\n"
        "- **Local:** Create a `.env` file with `CO_API_KEY=your_key`\n"
        "- **Cloud:** Add `CO_API_KEY` to Streamlit secrets"
    )
    st.stop()
    return ""  # unreachable, keeps type checker happy


# ── Chunking ─────────────────────────────────────────────────────────────────


def _load_document(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks on paragraph boundaries."""
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 1 > chunk_size and current:
            chunks.append(current.strip())
            # keep tail as overlap seed
            current = current[-overlap:] + "\n" + para if overlap else para
        else:
            current = current + "\n" + para if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks


@st.cache_data(show_spinner="Loading and chunking papers…")
def load_all_chunks() -> tuple[list[str], list[dict]]:
    """Return (chunks, metadata) for every document in docs/."""
    all_chunks: list[str] = []
    all_meta: list[dict] = []
    for txt_file in sorted(DOCS_DIR.glob("*.txt")):
        doc_id = txt_file.stem
        info = PAPERS.get(doc_id, {"title": doc_id, "short": doc_id, "year": 0})
        text = _load_document(txt_file)
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_meta.append(
                {
                    "doc_id": doc_id,
                    "title": info["title"],
                    "short": info["short"],
                    "year": info["year"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            )
    return all_chunks, all_meta


# ── Embedding ────────────────────────────────────────────────────────────────


def _cache_key(chunks: list[str]) -> str:
    h = hashlib.sha256("\n".join(chunks).encode()).hexdigest()[:16]
    return h


def _embed_with_retry(
    co: cohere.ClientV2,
    texts: list[str],
    input_type: str,
    max_retries: int = 5,
) -> np.ndarray:
    """Call co.embed with exponential backoff on 429 rate-limit errors."""
    for attempt in range(max_retries):
        try:
            resp = co.embed(
                model=EMBED_MODEL,
                texts=texts,
                input_type=input_type,
                embedding_types=["float"],
            )
            return np.array(resp.embeddings.float_)
        except cohere.errors.TooManyRequestsError:
            wait = 2 ** attempt * 10  # 10s, 20s, 40s, 80s, 160s
            st.toast(f"Rate limited — waiting {wait}s before retry ({attempt + 1}/{max_retries})…")
            time.sleep(wait)
    raise RuntimeError("Embedding failed after max retries — trial rate limit exceeded.")


def _embed_batch(
    co: cohere.ClientV2,
    texts: list[str],
    input_type: str,
    batch_size: int = 16,
    pause: float = 12.0,
) -> np.ndarray:
    """Embed a list of texts with rate-limit-friendly batching.

    Uses small batches (16 texts) with generous pauses and automatic
    retry with exponential backoff to stay within trial-tier limits
    (100k tokens/min). Embeddings are cached to disk after completion
    so this only runs once.
    """
    arrays = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    progress = st.progress(0, text="Embedding chunks (one-time)…")
    for idx, i in enumerate(range(0, len(texts), batch_size)):
        progress.progress(
            (idx + 1) / total_batches,
            text=f"Embedding batch {idx + 1}/{total_batches} (one-time, cached after)…",
        )
        emb = _embed_with_retry(co, texts[i : i + batch_size], input_type)
        arrays.append(emb)
        # pause between batches to respect rate limits (skip after last)
        if idx < total_batches - 1:
            time.sleep(pause)
    progress.empty()
    return np.vstack(arrays)


@st.cache_resource(show_spinner="Embedding document chunks with Cohere…")
def get_chunk_embeddings(_co: cohere.ClientV2, chunks: tuple[str, ...]) -> np.ndarray:
    """Embed all chunks (cached to disk + Streamlit cache)."""
    key = _cache_key(list(chunks))
    cache_path = CACHE_DIR / f"embeddings_{key}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    emb = _embed_batch(_co, list(chunks), input_type="search_document")
    np.save(cache_path, emb)
    return emb


def embed_query(co: cohere.ClientV2, query: str) -> np.ndarray:
    resp = co.embed(
        model=EMBED_MODEL,
        texts=[query],
        input_type="search_query",
        embedding_types=["float"],
    )
    return np.array(resp.embeddings.float_[0])


# ── Retrieval ────────────────────────────────────────────────────────────────


def cosine_similarity(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity between vector a and matrix B."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return B_norm @ a_norm


def retrieve_and_rerank(
    co: cohere.ClientV2,
    query: str,
    chunks: list[str],
    meta: list[dict],
    embeddings: np.ndarray,
    top_k: int = TOP_K_RETRIEVE,
    top_n: int = TOP_N_RERANK,
    paper_filter: list[str] | None = None,
) -> list[dict]:
    """Dense retrieval → Rerank → return top results with metadata."""
    # optional paper filter
    if paper_filter:
        indices = [i for i, m in enumerate(meta) if m["doc_id"] in paper_filter]
    else:
        indices = list(range(len(chunks)))

    if not indices:
        return []

    sub_embeddings = embeddings[indices]
    q_emb = embed_query(co, query)
    sims = cosine_similarity(q_emb, sub_embeddings)
    top_idx = np.argsort(sims)[::-1][:top_k]

    # map back to global indices
    candidate_global = [indices[i] for i in top_idx]
    candidate_chunks = [chunks[i] for i in candidate_global]

    # rerank
    rerank_resp = co.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=candidate_chunks,
        top_n=top_n,
    )

    results = []
    for r in rerank_resp.results:
        gi = candidate_global[r.index]
        results.append(
            {
                "text": chunks[gi],
                "meta": meta[gi],
                "relevance": r.relevance_score,
            }
        )
    return results


# ── Chat with RAG ────────────────────────────────────────────────────────────


def chat_rag(co: cohere.ClientV2, query: str, documents: list[dict], system: str = "") -> dict:
    """Chat with RAG documents. Returns {text, citations}."""
    doc_payloads = []
    for i, doc in enumerate(documents):
        doc_payloads.append(
            {
                "id": f"doc_{i}",
                "data": {
                    "snippet": doc["text"],
                    "title": doc["meta"]["title"],
                    "paper": doc["meta"]["short"],
                },
            }
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": query})

    resp = co.chat(
        model=CHAT_MODEL,
        messages=messages,
        documents=doc_payloads,
    )

    text = resp.message.content[0].text if resp.message.content else ""
    citations = []
    if resp.message.citations:
        for c in resp.message.citations:
            sources = []
            if c.sources:
                for s in c.sources:
                    doc_id = getattr(s, "id", None) or getattr(s, "document_id", None) or ""
                    sources.append(doc_id)
            citations.append(
                {
                    "start": c.start,
                    "end": c.end,
                    "text": text[c.start : c.end],
                    "sources": sources,
                }
            )

    return {"text": text, "citations": citations, "documents": doc_payloads}


# ── Streamlit UI ─────────────────────────────────────────────────────────────


def _source_label(doc_payload: dict, doc_id: str) -> str:
    """Human-readable label for a cited document."""
    idx = int(doc_id.split("_")[1]) if doc_id.startswith("doc_") else 0
    if idx < len(doc_payload):
        return doc_payload[idx]["data"]["paper"]
    return doc_id


def render_citations(result: dict):
    """Show cited sources in an expander."""
    if not result["citations"]:
        return
    st.markdown("---")
    st.markdown("**Cited Sources**")
    seen = set()
    for c in result["citations"]:
        for src in c["sources"]:
            if src in seen:
                continue
            seen.add(src)
            label = _source_label(result["documents"], src)
            idx = int(src.split("_")[1]) if src.startswith("doc_") else 0
            if idx < len(result["documents"]):
                with st.expander(f"📄 {label}"):
                    st.caption(result["documents"][idx]["data"]["title"])
                    st.text(result["documents"][idx]["data"]["snippet"][:600] + "…")


def main():
    st.set_page_config(page_title="Research Q&A — Adam Sorrenti", page_icon="🔬", layout="wide")

    st.title("🔬 Research Paper Q&A")
    st.caption(
        "Powered by Cohere **embed-v4.0 → rerank-v4.0-pro → command-a-03-2025** · "
        "Ask questions about Adam Sorrenti's research papers or synthesize themes across them. "
        "· [GitHub](https://github.com/mbrotos/cohere-explore)"
    )

    # ── API key: Streamlit secrets (cloud) → .env (local) ────────────
    api_key = _get_api_key()

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Mode")
        mode = st.radio(
            "Choose a mode:",
            ["📖 Paper Q&A", "🔗 Cross-Paper Synthesis"],
            index=0,
            label_visibility="collapsed",
        )
        if mode == "📖 Paper Q&A":
            st.caption(
                "Ask any question about the papers below. "
                "Relevant passages are retrieved, reranked, and "
                "used as context for a grounded, cited answer."
            )
        else:
            st.caption(
                "Discover shared themes, contrasting methods, and "
                "overarching narratives across two or more papers. "
                "Chunks are retrieved from each selected paper for "
                "balanced cross-document reasoning."
            )

        st.markdown("---")
        st.subheader("Papers")
        selected_papers: list[str] = []
        for doc_id, info in PAPERS.items():
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.checkbox(
                    f"{info['short']} ({info['year']})",
                    value=True,
                    key=f"cb_{doc_id}",
                ):
                    selected_papers.append(doc_id)
            with col2:
                url = info.get("url", "")
                if url:
                    st.link_button("🔗", url, help=info["title"])

        st.markdown("---")
        st.subheader("Retrieval")
        top_k = st.slider("Dense retrieval (top-k)", 5, 50, TOP_K_RETRIEVE)
        top_n = st.slider("Rerank (top-n)", 2, 20, TOP_N_RERANK)

    co = cohere.ClientV2(api_key=api_key)

    # ── Load data ────────────────────────────────────────────────────────
    chunks, meta = load_all_chunks()
    embeddings = get_chunk_embeddings(co, tuple(chunks))

    # ── Mode: Paper Q&A ─────────────────────────────────────────────────
    if mode == "📖 Paper Q&A":
        st.markdown("### Ask a question about the research papers")

        example_qs = [
            "What architectures were compared for log parsing and which performed best?",
            "How does the U-Net model separate vocals from music?",
            "What is the role of QRNG in neural network weight initialization?",
            "How do Transformer and LSTM models compare for lock sequence prediction?",
            "What datasets were used across the different studies?",
        ]
        with st.expander("💡 Example questions"):
            for q in example_qs:
                st.markdown(f"- {q}")

        query = st.text_input("Your question:", key="qa_input")

        if query:
            with st.spinner("Retrieving & reranking…"):
                results = retrieve_and_rerank(
                    co, query, chunks, meta, embeddings,
                    top_k=top_k, top_n=top_n,
                    paper_filter=selected_papers or None,
                )

            if not results:
                st.warning("No relevant chunks found. Try broadening your paper filter.")
                return

            system_prompt = dedent("""\
                You are a research assistant for Adam Sorrenti's academic work.
                Answer the user's question using ONLY the provided documents.
                Be precise, cite specific results (metrics, numbers) when available,
                and indicate which paper each fact comes from.
            """)

            with st.spinner("Generating answer with Cohere RAG…"):
                result = chat_rag(co, query, results, system=system_prompt)

            st.markdown("### Answer")
            st.markdown(result["text"])
            render_citations(result)

            with st.expander("🔍 Retrieved chunks (debug)"):
                for i, r in enumerate(results):
                    st.markdown(f"**[{i+1}] {r['meta']['short']}** (relevance: {r['relevance']:.3f})")
                    st.text(r["text"][:400] + "…")
                    st.markdown("---")

    # ── Mode: Cross-Paper Synthesis ──────────────────────────────────────
    elif mode == "🔗 Cross-Paper Synthesis":
        st.markdown("### Cross-Paper Synthesis")
        st.markdown(
            "Discover themes, shared techniques, and connections across multiple papers. "
            "Select at least two papers in the sidebar."
        )

        synthesis_prompts = [
            "What deep learning architectures appear across these papers and how are they used differently?",
            "What shared evaluation methodologies or metrics connect these studies?",
            "How do these papers collectively advance the state of sequence modeling?",
            "What common challenges or limitations are discussed across this body of work?",
            "Summarize each paper in 2-3 sentences, then identify overarching research themes.",
        ]

        preset = st.selectbox("Choose a synthesis prompt or write your own:", ["Custom…"] + synthesis_prompts)
        if preset == "Custom…":
            query = st.text_area("Your synthesis question:", height=100, key="synth_input")
        else:
            query = preset

        if st.button("Synthesize", type="primary") and query:
            if len(selected_papers) < 2:
                st.warning("Select at least 2 papers for cross-paper synthesis.")
                return

            # Retrieve top chunks per paper to ensure balanced representation
            per_paper_n = max(2, top_n // len(selected_papers))
            all_results: list[dict] = []

            with st.spinner("Retrieving from each paper…"):
                for doc_id in selected_papers:
                    paper_results = retrieve_and_rerank(
                        co, query, chunks, meta, embeddings,
                        top_k=top_k, top_n=per_paper_n,
                        paper_filter=[doc_id],
                    )
                    all_results.extend(paper_results)

            if not all_results:
                st.warning("No relevant chunks found.")
                return

            system_prompt = dedent("""\
                You are a research synthesis assistant analyzing Adam Sorrenti's body
                of academic work. Your job is to find connections, shared themes,
                contrasts, and overarching narratives across multiple papers.

                Guidelines:
                - Reference specific papers by name when making claims.
                - Highlight both similarities and differences in methods, architectures,
                  datasets, and findings.
                - Be analytical, not just descriptive — draw insight from the connections.
                - Use specific numbers and results where relevant.
            """)

            with st.spinner("Synthesizing across papers…"):
                result = chat_rag(co, query, all_results, system=system_prompt)

            st.markdown("### Synthesis")
            st.markdown(result["text"])
            render_citations(result)

            # Show which papers contributed
            contributing = set(r["meta"]["short"] for r in all_results)
            st.markdown("---")
            st.markdown(f"**Papers included:** {', '.join(sorted(contributing))}")

            with st.expander("🔍 Retrieved chunks (debug)"):
                for i, r in enumerate(all_results):
                    st.markdown(f"**[{i+1}] {r['meta']['short']}** (relevance: {r['relevance']:.3f})")
                    st.text(r["text"][:400] + "…")
                    st.markdown("---")


if __name__ == "__main__":
    main()
