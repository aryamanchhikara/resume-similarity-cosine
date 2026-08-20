from dataclasses import dataclass
from typing import Sequence

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class TfidfResult:
    vocab: list[str]
    matrix: np.ndarray            # shape (n_docs, vocab)
    norms: list[float]
    dot: float
    cosine: float
    nonzero_per_doc: list[int]
    vocab_size: int


@dataclass
class EmbedResult:
    vectors: np.ndarray           # shape (n_docs, dim)
    norms: list[float]
    dot: float
    cosine: float
    dim: int
    model: str


def _safe_cosine(u: np.ndarray, v: np.ndarray) -> tuple[float, float, float, float]:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    dot = float(np.dot(u, v))
    if nu == 0.0 or nv == 0.0:
        return dot, nu, nv, 0.0
    return dot, nu, nv, dot / (nu * nv)


def tfidf_pair(text_a: str, text_b: str) -> TfidfResult:
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.\-]{1,}\b",
        min_df=1,
    )
    try:
        m = vec.fit_transform([text_a or " ", text_b or " "]).toarray()
        vocab = vec.get_feature_names_out().tolist()
    except ValueError:
        # both docs contain only stopwords / whitespace — vocab is empty
        m = np.zeros((2, 0), dtype=np.float64)
        vocab = []
    u, v = m[0], m[1]
    dot, nu, nv, cos = _safe_cosine(u, v)
    nonzero = [int((u != 0).sum()), int((v != 0).sum())]
    return TfidfResult(
        vocab=vocab,
        matrix=m,
        norms=[nu, nv],
        dot=dot,
        cosine=cos,
        nonzero_per_doc=nonzero,
        vocab_size=len(vocab),
    )


def tfidf_matrix(texts: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.\-]{1,}\b",
        min_df=1,
    )
    try:
        m = vec.fit_transform([t or " " for t in texts]).toarray()
        return m, vec.get_feature_names_out().tolist()
    except ValueError:
        return np.zeros((len(texts), 0), dtype=np.float64), []


def cosine_matrix(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = mat / norms
    return normed @ normed.T


@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    model = load_embedder()
    vecs = model.encode(list(texts), normalize_embeddings=False, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def embed_pair(text_a: str, text_b: str) -> EmbedResult:
    vecs = embed_texts([text_a or " ", text_b or " "])
    u, v = vecs[0], vecs[1]
    dot, nu, nv, cos = _safe_cosine(u, v)
    return EmbedResult(
        vectors=vecs,
        norms=[nu, nv],
        dot=dot,
        cosine=cos,
        dim=int(vecs.shape[1]),
        model=EMBED_MODEL_NAME,
    )


def top_term_contributions(res: TfidfResult, k: int = 15) -> list[tuple[str, float, float, float]]:
    """Return list of (term, weight_a, weight_b, contribution) sorted by contribution."""
    u, v = res.matrix[0], res.matrix[1]
    contribs = u * v
    idx = np.argsort(-contribs)
    out: list[tuple[str, float, float, float]] = []
    for i in idx:
        c = float(contribs[i])
        if c <= 0:
            break
        out.append((res.vocab[i], float(u[i]), float(v[i]), c))
        if len(out) >= k:
            break
    return out


def token_count(text: str) -> int:
    """Approx token count using MiniLM's tokenizer without loading the full model."""
    try:
        from transformers import AutoTokenizer
        tok = _cached_tokenizer()
        return len(tok.encode(text, add_special_tokens=False))
    except Exception:
        return len(text.split())


@st.cache_resource(show_spinner=False)
def _cached_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
