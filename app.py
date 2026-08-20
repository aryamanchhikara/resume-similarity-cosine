import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from karrou import (
    KarrouDoc,
    KarrouFetchError,
    KarrouNotFound,
    STANDARD_SECTIONS,
    load_karrou,
)
from pdf_extract import PDFDoc, extract_pdf
from similarity import (
    embed_pair,
    embed_texts,
    tfidf_matrix,
    tfidf_pair,
    token_count,
    top_term_contributions,
)
from plots import (
    angle_plot,
    contribution_bars,
    pca_scatter,
    section_heatmap,
)


st.set_page_config(page_title="Karrou similarity, made visible", layout="wide")


# ─── data model ────────────────────────────────────────────────────────────────

@dataclass
class Document:
    label: str
    kind: str                        # "karrou" or "resume"
    full_text: str
    sections: dict                   # {name: body}
    fetch_ms: float
    parse_ms: float
    source_note: str
    pdf_meta: Optional[PDFDoc] = None
    karrou_doc: Optional[KarrouDoc] = None

    def has_sections(self) -> bool:
        return bool(self.sections) and any(v.strip() for v in self.sections.values())


# ─── input widgets ─────────────────────────────────────────────────────────────

def input_slot(slot_key: str, default_username: str = "") -> Optional[Document]:
    mode = st.radio(
        f"Source {slot_key}",
        ["Karrou username", "Upload PDF résumé"],
        key=f"mode_{slot_key}",
        horizontal=True,
    )
    if mode == "Karrou username":
        u = st.text_input(
            f"@username for {slot_key}",
            value=default_username,
            key=f"user_{slot_key}",
            placeholder="e.g. aryamanchhikara",
        )
        if not u.strip():
            return None
        t0 = time.perf_counter()
        try:
            kd = load_karrou(u)
        except KarrouNotFound as e:
            st.error(str(e))
            return None
        except KarrouFetchError as e:
            st.error(f"network error fetching @{u}: {e}")
            return None
        fetch_ms = (time.perf_counter() - t0) * 1000
        return Document(
            label=f"@{kd.username}",
            kind="karrou",
            full_text=kd.full_text,
            sections={s: kd.sections.get(s, "") for s in STANDARD_SECTIONS},
            fetch_ms=fetch_ms,
            parse_ms=0.0,
            source_note=f"karrouhq.com/{kd.username}.md",
            karrou_doc=kd,
        )
    else:
        f = st.file_uploader(
            f"PDF résumé for {slot_key}",
            type=["pdf"],
            key=f"pdf_{slot_key}",
        )
        if f is None:
            return None
        t0 = time.perf_counter()
        pd_ = extract_pdf(f)
        parse_ms = (time.perf_counter() - t0) * 1000
        return Document(
            label=f.name,
            kind="resume",
            full_text=pd_.full_text,
            sections={},
            fetch_ms=0.0,
            parse_ms=parse_ms,
            source_note=f"{f.name} · {pd_.pages} page(s)",
            pdf_meta=pd_,
        )


# ─── UI ────────────────────────────────────────────────────────────────────────

st.title("Karrou similarity, made visible")
st.markdown(
    "Karrou publishes every profile as a canonical machine-readable document at "
    "`karrouhq.com/{username}.md`. That turns *comparing two people* from a "
    "scraping problem into a text problem. This app compares two Karrou "
    "profiles (or two résumés, or one of each) with two cosine methods, "
    "showing every step of the arithmetic — no black boxes."
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("A")
    doc_a = input_slot("A")
with c2:
    st.subheader("B")
    doc_b = input_slot("B")

if not (doc_a and doc_b):
    st.info("Pick a source on both sides to begin.")
    st.stop()

# ─── vectorize & score ────────────────────────────────────────────────────────

with st.spinner("Vectorizing..."):
    t0 = time.perf_counter()
    tfidf = tfidf_pair(doc_a.full_text, doc_b.full_text)
    t_tfidf_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    embed = embed_pair(doc_a.full_text, doc_b.full_text)
    t_embed_ms = (time.perf_counter() - t0) * 1000

# ─── side-by-side scores ─────────────────────────────────────────────────────

st.header("Two methods, side by side")
st.caption(
    "The contrast is the point. TF-IDF asks: *do these documents use the same "
    "words?* Embeddings ask: *do these documents mean similar things?*"
)

sc1, sc2 = st.columns(2)
with sc1:
    st.metric("TF-IDF cosine", f"{tfidf.cosine:.4f}")
    st.caption(
        f"vocabulary: {tfidf.vocab_size} terms · "
        f"non-zero dims A={tfidf.nonzero_per_doc[0]}, B={tfidf.nonzero_per_doc[1]}"
    )
    st.plotly_chart(
        angle_plot(
            tfidf.matrix[0], tfidf.matrix[1],
            doc_a.label, doc_b.label,
            "TF-IDF vectors in their exact spanned 2D plane",
        ),
        use_container_width=True,
    )
    st.caption(
        "Both vectors live in a very high-dimensional TF-IDF space, but any two "
        "vectors span at most a 2D plane. This plot places them inside that "
        "plane via Gram-Schmidt, so the angle you see is exact — not a "
        "dimensionality-reduction approximation."
    )

with sc2:
    st.metric("Embedding cosine", f"{embed.cosine:.4f}")
    st.caption(f"model: {embed.model} · dim = {embed.dim}")
    st.plotly_chart(
        angle_plot(
            embed.vectors[0], embed.vectors[1],
            doc_a.label, doc_b.label,
            "Embedding vectors in their exact spanned 2D plane",
        ),
        use_container_width=True,
    )
    st.caption(
        "Same trick, different vector space. The 384-dim MiniLM embeddings "
        "collapse onto the plane they jointly span — exact angle, again."
    )

# ─── term contributions ─────────────────────────────────────────────────────

st.header("Where the TF-IDF score comes from")
st.caption(
    "Every term either appears in both documents (positive contribution) "
    "or it doesn't (zero). The score is just the sum of the bars below "
    "divided by the product of the two L2 norms."
)
rows = top_term_contributions(tfidf, k=15)
st.plotly_chart(
    contribution_bars(rows, "Top 15 terms by contribution to A·B"),
    use_container_width=True,
)

# ─── section heatmap ────────────────────────────────────────────────────────

st.header("Section-by-section")
if doc_a.has_sections() and doc_b.has_sections():
    labels_a = [s for s in STANDARD_SECTIONS if doc_a.sections.get(s, "").strip()]
    labels_b = [s for s in STANDARD_SECTIONS if doc_b.sections.get(s, "").strip()]
    texts_a = [doc_a.sections[s] for s in labels_a]
    texts_b = [doc_b.sections[s] for s in labels_b]
    all_texts = texts_a + texts_b
    all_vecs = embed_texts(all_texts)
    va = all_vecs[: len(texts_a)]
    vb = all_vecs[len(texts_a):]
    na = np.linalg.norm(va, axis=1, keepdims=True); na[na == 0] = 1
    nb = np.linalg.norm(vb, axis=1, keepdims=True); nb[nb == 0] = 1
    sim = (va / na) @ (vb / nb).T
    st.plotly_chart(
        section_heatmap(sim, labels_a, labels_b, "Cross-section embedding cosine"),
        use_container_width=True,
    )
    st.caption(
        "The diagonal is like-for-like (About↔About, Skills↔Skills). The off-"
        "diagonal is more interesting: one person's Skills lighting up against "
        "another's Experience is a real signal that gets lost in a whole-document score."
    )
else:
    st.info(
        "Section-by-section comparison needs both sides to be Karrou profiles. "
        "PDF résumés don't carry stable section boundaries, so this degrades to "
        "whole-document only."
    )

# ─── optional PCA scatter ───────────────────────────────────────────────────

st.header("Optional: put them in a crowd")
extra = st.text_input(
    "Comma-separated Karrou usernames to include as background points",
    placeholder="alex, sam, jordan",
    help="Fetches each username and embeds it; PCA to 2D. This projection IS lossy — unlike the exact 2D plots above.",
)
if extra.strip():
    others = [u.strip().lstrip("@") for u in extra.split(",") if u.strip()]
    docs_texts = [doc_a.full_text, doc_b.full_text]
    docs_labels = [doc_a.label, doc_b.label]
    for u in others:
        try:
            kd = load_karrou(u)
            docs_texts.append(kd.full_text)
            docs_labels.append(f"@{kd.username}")
        except (KarrouNotFound, KarrouFetchError) as e:
            st.warning(f"skipped @{u}: {e}")
    if len(docs_texts) >= 3:
        vecs = embed_texts(docs_texts)
        st.plotly_chart(
            pca_scatter(vecs, docs_labels, [0, 1], "PCA of embeddings (lossy 2D)"),
            use_container_width=True,
        )
        st.caption("PCA compresses 384 dimensions into 2, so unlike the exact-angle plots above, distances here are approximations.")

# ─── metrics panel ──────────────────────────────────────────────────────────

st.header("Internal metrics — the arithmetic, spelled out")

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown("**TF-IDF**")
    st.write({
        "vocabulary size": tfidf.vocab_size,
        f"non-zero dims [{doc_a.label}]": tfidf.nonzero_per_doc[0],
        f"non-zero dims [{doc_b.label}]": tfidf.nonzero_per_doc[1],
        "‖a‖": round(tfidf.norms[0], 6),
        "‖b‖": round(tfidf.norms[1], 6),
        "a · b": round(tfidf.dot, 6),
        "cos = (a·b) / (‖a‖·‖b‖)": round(tfidf.cosine, 6),
    })
    st.code(
        f"cos = {tfidf.dot:.6f} / ({tfidf.norms[0]:.6f} × {tfidf.norms[1]:.6f})\n"
        f"    = {tfidf.dot:.6f} / {tfidf.norms[0] * tfidf.norms[1]:.6f}\n"
        f"    = {tfidf.cosine:.6f}",
        language="text",
    )

with m2:
    st.markdown("**Embeddings**")
    st.write({
        "model": embed.model,
        "dimensionality": embed.dim,
        "‖a‖": round(embed.norms[0], 6),
        "‖b‖": round(embed.norms[1], 6),
        "a · b": round(embed.dot, 6),
        "cos = (a·b) / (‖a‖·‖b‖)": round(embed.cosine, 6),
    })
    st.code(
        f"cos = {embed.dot:.6f} / ({embed.norms[0]:.6f} × {embed.norms[1]:.6f})\n"
        f"    = {embed.dot:.6f} / {embed.norms[0] * embed.norms[1]:.6f}\n"
        f"    = {embed.cosine:.6f}",
        language="text",
    )

with m3:
    st.markdown("**Per-stage timings (ms)**")
    st.write({
        f"fetch [{doc_a.label}]": round(doc_a.fetch_ms, 1),
        f"fetch [{doc_b.label}]": round(doc_b.fetch_ms, 1),
        f"parse [{doc_a.label}]": round(doc_a.parse_ms, 1),
        f"parse [{doc_b.label}]": round(doc_b.parse_ms, 1),
        "vectorize (tf-idf)": round(t_tfidf_ms, 1),
        "vectorize + score (embed)": round(t_embed_ms, 1),
    })

st.markdown("**Per-section character and token counts**")
rows_ = []
for label, doc in [(doc_a.label, doc_a), (doc_b.label, doc_b)]:
    if doc.has_sections():
        for name in STANDARD_SECTIONS:
            body = doc.sections.get(name, "").strip()
            if body:
                rows_.append({
                    "doc": label,
                    "section": name,
                    "chars": len(body),
                    "tokens (MiniLM)": token_count(body),
                })
    else:
        rows_.append({
            "doc": label,
            "section": "(whole document)",
            "chars": len(doc.full_text),
            "tokens (MiniLM)": token_count(doc.full_text),
        })
st.dataframe(pd.DataFrame(rows_), use_container_width=True, hide_index=True)

# ─── résumé vs karrou quality panel ────────────────────────────────────────

kinds = {doc_a.kind, doc_b.kind}
if kinds == {"karrou", "resume"}:
    st.header("Résumé vs Karrou: the same person, two projections")
    st.caption(
        "A canonical machine-readable projection is cleaner input for any "
        "downstream model than a PDF built for human eyes. That's the reason the "
        "machine view exists — shown as a measurement below, not a claim."
    )
    karrou_doc = doc_a if doc_a.kind == "karrou" else doc_b
    resume_doc = doc_a if doc_a.kind == "resume" else doc_b
    _, k_vocab = tfidf_matrix([karrou_doc.full_text])
    _, r_vocab = tfidf_matrix([resume_doc.full_text])
    q_rows = [
        {"metric": "characters",
         karrou_doc.label: len(karrou_doc.full_text),
         resume_doc.label: len(resume_doc.full_text)},
        {"metric": "tokens (MiniLM)",
         karrou_doc.label: token_count(karrou_doc.full_text),
         resume_doc.label: token_count(resume_doc.full_text)},
        {"metric": "vocabulary size (solo TF-IDF)",
         karrou_doc.label: len(k_vocab),
         resume_doc.label: len(r_vocab)},
        {"metric": "usable sections",
         karrou_doc.label: sum(1 for s in STANDARD_SECTIONS if karrou_doc.sections.get(s, "").strip()),
         resume_doc.label: (len(resume_doc.pdf_meta.detected_sections) if resume_doc.pdf_meta else 0)},
        {"metric": "structural-noise ratio",
         karrou_doc.label: 0.0,
         resume_doc.label: round(resume_doc.pdf_meta.noise_ratio, 3) if resume_doc.pdf_meta else 0.0},
    ]
    st.dataframe(pd.DataFrame(q_rows), use_container_width=True, hide_index=True)
    st.caption(
        "*Structural-noise ratio* counts characters that are page numbers, "
        "contact-line boilerplate, bullet glyphs, and all-caps banners — text "
        "that carries formatting intent for a human reader but no meaning for "
        "a vectorizer. The Karrou machine view has none by construction."
    )

# ─── honest caveats ─────────────────────────────────────────────────────────

st.header("Where each method is wrong")
st.markdown(
    "- **TF-IDF** counts exact tokens. It will score two documents as *unrelated* "
    "when one says \"Postgres\" and the other says \"relational database\", "
    "and it will score two documents as *related* if they share a lot of "
    "common jargon that neither person actually specializes in.\n"
    "- **Embeddings** score meaning, but *meaning* is entangled with register. "
    "Two people who both write in polished consultant-ese can score high without "
    "doing similar work; two people doing the same job in different vocabularies "
    "(one academic, one industry) can score lower than they should.\n"
    "- **Both** methods reduce a person to their text. Neither sees anything about "
    "impact, taste, or ability to actually ship."
)
st.caption(
    "Similarity across a handful of profiles demonstrates the method. It is "
    "not a finding about the people."
)
