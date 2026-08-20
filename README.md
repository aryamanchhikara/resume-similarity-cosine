# resume-similarity-cosine

Cosine similarity over [Karrou](https://karrouhq.com) profiles, made visible
end to end.

Karrou publishes every profile as a canonical machine-readable document at
`karrouhq.com/{username}.md`. That turns *comparing two people* from a
scraping problem into a text problem. This app takes two Karrou profiles (or
two résumés, or one of each) and shows exactly how the similarity score is
computed — every vector, every norm, every division.

**Live app:** _add your Streamlit Community Cloud URL here after deploy_

## What it does

- Fetches `karrouhq.com/{username}.md`, or extracts text from an uploaded
  PDF résumé (in memory, never to disk).
- Parses the machine view into its H2 sections: About, Experience, Skills,
  Work, Widgets, Links, GitHub, Media, Projects.
- Scores similarity with **two methods, side by side**:
  1. TF-IDF cosine (scikit-learn) — lexical, fully interpretable.
  2. Embedding cosine (sentence-transformers `all-MiniLM-L6-v2`) — semantic,
     no API key.
- Draws **the actual angle**: places both vectors inside the exact 2D plane
  they span via Gram-Schmidt, so the rendered angle is exact — not a
  dimensionality-reduction approximation. `cos θ` on the plot *is* the score.
- Ranks the top-15 terms by contribution to the TF-IDF dot product.
- Cross-section heatmap when both sides are Karrou (one person's Skills
  against another's Experience is a real signal).
- Optional PCA scatter when several profiles are provided.
- Internal metrics panel: vocabulary size, non-zero dims, L2 norms, the raw
  dot product, and the division spelled out; embedding model + dimensionality;
  per-stage timings; per-section char/token counts.
- Résumé-vs-Karrou quality panel: vocabulary size, token count, structural-
  noise ratio (page headers, contact lines, bullet glyphs, hyphenation
  artifacts), usable sections detected. Shows *why* a canonical machine view
  is cleaner input than a PDF built for human eyes — as a measurement, not a
  claim.

## Screenshot

_add screenshot at `docs/screenshot.png` after first run_

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

First launch downloads the MiniLM weights (~90 MB) and takes a minute.
Subsequent launches use the cached model.

## Deploy

Deploy directly to [Streamlit Community Cloud](https://streamlit.io/cloud)
pointing at `app.py`. If the deploy fails on size or memory (torch is heavy),
swap the local model for an embeddings API and put the key in Streamlit
secrets — the `embed_pair` / `embed_texts` boundary in `similarity.py` is the
only surface that needs to change.

## Caveats

- Similarity across a handful of profiles demonstrates a **method**, not a
  finding about the people.
- TF-IDF misses "Postgres" ≈ "relational database". Embeddings can score
  two unrelated people as similar because they write in the same register.
- Both methods reduce a person to their text. Neither sees impact, taste, or
  ability to actually ship.

## Files

- `app.py` — Streamlit UI, orchestration
- `karrou.py` — fetch + section parse
- `pdf_extract.py` — pdfplumber extraction + noise/section heuristics
- `similarity.py` — TF-IDF + MiniLM cosines
- `plots.py` — exact-angle plot, contribution bars, heatmap, PCA scatter
