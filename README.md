# resume-similarity-cosine

Cosine similarity over [Karrou](https://karrouhq.com) profiles, made visible
end to end.

Karrou publishes every profile as a canonical machine-readable document at
`karrouhq.com/{username}.md`. That turns *comparing two people* from a
scraping problem into a text problem. This app takes two Karrou profiles (or
two résumés, or one of each) and shows exactly how the similarity score is
computed — every vector, every norm, every division.

**Live app:** [resume-similarity-cosine](https://resume-similarity-cosine.streamlit.app/)

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

## Method

For those who want to see the arithmetic before running the app.

### 1. Cosine similarity

Given two document vectors $a, b \in \mathbb{R}^n$:

$$\text{sim}(a,b) = \cos\theta = \frac{a \cdot b}{\|a\|\,\|b\|} = \frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2}\,\sqrt{\sum_i b_i^2}}$$

Where:

- $a_i, b_i$ are the $i$-th coordinates of the two vectors
- $a \cdot b = \sum_i a_i b_i$ is the dot product
- $\|a\| = \sqrt{\sum_i a_i^2}$ is the L2 (Euclidean) norm
- $\theta$ is the angle between $a$ and $b$ in whatever space they live in

Why cosine and not Euclidean distance? Cosine is **scale-invariant** — it depends only on the *direction* of each vector, not its magnitude. A long résumé and a short profile do not produce vectors of comparable length, and we don't want that length gap to dominate the score. Euclidean would penalise the longer document even if its content matched perfectly.

### 2. TF-IDF (what scikit-learn actually computes)

`TfidfVectorizer` is invoked with all defaults intact for `smooth_idf`, `norm`, `use_idf`, and `sublinear_tf`. That means:

Term frequency is the raw count:

$$\text{tf}(t, d) = \text{number of times term } t \text{ appears in document } d$$

Inverse document frequency uses **smoothed** idf (adds one to both numerator and denominator to avoid zero-division):

$$\text{idf}(t) = \ln\!\left(\frac{1+n}{1+\text{df}(t)}\right) + 1$$

Where $n$ is the number of documents, $\text{df}(t)$ is the number of documents containing $t$, and $\ln$ is the natural log.

The unnormalised weight is:

$$\tilde{w}_{t,d} = \text{tf}(t,d) \cdot \text{idf}(t)$$

Then `TfidfVectorizer` L2-normalises each row:

$$w_{t,d} = \frac{\tilde{w}_{t,d}}{\sqrt{\sum_{t'} \tilde{w}_{t',d}^2}}$$

**Consequence:** every TF-IDF row vector has $\|a\| = \|b\| = 1$ by construction. The cosine formula collapses to a plain dot product:

$$\cos\theta = a \cdot b \qquad \text{(because } \|a\|\,\|b\| = 1\text{)}$$

This is why the arithmetic in the metrics panel divides by $1.0 \times 1.0$ for TF-IDF — the denominator is decorative.

### 3. Term contribution

Each vocabulary term contributes:

$$c_t = a_t \cdot b_t$$

Since the TF-IDF vectors are L2-normalised, the contributions sum exactly to the score:

$$\cos\theta = a \cdot b = \sum_t a_t b_t = \sum_t c_t$$

That is what the bar chart is ranking, and it is exactly why the bars add up to the number in the metrics panel.

### 4. Exact-angle 2D projection

Two vectors span at most a 2D plane, regardless of the dimensionality of the ambient space. An orthonormal basis for that plane is built via Gram-Schmidt:

$$u_1 = \frac{a}{\|a\|}, \qquad w = b - (b \cdot u_1)\,u_1, \qquad u_2 = \frac{w}{\|w\|}$$

In the $(u_1, u_2)$ basis, the two vectors have coordinates:

$$a_{\text{2D}} = \bigl(\|a\|,\ 0\bigr), \qquad b_{\text{2D}} = \bigl(b \cdot u_1,\ \|w\|\bigr)$$

Because $(u_1, u_2)$ spans the same plane as $(a, b)$, this projection preserves the angle between the two vectors **exactly**. The angle you see on the plot is the angle cosine is computing — not a dimensionality-reduction approximation.

Contrast the PCA scatter: PCA takes many high-dimensional vectors and forces them into 2D by discarding variance along the smaller components. PCA distances are approximations; the two-vector projection above is not.

### 5. Embedding similarity

Same cosine formula, dense vectors instead of sparse.

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector space: $\mathbb{R}^{384}$

Unlike TF-IDF, the embeddings are **not** L2-normalised in the code (`normalize_embeddings=False`), so $\|a\|$ and $\|b\|$ are not 1. The denominator in $\cos\theta = (a \cdot b) / (\|a\|\,\|b\|)$ actually does arithmetic work here, and the metrics panel spells it out.

### 6. Worked example

Two three-word documents:

- $A = $ "python data python"
- $B = $ "python data sql"

**Counts.** $n = 2$ documents; vocabulary $= \\{\text{data}, \text{python}, \text{sql}\\}$.

| term   | df | tf in A | tf in B |
|--------|----|---------|---------|
| data   | 2  | 1       | 1       |
| python | 2  | 2       | 1       |
| sql    | 1  | 0       | 1       |

**IDF** (smoothed):

$$\text{idf}(\text{data}) = \ln\tfrac{1+2}{1+2} + 1 = 1$$

$$\text{idf}(\text{python}) = \ln\tfrac{1+2}{1+2} + 1 = 1$$

$$\text{idf}(\text{sql}) = \ln\tfrac{1+2}{1+1} + 1 = \ln\tfrac{3}{2} + 1 \approx 1.4055$$

**Unnormalised tf-idf** (component order = data, python, sql):

$$\tilde{A} = (1,\ 2,\ 0), \qquad \tilde{B} = (1,\ 1,\ 1.4055)$$

**L2 norms:**

$$\|\tilde{A}\| = \sqrt{1^2 + 2^2 + 0^2} = \sqrt{5} \approx 2.2361$$

$$\|\tilde{B}\| = \sqrt{1^2 + 1^2 + 1.4055^2} \approx \sqrt{3.9753} \approx 1.9938$$

**Normalised rows:**

$$A \approx (0.4472,\ 0.8944,\ 0)$$

$$B \approx (0.5015,\ 0.5015,\ 0.7049)$$

**Score.** Both are unit length, so cosine collapses to the dot product:

$$\cos\theta = (0.4472)(0.5015) + (0.8944)(0.5015) + (0)(0.7049)$$

$$= 0.2243 + 0.4485 + 0 \approx 0.6729$$

Cross-checked against `TfidfVectorizer` on the same two documents: `0.6728984701822545`. Agreement to 15 significant figures.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

First launch downloads the MiniLM weights (~90 MB) and takes a minute.
Subsequent launches use the cached model.


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
