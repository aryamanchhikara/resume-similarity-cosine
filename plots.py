import math
from typing import Sequence

import numpy as np
import plotly.graph_objects as go


def project_to_span(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gram-Schmidt: place u and v inside the exact 2D plane they span.
    Returns (u_2d, v_2d) — coordinates in that plane. Angle is preserved exactly."""
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    nu = float(np.linalg.norm(u))
    if nu == 0.0:
        return np.array([0.0, 0.0]), np.array([float(np.linalg.norm(v)), 0.0])
    e1 = u / nu
    v_along = float(np.dot(v, e1))
    perp = v - v_along * e1
    perp_norm = float(np.linalg.norm(perp))
    u_2d = np.array([nu, 0.0])
    v_2d = np.array([v_along, perp_norm])
    return u_2d, v_2d


def angle_plot(u: np.ndarray, v: np.ndarray, label_a: str, label_b: str, title: str) -> go.Figure:
    u2, v2 = project_to_span(u, v)
    nu = float(np.linalg.norm(u2))
    nv = float(np.linalg.norm(v2))
    if nu == 0.0 or nv == 0.0:
        cos = 0.0
    else:
        cos = float(np.clip(np.dot(u2, v2) / (nu * nv), -1.0, 1.0))
    theta_deg = math.degrees(math.acos(cos))

    lim = max(nu, nv) * 1.15 + 1e-9

    fig = go.Figure()
    for tip, name, color in [
        (u2, label_a, "#1f77b4"),
        (v2, label_b, "#ff7f0e"),
    ]:
        fig.add_trace(go.Scatter(
            x=[0, tip[0]], y=[0, tip[1]],
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=3),
            marker=dict(size=[0, 10], color=color),
            hovertemplate=f"{name}<br>‖v‖=%{{customdata:.4f}}<extra></extra>",
            customdata=[0, float(np.linalg.norm(tip))],
        ))

    # angle arc
    if nu > 0 and nv > 0:
        r = min(nu, nv) * 0.25
        a0 = math.atan2(u2[1], u2[0])
        a1 = math.atan2(v2[1], v2[0])
        if a1 < a0:
            a0, a1 = a1, a0
        arc = np.linspace(a0, a1, 40)
        fig.add_trace(go.Scatter(
            x=r * np.cos(arc), y=r * np.sin(arc),
            mode="lines", line=dict(color="#666", dash="dot", width=1),
            showlegend=False, hoverinfo="skip",
        ))
        mid = (a0 + a1) / 2
        fig.add_annotation(
            x=r * 1.4 * math.cos(mid), y=r * 1.4 * math.sin(mid),
            text=f"θ ≈ {theta_deg:.1f}°<br>cos θ = {cos:.4f}",
            showarrow=False, font=dict(size=13, color="#333"),
        )

    fig.update_layout(
        title=title,
        xaxis=dict(range=[-lim * 0.15, lim], zeroline=True, showgrid=True, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-lim * 0.15, lim], zeroline=True, showgrid=True),
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    return fig


def contribution_bars(rows: list[tuple[str, float, float, float]], title: str) -> go.Figure:
    if not rows:
        fig = go.Figure()
        fig.update_layout(title=title, annotations=[dict(
            text="No overlapping terms with positive weight in both documents.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )])
        return fig
    terms = [r[0] for r in rows][::-1]
    contribs = [r[3] for r in rows][::-1]
    wa = [r[1] for r in rows][::-1]
    wb = [r[2] for r in rows][::-1]
    fig = go.Figure(go.Bar(
        x=contribs, y=terms, orientation="h",
        marker=dict(color=contribs, colorscale="Viridis"),
        customdata=list(zip(wa, wb)),
        hovertemplate="<b>%{y}</b><br>w_a=%{customdata[0]:.4f}"
                      "<br>w_b=%{customdata[1]:.4f}"
                      "<br>contribution=%{x:.5f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="w_a × w_b (contribution to dot product)",
        height=420,
        margin=dict(l=20, r=20, t=60, b=40),
    )
    return fig


def section_heatmap(matrix: np.ndarray, labels_a: Sequence[str], labels_b: Sequence[str], title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=list(labels_b),
        y=list(labels_a),
        colorscale="Viridis",
        zmin=0.0, zmax=1.0,
        colorbar=dict(title="cosine"),
        hovertemplate="A:%{y}<br>B:%{x}<br>cos=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=20, r=20, t=60, b=60),
        xaxis=dict(side="bottom", tickangle=-30),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def pca_scatter(vectors: np.ndarray, labels: Sequence[str], highlight_idx: Sequence[int], title: str) -> go.Figure:
    from sklearn.decomposition import PCA

    if vectors.shape[0] < 2:
        fig = go.Figure()
        fig.update_layout(title=title, annotations=[dict(
            text="Need at least 2 profiles for PCA.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )])
        return fig

    n_components = 2 if vectors.shape[0] >= 2 and vectors.shape[1] >= 2 else 1
    pca = PCA(n_components=n_components)
    xy = pca.fit_transform(vectors)
    if n_components == 1:
        xy = np.hstack([xy, np.zeros_like(xy)])

    highlight_set = set(highlight_idx)
    colors = ["#ff7f0e" if i in highlight_set else "#8888aa" for i in range(len(labels))]
    sizes = [16 if i in highlight_set else 8 for i in range(len(labels))]

    fig = go.Figure(go.Scatter(
        x=xy[:, 0], y=xy[:, 1],
        mode="markers+text",
        text=list(labels),
        textposition="top center",
        marker=dict(color=colors, size=sizes, line=dict(color="#333", width=1)),
        hovertemplate="%{text}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)" if n_components == 2 else "",
        height=420,
        margin=dict(l=20, r=20, t=60, b=40),
    )
    return fig
