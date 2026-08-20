import re
from dataclasses import dataclass
from typing import Optional

import requests
import streamlit as st

KARROU_BASE = "https://karrouhq.com"
UA = "resume-similarity-cosine/0.1 (+https://github.com/aryamanchhikara)"

STANDARD_SECTIONS = [
    "About",
    "Experience",
    "Skills",
    "Work",
    "Widgets",
    "Links",
    "GitHub",
    "Media",
    "Projects",
]


class KarrouNotFound(Exception):
    pass


class KarrouFetchError(Exception):
    pass


@dataclass
class KarrouDoc:
    username: str
    header: str
    sections: dict
    raw: str

    @property
    def full_text(self) -> str:
        parts = [self.header] if self.header else []
        for name in STANDARD_SECTIONS:
            body = self.sections.get(name, "").strip()
            if body:
                parts.append(f"{name}\n{body}")
        return "\n\n".join(parts).strip() or self.raw


@st.cache_data(ttl=300, show_spinner=False)
def fetch_karrou_md(username: str) -> str:
    username = username.strip().lstrip("@")
    if not username or not re.match(r"^[A-Za-z0-9_.-]+$", username):
        raise KarrouFetchError(f"invalid username: {username!r}")
    url = f"{KARROU_BASE}/{username}.md"
    try:
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=10)
    except requests.RequestException as e:
        raise KarrouFetchError(str(e)) from e
    if resp.status_code == 404:
        raise KarrouNotFound(
            f"profile @{username} not found — either it doesn't exist "
            "or the owner disabled the machine-readable view"
        )
    if resp.status_code >= 400:
        raise KarrouFetchError(f"HTTP {resp.status_code} fetching {url}")
    return resp.text


def parse_sections(md: str) -> tuple[str, dict]:
    """Split a Karrou .md into (header_block, {section_name: body})."""
    lines = md.splitlines()
    header_lines: list[str] = []
    sections: dict[str, list[str]] = {}
    current: Optional[str] = None

    for line in lines:
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            current = m.group(1).strip()
            sections.setdefault(current, [])
            continue
        if current is None:
            header_lines.append(line)
        else:
            sections[current].append(line)

    header = _clean("\n".join(header_lines))
    sec_out = {name: _clean("\n".join(body)) for name, body in sections.items()}
    return header, sec_out


def _clean(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_karrou(username: str) -> KarrouDoc:
    raw = fetch_karrou_md(username)
    header, sections = parse_sections(raw)
    return KarrouDoc(
        username=username.strip().lstrip("@"),
        header=header,
        sections=sections,
        raw=raw,
    )
