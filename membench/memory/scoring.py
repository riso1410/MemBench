from __future__ import annotations

import re
from collections import Counter


TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def lexical_score(query: str, text: str) -> float:
    query_terms = Counter(tokenize(query))
    if not query_terms:
        return 0.0
    text_terms = Counter(tokenize(text))
    if not text_terms:
        return 0.0
    overlap = 0
    for term, count in query_terms.items():
        overlap += min(count, text_terms.get(term, 0))
    unique_overlap = len(set(query_terms) & set(text_terms))
    return float(overlap + unique_overlap * 2)

