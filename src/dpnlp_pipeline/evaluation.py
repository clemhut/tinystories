from __future__ import annotations

from collections import Counter


def build_ngram_counter(text: str, n: int = 6) -> Counter[str]:
    if len(text) < n:
        return Counter()
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def ngram_overlap_against_counts(reference_ngrams: Counter[str], candidate: str, n: int = 6) -> float:
    if len(candidate) < n:
        return 0.0
    candidate_ngrams = build_ngram_counter(candidate, n=n)
    shared = sum((reference_ngrams & candidate_ngrams).values())
    total = sum(candidate_ngrams.values())
    return 0.0 if total == 0 else shared / total


def ngram_overlap(reference: str, candidate: str, n: int = 6) -> float:
    reference_ngrams = build_ngram_counter(reference, n=n)
    return ngram_overlap_against_counts(reference_ngrams, candidate, n=n)


def evaluate_memorization(reference: str, candidate: str, n: int = 6) -> float:
    return ngram_overlap(reference, candidate, n=n)
