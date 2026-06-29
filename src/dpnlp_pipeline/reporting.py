from __future__ import annotations

from src.dpnlp_pipeline.config import MODEL_SPECS


def _sort_evaluation_rows(evaluation_rows: list[dict]) -> list[dict]:
    model_order = {name: index for index, name in enumerate(MODEL_SPECS.keys())}
    return sorted(
        evaluation_rows,
        key=lambda row: (
            model_order.get(row.get("model_size"), len(model_order)),
            row.get("parameter_count", float("inf")),
            row.get("model_size", ""),
        ),
    )


def build_claim_summary(evaluation_rows: list[dict]) -> dict[str, str]:
    if not evaluation_rows:
        return {
            "larger_models_more_coherent": "unsupported",
            "small_models_grammar_before_creativity_consistency": "unsupported",
            "outputs_not_simple_memorization": "unsupported",
        }

    ordered_rows = _sort_evaluation_rows(evaluation_rows)
    consistencies = [row["consistency"] for row in ordered_rows]
    overlaps = [row["max_overlap"] for row in ordered_rows]
    smallest = ordered_rows[0]

    larger_models_more_coherent = "supported" if consistencies == sorted(consistencies) else "partial"
    grammar_gap = smallest["grammar"] > smallest["creativity"] and smallest["grammar"] > smallest["consistency"]
    small_models_grammar_first = "supported" if grammar_gap else "partial"
    not_memorized = "supported" if max(overlaps) < 0.2 else "partial"

    return {
        "larger_models_more_coherent": larger_models_more_coherent,
        "small_models_grammar_before_creativity_consistency": small_models_grammar_first,
        "outputs_not_simple_memorization": not_memorized,
    }
