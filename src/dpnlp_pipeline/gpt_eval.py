from __future__ import annotations

import json
import os


DEFAULT_GRADING_RUBRIC = """The following exercise, the student is given a beginning of a story. The student needs to complete it into a full story. The exercise tests the student's language abilities and creativity. The symbol *** marks the separator between the prescribed beginning and the student's completion:

{prompt_prefix}
***
{completion}

Please provide your general assessment about the part written by the student (the one after the *** symbol). Is it gramatically correct? Is it consistent with the beginning of the story? Pay special attention to whether the student manages to complete the sentence which is split in the middle by the separator ***.

Now, grade the student's completion in terms of grammar, creativity, consistency with the story's beginning and whether the plot makes sense. Moreover, please provide your best guess of what the age of the student might be, as reflected from the complection. Choose from possible age groups: A: 3 or under. B: 4-5. C: 6-7. D: 8-9. E: 10-12. F: 13-16. Return strict JSON with the following fields:
- general_assessment: string
- grammar: number from 1 to 10
- creativity: number from 1 to 10
- consistency: number from 1 to 10
- plot_sense: number from 1 to 10
- age_guess: short string
"""


def build_gpt_eval_prompt(prompt_prefix: str, completion: str) -> str:
    return DEFAULT_GRADING_RUBRIC.format(prompt_prefix=prompt_prefix, completion=completion)


def _normalize_json_content(content: str) -> str:
    normalized = content.strip()
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        normalized = "\n".join(lines).strip()
    return normalized


def grade_completion(prompt_prefix: str, completion: str, client=None, model: str = "gpt-4o") -> dict[str, float | str]:
    client = client or _build_openai_client()
    response = client.responses.create(
        model=model,
        input=build_gpt_eval_prompt(prompt_prefix, completion),
    )
    content = _normalize_json_content(response.output_text)
    payload = json.loads(content)
    return {
        "general_assessment": str(payload["general_assessment"]),
        "grammar": float(payload["grammar"]),
        "creativity": float(payload["creativity"]),
        "consistency": float(payload["consistency"]),
        "plot_sense": float(payload["plot_sense"]),
        "age_guess": str(payload["age_guess"]),
    }


def _build_openai_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for GPT evaluation")
    return OpenAI(api_key=api_key)
