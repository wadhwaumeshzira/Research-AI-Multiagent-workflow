"""
compare.py — Compare two topics side-by-side.

Runs the full pipeline independently on each topic (so neither research
biases the other), then asks the LLM to synthesize a structured comparison
from the two finished reports.
"""

import os
from dotenv import load_dotenv
from groq import Groq

from orchestrator import run_pipeline

load_dotenv()

COMPARE_MODEL = "openai/gpt-oss-120b"

COMPARE_SYSTEM_PROMPT = """You are a comparison agent. You are given two independent \
research reports on two different topics. Produce a structured comparison between them.

Use this format:

## Comparison: {topic_a} vs {topic_b}

### Overview
2-3 sentences framing how these two topics relate or differ.

### Side-by-Side Comparison
A markdown table comparing key dimensions (choose dimensions relevant to both topics — \
e.g. scale, impact, recent developments, outlook — whatever fits).

| Dimension | {topic_a} | {topic_b} |
|---|---|---|
| ... | ... | ... |

### Key Differences
Bullet points on the most important distinctions.

### Key Similarities
Bullet points on what they have in common, if anything.

Only use information present in the two reports below — do not invent new facts."""


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")
    return Groq(api_key=api_key)


def run_comparison(topic_a: str, topic_b: str, verbose: bool = True) -> dict:
    """
    Researches both topics independently (full pipeline each), then generates
    a structured comparison. Returns dict with: comparison, report_a, report_b.
    """
    if verbose:
        print(f"\n[Compare] Researching topic A: {topic_a}")
    result_a = run_pipeline(topic_a, verbose=verbose)

    if verbose:
        print(f"\n[Compare] Researching topic B: {topic_b}")
    result_b = run_pipeline(topic_b, verbose=verbose)

    system_prompt = COMPARE_SYSTEM_PROMPT.format(topic_a=topic_a, topic_b=topic_b)
    user_prompt = (
        f"REPORT ON {topic_a}:\n{result_a['report']}\n\n"
        f"REPORT ON {topic_b}:\n{result_b['report']}"
    )

    if verbose:
        print("\n[Compare] Generating structured comparison...")

    client = get_client()
    response = client.chat.completions.create(
        model=COMPARE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        reasoning_effort="low",
    )

    return {
        "comparison": response.choices[0].message.content,
        "report_a": result_a["report"],
        "report_b": result_b["report"],
        "pipeline_a": result_a,
        "pipeline_b": result_b,
    }


if __name__ == "__main__":
    result = run_comparison("Electric vehicles in India", "Electric vehicles in China")
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(result["comparison"])
