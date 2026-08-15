"""
evaluation.py — LLM-as-judge evaluation of the research pipeline.

Runs the pipeline on a set of test topics, then uses a separate LLM call
(the "judge") to score each report on 4 criteria: Factual Specificity,
Source Quality, Completeness, Clarity. Aggregates into a summary.

This answers: "is my agent actually good?" with a number, not a guess.
"""

import json
import re
import time
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
import os

from orchestrator import run_pipeline

load_dotenv()

JUDGE_MODEL = "openai/gpt-oss-120b"

TEST_TOPICS = [
    "Latest developments in renewable energy 2026",
    "Impact of remote work on company productivity",
    "Current state of electric vehicle adoption in India",
    "Effects of social media on mental health",
]

JUDGE_PROMPT = """You are an evaluation judge assessing the quality of an AI-generated research report.

TOPIC: {topic}

REPORT TO EVALUATE:
{report}

Score this report on these 4 criteria, each from 1-10 (10 = excellent, 1 = very poor):

1. FACTUAL_SPECIFICITY: Does it contain concrete facts, numbers, dates, named entities? (vs vague generalities)
2. SOURCE_QUALITY: Are sources cited? Do they look credible (real URLs, named publications)?
3. COMPLETENESS: Does it cover the topic thoroughly across multiple angles?
4. CLARITY: Is it well-organized, readable, and free of contradictions?

Output ONLY valid JSON, nothing else, no markdown fences:
{{"factual_specificity": <1-10>, "source_quality": <1-10>, "completeness": <1-10>, "clarity": <1-10>, "comments": "1-2 sentence summary"}}"""


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")
    return Groq(api_key=api_key)


def judge_report(topic: str, report: str) -> dict:
    """Scores one report via the judge LLM. Returns a default low score on failure."""
    client = get_client()
    try:
        prompt = JUDGE_PROMPT.format(topic=topic, report=report)
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You output only valid JSON, no other text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            reasoning_effort="low",  # bump to "medium" if scoring feels inconsistent
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        data = json.loads(raw)
        return {
            "factual_specificity": data.get("factual_specificity", 0),
            "source_quality": data.get("source_quality", 0),
            "completeness": data.get("completeness", 0),
            "clarity": data.get("clarity", 0),
            "comments": data.get("comments", ""),
        }
    except Exception as e:
        print(f"Judge failed for topic '{topic}': {e}")
        return {"factual_specificity": 0, "source_quality": 0, "completeness": 0, "clarity": 0,
                 "comments": f"Judge evaluation failed: {e}"}


def run_evaluation(topics: list = None, verbose: bool = True) -> dict:
    """Runs the pipeline + judge on each topic, returns an aggregated summary."""
    topics = topics or TEST_TOPICS
    results = []

    for i, topic in enumerate(topics, 1):
        if verbose:
            print(f"\n[{i}/{len(topics)}] Researching: {topic}")

        try:
            pipeline_result = run_pipeline(topic, verbose=False)
            report = pipeline_result["report"]
        except Exception as e:
            print(f"  Research failed: {e}")
            results.append({
                "topic": topic,
                "scores": {"factual_specificity": 0, "source_quality": 0, "completeness": 0, "clarity": 0},
                "comments": f"Research failed: {e}",
                "report_length": 0,
            })
            continue

        if verbose:
            print(f"  Report generated ({len(report)} chars). Judging...")

        scores = judge_report(topic, report)
        results.append({
            "topic": topic,
            "scores": {k: scores[k] for k in ["factual_specificity", "source_quality", "completeness", "clarity"]},
            "comments": scores["comments"],
            "report_length": len(report),
        })

        if verbose:
            avg = sum(scores[k] for k in ["factual_specificity", "source_quality", "completeness", "clarity"]) / 4
            print(f"  Scores: factual={scores['factual_specificity']}, sources={scores['source_quality']}, "
                  f"completeness={scores['completeness']}, clarity={scores['clarity']} (avg: {avg:.1f})")

        time.sleep(1)  # small gap between calls

    criteria = ["factual_specificity", "source_quality", "completeness", "clarity"]
    aggregate = {}
    for c in criteria:
        valid = [r["scores"][c] for r in results if r["scores"][c] > 0]
        aggregate[c] = round(sum(valid) / len(valid), 2) if valid else 0

    overall_avg = round(sum(aggregate.values()) / len(aggregate), 2) if aggregate else 0

    return {
        "evaluated_at": datetime.now().isoformat(),
        "total_topics": len(topics),
        "successful_topics": len([r for r in results if r["scores"]["factual_specificity"] > 0]),
        "aggregate_scores": aggregate,
        "overall_average": overall_avg,
        "per_topic_results": results,
    }


def save_results(summary: dict, path: str = "evaluation_results.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {path}")


def print_summary_table(summary: dict):
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Topics evaluated: {summary['successful_topics']}/{summary['total_topics']}")
    print(f"Overall average score: {summary['overall_average']}/10\n")
    print("Per-criterion averages:")
    for criterion, score in summary["aggregate_scores"].items():
        bar = "#" * int(score) + "-" * (10 - int(score))
        print(f"  {criterion:22s} {score:5.2f}/10  {bar}")

    print("\nPer-topic breakdown:")
    for r in summary["per_topic_results"]:
        avg = sum(r["scores"].values()) / 4
        print(f"  [{avg:.1f}/10] {r['topic']}")
    print("=" * 70)


if __name__ == "__main__":
    print("Starting Research Agent Evaluation\n")
    print(f"Testing {len(TEST_TOPICS)} topics...")

    summary = run_evaluation(TEST_TOPICS, verbose=True)
    print_summary_table(summary)
    save_results(summary)
