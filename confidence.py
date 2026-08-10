"""
confidence.py — STAGE 6: Confidence Scoring

A heuristic score (0-100) reflecting how much we should trust the report,
based on measurable signals from the pipeline — not another LLM call,
just math over what already happened. Cheap, fast, deterministic.

Signals used:
- How many sub-questions actually got real tool-based research (vs failed)
- How many unique sources were cited
- Critic verdict (PASS vs needed revision)
- Fact-checker: how many unsupported claims were flagged
- Report length (very short reports are usually shallow)
"""

import re


def _count_unique_sources(report: str) -> int:
    urls = re.findall(r"https?://[^\s)\]]+", report)
    return len(set(urls))


def score_confidence(pipeline_result: dict) -> dict:
    """
    Takes the full dict returned by orchestrator.run_pipeline() and computes
    a confidence score + label + short explanation.
    """
    findings = pipeline_result["findings"]
    report = pipeline_result["report"]
    critic_verdict = pipeline_result["critic_verdict"]
    fact_check = pipeline_result.get("fact_check", {"unsupported_claims": []})
    retries_used = pipeline_result.get("retries_used", 0)

    score = 100
    reasons = []

    # Signal 1: research coverage — did sub-questions actually get answered with tool steps?
    total = len(findings)
    with_steps = sum(1 for f in findings if f.get("steps"))
    if total > 0 and with_steps < total:
        penalty = round(((total - with_steps) / total) * 25)
        score -= penalty
        reasons.append(f"{total - with_steps}/{total} sub-questions had no successful tool research (-{penalty})")

    # Signal 2: source count
    n_sources = _count_unique_sources(report)
    if n_sources == 0:
        score -= 25
        reasons.append("No sources cited in the final report (-25)")
    elif n_sources < 3:
        score -= 10
        reasons.append(f"Only {n_sources} unique source(s) cited (-10)")

    # Signal 3: critic verdict
    if critic_verdict.get("verdict") == "NEEDS_REVISION":
        score -= 15
        reasons.append("Critic still flagged issues after retries (-15)")

    # Signal 4: fact-check
    n_unsupported = len(fact_check.get("unsupported_claims", []))
    if n_unsupported > 0:
        penalty = min(20, n_unsupported * 7)
        score -= penalty
        reasons.append(f"{n_unsupported} claim(s) flagged as unsupported by fact-checker (-{penalty})")

    # Signal 5: report depth
    if len(report) < 800:
        score -= 10
        reasons.append("Report is quite short, may lack depth (-10)")

    # Small bonus: retries used and still passed = system self-corrected successfully
    if retries_used > 0 and critic_verdict.get("verdict") == "PASS":
        score += 5
        reasons.append("Pipeline self-corrected via retry and passed final review (+5)")

    score = max(0, min(100, score))

    if score >= 85:
        label = "High"
    elif score >= 65:
        label = "Moderate"
    else:
        label = "Low"

    return {
        "score": score,
        "label": label,
        "reasons": reasons,
        "unique_sources": n_sources,
    }


if __name__ == "__main__":
    from orchestrator import run_pipeline

    topic = "Impact of Artificial Intelligence on jobs in 2026"
    result = run_pipeline(topic)
    conf = score_confidence(result)

    print("\n" + "=" * 60)
    print("CONFIDENCE SCORE")
    print("=" * 60)
    print(f"Score: {conf['score']}/100  ({conf['label']})")
    print(f"Unique sources cited: {conf['unique_sources']}")
    print("Reasons:")
    for r in conf["reasons"]:
        print(f"  - {r}")
