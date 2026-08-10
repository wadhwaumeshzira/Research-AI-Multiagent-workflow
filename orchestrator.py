"""
orchestrator.py — STAGE 4: Full Pipeline

Runs the complete multi-agent pipeline:
  Planner -> Researcher -> Synthesizer -> Critic
            (loops back to Researcher on missing points, up to MAX_RETRIES times)

This is the main entry point for the whole project.
"""

from planner import plan_sub_questions
from researcher import research_all
from synthesizer import synthesize
from critic import review
from fact_checker import fact_check, annotate_report
from observability import log_pipeline_stage
from cache import get_cached, set_cached
import time

MAX_RETRIES = 2  # configurable — reference project only allowed 1


def run_pipeline(topic: str, verbose: bool = True, use_cache: bool = True, theme: str = "standard") -> dict:
    """
    Runs the full research pipeline for one topic.
    Returns dict with: report, sub_questions, findings, critic_verdict, retries_used
    If use_cache is True and a fresh cached result exists for this exact topic,
    returns it immediately without running the pipeline again.
    """
    if use_cache:
        cached = get_cached(topic)
        if cached:
            if verbose:
                print(f"\n[Cache] Found a cached result for this topic (within {24}h). Skipping pipeline.")
            return cached

    if verbose:
        print(f"\n{'='*60}\nRESEARCHING: {topic}\n{'='*60}")

    _pipeline_start = time.time()

    # 1. PLANNER
    _t = time.time()
    sub_questions = plan_sub_questions(topic, verbose=verbose)
    log_pipeline_stage("planner", topic=topic, n_sub_questions=len(sub_questions), duration_sec=round(time.time() - _t, 3))

    # 2. RESEARCHER
    _t = time.time()
    findings = research_all(sub_questions, verbose=verbose)
    log_pipeline_stage("researcher", n_findings=len(findings), duration_sec=round(time.time() - _t, 3))

    # 3. SYNTHESIZER
    _t = time.time()
    report = synthesize(topic, findings, theme=theme, verbose=verbose)
    log_pipeline_stage("synthesizer", report_length=len(report), duration_sec=round(time.time() - _t, 3))

    # 4. CRITIC (+ retry loop)
    retries_used = 0
    _t = time.time()
    verdict = review(sub_questions, report, verbose=verbose)
    log_pipeline_stage("critic", verdict=verdict["verdict"], duration_sec=round(time.time() - _t, 3))

    while verdict["verdict"] == "NEEDS_REVISION" and verdict["missing_points"] and retries_used < MAX_RETRIES:
        retries_used += 1
        if verbose:
            print(f"\n[Orchestrator] Retry {retries_used}/{MAX_RETRIES} — addressing gaps:")
            for p in verdict["missing_points"]:
                print(f"  - {p}")

        # Research the missing points as new sub-questions
        gap_findings = research_all(verdict["missing_points"], verbose=verbose)
        findings.extend(gap_findings)

        # Re-synthesize with the extra findings included
        report = synthesize(topic, findings, theme=theme, verbose=verbose)

        # Re-check
        verdict = review(sub_questions, report, verbose=verbose)

    if verbose:
        print(f"\n[Orchestrator] Pipeline complete. Retries used: {retries_used}. Final verdict: {verdict['verdict']}")

    # 5. FACT-CHECKER (our addition) — cross-check the final report against raw research notes
    _t = time.time()
    fc_result = fact_check(report, findings, verbose=verbose)
    log_pipeline_stage("fact_checker", n_unsupported=len(fc_result.get("unsupported_claims", [])), duration_sec=round(time.time() - _t, 3))
    annotated_report = annotate_report(report, fc_result)

    log_pipeline_stage("pipeline_complete", topic=topic, total_duration_sec=round(time.time() - _pipeline_start, 3), retries_used=retries_used)

    pipeline_result = {
        "report": annotated_report,
        "sub_questions": sub_questions,
        "findings": findings,
        "critic_verdict": verdict,
        "fact_check": fc_result,
        "retries_used": retries_used,
    }

    if use_cache:
        set_cached(topic, pipeline_result)

    return pipeline_result


if __name__ == "__main__":
    from confidence import score_confidence
    from export import export_markdown, export_pdf

    topic = input("Enter a research topic: ").strip()
    if not topic:
        topic = "Impact of Artificial Intelligence on jobs in 2026"
        print(f"(No input given — using default topic: {topic})")

    result = run_pipeline(topic)
    conf = score_confidence(result)

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result["report"])

    print("\n" + "=" * 60)
    print(f"Critic verdict: {result['critic_verdict']['verdict']}  |  Retries used: {result['retries_used']}")
    print(f"Confidence: {conf['score']}/100 ({conf['label']})")
    for r in conf["reasons"]:
        print(f"  - {r}")
    print("=" * 60)

    choice = input("\nExport this report? [m]arkdown / [p]df / [b]oth / [n]o: ").strip().lower()
    if choice in ("m", "b"):
        export_markdown(topic, result, conf)
    if choice in ("p", "b"):
        export_pdf(topic, result, conf)
