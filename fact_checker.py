"""
fact_checker.py — STAGE 5: Fact-Checker Agent (our addition beyond the reference project)

The Synthesizer combines findings into a report, but LLMs tend to blend in
things they "remember" from training alongside what was actually found via
search. This agent checks: for each factual claim in the report, is there
actual supporting evidence in the raw search results the Researcher gathered?

This directly fixes the issue you saw earlier — vague sources like
"forbes.com/" instead of a specific traceable article, or facts that don't
match any retrieved search snippet.
"""

import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
from observability import log_event

load_dotenv()

FACT_CHECK_MODEL = "openai/gpt-oss-120b"

FACT_CHECK_SYSTEM_PROMPT = """You are a fact-checking agent. You will be given a research \
report and the raw research notes (tool outputs) that were actually gathered during research.

Your job: identify any claims in the report that are NOT supported by the raw research \
notes — meaning the report likely invented them or pulled them from general training \
knowledge rather than the actual search results.

Be reasonable: general framing sentences, summaries, and transitions don't need direct \
support. Focus on specific factual claims — numbers, statistics, named events, dates, \
named entities — that should be traceable to the research notes but aren't.

Output ONLY valid JSON, nothing else, no markdown fences:
{"unsupported_claims": ["claim 1", "claim 2"], "confidence_note": "1 sentence overall assessment"}

If everything is well-supported, unsupported_claims should be an empty list."""


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")
    return Groq(api_key=api_key)


def _raw_notes_from_findings(findings: list[dict]) -> str:
    """Reconstructs the raw research trail (questions + answers) that fed the Synthesizer,
    used here as the 'ground truth' to check the final report against."""
    notes = ""
    for f in findings:
        notes += f"\n--- Research on: {f['question']} ---\n{f['answer']}\n"
    return notes


def fact_check(report: str, findings: list[dict], verbose: bool = True) -> dict:
    """
    Checks the report's claims against the raw research notes.
    Returns {"unsupported_claims": [...], "confidence_note": ...}
    Defaults to an empty (pass) result on failure — never blocks the pipeline.
    """
    client = get_client()
    raw_notes = _raw_notes_from_findings(findings)
    user_prompt = f"RAW RESEARCH NOTES:\n{raw_notes}\n\nFINAL REPORT TO CHECK:\n{report}"

    if verbose:
        print("[Fact-Checker] Cross-checking report claims against raw research notes...")

    try:
        response = client.chat.completions.create(
            model=FACT_CHECK_MODEL,
            messages=[
                {"role": "system", "content": FACT_CHECK_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            reasoning_effort="low"
        )
        usage = getattr(response, "usage", None)
        log_event("llm_call", model=FACT_CHECK_MODEL, function="fact_check", success=True,
                   prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                   completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                   total_tokens=getattr(usage, "total_tokens", 0) if usage else 0)
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        data = json.loads(raw)

        result = {
            "unsupported_claims": data.get("unsupported_claims", []),
            "confidence_note": data.get("confidence_note", ""),
        }

        if verbose:
            n = len(result["unsupported_claims"])
            print(f"[Fact-Checker] Found {n} potentially unsupported claim(s)")
            for c in result["unsupported_claims"]:
                print(f"  ⚠ {c}")

        return result

    except Exception as e:
        if verbose:
            print(f"[Fact-Checker] Failed ({e}), skipping fact-check")
        return {"unsupported_claims": [], "confidence_note": "Fact-check skipped due to an error."}


def annotate_report(report: str, fact_check_result: dict) -> str:
    """Appends a fact-check disclosure section to the report if any unsupported claims were found."""
    claims = fact_check_result.get("unsupported_claims", [])
    if not claims:
        return report

    note = "\n\n### ⚠ Fact-Check Notes\nThe following claims could not be directly verified against the retrieved search results and should be treated with caution:\n"
    note += "\n".join(f"- {c}" for c in claims)
    return report + note


if __name__ == "__main__":
    from planner import plan_sub_questions
    from researcher import research_all
    from synthesizer import synthesize

    topic = "Impact of Artificial Intelligence on jobs in 2026"
    questions = plan_sub_questions(topic)
    findings = research_all(questions)
    report = synthesize(topic, findings)
    fc_result = fact_check(report, findings)

    print("\n" + "=" * 60)
    print("FACT-CHECK RESULT")
    print("=" * 60)
    print(json.dumps(fc_result, indent=2))

    final = annotate_report(report, fc_result)
    print("\n" + "=" * 60)
    print("ANNOTATED REPORT")
    print("=" * 60)
    print(final)
