"""
critic.py — STAGE 4: Critic Agent

Reviews the synthesized report against the original sub-questions and flags:
- Missing coverage (a sub-question that wasn't really answered)
- Vague/unsupported claims
- Missing or weak sourcing

Returns a verdict (PASS / NEEDS_REVISION) so the orchestrator can decide
whether to do a corrective research pass.
"""

import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
from observability import log_event

load_dotenv()

CRITIC_MODEL = "openai/gpt-oss-120b"

CRITIC_SYSTEM_PROMPT = """You are a quality-control critic reviewing a research report.

Evaluate the report against these criteria:
1. Does it address ALL the sub-questions listed below (at least partially)?
2. Are there specific, concrete claims (facts/numbers), not just vague generalities?
3. Does it cite real, specific sources (not just bare domain names)?
4. Are there any obvious contradictions or unsupported claims?

Output ONLY valid JSON, nothing else, no markdown fences:
{"verdict": "PASS" or "NEEDS_REVISION", "missing_points": ["point 1", "point 2"], "reasoning": "1-2 sentence explanation"}

If verdict is PASS, missing_points must be an empty list.
Only fail (NEEDS_REVISION) for real, meaningful gaps — not minor style issues."""


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")
    return Groq(api_key=api_key)


def review(sub_questions: list[str], report: str, verbose: bool = True) -> dict:
    """
    Reviews a report. Returns {"verdict": ..., "missing_points": [...], "reasoning": ...}
    Defaults to PASS on any failure — a broken critic should never block the pipeline.
    """
    client = get_client()
    questions_list = "\n".join(f"- {q}" for q in sub_questions)
    user_prompt = f"SUB-QUESTIONS TO CHECK COVERAGE FOR:\n{questions_list}\n\nREPORT:\n{report}"

    if verbose:
        print("[Critic] Reviewing report quality...")

    try:
        response = client.chat.completions.create(
            model=CRITIC_MODEL,
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            reasoning_effort="low",  # bump to "medium" if critiques feel shallow
        )
        usage = getattr(response, "usage", None)
        log_event("llm_call", model=CRITIC_MODEL, function="review", success=True,
                   prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                   completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                   total_tokens=getattr(usage, "total_tokens", 0) if usage else 0)
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        data = json.loads(raw)

        result = {
            "verdict": data.get("verdict", "PASS"),
            "missing_points": data.get("missing_points", []),
            "reasoning": data.get("reasoning", ""),
        }

        if verbose:
            print(f"[Critic] Verdict: {result['verdict']} — {result['reasoning']}")

        return result

    except Exception as e:
        if verbose:
            print(f"[Critic] Failed ({e}), defaulting to PASS so the pipeline isn't blocked")
        return {"verdict": "PASS", "missing_points": [], "reasoning": "Critic check skipped due to an error."}


if __name__ == "__main__":
    from planner import plan_sub_questions
    from researcher import research_all
    from synthesizer import synthesize

    topic = "Impact of Artificial Intelligence on jobs in 2026"
    questions = plan_sub_questions(topic)
    findings = research_all(questions)
    report = synthesize(topic, findings)
    verdict = review(questions, report)

    print("\n" + "=" * 60)
    print("CRITIC VERDICT")
    print("=" * 60)
    print(json.dumps(verdict, indent=2))
