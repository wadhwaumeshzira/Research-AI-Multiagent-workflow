"""
synthesizer.py — STAGE 3: Synthesizer Agent

Takes the Researcher's separate findings (one per sub-question) and combines
them into a single, well-structured report — with proper sections, not just
the Q&A pairs pasted together.
"""

import os
from dotenv import load_dotenv
from groq import Groq
from observability import log_event

load_dotenv()

SYNTHESIZER_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "qwen/qwen3.6-27b" # used if primary model fails

SYNTHESIZER_SYSTEM_PROMPT = """You are a research synthesis agent. You will be given \
research findings for several sub-questions about one broader topic. Combine them \
into ONE cohesive, well-structured report — do not just list the Q&A pairs, actually \
weave the findings into flowing sections.

{theme_instruction}

Use this exact structure:

## Research Report: {topic}

### Executive Summary
2-3 sentences summarizing the overall picture across all findings.

### Key Findings
The most important, concrete points from across all sub-questions (bullet points, \
with facts/numbers where available).

### Detailed Analysis
A few paragraphs connecting the different sub-question findings into a coherent narrative.

### Future Outlook
Forward-looking trends found across the research.

### Sources
List all unique URLs mentioned across the findings, deduplicated.

Do not invent new information — only synthesize what's given to you below."""

THEME_INSTRUCTIONS = {
    "standard": "Write in a clear, professional, neutral tone — suitable for a general audience.",
    "executive": "Write concisely and to the point, as if for a busy executive — favor bullet points over long paragraphs, lead with the bottom line, minimize jargon.",
    "academic": "Write in a formal, detailed, academic tone — thorough analysis, precise language, hedge claims appropriately (e.g. 'suggests', 'indicates').",
    "casual": "Write in a friendly, conversational tone, as if explaining to a curious friend — avoid jargon, use approachable language, keep it engaging.",
}


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")
    return Groq(api_key=api_key)


def synthesize(topic: str, findings: list[dict], theme: str = "standard", verbose: bool = True) -> str:
    """
    Combines a list of {"question": ..., "answer": ...} dicts into one report.
    theme: one of "standard", "executive", "academic", "casual" — controls tone/style.
    """
    findings_text = ""
    for f in findings:
        findings_text += f"\n--- Sub-question: {f['question']} ---\n{f['answer']}\n"

    user_prompt = f"TOPIC: {topic}\n\nFINDINGS:\n{findings_text}"
    theme_instruction = THEME_INSTRUCTIONS.get(theme, THEME_INSTRUCTIONS["standard"])
    system_prompt = SYNTHESIZER_SYSTEM_PROMPT.format(topic=topic, theme_instruction=theme_instruction)

    client = get_client()

    if verbose:
        print("[Synthesizer] Combining findings into final report...")

    try:
        response = client.chat.completions.create(
            model=SYNTHESIZER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            reasoning_effort="low",  # bump to "medium" if report quality drops
        )
        usage = getattr(response, "usage", None)
        log_event("llm_call", model=SYNTHESIZER_MODEL, function="synthesize", success=True,
                   prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                   completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                   total_tokens=getattr(usage, "total_tokens", 0) if usage else 0)
        return response.choices[0].message.content

    except Exception as e:
        if verbose:
            print(f"[Synthesizer] Primary model failed: {type(e).__name__}: {e}")
            print(f"[Synthesizer] Retrying with fallback model ({FALLBACK_MODEL})...")
        try:
            response = client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                # note: no reasoning_effort here — that param is gpt-oss-specific;
                # FALLBACK_MODEL (qwen) may reject it
            )
            usage = getattr(response, "usage", None)
            log_event("llm_call", model=FALLBACK_MODEL, function="synthesize", success=True,
                       prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                       completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                       total_tokens=getattr(usage, "total_tokens", 0) if usage else 0)
            return response.choices[0].message.content
        except Exception as e2:
            if verbose:
                print(f"[Synthesizer] Fallback model also failed: {type(e2).__name__}: {e2}")
            log_event("llm_call", model=FALLBACK_MODEL, function="synthesize", success=False,
                       error=str(e2))
            raise RuntimeError(
                f"Both primary ({SYNTHESIZER_MODEL}) and fallback ({FALLBACK_MODEL}) "
                f"models failed. Primary error: {e}. Fallback error: {e2}"
            ) from e2


if __name__ == "__main__":
    from planner import plan_sub_questions
    from researcher import research_all

    topic = "Impact of Artificial Intelligence on jobs in 2026"
    questions = plan_sub_questions(topic)
    findings = research_all(questions)
    report = synthesize(topic, findings)

    print("\n" + "=" * 60)
    print("FINAL SYNTHESIZED REPORT")
    print("=" * 60)
    print(report)
