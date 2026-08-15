"""
planner.py — STAGE 2: Planner Agent

Takes a broad research topic and breaks it into 2-4 specific, focused
sub-questions that together give comprehensive coverage.

Why this helps: a single broad search ("renewable energy 2026") tends to
return shallow, scattered results. Specific sub-questions ("What new solar
technologies launched in 2026?", "How are governments funding renewables
in 2026?") each get a focused, deep search — then we combine them later.
"""

import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
from observability import log_event

load_dotenv()

PLANNER_MODEL = "openai/gpt-oss-120b"

PLANNER_SYSTEM_PROMPT = """You are a research planning agent. Break down a broad \
research topic into 2-4 specific, focused sub-questions that together give \
comprehensive coverage of the topic.

Rules:
- Each sub-question must be specific and searchable (good for a web search engine)
- Cover different angles: current state, causes/background, impact, future trends
- Do NOT answer the questions — only generate them
- Output ONLY valid JSON, nothing else. No markdown fences, no preamble, no explanation.

Output format:
{"sub_questions": ["question 1", "question 2", "question 3"]}"""


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")
    return Groq(api_key=api_key)


def plan_sub_questions(topic: str, verbose: bool = True) -> list[str]:
    """
    Breaks a topic into 2-4 sub-questions.
    Falls back to [topic] itself if the planner fails for any reason —
    the pipeline should never hard-crash just because planning failed.
    """
    client = get_client()

    try:
        response = client.chat.completions.create(
            model=PLANNER_MODEL,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Topic: {topic}"},
            ],
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        usage = getattr(response, "usage", None)
        log_event("llm_call", model=PLANNER_MODEL, function="plan_sub_questions", success=True,
                   prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                   completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                   total_tokens=getattr(usage, "total_tokens", 0) if usage else 0)
        # Strip markdown fences in case the model adds them anyway
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        data = json.loads(raw)

        sub_questions = [q.strip() for q in data.get("sub_questions", []) if q.strip()]
        if not sub_questions:
            raise ValueError("Planner returned an empty sub_questions list")

        sub_questions = sub_questions[:4]  # cap at 4 to control cost/time

        if verbose:
            print(f"[Planner] Generated {len(sub_questions)} sub-questions:")
            for i, q in enumerate(sub_questions, 1):
                print(f"  {i}. {q}")

        return sub_questions

    except Exception as e:
        if verbose:
            print(f"[Planner] Failed ({e}), falling back to the original topic as a single question")
        return [topic]


if __name__ == "__main__":
    test_topic = "Impact of Artificial Intelligence on jobs in 2026"
    questions = plan_sub_questions(test_topic)
    print("\nFinal sub-questions:", questions)
