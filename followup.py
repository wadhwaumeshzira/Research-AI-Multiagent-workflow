"""
followup.py — Ask follow-up questions about an existing report.

Answers questions using ONLY the report's own content (and its underlying
research notes if available) — no new web search. Fast, cheap, and keeps
answers grounded in what was already researched instead of inventing new info.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

FOLLOWUP_MODEL = "llama-3.1-8b-instant"  # fast + cheap is fine for this, no tool use needed

FOLLOWUP_SYSTEM_PROMPT = """You are answering a follow-up question about a research report \
that has already been generated. Answer using ONLY the information in the report below — \
do not invent new facts or use outside knowledge.

If the report doesn't contain enough information to answer the question, say so clearly \
instead of guessing.

REPORT:
{report}"""


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")
    return Groq(api_key=api_key)


def ask_followup(report: str, question: str, chat_history: list[tuple[str, str]] = None) -> str:
    """
    Answers a follow-up question grounded in the given report.
    chat_history: optional list of (question, answer) pairs from earlier in this session,
                  so multi-turn follow-ups keep context.
    """
    client = get_client()
    system_prompt = FOLLOWUP_SYSTEM_PROMPT.format(report=report)

    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        for q, a in chat_history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=FOLLOWUP_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    sample_report = """## Research Report: AI and Jobs
### Key Findings
* AI is automating repetitive tasks in manufacturing and data entry.
* New roles are emerging in AI oversight, prompt engineering, and data science.
* McKinsey estimates up to 30% of hours worked could be automated by 2030."""

    q = "What percentage of hours could be automated?"
    answer = ask_followup(sample_report, q)
    print(f"Q: {q}\nA: {answer}")
