"""
researcher.py — Researcher Agent (now parallel)

Researches each sub-question CONCURRENTLY instead of one at a time, using a
thread pool. Since each call is mostly waiting on network I/O (Groq API +
web search), threads give a real speedup without needing async/await syntax.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from agent import run_agent_quiet

MAX_WORKERS = 4  # cap concurrent requests so we don't blow through rate limits


def research_all(sub_questions: list[str], verbose: bool = True) -> list[dict]:
    """
    Researches each sub-question concurrently.
    Returns findings in the SAME ORDER as sub_questions (order is preserved
    even though execution is parallel).
    """
    if not sub_questions:
        return []

    results = [None] * len(sub_questions)

    def _research_one(index_question):
        index, question = index_question
        try:
            result = run_agent_quiet(question)
            return index, {
                "question": question,
                "answer": result["answer"],
                "steps": result["steps"],
            }
        except Exception as e:
            return index, {
                "question": question,
                "answer": f"(Research failed for this sub-question: {e})",
                "steps": [],
            }

    if verbose:
        print(f"[Researcher] Researching {len(sub_questions)} sub-questions in parallel (max {MAX_WORKERS} at once)...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_research_one, (i, q)) for i, q in enumerate(sub_questions)]
        for future in as_completed(futures):
            index, finding = future.result()
            results[index] = finding
            if verbose:
                print(f"[Researcher] Done: {finding['question'][:60]}")

    return results


if __name__ == "__main__":
    from planner import plan_sub_questions
    import time

    topic = "Impact of Artificial Intelligence on jobs in 2026"
    questions = plan_sub_questions(topic)

    start = time.time()
    results = research_all(questions)
    print(f"\nTotal time: {time.time() - start:.1f}s (parallel)")

    print("\n" + "=" * 60)
    print("RESEARCH FINDINGS")
    print("=" * 60)
    for r in results:
        print(f"\nQ: {r['question']}")
        print(f"A: {r['answer'][:300]}...")
