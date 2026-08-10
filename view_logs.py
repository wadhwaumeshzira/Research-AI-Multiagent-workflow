"""
view_logs.py — Summarizes logs/events.jsonl into a readable report.

Run standalone: python view_logs.py
"""

import json
import os
from collections import defaultdict

LOG_FILE = "logs/events.jsonl"


def load_events():
    if not os.path.exists(LOG_FILE):
        print("No logs found yet. Run the agent first.")
        return []
    events = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def print_summary(events):
    llm_calls = [e for e in events if e["event_type"] == "llm_call"]
    tool_calls = [e for e in events if e["event_type"] == "tool_call"]

    print("=" * 70)
    print("OBSERVABILITY SUMMARY")
    print("=" * 70)
    print(f"Total events logged: {len(events)}")
    print(f"LLM calls: {len(llm_calls)}  |  Tool calls: {len(tool_calls)}\n")

    print("-" * 70)
    print("LLM CALLS BY MODEL")
    print("-" * 70)
    by_model = defaultdict(lambda: {"success": 0, "fail": 0, "durations": []})
    for e in llm_calls:
        model = e.get("model", "unknown")
        by_model[model]["success" if e.get("success") else "fail"] += 1
        by_model[model]["durations"].append(e.get("duration_sec", 0))

    print(f"{'Model':<30} {'Success':<10} {'Fail':<8} {'Avg Duration':<12}")
    for model, stats in by_model.items():
        avg_dur = sum(stats["durations"]) / len(stats["durations"]) if stats["durations"] else 0
        print(f"{model:<30} {stats['success']:<10} {stats['fail']:<8} {avg_dur:.2f}s")

    print("\n" + "-" * 70)
    print("TOKEN USAGE (by model)")
    print("-" * 70)
    by_model_tokens = defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0, "calls": 0})
    for e in llm_calls:
        model = e.get("model", "unknown")
        by_model_tokens[model]["prompt"] += e.get("prompt_tokens", 0) or 0
        by_model_tokens[model]["completion"] += e.get("completion_tokens", 0) or 0
        by_model_tokens[model]["total"] += e.get("total_tokens", 0) or 0
        by_model_tokens[model]["calls"] += 1

    grand_total = 0
    print(f"{'Model':<30} {'Calls':<8} {'Prompt':<10} {'Completion':<12} {'Total':<10}")
    for model, t in by_model_tokens.items():
        grand_total += t["total"]
        print(f"{model:<30} {t['calls']:<8} {t['prompt']:<10} {t['completion']:<12} {t['total']:<10}")
    print(f"\nGrand total tokens used across all calls: {grand_total:,}")

    print("\n" + "-" * 70)
    print("TOOL CALLS BY TOOL")
    print("-" * 70)
    by_tool = defaultdict(lambda: {"success": 0, "fail": 0, "durations": []})
    for e in tool_calls:
        tool = e.get("tool", "unknown")
        by_tool[tool]["success" if e.get("success") else "fail"] += 1
        by_tool[tool]["durations"].append(e.get("duration_sec", 0))

    print(f"{'Tool':<25} {'Success':<10} {'Fail':<8} {'Avg Duration':<12}")
    for tool, stats in by_tool.items():
        avg_dur = sum(stats["durations"]) / len(stats["durations"]) if stats["durations"] else 0
        print(f"{tool:<25} {stats['success']:<10} {stats['fail']:<8} {avg_dur:.2f}s")

    errors = [e for e in events if not e.get("success", True) and e.get("error")]
    if errors:
        print("\n" + "-" * 70)
        print(f"RECENT ERRORS (last {min(5, len(errors))})")
        print("-" * 70)
        for e in errors[-5:]:
            ts = e.get("timestamp", "")[:19]
            src = e.get("model") or e.get("tool", "unknown")
            print(f"[{ts}] {src}: {e['error'][:100]}")

    print("=" * 70)


if __name__ == "__main__":
    events = load_events()
    if events:
        print_summary(events)
