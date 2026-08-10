"""
tools.py
Tools available to research agents.

Using DuckDuckGo search (via ddgs) instead of Tavily because it needs
NO API key — only requirement for this project is a Groq key.
"""

import re
from datetime import datetime
from ddgs import DDGS

CURRENT_YEAR = datetime.now().year


def _looks_stale(text: str) -> bool:
    """Flags results that mention a year 3+ years old and no recent year — a rough
    heuristic to catch outdated content mixed into search results."""
    years_mentioned = [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]
    if not years_mentioned:
        return False
    newest_mentioned = max(years_mentioned)
    return newest_mentioned <= CURRENT_YEAR - 3


def search_web(query: str, max_results: int = 5) -> str:
    """
    Searches the web for a query and returns formatted results with sources.
    Results are re-ordered to put likely-current content first, and stale-looking
    results (old years mentioned, nothing recent) are labeled so the model can
    weigh them appropriately instead of treating everything as equally current.
    """
    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return "No results found."

    scored = []
    for r in results:
        text = f"{r.get('title', '')} {r.get('body', '')}"
        scored.append((_looks_stale(text), r))
    scored.sort(key=lambda x: x[0])  # non-stale (False) first

    formatted = []
    for i, (stale, r) in enumerate(scored, 1):
        title = r.get("title", "Untitled")
        body = r.get("body", "")
        url = r.get("href", "")
        tag = " [POSSIBLY OUTDATED]" if stale else ""
        formatted.append(f"[{i}]{tag} {title}\n{body}\nSource: {url}")

    return "\n\n".join(formatted)


# Tool schema in the format Groq's function-calling API expects
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the live web for current information on a topic. Use this for facts, news, statistics, or anything that might have changed recently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific and targeted.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# Maps tool name -> actual Python function, so the agent loop can dispatch calls
TOOL_REGISTRY = {
    "search_web": search_web,
}
