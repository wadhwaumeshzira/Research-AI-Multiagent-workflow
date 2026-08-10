"""
observability.py — Logs every LLM call and tool call to a JSON-lines file.

Each line is one event: timestamp, type, duration, success/failure, and
whatever extra fields are relevant (model name, tool name, error message).
Cheap to add, genuinely useful when debugging intermittent failures later.
"""

import json
import time
import os
from datetime import datetime
from functools import wraps

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "events.jsonl")


def _ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def log_event(event_type: str, **fields):
    """Writes one event as a JSON line. event_type: 'llm_call', 'tool_call', 'pipeline_stage'."""
    _ensure_log_dir()
    entry = {"timestamp": datetime.now().isoformat(), "event_type": event_type, **fields}
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def track_llm_call(model_name: str):
    """
    Decorator for functions that make an LLM call. Logs duration and success/failure
    automatically. Usage: @track_llm_call("llama-3.3-70b-versatile")
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                log_event("llm_call", model=model_name, function=func.__name__,
                           duration_sec=round(time.time() - start, 3), success=True)
                return result
            except Exception as e:
                log_event("llm_call", model=model_name, function=func.__name__,
                           duration_sec=round(time.time() - start, 3), success=False, error=str(e)[:200])
                raise
        return wrapper
    return decorator


def track_tool_call(tool_name: str, duration_sec: float, success: bool = True, error: str = None):
    """Manual logging for tool calls (can't use a decorator since tools are dispatched dynamically)."""
    log_event("tool_call", tool=tool_name, duration_sec=round(duration_sec, 3),
               success=success, error=error[:200] if error else None)


def log_pipeline_stage(stage_name: str, **fields):
    """Logs a pipeline stage transition (planner, researcher, synthesizer, etc.)"""
    log_event("pipeline_stage", stage=stage_name, **fields)


def extract_tokens(response) -> dict:
    """
    Pulls token usage out of a Groq API response (OpenAI-compatible format).
    Returns {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    or zeros if usage info isn't available for some reason.
    """
    usage = getattr(response, "usage", None)
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }
