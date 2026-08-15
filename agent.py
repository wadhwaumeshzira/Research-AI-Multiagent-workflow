"""
agent.py — STAGE 1: Single Agent Baseline

This is the simplest possible "agentic" loop:
1. Send the user's question + available tools to the LLM
2. If the LLM wants to call a tool, run it and feed the result back
3. Repeat until the LLM gives a final text answer (no more tool calls)

No framework. This IS what LangGraph/CrewAI do under the hood — we're
building it by hand first so it's never a black box.
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

from tools import TOOL_DEFINITIONS, TOOL_REGISTRY
from observability import log_event, track_tool_call, extract_tokens
import time

load_dotenv()

MODEL = "openai/gpt-oss-120b"
MAX_ITERATIONS = 6  # safety cap so a confused agent can't loop forever


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to a .env file in this folder.")
    return Groq(api_key=api_key)


SYSTEM_PROMPT = """You are a research assistant. Use the search_web tool to find \
current, accurate information before answering. Always cite your sources with \
URLs. Don't make up facts — if the search results don't cover something, say so.

When you have enough information, give a clear, well-organized final answer."""


def run_agent(question: str, verbose: bool = True) -> dict:
    """
    Runs the single-agent tool-calling loop for one question.
    Returns dict with: answer (str), steps (list of str, for logging/UI)
    """
    client = get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps = []

    for iteration in range(MAX_ITERATIONS):
        _llm_start = time.time()
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.2,
                parallel_tool_calls=False,  # Groq's tool-calling is flaky with parallel calls
            )
            tokens = extract_tokens(response)
            log_event("llm_call", model=MODEL, function="run_agent",
                      duration_sec=round(time.time() - _llm_start, 3), success=True, **tokens)
        except Exception as e:
            log_event("llm_call", model=MODEL, function="run_agent",
                      duration_sec=round(time.time() - _llm_start, 3), success=False, error=str(e)[:200])
            # Groq sometimes rejects a malformed function call ("BadRequestError: Failed to
            # call a function"). Retry once without tools so we still get an answer.
            steps.append(f"[Tool-call error, retrying without tools] {e}")
            if verbose:
                print(f"[Tool-call error, retrying without tools] {e}")
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages + [{
                    "role": "user",
                    "content": "(A tool call failed. Please answer directly using what you already know, without calling any tools.)"
                }],
                temperature=0.2,
            )
            msg = response.choices[0].message
            steps.append("[Final answer generated after fallback]")
            return {"answer": msg.content, "steps": steps}

        msg = response.choices[0].message

        # Case 1: model wants to call one or more tools
        if msg.tool_calls:
            # Append the assistant's tool-call request to history
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                step_msg = f"[Tool call] {tool_name}({args})"
                steps.append(step_msg)
                if verbose:
                    print(step_msg)

                tool_fn = TOOL_REGISTRY.get(tool_name)
                _tool_start = time.time()
                if tool_fn is None:
                    result = f"Error: unknown tool '{tool_name}'"
                    track_tool_call(tool_name, time.time() - _tool_start, success=False, error="unknown tool")
                else:
                    try:
                        result = tool_fn(**args)
                        track_tool_call(tool_name, time.time() - _tool_start, success=True)
                    except Exception as e:
                        result = f"Tool error: {e}"
                        track_tool_call(tool_name, time.time() - _tool_start, success=False, error=str(e))

                # Feed the tool's result back to the model
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

            continue  # loop again — model sees tool results next

        # Case 2: model gave a final answer, no more tools needed
        answer = msg.content
        steps.append("[Final answer generated]")
        return {"answer": answer, "steps": steps}

    # Safety fallback if we hit MAX_ITERATIONS without a final answer
    return {
        "answer": "I wasn't able to reach a final answer within the step limit. Try a more specific question.",
        "steps": steps,
    }


if __name__ == "__main__":
    topic = "What are the latest developments in renewable energy in 2026?"
    print(f"Researching: {topic}\n{'='*60}")
    result = run_agent(topic)
    print(f"\n{'='*60}\nFINAL ANSWER:\n{'='*60}")
    print(result["answer"])


# ── Used by researcher.py to run this same loop per sub-question ──
def run_agent_quiet(question: str) -> dict:
    """Same as run_agent but with verbose=False — used inside the multi-agent pipeline
    where we don't want every sub-question's tool calls printed to stdout directly;
    the caller (researcher.py) prints its own progress labels instead."""
    return run_agent(question, verbose=False)
