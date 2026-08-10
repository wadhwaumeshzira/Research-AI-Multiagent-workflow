"""
api.py — REST API for the research agent.

Exposes the full pipeline over HTTP so it can be called from anything —
a website, Postman, curl, a mobile app, or later a Streamlit frontend.

Run: python api.py   (or: uvicorn api:app --reload)
Docs auto-generated at: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

from orchestrator import run_pipeline
from confidence import score_confidence
from export import export_markdown, export_pdf
from compare import run_comparison
from followup import ask_followup
import memory_mongo as memory
import auth
import rate_limit

app = FastAPI(
    title="Research AI Agent API",
    description="Multi-agent research pipeline — Planner, Researcher, Synthesizer, Critic, Fact-Checker",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ──────────────────────────────────────
class SignupRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class ResearchRequest(BaseModel):
    user_id: str
    topic: str
    theme: Optional[str] = "standard"  # "standard", "executive", "academic", "casual"


class CompareRequest(BaseModel):
    user_id: str
    topic_a: str
    topic_b: str


class FollowupRequest(BaseModel):
    report: str
    question: str
    chat_history: Optional[list] = None


# ── Root / health ────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Research AI Agent API",
        "status": "running",
        "endpoints": ["/signup", "/login", "/research", "/compare", "/followup",
                      "/history/{user_id}", "/session/{user_id}/{session_id}", "/health"],
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


# ── Auth ─────────────────────────────────────────────────
@app.post("/signup")
def signup(request: SignupRequest):
    result = auth.signup(request.username, request.password)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/login")
def login(request: LoginRequest):
    result = auth.login(request.username, request.password)
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["message"])
    return result


# ── Research ─────────────────────────────────────────────
@app.post("/research")
def research(request: ResearchRequest):
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty")

    limit_check = rate_limit.check_and_record(request.user_id)
    if not limit_check["allowed"]:
        raise HTTPException(status_code=429, detail=f"Daily research limit reached ({limit_check['limit']}/day). Try again tomorrow.")

    try:
        result = run_pipeline(request.topic, verbose=False, theme=request.theme)
        conf = score_confidence(result)
        session_id = memory.save_session(request.user_id, request.topic, result["report"], metadata={
            "confidence": conf, "critic_verdict": result["critic_verdict"], "retries_used": result["retries_used"]
        })

        return {
            "session_id": session_id,
            "topic": request.topic,
            "report": result["report"],
            "critic_verdict": result["critic_verdict"],
            "confidence": conf,
            "retries_used": result["retries_used"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")


@app.post("/compare")
def compare(request: CompareRequest):
    if not request.topic_a.strip() or not request.topic_b.strip():
        raise HTTPException(status_code=400, detail="Both topics are required")

    try:
        result = run_comparison(request.topic_a, request.topic_b, verbose=False)
        memory.save_session(request.user_id, request.topic_a, result["report_a"])
        memory.save_session(request.user_id, request.topic_b, result["report_b"])
        memory.save_session(request.user_id, f"Comparison: {request.topic_a} vs {request.topic_b}", result["comparison"])

        return {
            "comparison": result["comparison"],
            "report_a": result["report_a"],
            "report_b": result["report_b"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.post("/followup")
def followup(request: FollowupRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        history_pairs = [(h["q"], h["a"]) for h in request.chat_history] if request.chat_history else None
        answer = ask_followup(request.report, request.question, chat_history=history_pairs)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Follow-up failed: {str(e)}")


# ── History / memory ─────────────────────────────────────
@app.get("/history/{user_id}")
def get_history(user_id: str, limit: int = 15):
    return {"sessions": memory.list_recent(user_id, limit=limit)}


@app.get("/session/{user_id}/{session_id}")
def get_session(user_id: str, session_id: str):
    session = memory.get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.get("/share/{session_id}")
def get_shared_report(session_id: str):
    """
    Public, no-auth endpoint for sharing a report via link. Looks up the
    session across all users by its raw MongoDB ID (the session_id itself
    already acts as an unguessable token — a 24-char hex ObjectId).
    Only returns the report content, not the owner's identity.
    """
    session = memory.get_session_public(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Shared report not found")
    return {"topic": session["topic"], "report": session["report"], "created_at": session["created_at"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
