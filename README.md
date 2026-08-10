# Research AI Agent

A multi-agent AI research system that takes a topic and produces a sourced, structured report — planning sub-questions, researching them in parallel, synthesizing the findings, critiquing its own output, fact-checking claims against real search results, and scoring its own confidence before handing the report back.

**[Live demo →](https://research-ai-agent-ui-3tcc.onrender.com)**

---

## What this is

Most "AI research tools" are a single prompt to a chatbot. This is different: it's an engineered pipeline where planning, research, synthesis, quality control, and fact-checking are explicit, separate stages — not one black-box call.

Give it a topic, and it runs through six stages automatically:

1. **Planner** — breaks the topic into 2–4 focused sub-questions covering different angles (current state, causes, impact, outlook)
2. **Researcher** — investigates each sub-question independently and *in parallel* (concurrent threads, not one at a time), using a real web-search tool
3. **Synthesizer** — combines all findings into one cohesive report, in a tone you choose (standard / executive / academic / casual)
4. **Critic** — reviews the report against the original sub-questions and flags gaps; if it finds real issues, the system automatically does one corrective research pass and re-synthesizes
5. **Fact-Checker** — cross-checks every claim in the final report against the raw research notes actually gathered, flagging anything that looks invented rather than sourced
6. **Confidence Scorer** — a deterministic, explainable score (0–100) based on source count, research coverage, critic verdict, and fact-check results — not another LLM call, just math over what already happened

Every session is saved per-user, can be revisited later, can be asked follow-up questions without re-researching, and two topics can be researched independently and diffed into a structured comparison.

---

## Why it's built this way

- **Raw Python for the pipeline, no agent framework.** Planner → Researcher → Synthesizer → Critic → Fact-Checker is a linear dependency chain — each stage needs the previous stage's output and nothing more. Writing the orchestration loop by hand (rather than reaching for LangGraph/CrewAI) means every part of the system is inspectable, not framework magic.
- **Parallel research, not sequential.** Sub-questions are independent of each other, so they're researched concurrently via a thread pool instead of one at a time — a genuine multi-second-per-request speedup with no downside, since each call is I/O-bound (waiting on the LLM API and web search).
- **A dedicated Fact-Checker stage, separate from the Critic.** The Critic checks *structure and coverage* ("did you answer all the sub-questions, are there real sources"). The Fact-Checker checks *grounding* ("is this specific claim actually traceable to something the Researcher found, or did the model just recall it from training"). Splitting these catches a real failure mode: reports that look well-formed and well-sourced but quietly blend in unverified claims.
- **Confidence scoring is math, not a model call.** It's computed from concrete signals already produced by the pipeline (source count, retry count, fact-check flags, report length) — free, instant, deterministic, and fully explainable (every point deducted has a stated reason).
- **MongoDB Atlas for persistence, with per-user isolation.** Every research session, follow-up conversation, and comparison is scoped to a `user_id`, so accounts are genuinely private, not just a UI convenience.
- **Caching by topic hash.** Identical research requests within 24 hours return instantly from a shared cache instead of re-running the whole pipeline — saves both time and free-tier API quota.
- **Rate limiting per user.** A daily cap on research requests (MongoDB-backed, survives restarts) protects the shared free-tier Groq quota from being exhausted by one user.
- **Recency-aware search.** Search results are scanned for stale year references and flagged `[POSSIBLY OUTDATED]` so the Synthesizer can weigh them appropriately instead of treating a 2020 article and a 2026 article as equally current.
- **Full observability.** Every LLM call and tool call is logged with duration, success/failure, and token usage to a flat JSON-lines file — no external account needed, and `view_logs.py` summarizes it into a readable report (by model, by tool, token totals, recent errors).
- **Self-evaluation via LLM-as-judge.** A separate model scores generated reports on factual specificity, source quality, completeness, and clarity — turning "is this agent any good?" into a measurable number instead of a guess.

---

## Architecture

```
                              ┌─────────────┐
                              │    Topic     │
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │   Planner    │  → 2-4 sub-questions
                              └──────┬──────┘
                                     │
                       ┌─────────────▼─────────────┐
                       │   Researcher (parallel)     │
                       │  tool-calling agent loop     │
                       │  per sub-question, N at once │
                       └─────────────┬─────────────┘
                                     │
                              ┌──────▼──────┐
                              │ Synthesizer  │  → combined report (themed)
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                        ┌────▶│    Critic    │
                        │     └──────┬──────┘
                        │            │ NEEDS_REVISION (max N retries)
                        │            ▼
                        └───── re-research gaps, re-synthesize
                                     │ PASS
                              ┌──────▼──────┐
                              │ Fact-Checker │  → flags unsupported claims
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │  Confidence   │  → 0-100 score, explained
                              │    Scoring    │
                              └──────┬──────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              MongoDB (per-user)   Markdown/PDF     Cache (24h TTL)
```

---

## Core features

| Feature | What it does |
|---|---|
| **6-stage self-correcting pipeline** | Planner, parallel Researcher, Synthesizer, Critic with retry loop, Fact-Checker, Confidence Scoring |
| **Accounts** | Signup/login with bcrypt-hashed passwords; every user's research is private |
| **Persistent history** | All sessions saved to MongoDB, browsable and reloadable at any time |
| **Follow-up Q&A** | Ask questions about a saved report without re-researching — grounded only in that report's content |
| **Compare mode** | Research two topics independently, then get a structured side-by-side comparison table |
| **Report themes** | Standard, executive, academic, or casual tone — chosen per request |
| **Confidence scoring** | Transparent 0–100 score with a stated reason for every point deducted |
| **Fact-checking** | Flags specific claims not traceable to actual search results, appended as a report annotation |
| **Caching** | Identical topics researched within 24h return instantly, no re-run |
| **Rate limiting** | Per-user daily request cap, protects shared API quota |
| **Recency filtering** | Search results with only stale year references are flagged and deprioritized |
| **Export** | Markdown and formatted PDF, both downloadable |
| **Shareable links** | Public, no-login URL to view a specific report |
| **REST API** | Full pipeline exposed over FastAPI — `/research`, `/compare`, `/followup`, `/signup`, `/login`, `/history`, `/share` |
| **Streamlit UI** | Browser interface for everything above — no terminal required |
| **Observability** | Every LLM/tool call logged with duration, success/failure, and token usage |
| **Self-evaluation** | LLM-as-judge scoring across factual specificity, source quality, completeness, clarity |

---

## Tech stack

**LLM** — Groq (Llama 3.3 70B for reasoning stages, Llama 3.1 8B for lightweight follow-ups)
**Agent orchestration** — Plain Python (no framework) with `ThreadPoolExecutor` for parallel research
**Search** — DuckDuckGo (`ddgs`), no API key required
**Backend API** — FastAPI
**UI** — Streamlit
**Database** — MongoDB Atlas (accounts, research history, cache, rate limiting)
**Auth** — bcrypt password hashing
**PDF generation** — ReportLab
**Deployment** — Render (Blueprint, two services: UI + API)

---

## Project structure

```
tools.py            Web search tool (with recency flagging)
agent.py             Single tool-calling agent loop (used per sub-question)
planner.py           Breaks a topic into sub-questions
researcher.py         Researches sub-questions in parallel
synthesizer.py        Combines findings into one themed report
critic.py             Reviews report quality, triggers retries
fact_checker.py        Cross-checks claims against raw research notes
confidence.py          Deterministic 0-100 confidence scoring
orchestrator.py        Runs the full 6-stage pipeline end-to-end
cache.py              24h result caching by topic hash
rate_limit.py          Per-user daily request limits
export.py             Markdown / PDF export
db.py                MongoDB connection
auth.py               Signup / login (bcrypt)
memory_mongo.py        Per-user research history (MongoDB)
followup.py           Follow-up Q&A grounded in a saved report
compare.py            Two-topic research + structured comparison
observability.py       Logs every LLM/tool call (duration, tokens, success)
view_logs.py           Summarizes the observability log
evaluation.py          LLM-as-judge self-evaluation across test topics
main.py               Terminal CLI (menu-driven)
app.py                Streamlit UI
api.py                FastAPI REST API
render.yaml            Render Blueprint (deploys both app.py and api.py)
```

---

## Running it locally

```bash
git clone https://github.com/wadhwaumeshzira/Research-AI-Multiagent-workflow.git
cd Research-AI-Multiagent-workflow
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/
```

Then run any of:

```bash
python main.py          # Terminal CLI (menu-driven)
streamlit run app.py    # Browser UI
python api.py           # REST API on http://localhost:8000/docs
```

Utility scripts:

```bash
python view_logs.py     # Summarize observability logs
python evaluation.py    # Run LLM-as-judge self-evaluation
```

---

## Deployment

Deployed on [Render](https://render.com) as two services defined in `render.yaml` — the Streamlit UI and the FastAPI backend, deployed together via Render's Blueprint feature from a single push. Both read `GROQ_API_KEY` and `MONGODB_URI` from environment variables set in the Render dashboard, never committed to the repo.

---

## Known limitations

- **Free-tier cold starts.** The Render free tier spins down after inactivity; the first request after idle time can take ~50 seconds to wake up.
- **Free-tier token limits.** Groq's free tier has daily token caps — heavy use of the pipeline (especially Compare mode, which runs two full pipelines) can exhaust it. The rate limiter and caching layer both exist specifically to reduce this risk.
- **Web search has no API key**, so it relies on DuckDuckGo's public search — results can occasionally be sparser than a paid search API would provide.
- **Streaming is not implemented.** Reports are generated in full before being returned, not streamed token-by-token.

---

## License

MIT
