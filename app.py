"""
app.py — Streamlit UI for the Research AI Agent.

Wraps everything we've built: auth, the full pipeline, memory/history,
follow-up Q&A, compare mode, themes, confidence scoring, and export —
all in a browser interface instead of the terminal.

Run: streamlit run app.py
"""

import streamlit as st
from datetime import datetime

from orchestrator import run_pipeline
from confidence import score_confidence
from export import export_markdown, export_pdf
import memory_mongo as memory
from followup import ask_followup
from compare import run_comparison
import auth
import rate_limit

st.set_page_config(
    page_title="Research AI Agent",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom styling ──────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0F172A; }
    section[data-testid="stSidebar"] { background-color: #111827; }
    .confidence-badge {
        display: inline-block; padding: 4px 14px; border-radius: 999px;
        font-weight: 600; font-size: 0.85rem;
    }
    .conf-high { background-color: #064E3B; color: #6EE7B7; }
    .conf-moderate { background-color: #78350F; color: #FCD34D; }
    .conf-low { background-color: #7F1D1D; color: #FCA5A5; }
    .stage-pill {
        display: inline-block; background-color: #1E293B; color: #93C5FD;
        padding: 3px 10px; border-radius: 6px; font-size: 0.8rem; margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ──────────────────────────────────
def init_state():
    defaults = {
        "logged_in": False,
        "user_id": None,
        "username": None,
        "current_result": None,
        "current_topic": None,
        "current_confidence": None,
        "followup_history": [],
        "followup_session": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def confidence_badge(conf: dict) -> str:
    cls = {"High": "conf-high", "Moderate": "conf-moderate", "Low": "conf-low"}.get(conf["label"], "conf-moderate")
    return f'<span class="confidence-badge {cls}">Confidence: {conf["score"]}/100 ({conf["label"]})</span>'


# ── Auth screen ──────────────────────────────────────────────
def render_auth():
    st.title("🔎 Research AI Agent")
    st.caption("Multi-agent research pipeline — Planner → Researcher → Synthesizer → Critic → Fact-Checker")

    tab_login, tab_signup = st.tabs(["Login", "Sign up"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", use_container_width=True)
            if submitted:
                result = auth.login(username, password)
                if result["success"]:
                    st.session_state.logged_in = True
                    st.session_state.user_id = result["user_id"]
                    st.session_state.username = username.strip().lower()
                    st.rerun()
                else:
                    st.error(result["message"])

    with tab_signup:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username", key="signup_username")
            new_password = st.text_input("Choose a password (min 6 chars)", type="password", key="signup_password")
            submitted = st.form_submit_button("Sign up", use_container_width=True)
            if submitted:
                result = auth.signup(new_username, new_password)
                if result["success"]:
                    st.success("Account created! Switch to the Login tab to sign in.")
                else:
                    st.error(result["message"])


# ── Sidebar nav ──────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        used_today = rate_limit.get_usage_today(st.session_state.user_id)
        remaining = max(0, rate_limit.DAILY_LIMIT - used_today)
        st.caption(f"Research requests left today: **{remaining}/{rate_limit.DAILY_LIMIT}**")

        st.divider()
        page = st.radio(
            "Navigate",
            ["🔍 New Research", "📚 History", "💬 Follow-up Q&A", "⚖️ Compare Topics"],
            label_visibility="collapsed",
        )
        st.divider()

        if st.button("Log out", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        return page


# ── Page: New Research ───────────────────────────────────────
def render_new_research():
    st.header("🔍 New Research")

    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input("Research topic", placeholder="e.g. Latest developments in renewable energy 2026")
    with col2:
        theme = st.selectbox("Report style", ["standard", "executive", "academic", "casual"])

    use_cache = st.checkbox("Use cached result if available (faster)", value=True)

    if st.button("Run Research", type="primary", use_container_width=True):
        if not topic.strip():
            st.warning("Enter a topic first.")
            return

        limit_check = rate_limit.check_and_record(st.session_state.user_id)
        if not limit_check["allowed"]:
            st.error(f"Daily research limit reached ({limit_check['limit']}/day). Try again tomorrow.")
            return

        progress = st.empty()
        with st.spinner("Running the research pipeline..."):
            progress.info("Planning sub-questions → Researching → Synthesizing → Critiquing → Fact-checking...")
            result = run_pipeline(topic, verbose=False, use_cache=use_cache, theme=theme)
            conf = score_confidence(result)

        progress.empty()

        session_id = memory.save_session(st.session_state.user_id, topic, result["report"], metadata={
            "confidence": conf, "critic_verdict": result["critic_verdict"], "retries_used": result["retries_used"]
        })

        st.session_state.current_result = result
        st.session_state.current_topic = topic
        st.session_state.current_confidence = conf
        st.session_state.current_session_id = session_id
        st.success(f"Research complete — saved as session {session_id}")

    # Show the most recent result, if any
    if st.session_state.current_result:
        result = st.session_state.current_result
        conf = st.session_state.current_confidence

        st.divider()
        st.markdown(confidence_badge(conf), unsafe_allow_html=True)
        st.markdown(
            f'<span class="stage-pill">Critic: {result["critic_verdict"]["verdict"]}</span> '
            f'<span class="stage-pill">Retries: {result["retries_used"]}</span> '
            f'<span class="stage-pill">Sources: {conf["unique_sources"]}</span>',
            unsafe_allow_html=True,
        )

        with st.expander("Why this confidence score?"):
            for r in conf["reasons"]:
                st.write(f"- {r}")
            if not conf["reasons"]:
                st.write("No penalties applied — report passed all checks cleanly.")

        st.markdown("### 📄 Report")
        st.markdown(result["report"])

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("📥 Export as Markdown", use_container_width=True):
                path = export_markdown(st.session_state.current_topic, result, conf)
                st.success(f"Saved: {path}")
        with col_b:
            if st.button("📥 Export as PDF", use_container_width=True):
                path = export_pdf(st.session_state.current_topic, result, conf)
                st.success(f"Saved: {path}")
        with col_c:
            share_url = f"http://localhost:8000/share/{st.session_state.get('current_session_id', '')}"
            st.text_input("Share link (requires api.py running)", value=share_url, disabled=True)


# ── Page: History ─────────────────────────────────────────────
def render_history():
    st.header("📚 Your Research History")

    sessions = memory.list_recent(st.session_state.user_id, limit=25)
    if not sessions:
        st.info("No saved research yet — run something in 'New Research' first.")
        return

    for s in sessions:
        with st.expander(f"**{s['topic']}**  ·  {s['created_at'][:19]}  ·  `{s['session_id']}`"):
            full = memory.get_session(st.session_state.user_id, s["session_id"])
            if full:
                meta = full.get("metadata", {})
                if meta.get("confidence"):
                    st.markdown(confidence_badge(meta["confidence"]), unsafe_allow_html=True)
                st.markdown(full["report"])

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Ask a follow-up about this", key=f"followup_{s['session_id']}"):
                        st.session_state.followup_session = s["session_id"]
                        st.session_state.followup_history = []
                        st.info("Switch to the 'Follow-up Q&A' tab in the sidebar.")
                with col_b:
                    if st.button("🗑️ Delete", key=f"delete_{s['session_id']}"):
                        memory.delete_session(st.session_state.user_id, s["session_id"])
                        st.rerun()


# ── Page: Follow-up Q&A ─────────────────────────────────────────
def render_followup():
    st.header("💬 Follow-up Q&A")

    sessions = memory.list_recent(st.session_state.user_id, limit=25)
    if not sessions:
        st.info("No saved research yet — run something in 'New Research' first.")
        return

    options = {f"{s['topic']} ({s['created_at'][:10]})": s["session_id"] for s in sessions}
    default_index = 0
    if st.session_state.followup_session:
        ids = list(options.values())
        if st.session_state.followup_session in ids:
            default_index = ids.index(st.session_state.followup_session)

    selected_label = st.selectbox("Choose a report to ask about", list(options.keys()), index=default_index)
    selected_id = options[selected_label]

    if selected_id != st.session_state.followup_session:
        st.session_state.followup_session = selected_id
        st.session_state.followup_history = []

    session = memory.get_session(st.session_state.user_id, selected_id)
    if not session:
        st.error("Session not found.")
        return

    with st.expander("View full report"):
        st.markdown(session["report"])

    st.divider()
    for q, a in st.session_state.followup_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)

    question = st.chat_input("Ask a question about this report...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.spinner("Thinking..."):
            answer = ask_followup(session["report"], question, chat_history=st.session_state.followup_history)
        with st.chat_message("assistant"):
            st.write(answer)
        st.session_state.followup_history.append((question, answer))


# ── Page: Compare ─────────────────────────────────────────────
def render_compare():
    st.header("⚖️ Compare Two Topics")

    col1, col2 = st.columns(2)
    with col1:
        topic_a = st.text_input("First topic", placeholder="e.g. Electric vehicles in India")
    with col2:
        topic_b = st.text_input("Second topic", placeholder="e.g. Electric vehicles in China")

    if st.button("Run Comparison", type="primary", use_container_width=True):
        if not topic_a.strip() or not topic_b.strip():
            st.warning("Enter both topics first.")
            return

        with st.spinner(f"Researching '{topic_a}' and '{topic_b}' independently, then comparing..."):
            result = run_comparison(topic_a, topic_b, verbose=False)

        memory.save_session(st.session_state.user_id, topic_a, result["report_a"])
        memory.save_session(st.session_state.user_id, topic_b, result["report_b"])
        memory.save_session(st.session_state.user_id, f"Comparison: {topic_a} vs {topic_b}", result["comparison"])

        st.success("Comparison complete and saved to history.")
        st.markdown("### Comparison")
        st.markdown(result["comparison"])

        col_a, col_b = st.columns(2)
        with col_a:
            with st.expander(f"Full report: {topic_a}"):
                st.markdown(result["report_a"])
        with col_b:
            with st.expander(f"Full report: {topic_b}"):
                st.markdown(result["report_b"])


# ── Main ────────────────────────────────────────────────────
def main():
    if not st.session_state.logged_in:
        render_auth()
        return

    page = render_sidebar()

    if page == "🔍 New Research":
        render_new_research()
    elif page == "📚 History":
        render_history()
    elif page == "💬 Follow-up Q&A":
        render_followup()
    elif page == "⚖️ Compare Topics":
        render_compare()


if __name__ == "__main__":
    main()
