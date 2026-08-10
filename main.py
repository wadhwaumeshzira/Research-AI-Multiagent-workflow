"""
main.py — Unified CLI for the whole research agent system, now with accounts.

Flow: login/signup first -> then the same menu as before, but everything
(research history, follow-ups) is now scoped to the logged-in user via MongoDB.
"""

from orchestrator import run_pipeline
from confidence import score_confidence
from export import export_markdown, export_pdf
import memory_mongo as memory
from followup import ask_followup
from compare import run_comparison
import auth
import rate_limit


def do_login_or_signup():
    """Returns (user_id, username) once login/signup succeeds."""
    while True:
        print("\n" + "=" * 60)
        print("WELCOME — RESEARCH AI AGENT")
        print("=" * 60)
        print("1. Login")
        print("2. Sign up")
        print("3. Quit")
        choice = input("\nChoose an option (1-3): ").strip()

        if choice == "1":
            username = input("Username: ").strip()
            password = input("Password: ").strip()
            result = auth.login(username, password)
            if result["success"]:
                print(f"\nWelcome back, {username}!")
                return result["user_id"], username
            else:
                print(f"\nLogin failed: {result['message']}")

        elif choice == "2":
            username = input("Choose a username: ").strip()
            password = input("Choose a password (min 6 chars): ").strip()
            result = auth.signup(username, password)
            if result["success"]:
                print(f"\nAccount created! You can now log in.")
            else:
                print(f"\nSignup failed: {result['message']}")

        elif choice == "3":
            print("Goodbye!")
            exit(0)
        else:
            print("Invalid choice, try again.")


def do_new_research(user_id: str):
    limit_check = rate_limit.check_and_record(user_id)
    if not limit_check["allowed"]:
        print(f"\nYou've hit your daily research limit ({limit_check['limit']}/day). Try again tomorrow.")
        return
    print(f"\n[Rate limit] {limit_check['remaining']} research request(s) left today.")

    topic = input("\nEnter a research topic: ").strip()
    if not topic:
        print("Topic cannot be empty.")
        return

    similar = memory.find_similar(user_id, topic)
    if similar:
        print("\n[Memory] Found similar past research of yours:")
        for s in similar:
            print(f"  [{s['session_id']}] {s['topic']} (similarity: {s['similarity']})")
        reuse = input("Reuse one of these instead of researching again? Enter session ID or press Enter to research fresh: ").strip()
        if reuse:
            session = memory.get_session(user_id, reuse)
            if session:
                print(f"\n{'='*60}\n{session['report']}\n{'='*60}")
                return
            else:
                print("Session ID not found, proceeding with fresh research.")

    result = run_pipeline(topic)
    conf = score_confidence(result)

    print(f"\n{'='*60}\nFINAL REPORT\n{'='*60}")
    print(result["report"])
    print(f"\n{'='*60}")
    print(f"Critic verdict: {result['critic_verdict']['verdict']}  |  Retries used: {result['retries_used']}")
    print(f"Confidence: {conf['score']}/100 ({conf['label']})")
    print(f"{'='*60}")

    session_id = memory.save_session(user_id, topic, result["report"], metadata={
        "confidence": conf, "critic_verdict": result["critic_verdict"], "retries_used": result["retries_used"]
    })
    print(f"\n[Memory] Saved as session {session_id}")

    choice = input("\nExport this report? [m]arkdown / [p]df / [b]oth / [n]o: ").strip().lower()
    if choice in ("m", "b"):
        export_markdown(topic, result, conf)
    if choice in ("p", "b"):
        export_pdf(topic, result, conf)


def do_browse_memory(user_id: str):
    sessions = memory.list_recent(user_id, limit=15)
    if not sessions:
        print("\nNo saved research yet.")
        return

    print("\nYour recent research sessions:")
    for s in sessions:
        print(f"  [{s['session_id']}] {s['topic']}  ({s['created_at'][:19]})")

    sid = input("\nEnter a session ID to view full report, or press Enter to go back: ").strip()
    if sid:
        session = memory.get_session(user_id, sid)
        if session:
            print(f"\n{'='*60}\n{session['report']}\n{'='*60}")
        else:
            print("Session not found (or it doesn't belong to you).")


def do_followup(user_id: str):
    sid = input("\nEnter the session ID to ask about (see menu 2 to browse): ").strip()
    session = memory.get_session(user_id, sid)
    if not session:
        print("Session not found (or it doesn't belong to you).")
        return

    print(f"\nAsking questions about: {session['topic']}")
    print("Type 'back' to return to the main menu.\n")

    chat_history = []
    while True:
        question = input("Your question: ").strip()
        if not question or question.lower() == "back":
            break
        answer = ask_followup(session["report"], question, chat_history=chat_history)
        print(f"\nA: {answer}\n")
        chat_history.append((question, answer))


def do_compare(user_id: str):
    topic_a = input("\nEnter first topic: ").strip()
    topic_b = input("Enter second topic: ").strip()
    if not topic_a or not topic_b:
        print("Both topics are required.")
        return

    result = run_comparison(topic_a, topic_b)
    print(f"\n{'='*60}\nCOMPARISON\n{'='*60}")
    print(result["comparison"])

    memory.save_session(user_id, topic_a, result["report_a"])
    memory.save_session(user_id, topic_b, result["report_b"])
    memory.save_session(user_id, f"Comparison: {topic_a} vs {topic_b}", result["comparison"])
    print("\n[Memory] All three (report A, report B, comparison) saved.")


def main():
    user_id, username = do_login_or_signup()

    while True:
        print("\n" + "=" * 60)
        print(f"RESEARCH AI AGENT — logged in as {username}")
        print("=" * 60)
        print("1. New research")
        print("2. Browse past research")
        print("3. Ask a follow-up question about a saved report")
        print("4. Compare two topics")
        print("5. Logout / Quit")

        choice = input("\nChoose an option (1-5): ").strip()

        if choice == "1":
            do_new_research(user_id)
        elif choice == "2":
            do_browse_memory(user_id)
        elif choice == "3":
            do_followup(user_id)
        elif choice == "4":
            do_compare(user_id)
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    main()
