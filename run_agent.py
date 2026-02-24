"""
Run the RAG agent from the command line.
  python run_agent.py "Your question here"

Requires the index to be built first: python build_index.py
Requires GOOGLE_API_KEY in .env for Gemini.
"""
import asyncio
import sys

# Suppress known multiprocess ResourceTracker shutdown noise on Windows (Python 3.12)
def _quiet_unraisable(hook):
    def wrapper(unraisable):
        try:
            exc_value = getattr(unraisable, "exc_value", None)
            exc_type = getattr(unraisable, "exc_type", None)
            msg = str(exc_value or "")
            # Suppress ResourceTracker / RLock recursion_count errors
            if ("_recursion_count" in msg or 
                "ResourceTracker" in msg or 
                (exc_type and "ResourceTracker" in str(exc_type)) or
                (exc_value and isinstance(exc_value, AttributeError) and "_recursion_count" in str(exc_value))):
                return  # Silently ignore
        except Exception:
            pass  # If suppression itself fails, ignore
        if hook is not None:
            hook(unraisable)
    return wrapper
_default_hook = getattr(sys, "unraisablehook", None)
sys.unraisablehook = _quiet_unraisable(_default_hook)

import os
from dotenv import load_dotenv
load_dotenv()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agent import root_agent

APP_NAME = "rag_app"
USER_ID = "default_user"


async def run_query(query: str) -> str:
    try:
        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
        )
        runner = Runner(
            agent=root_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )
        content = types.Content(role="user", parts=[types.Part(text=query)])
        final_text = ""
        events_received = 0
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session.id if hasattr(session, "id") else getattr(session.session, "id"),
            new_message=content,
        ):
            events_received += 1
            if event.is_final_response() and event.content and event.content.parts:
                final_text = event.content.parts[0].text
                break
        
        if not final_text:
            return f"(No response - received {events_received} events. Check API key and index.)"
        return final_text
    except Exception as e:
        return f"(Error: {str(e)})"


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_agent.py \"Your question\"")
        sys.exit(1)
    
    # Check API key
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment. Check your .env file.")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    print("Query:", query)
    print("Processing...")
    answer = asyncio.run(run_query(query))
    print("Answer:", answer)


if __name__ == "__main__":
    main()
