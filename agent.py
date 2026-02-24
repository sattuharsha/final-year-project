import sys

# Suppress known multiprocess ResourceTracker shutdown noise on Windows (Python 3.12)
def _quiet_unraisable(hook):
    def wrapper(unraisable):
        try:
            exc_value = getattr(unraisable, "exc_value", None)
            exc_type = getattr(unraisable, "exc_type", None)
            msg = str(exc_value or "")
            if ("_recursion_count" in msg or 
                "ResourceTracker" in msg or 
                (exc_type and "ResourceTracker" in str(exc_type)) or
                (exc_value and isinstance(exc_value, AttributeError) and "_recursion_count" in str(exc_value))):
                return
        except Exception:
            pass
        if hook is not None:
            hook(unraisable)
    return wrapper
_default_hook = getattr(sys, "unraisablehook", None)
sys.unraisablehook = _quiet_unraisable(_default_hook)

from google.adk.agents.llm_agent import Agent

from rag.retrieval import search_knowledge_base

root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='A helpful assistant that answers questions using a knowledge base of FEVER (fact verification) and RAGTruth (QA/summary) data. Use the search tool for factual or verification questions.',
    instruction=(
        'You are a helpful assistant. For factual or verification questions, call search_knowledge_base first. '
        'If the retrieved passages contain a clear answer, use them and say your answer is based on the retrieved evidence. '
        'If the retrieved content does not contain the answer or is empty, you may still answer from your general knowledge and say so (e.g. "The knowledge base did not have this; from general knowledge: ..."). '
        'Do not refuse to answer common factual questions (e.g. capitals, dates) when retrieval has nothing relevant—answer from your knowledge and note that it was not from the index.'
    ),
    tools=[search_knowledge_base],
)
