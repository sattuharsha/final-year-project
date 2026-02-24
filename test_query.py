"""
Quick test script to diagnose query issues.
Run: python test_query.py "Your question"
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

def test_basic():
    """Test basic components."""
    print("=== Testing Basic Components ===\n")
    
    # 1. Check API key
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if api_key:
        print(f"✓ API key found: {api_key[:10]}...")
    else:
        print("✗ API key NOT found in .env")
        return False
    
    # 2. Check ChromaDB collection
    try:
        from rag.retrieval import get_collection
        coll = get_collection()
        count = coll.count()
        print(f"✓ ChromaDB collection found: {count} documents")
        if count == 0:
            print("  WARNING: Collection is empty! Run: python build_index.py")
            return False
    except Exception as e:
        print(f"✗ ChromaDB error: {e}")
        return False
    
    # 3. Test retrieval
    try:
        from rag.retrieval import search_knowledge_base
        result = search_knowledge_base("test", top_k=1)
        if result.get("status") == "success" and result.get("count", 0) > 0:
            print("✓ Retrieval working")
        else:
            print(f"✗ Retrieval returned: {result.get('status')}, count: {result.get('count', 0)}")
    except Exception as e:
        print(f"✗ Retrieval error: {e}")
        return False
    
    return True

def test_pipeline(query: str):
    """Test the Dual RAG+ pipeline."""
    print(f"\n=== Testing Pipeline with: '{query}' ===\n")
    try:
        from pipeline import drag_plus_pipeline
        result = drag_plus_pipeline(query)
        
        # Check each step
        draft = result.get("draft", "")
        answer = result.get("answer", "")
        evidence_count = len(result.get("evidence_results", []))
        draft_meta = result.get("draft_metadata", {})
        revision_result = result.get("revision_result", {})
        
        print(f"Evidence retrieved: {evidence_count} chunks")
        if draft_meta.get("error"):
            print(f"✗ Draft generation error: {draft_meta['error']}")
        
        print(f"Draft length: {len(draft)} chars")
        if draft:
            print(f"  Draft preview: {draft[:100]}...")
        else:
            print("  ✗ Draft is empty!")
        
        print(f"Final answer length: {len(answer)} chars")
        if answer:
            print(f"✓ Got answer ({len(answer)} chars):")
            print(answer)
            return True
        else:
            print("✗ No answer returned")
            print(f"  Draft was: {draft[:100] if draft else '(empty)'}")
            print(f"  Revision result: {revision_result.get('revised_answer', '')[:100] if revision_result.get('revised_answer') else '(empty)'}")
            print(f"  Revisions applied: {revision_result.get('revisions_applied', 0)}")
            if "error" in result:
                print(f"  Error: {result['error']}")
            return False
    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        query = "What is the capital of France?"
        print(f"No query provided, using default: '{query}'\n")
    else:
        query = " ".join(sys.argv[1:])
    
    if test_basic():
        test_pipeline(query)
    else:
        print("\nFix the issues above first.")
