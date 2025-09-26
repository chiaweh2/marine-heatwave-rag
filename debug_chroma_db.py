"""
Debug script to examine the Chroma database and understand why there are no matches
"""
from langchain_ollama import OllamaEmbeddings
# Use the new langchain-chroma import
try:
    from langchain_chroma import Chroma
except ImportError:
    # Fall back to old import if new one not available
    from langchain_community.vectorstores import Chroma
from langchain.evaluation import load_evaluator
import numpy as np

def debug_chroma_database():
    """Debug the Chroma database to understand the vector storage and retrieval"""
    
    # Initialize the same embeddings as used in creation
    print("🔍 Initializing embeddings model...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Same as in your creation
    
    # Load the database
    print("📂 Loading Chroma database...")
    try:
        chroma_db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
            collection_name="marine_heatwave_discussions"  # Same as creation script!
        )
        print("✅ Database loaded successfully")
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        return
    
    # Check database contents
    print("\n📊 Database Statistics:")
    try:
        # Get collection info
        collection = chroma_db._collection
        print(f"Collection name: {collection.name}")
        print(f"Number of documents: {collection.count()}")
        
        # Get a few sample documents
        print("\n📄 Sample documents in database:")
        sample_results = chroma_db.similarity_search("", k=5)  # Get any 5 docs
        for i, doc in enumerate(sample_results[:3]):  # Show first 3
            print(f"\nDocument {i+1}:")
            print(f"Content preview: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
            
    except Exception as e:
        print(f"❌ Error getting database stats: {e}")
    
    # Test the specific query
    query = "what is the marine heatwave(MHW) coverage forecast?"
    text1 = "Forecasts predict that global MHW coverage will remain at ~25-30% [12-14%] over the coming year, with a slight decrease through the rest of 2025 and increase in the first half of 2026."
    
    print(f"\n🔍 Testing specific query:")
    print(f"Query: {query}")
    print(f"Expected text: {text1}")
    
    # Test 1: Direct similarity search
    print(f"\n🔬 Test 1: Direct similarity search")
    try:
        results = chroma_db.similarity_search(query, k=10)
        print(f"Found {len(results)} results")
        
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Content: {doc.page_content[:300]}...")
            print(f"Metadata: {doc.metadata}")
    except Exception as e:
        print(f"❌ Error in similarity search: {e}")
    
    # Test 2: Similarity search with scores
    print(f"\n🔬 Test 2: Similarity search with relevance scores")
    try:
        results_with_scores = chroma_db.similarity_search_with_relevance_scores(query, k=10)
        print(f"Found {len(results_with_scores)} results with scores")
        
        for i, (doc, score) in enumerate(results_with_scores):
            print(f"\nResult {i+1} (Score: {score:.4f}):")
            print(f"Content: {doc.page_content[:200]}...")
            
            # Check if this matches our expected text
            if text1.strip() in doc.page_content or doc.page_content in text1.strip():
                print("🎯 MATCH FOUND! This document contains the expected text")
    except Exception as e:
        print(f"❌ Error in similarity search with scores: {e}")
    
    # Test 3: Manual embedding comparison
    print(f"\n🔬 Test 3: Manual embedding comparison")
    try:
        # Get embedding for query
        query_embedding = embeddings.embed_query(query)
        text1_embedding = embeddings.embed_query(text1)
        
        print(f"Query embedding length: {len(query_embedding)}")
        print(f"Text1 embedding length: {len(text1_embedding)}")
        
        # Calculate cosine similarity manually
        query_vec = np.array(query_embedding)
        text1_vec = np.array(text1_embedding)
        
        cosine_sim = np.dot(query_vec, text1_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(text1_vec))
        print(f"Manual cosine similarity: {cosine_sim:.4f}")
        
        # Use evaluator for comparison
        evaluator = load_evaluator("pairwise_embedding_distance", embeddings=embeddings)
        distance_result = evaluator.evaluate_string_pairs(prediction=query, prediction_b=text1)
        print(f"Evaluator distance result: {distance_result}")
        
    except Exception as e:
        print(f"❌ Error in manual embedding comparison: {e}")
    
    # Test 4: Search for the expected text directly
    print(f"\n🔬 Test 4: Search for expected text directly")
    try:
        # Search using part of the expected text
        search_terms = ["MHW coverage", "25-30%", "global MHW", "2025", "2026"]
        
        for term in search_terms:
            results = chroma_db.similarity_search(term, k=5)
            print(f"\nSearching for '{term}': {len(results)} results")
            if results:
                for i, doc in enumerate(results[:2]):  # Show top 2
                    if text1.strip() in doc.page_content or doc.page_content.strip() in text1.strip():
                        print(f"🎯 FOUND! Result {i+1} contains expected text")
                        print(f"Content: {doc.page_content[:300]}...")
                        break
    except Exception as e:
        print(f"❌ Error in direct text search: {e}")
    
    # Test 5: Check if the expected text exists at all
    print(f"\n🔬 Test 5: Check if expected text exists in database")
    try:
        # Get all documents and search for the text
        all_docs = chroma_db.similarity_search("", k=1000)  # Get many docs
        print(f"Total documents retrieved: {len(all_docs)}")
        
        found_exact = False
        found_partial = False
        
        for doc in all_docs:
            if text1.strip() in doc.page_content:
                found_exact = True
                print("🎯 EXACT MATCH FOUND!")
                print(f"Document content: {doc.page_content}")
                break
            elif any(word in doc.page_content for word in ["MHW coverage", "25-30%", "global MHW"]):
                if not found_partial:
                    found_partial = True
                    print("🔍 PARTIAL MATCH FOUND!")
                    print(f"Document content: {doc.page_content[:300]}...")
        
        if not found_exact and not found_partial:
            print("❌ No matching documents found in database")
        
    except Exception as e:
        print(f"❌ Error checking for expected text: {e}")

if __name__ == "__main__":
    debug_chroma_database()