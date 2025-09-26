from datetime import datetime
import argparse
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM


def rag_query_template(
    rag_context: str,
    user_query: str,
    cached_chat_history: str='',
) -> str:
    """
    RAG query template to combine the retrieved context with the user query.
    Parameters
    ----------
    rag_context : str
        The context retrieved from the embedding database.
    user_query : str
        The user query.
    Returns
    -------
    str
        The combined prompt for the LLM.
    """
    return f"""You are a knowledgeable assistant.
    Use the following context and previous cached chat history to answer the question.

    Context: {rag_context}

    Cached Chat History: {cached_chat_history}
    (If the chat history is empty, just ignore it)

    Question: {user_query}

    """

def cache_query_answer_template(
    rag_query: str,
    llm_answer: str
) -> str:
    """
    Cache the previous query and its answer with a timestamp.
    
    Parameters
    ----------
    rag_query : str
        The previous RAG query.
    llm_answer : str
        The answer from the LLM.

    Returns
    -------
    str
        The cached query and answer with a timestamp.
    """
    time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""The following is a previous query and its answer at the following time: {time_stamp}.

    Cached Query: {rag_query}

    Your previous answer: {llm_answer}

    """




def main(LLM_EMBEDDING_MODEL:str):
    """
    Interactive RAG query system that allows continuous questioning.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Interactive RAG Query System for Marine Heatwave Discussions"
    )
    parser.add_argument(
        '--num_top_ans',
        type=int, default=3, 
        help='The number of top similar chunks to retrieve'
    )
    parser.add_argument(
        '--embedding_db_path',
        type=str, default='./chroma_db',
        help='Path to the Chroma embedding database'
    )
    parser.add_argument(
        '--model',
        type=str, default='llama3',
        help='Ollama model to use for responses'
    )
    args = parser.parse_args()

    print("🌊 Marine Heatwave Discussion RAG System 🌊")
    print("=" * 40)

    # Initialize the embedding vector database
    print("Loading embedding database...")
    try:
        # this is hard coded for the embedding model used
        # need to be consistent with the one used in create_embedding_db.py
        embeddings = OllamaEmbeddings(model=LLM_EMBEDDING_MODEL)
        chroma_db = Chroma(
            persist_directory=args.embedding_db_path,
            embedding_function=embeddings,
            collection_name="marine_heatwave_discussions"  # Must match create_embedding_db.py!
        )
        print(f"✅ Database loaded from: {args.embedding_db_path}")
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        print("Make sure you've run create_embedding_db.py first!")
        return

    # Initialize the LLM
    print("Initializing language model...")
    try:
        llm = OllamaLLM(model=args.model)
        print(f"✅ Model loaded: {args.model}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Make sure {args.model} is installed in Ollama!")
        return

    # Initialize chat history
    chat_history = ""

    print("\n" + "=" * 40)
    print("🤖 Interactive Mode Started")
    print("Type 'quit', 'exit', or press Ctrl+C to stop")
    print("=" * 40 + "\n")

    try:
        while True:
            # Get user input
            user_query = input("🔍 Enter your question about marine heatwaves: ").strip()

            # Check for exit commands
            if user_query.lower() in ['quit', 'exit']:
                break

            print("\n🔎 Searching relevant documents...")

            # Retrieve relevant documents
            try:
                result = chroma_db.similarity_search_with_relevance_scores(
                    user_query,
                    k=args.num_top_ans
                )
                print(f"✅ Retrieved {len(result)} documents")
                # remove content that has score lower than 0.7
                result_screen = []
                for doc, score in result:
                    if score >= 0.001:
                        result_screen.append((doc, score))

                if not result_screen:
                    print("📄 No relevant documents found.")
                    continue

                doc_contents, _ = zip(*result_screen)  # Unpack the tuples

                rag_context = "\n\n****\n\n".join(doc.page_content for doc in doc_contents)
                print(f"📄 Found {len(result_screen)} relevant documents")
            except Exception as e:
                print(f"❌ Error retrieving documents: {e}")
                continue

            # Generate the prompt
            prompt = rag_query_template(rag_context, user_query, chat_history)

            # Display the response
            print("-" * 20)
            print("📝 RAG prompt:")
            print("-" * 20)
            print(prompt)
            print("-" * 20)

            print("🤖 Generating response...\n")

            # Get LLM response
            try:
                response = llm.invoke(prompt)

                # Display the response
                print("📝 Answer:")
                print("-" * 20)
                print(response)
                print("-" * 20)

                # Update chat history
                chat_history += cache_query_answer_template(user_query, response)

            except Exception as e:
                print(f"❌ Error generating response: {e}")
                continue

            print("\n" + "=" * 40 + "\n")

    except KeyboardInterrupt:
        pass

    print("\n👋 Thanks for using the Marine Heatwave RAG System!")


if __name__ == "__main__":
    LLM_EMBEDDING_MODEL = 'nomic-embed-text'
    main(LLM_EMBEDDING_MODEL)
