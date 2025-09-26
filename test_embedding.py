from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.evaluation import load_evaluator

# embedding the query
query = "what is the marine heatwave(MHW) coverage forecast?"

# embedding the original text
text1 = "Forecasts predict that global MHW coverage will remain at ~25-30% [12-14%] over the coming year, with a slight decrease through the rest of 2025 and increase in the first half of 2026."

# find similarity in the text
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector = embeddings.embed_query("apple")
# print(f"Vector for 'apple': {vector}")
# print(f"Vector length: {len(vector)}")

# Compare vector of two words
evaluator = load_evaluator("pairwise_embedding_distance", embeddings=embeddings)
words = (query, text1)
x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
print(f"Comparing ({words[0]}, {words[1]}): {x}")