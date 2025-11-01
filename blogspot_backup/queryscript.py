""""
Use this script to query json files.
1. Read in the json files (docs)
2. Create doc embeddings with Ollama's embedding model and store in ChromaDB
3. Get the top matching docs
4. Query with Ollama's LLM model using the context of the top matching docs
"""


import json, os

import chromadb
import ollama

from chromadb.utils import embedding_functions

"""
1. Read in json files.
"""

docs = []
for file in os.listdir("posts"):
    if file.endswith(".json"):
        with open(os.path.join("posts", file)) as f:
            data = json.load(f)
            docs.append({
                "title": data.get("title", ""),
                "text": data.get("content", ""),
                "pub_date": data.get("published", ""),
                "source": file
            })

print(f"Loaded {len(docs)} json files representing blog posts.")

"""
2. Create embeddings with Ollama's embedding model "nomic-embed-text", and store in ChromaDB.
"""

# Create a function that converts text into numerical vectors (embeddings) using model "nomic-embed-text"
# Timeout set to 1 hour to to handle tofa specific use case
embedding_fn = embedding_functions.OllamaEmbeddingFunction(model_name="nomic-embed-text", timeout=3600)

# Create a client connection to the local ChromaDB instance database
chroma_client = chromadb.Client()

# Create a “collection” (similar to a table in SQL or an index in Elasticsearch)
collection = chroma_client.create_collection("blog_posts", embedding_function=embedding_fn)

# Vectorize the documents and add them to the ChromaDB collection in batches
BATCH_SIZE = 5000  # Anything < 5461

for i in range(0, len(docs), BATCH_SIZE):
    batch_docs = [d["text"] for d in docs[i:i + BATCH_SIZE]] 
    batch_metas = [{"title": d["title"], "source": d["source"], "pub_date": d["pub_date"]} for d in docs[i:i + BATCH_SIZE]]
    batch_ids = [str(i) for i in range(len(docs[i:i + BATCH_SIZE]))]

    print(f"Adding batch {i//BATCH_SIZE + 1} ({len(batch_docs)} docs)")

    collection.add(
        documents=batch_docs,
        metadatas=batch_metas,
        ids=batch_ids
    )

print(f"Finished adding posts to ChromaDB.")


def ask_question(question):
    query = question + " in my blog posts"

    """
    3. Get the top matching docs
    """
    # Return the n most semantically similar posts
    results = collection.query(query_texts=[query], n_results=30)

    # Concatenate the top-matching blog posts into one string
    context = "\n\n".join([doc for doc in results['documents'][0]])

    # Build a prompt for the LLM
    prompt = f"Using my blog posts, answer the question:\n\n{query}\n\nContext:\n{context}"

    """
    4. Query with Ollama's LLM model "llama3:8b"
    """
    response = ollama.chat(
        model="llama3:8b",
        messages=[{"role": "user", "content": prompt}]  
    )
    print(response['message']['content'])


ask_question("summarize posts mentioning danny")
ask_question("what did i write about mountains")
ask_question("what does our family do on birthdays")
ask_question("what are some recipes with almond")
ask_question("how many kids do i have")
