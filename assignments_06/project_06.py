import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


def show_sources(source_nodes):
    for source in source_nodes:
        name = source.node.metadata.get("file_name", "Unknown document")
        score = f"{source.score:.4f}" if source.score is not None else "unavailable"
        print(f"  {name} (similarity: {score})")
        print(f"  {source.node.get_content()[:200]}")
    if not source_nodes:
        print("  No sources returned.")


def main():
    # Step 1: Setup
    parser = argparse.ArgumentParser(description="Groundwork Coffee Co. Q&A assistant")
    parser.add_argument("--docs-dir", type=Path, default=Path(__file__).parent / "resources" / "groundwork_docs")
    docs_dir = parser.parse_args().docs_dir
    assert docs_dir.is_dir(), f"Document directory not found: {docs_dir}"
    load_dotenv(Path(__file__).with_name(".env"))
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in the .env file beside this script.")
    print("API key loaded successfully.")

    # Step 2: Load
    documents = SimpleDirectoryReader(str(docs_dir)).load_data()
    print(f"Loaded {len(documents)} documents.")
    for document in documents:
        print(f"  {document.metadata.get('file_name', 'Unknown document')}")

    # Step 3: Index
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=3)
    print("Index built successfully. Ready to answer questions.")

    # Step 4: Query
    questions = [
        "What are Groundwork's hours on weekends?",
        "Do you offer any dairy-free milk options?",
        "How does the loyalty program work?",
        "How did Groundwork Coffee get started?",
        "Do you offer catering or wholesale orders?",
    ]
    for question in questions:
        response = query_engine.query(question)
        print(f"\nQuestion: {question}\nAnswer: {response}")
        show_sources(response.source_nodes[:1])
    # All five answers sounded confident and agreed with the course documents.
    # The surprising result was that the company story ranked first for the
    # hours question and seasonal specials ranked first for dairy-free milk.
    # The FAQ and menu confirmed the answers. The first retrieved source
    # alone would give a reader the wrong place to verify either detail.

    # Step 5: Find a Failure
    question = "What is the Wi-Fi password at the original Groundwork location?"
    response = query_engine.query(question)
    print(f"\nHard question: {question}\nAnswer: {response}")
    show_sources(response.source_nodes)
    # I asked for the password at the original location because it combines
    # a location clue with a detail that may change. The story ranked first
    # and the FAQ ranked second. The FAQ says to ask a barista and omits the
    # password. The answer acknowledged the gap and gave that same next step.
    # A fluent answer still needs source support. I would rerank for direct
    # evidence and show the supporting FAQ passage next to the answer.


if __name__ == "__main__":
    main()


# Step 6: Reflection
# The retrieval setup uses five statements for loading, configuring the models,
# building the index, and creating the query engine. The warmup's keyword
# function takes substantially more code just to tokenize, score, and rank.
# LlamaIndex also handles chunking, embeddings, and indexing behind its API.
#
# A repair team could search maintenance manuals and service bulletins to find
# the correct inspection procedure for a particular machine model.
#
# Even when retrieval returns the correct manual, generation can confuse
# numbers from adjacent sections and report a value for the wrong model.
