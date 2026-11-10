import argparse
import asyncio
import os
from pathlib import Path
import string

from dotenv import load_dotenv


# --- RAG Concepts ---
# Concepts Q1
# A. RAG lets the legal team search hundreds of policies and reindex updated
#    documents each quarter. Answers can point back to the policy in force.
# B. Fine-tuning can learn the recurring brand voice from the 3,000 examples.
# C. Put the two-page report in the prompt so the model can refer to it directly.

# Concepts Q2
# A specific answer delivered with certainty can make people skip verification.
# For example, an invented password-reset procedure could persuade an employee
# to share a recovery code. An explicit admission of uncertainty invites a check.

# Concepts Q3
# 1. Extract text from source documents so the system can process it.
# 2. Split text into chunks small enough to retrieve individually.
# 3. Convert text chunks into embeddings and index them.
# 4. Receive the user's query.
# 5. Embed the user's query using the same embedding model.
# 6. Retrieve the most relevant chunks by comparing vectors.
# 7. Inject retrieved chunks into the prompt as evidence.
# 8. Generate a response from the LLM using the supplied context.


# --- Keyword RAG ---
def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

def run_keyword_questions():
    # Keyword Q1
    query = "What are your hours on weekends?"

    documents = {
        "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
        "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
        "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
        "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
    }
    print("\nKeyword Q1")
    print(f"Selected document: {simple_keyword_retrieval(query, documents)[0][0]}")
    # hours.txt matches "weekends"; hiring.txt and loyalty.txt match "your".
    # The three scores tie at 1, so reverse filename order selects loyalty.txt.

    # Keyword Q2
    print("\nKeyword Q2")
    query = "Do you have anything without caffeine?"
    print(f"Selected document: {simple_keyword_retrieval(query, documents)[0][0]}")
    # No match. Embedding search could surface the menu by topic, but the menu
    # still needs to say which drinks are caffeine-free to support an answer.

    # Keyword Q3
    # Prediction: loyalty.txt covers joining a program and earning rewards.
    print("\nKeyword Q3")
    query = "How do I sign up for rewards?"
    print(f"Selected document: {simple_keyword_retrieval(query, documents)[0][0]}")
    # No match. "Sign up" and "rewards" do not literally occur in loyalty.txt.
    # The shared word "for" is a stopword.


# --- Semantic RAG Concepts ---
# Semantic Q1
# An embedding is a list of numbers that locates a piece of text in a space
# learned from language. The 0.85 chunk ranks above the 0.30 chunk because its
# vector points more nearly in the query's direction. A score is not proof that
# a chunk contains the answer. Paraphrases can be near each other in this space
# because the model learns relationships from how words and phrases are used.

# Semantic Q2
# | Feature                 | Keyword RAG                | Semantic RAG                 |
# |-------------------------|----------------------------|------------------------------|
# | What is compared?       | Exact word overlap         | Query and chunk embeddings   |
# | What is retrieved?      | Full document              | Relevant chunks              |
# | Can it handle synonyms? | No                         | Often                        |
# | Storage format          | Plain text dictionary      | Vector index and chunk text  |
# | Relevance score         | Number of overlapping words | Vector similarity, e.g. cosine |


# --- LlamaIndex ---
def show_response(question, response, preview_chars=None):
    print(f"\nQuestion: {question}\nAnswer: {response}")
    for source in response.source_nodes:
        score = f"{source.score:.4f}" if source.score is not None else "unavailable"
        print(f"  Score: {score}")
        print(source.node.get_content()[:preview_chars])


def run_llamaindex_questions(docs_dir):
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    documents = SimpleDirectoryReader(str(docs_dir), required_exts=[".pdf"]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=3)

    # LlamaIndex Q1
    questions = [
        "What employee benefits does BrightLeaf offer?",
        "What are BrightLeaf's security policies?",
    ]
    responses = []
    for question in questions:
        response = query_engine.query(question)
        responses.append(response)
        show_response(question, response, 150)
    # The benefits document ranked first (about 0.76). The answer confidently
    # gave details such as 12 weeks of paid parental leave and a 5% 401(k)
    # match. The partnership and mission passages were unexpected extras.
    # The security policy ranked first (about 0.66). The answer confidently
    # described MFA and incident response. The benefits and mission passages
    # also ranked in the top three, even though neither answered the question.

    # LlamaIndex Q2
    for top_k in (1, 5):
        print(f"\nLlamaIndex Q2, top_k={top_k}")
        response = index.as_query_engine(similarity_top_k=top_k).query(questions[0])
        show_response(questions[0], response, 150)
    # Both runs gave the same core benefits, including the $600 annual wellness
    # reimbursement. The five-source run mentioned mindfulness sessions, but
    # also pulled in unrelated earnings and product material. The top chunk
    # already covered the main answer, so extra context added little here.

    # LlamaIndex Q3
    hard_question = "What is the Wi-Fi password for BrightLeaf's visitor network?"
    hard_response = query_engine.query(hard_question)
    show_response(hard_question, hard_response)
    # I chose a visitor password because a policy might mention security
    # without listing that credential. The top score was about 0.43, and the
    # answer said the password was unavailable. I would require direct source
    # support for specific details and keep an explicit fallback for gaps.

    # LlamaIndex Q4
    judge = OpenAI(model="gpt-4o-mini", temperature=0)
    faithfulness = FaithfulnessEvaluator(llm=judge)
    relevancy = RelevancyEvaluator(llm=judge)
    for question, response in ((questions[0], responses[0]), (hard_question, hard_response)):
        print(f"\nEvaluation: {question}")
        faithful_result = faithfulness.evaluate_response(query=question, response=response)
        relevant_result = relevancy.evaluate_response(query=question, response=response)
        print(f"Faithfulness: {faithful_result.score}")
        print(f"Relevancy: {relevant_result.score}")
    # A 1.0 faithfulness score means the judge found support in the retrieved
    # text. A 0.0 means it judged the answer unsupported. Relevancy asks whether
    # the response and context address the query. A separate model can assess
    # meaning and evidence even when valid answers use different wording.
    # The benefits answer scored 1.0 on both metrics. The Wi-Fi question scored
    # 0.0 on both even though the model appropriately declined to guess. I
    # would track correct abstentions separately from supported answers.


def main():
    parser = argparse.ArgumentParser(description="Week 6 RAG warmup")
    parser.add_argument("--docs-dir", type=Path, default=Path(__file__).parent / "resources" / "brightleaf_pdfs")
    parser.add_argument("--keyword-only", action="store_true")
    args = parser.parse_args()

    run_keyword_questions()
    if args.keyword_only:
        return
    load_dotenv(Path(__file__).with_name(".env"))
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in the .env file beside this script.")
    print("API key loaded successfully.")
    if not args.docs_dir.is_dir():
        parser.error(f"BrightLeaf PDF directory not found: {args.docs_dir}")
    # The default Windows loop can hit an invalid-handle error when the
    # evaluator connections are cleaned up. Use a selector loop for this CLI.
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    run_llamaindex_questions(args.docs_dir)


if __name__ == "__main__":
    main()
