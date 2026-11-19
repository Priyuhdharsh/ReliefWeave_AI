from pathlib import Path

# Ingestion
from src.ingestion.loader import load_txt
from src.ingestion.splitter import split_documents

# Retrieval
from src.retrieval.vector_store import create_vector_store
from src.retrieval.bm25_retriever import create_bm25_index
from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.reranker import load_reranker, rerank_documents

# Post-retrieval
from src.retrieval.context_distiller import distill_context
from src.retrieval.citation import generate_citations


def main():

    # --------------------------------------------------
    # STEP 1: FIND ACTUAL TXT DOCUMENTS
    # --------------------------------------------------

    project_root = Path(__file__).resolve().parents[1]

    data_path = project_root / "data" / "documents"

    files = sorted(data_path.glob("*.txt"))

    if not files:
        raise FileNotFoundError(
            f"No TXT documents found in {data_path}"
        )

    print("\n========== DOCUMENT INGESTION ==========\n")

    documents = []

    for file in files:

        doc = load_txt(str(file))

        documents.append(doc)

        print("Loaded:", file.name)

    print("\nTotal documents loaded:", len(documents))

    # --------------------------------------------------
    # STEP 2: CHUNKING
    # --------------------------------------------------

    print("\n========== DOCUMENT CHUNKING ==========\n")

    chunks = split_documents(documents)

    print("Total chunks:", len(chunks))

    for chunk in chunks:
        print("Source:", chunk.metadata["source"])
        print("Content:", chunk.page_content[:100])
        print("-" * 50)

    # --------------------------------------------------
    # STEP 3: CREATE FAISS VECTOR STORE
    # --------------------------------------------------

    print("\n========== FAISS VECTOR STORE ==========\n")

    vector_store = create_vector_store(chunks)

    print("FAISS vector store created successfully.")

    # --------------------------------------------------
    # STEP 4: CREATE BM25 INDEX
    # --------------------------------------------------

    print("\n========== BM25 INDEX ==========\n")

    bm25 = create_bm25_index(chunks)

    print("BM25 index created successfully.")

    # --------------------------------------------------
    # STEP 5: HYBRID RETRIEVAL
    # --------------------------------------------------

    query = (
        "How can disaster-affected families "
        "receive food assistance?"
    )

    print("\n========== HYBRID RETRIEVAL ==========\n")

    hybrid_results = hybrid_search(
        vector_store=vector_store,
        bm25=bm25,
        docs=chunks,
        query=query
    )

    if not hybrid_results:
        raise RuntimeError("Hybrid retrieval returned no results.")

    for item in hybrid_results:

        print("Hybrid Score:", item["hybrid_score"])
        print("Document:", item["document"].page_content)
        print("-" * 50)

    hybrid_documents = [
        item["document"]
        for item in hybrid_results
    ]

    # --------------------------------------------------
    # STEP 6: CROSS-ENCODER RERANKING
    # --------------------------------------------------

    print("\n========== RERANKING ==========\n")

    reranker = load_reranker()

    final_results = rerank_documents(
        model=reranker,
        query=query,
        documents=hybrid_documents,
        top_k=3
    )

    if not final_results:
        raise RuntimeError("Reranking returned no results.")

    for rank, item in enumerate(final_results, start=1):

        print("Rank:", rank)
        print("Score:", item["rerank_score"])
        print("Document:", item["document"].page_content)
        print("-" * 50)

    # --------------------------------------------------
    # STEP 7: CONTEXT DISTILLATION
    # --------------------------------------------------

    print("\n========== CONTEXT DISTILLATION ==========\n")

    reranked_documents = [
        item["document"]
        for item in final_results
    ]

    distilled_docs = distill_context(
        query=query,
        documents=reranked_documents,
        max_sentences=3
    )

    if not distilled_docs:
        raise RuntimeError("Context distillation returned no documents.")

    for doc in distilled_docs:

        print(doc.page_content)
        print("Metadata:", doc.metadata)
        print("-" * 50)

    # --------------------------------------------------
    # STEP 8: CITATION GENERATION
    # --------------------------------------------------

    print("\n========== CITATION ENGINE ==========\n")

    citation_result = generate_citations(distilled_docs)

    print("Cited Context:\n")
    print(citation_result["context"])

    print("\nReferences:\n")

    for citation_id, source in citation_result["references"].items():

        print(f"[{citation_id}] {source}")

    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

    print("\n========== PIPELINE SUMMARY ==========\n")

    print("Documents loaded:", len(documents))
    print("Chunks created:", len(chunks))
    print("Hybrid results:", len(hybrid_results))
    print("Reranked results:", len(final_results))
    print("Distilled documents:", len(distilled_docs))
    print("Citations generated:", len(citation_result["references"]))

    print("\nRELIEFWEAVE END-TO-END TEST COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()