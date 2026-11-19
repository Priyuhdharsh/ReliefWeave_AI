
from src.retrieval.reranker import (
    load_reranker,
    rerank_documents
)

from src.retrieval.hybrid_retriever import hybrid_search

from src.retrieval.vector_store import (
    create_vector_store
)

from src.retrieval.bm25_retriever import (
    create_bm25_index
)

from langchain_core.documents import Document


docs = [
    Document(
        page_content="Volunteers distribute food to families affected by disasters.",
        metadata={"source": "volunteer_support.txt"}
    ),
    Document(
        page_content="Emergency food assistance is available to disaster victims.",
        metadata={"source": "food_assistance.txt"}
    ),
    Document(
        page_content="Medical teams provide healthcare support to injured people.",
        metadata={"source": "medical_support.txt"}
    ),
    Document(
        page_content="Shelter facilities are provided to families who lost their homes.",
        metadata={"source": "shelter_support.txt"}
    ),
    Document(
        page_content="Relief organizations provide clean drinking water to affected communities.",
        metadata={"source": "water_support.txt"}
    )
]


query = "How can disaster-affected families receive food assistance?"


vector_store = create_vector_store(docs)


bm25 = create_bm25_index(docs)


hybrid_results = hybrid_search(
    vector_store=vector_store,
    bm25=bm25,
    docs=docs,
    query=query
)


print("\n========== HYBRID RESULTS ==========\n")

for item in hybrid_results:

    print("Hybrid Score:", item["hybrid_score"])
    print("Document:", item["document"].page_content)
    print("-" * 50)


hybrid_documents = [
    item["document"]
    for item in hybrid_results
]


reranker = load_reranker()


final_results = rerank_documents(
    model=reranker,
    query=query,
    documents=hybrid_documents,
    top_k=3
)




print("\n========== FINAL RERANKED RESULTS ==========\n")

for rank, item in enumerate(final_results, start=1):

    print("Rank:", rank)
    print("Rerank Score:", item["rerank_score"])
    print("Document:", item["document"].page_content)
    print("-" * 50)

from src.retrieval.context_distiller import distill_context


# Extract reranked documents
reranked_documents = [
    item["document"]
    for item in final_results
]


# Distill relevant context
distilled_docs = distill_context(
    query=query,
    documents=reranked_documents,
    max_sentences=3
)


print("\n========== DISTILLED CONTEXT ==========\n")

for doc in distilled_docs:
    print(doc.page_content)
    print("Metadata:", doc.metadata)
    print("-" * 50)
    
from src.retrieval.citation import generate_citations


# Generate citations
citation_result = generate_citations(distilled_docs)


print("\n========== CITATION ENGINE ==========\n")

print("Cited Context:\n")
print(citation_result["context"])

print("\nReferences:\n")

for citation_id, source in citation_result["references"].items():

    print(f"[{citation_id}] {source}")