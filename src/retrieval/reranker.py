
from sentence_transformers import CrossEncoder


# Load the pretrained Cross-Encoder model
def load_reranker():

    model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return model


# Rerank candidate documents
def rerank_documents(model, query, documents, top_k=3):

    if not documents:
        return []

    # Create query-document pairs
    pairs = [
        (query, doc.page_content)
        for doc in documents
    ]

    # Predict relevance scores
    scores = model.predict(pairs)

    # Attach scores to documents
    ranked_results = []

    for doc, score in zip(documents, scores):

        ranked_results.append({
            "document": doc,
            "rerank_score": float(score)
        })

    # Sort by relevance score
    ranked_results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    # Return top-k documents
    return ranked_results[:top_k]
