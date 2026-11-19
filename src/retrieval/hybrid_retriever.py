
from src.utils.text_utils import tokenize_text
from src.core.config import load_config


def normalize_scores(scores, lower_is_better=False):
    """
    Normalize scores between 0 and 1.

    lower_is_better=True is used for FAISS distances.
    lower_is_better=False is used for BM25 relevance scores.
    """

    if not scores:
        return {}

    min_score = min(scores.values())
    max_score = max(scores.values())

    # Handle the case where all scores are equal
    if max_score == min_score:
        return {
            key: 1.0
            for key in scores
        }

    normalized = {}

    for key, score in scores.items():

        if lower_is_better:
            normalized[key] = (
                (max_score - score) / (max_score - min_score)
            )

        else:
            normalized[key] = (
                (score - min_score) / (max_score - min_score)
            )

    return normalized


def hybrid_search(vector_store, bm25, docs, query):
    """
    Perform hybrid retrieval using FAISS and BM25.

    Steps:
    1. Retrieve documents from FAISS.
    2. Retrieve documents from BM25.
    3. Collect unique candidate documents.
    4. Normalize retrieval scores.
    5. Calculate weighted hybrid scores.
    6. Rank documents by hybrid score.
    """

    # Load configuration
    config = load_config()

    k_semantic = config.hybrid.k_semantic
    k_sparse = config.hybrid.k_sparse

    semantic_weight = config.hybrid.weights.semantic
    bm25_weight = config.hybrid.weights.bm25

    # --------------------------------------------------
    # STEP 1: FAISS retrieval
    # --------------------------------------------------

    faiss_results = vector_store.similarity_search_with_score(
        query,
        k=k_semantic
    )

    # Store FAISS distances using document content as key
    faiss_scores = {}

    for doc, distance in faiss_results:
        faiss_scores[doc.page_content] = float(distance)

    # --------------------------------------------------
    # STEP 2: BM25 retrieval
    # --------------------------------------------------

    tokenized_query = tokenize_text(query)

    bm25_all_scores = bm25.get_scores(tokenized_query)

    # Get the indices of the top BM25 documents
    bm25_ranked_indices = sorted(
        range(len(bm25_all_scores)),
        key=lambda i: bm25_all_scores[i],
        reverse=True
    )

    top_bm25_indices = bm25_ranked_indices[:k_sparse]

    # Store BM25 scores using document content as key
    bm25_scores = {}

    for index in top_bm25_indices:
        doc = docs[index]
        bm25_scores[doc.page_content] = float(
            bm25_all_scores[index]
        )

    # --------------------------------------------------
    # STEP 3: Candidate collection
    # --------------------------------------------------

    candidates = {}

    # Add FAISS documents
    for doc, distance in faiss_results:

        content = doc.page_content

        candidates[content] = {
            "document": doc,
            "faiss_score": distance,
            "bm25_score": None
        }

    # Add BM25 documents
    for index in top_bm25_indices:

        doc = docs[index]
        content = doc.page_content

        if content not in candidates:
            candidates[content] = {
                "document": doc,
                "faiss_score": None,
                "bm25_score": bm25_all_scores[index]
            }

        else:
            candidates[content]["bm25_score"] = (
                bm25_all_scores[index]
            )

    # --------------------------------------------------
    # STEP 4: Score normalization
    # --------------------------------------------------

    normalized_faiss = normalize_scores(
        faiss_scores,
        lower_is_better=True
    )

    normalized_bm25 = normalize_scores(
        bm25_scores,
        lower_is_better=False
    )

    # --------------------------------------------------
    # STEP 5: Weighted score fusion
    # --------------------------------------------------

    hybrid_results = []

    for content, candidate in candidates.items():

        faiss_score = normalized_faiss.get(content, 0.0)
        bm25_score = normalized_bm25.get(content, 0.0)

        hybrid_score = (
            semantic_weight * faiss_score
            +
            bm25_weight * bm25_score
        )

        hybrid_results.append({
            "document": candidate["document"],
            "faiss_score": faiss_score,
            "bm25_score": bm25_score,
            "hybrid_score": hybrid_score
        })

    # --------------------------------------------------
    # STEP 6: Final ranking
    # --------------------------------------------------

    hybrid_results.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True
    )

    return hybrid_results
