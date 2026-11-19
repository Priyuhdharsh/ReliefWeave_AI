from rank_bm25 import BM25Okapi
from src.utils.text_utils import tokenize_text


def create_bm25_index(docs):

    tokenized_corpus = [
        tokenize_text(doc.page_content)
        for doc in docs
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25


from rank_bm25 import BM25Okapi
from src.utils.text_utils import tokenize_text


def create_bm25_index(docs):

    tokenized_corpus = [
        tokenize_text(doc.page_content)
        for doc in docs
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25


def search_bm25(bm25, query, docs, k=4):

   
    tokenized_query = tokenize_text(query)

   
    scores = bm25.get_scores(tokenized_query)

  
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    top_indices = ranked_indices[:k]

    results = [docs[i] for i in top_indices]

    return results
