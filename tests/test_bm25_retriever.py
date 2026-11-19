from langchain_core.documents import Document

from src.retrieval.bm25_retriever import (
    create_bm25_index,
    search_bm25
)


docs = [
    Document(page_content="People affected by disasters can submit relief requests."),
    Document(page_content="Volunteers provide food, shelter and medical assistance."),
    Document(
        page_content="Emergency donations help families affected by natural disasters.")
]


def test_create_bm25_index():

    bm25 = create_bm25_index(docs)

    assert bm25 is not None


def test_search_bm25():

    bm25 = create_bm25_index(docs)

    results = search_bm25(
        bm25,
        query="How can affected families receive food?",
        docs=docs,
        k=2
    )

    assert len(results) > 0
    assert len(results) <= 2
