from langchain_core.documents import Document

from src.retrieval.vector_store import (
    create_vector_store,
    similarity_search,
    save_vector_store,
    load_vector_store
)

docs = [
    Document(page_content="People affected by disasters can submit relief requests."),
    Document(page_content="Volunteers provide food, shelter and medical assistance."),
    Document(page_content="Emergency donations help families affected by natural disasters.")
]

def test_create_and_search_vector_store():

    # Arrange - Create FAISS vector store
    vector_store = create_vector_store(docs)

    # Act - Perform similarity search
    results = similarity_search(
        vector_store,
        query="How can affected families receive food?",
        k=2
    )

    # Assert - Verify results
    assert len(results) > 0
    assert len(results) <= 2
    
def test_save_and_load_vector_store(tmp_path):

    # Arrange
    vector_store = create_vector_store(docs)

    save_path = str(tmp_path / "faiss_index")

    # Act - Save vector store
    save_vector_store(vector_store, save_path)

    # Act - Load vector store
    loaded_store = load_vector_store(save_path)

    # Assert
    assert loaded_store is not None

    # Verify search after loading
    results = similarity_search(
        loaded_store,
        query="How can affected families receive food?",
        k=2
    )

    assert len(results) > 0