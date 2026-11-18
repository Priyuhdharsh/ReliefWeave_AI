from langchain_community.vectorstores import FAISS
from src.ingestion.embedder import get_embedding_model

def create_vector_store(docs):

    embedding_model = get_embedding_model()

    vector_store = FAISS.from_documents(
        documents=docs,
        embedding=embedding_model
    )

    return vector_store

def save_vector_store(vector_store, path):

    vector_store.save_local(path)
    
def load_vector_store(path):

    embedding_model = get_embedding_model()

    vector_store = FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_store

def similarity_search(vector_store, query, k=4):

    results = vector_store.similarity_search(
        query,
        k=k
    )

    return results