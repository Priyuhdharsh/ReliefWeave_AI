from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_documents(docs, chunk_size=500, chunk_overlap=100):

    if not isinstance(docs, list):
        raise TypeError("docs must be a list of Document objects")

    if not docs:
        raise ValueError("docs cannot be empty")

    if not all(isinstance(doc, Document) for doc in docs):
        raise TypeError("All items in docs must be Document objects")

    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError("chunk_size must be an integer")

    if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
        raise TypeError("chunk_overlap must be an integer")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(docs)

    return chunks
