from src.ingestion.loader import load_txt, load_pdf, load_docx
from src.ingestion.splitter import split_documents
from langchain_core.documents import Document
from tests.ingestion.test_loader import create_test_pdf
from tests.ingestion.test_loader import create_test_docx
import pytest


def test_split_txt_document(tmp_path):

    file_path = tmp_path / "hospital_guidelines.txt"

    file_path.write_text(
        "Hospital A has 120 available beds. " * 40,
        encoding="utf-8"
    )

    # Step 1: Load the TXT file
    doc = load_txt(str(file_path))

    # Step 2: Send the document as a list to the splitter
    chunks = split_documents([doc])

    # Step 3: Verify the chunks
    assert len(chunks) > 1

    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert len(chunk.page_content) <= 500
        assert chunk.metadata["source"] == str(file_path.resolve())

    assert chunks[0].page_content[-20:] in chunks[1].page_content


def test_split_pdf_document(tmp_path):

    file_path = tmp_path / "hospital_guidelines.pdf"

    # Step 1: Generate a real PDF
    create_test_pdf(file_path)

    # Step 2: Load the PDF
    docs = load_pdf(str(file_path))

    # Step 3: Send the document list to the splitter
    chunks = split_documents(docs)

    # Step 4: Verify the chunks
    assert len(chunks) >= 2

    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert len(chunk.page_content) <= 500
        assert chunk.metadata["source"] == str(file_path.resolve())


def test_split_docx_document(tmp_path):

    file_path = tmp_path / "hospital_guidelines.docx"

    # Step 1: Generate a real DOCX file
    create_test_docx(file_path)

    # Step 2: Load the DOCX
    doc = load_docx(str(file_path))

    # Step 3: Send the document as a list
    chunks = split_documents([doc])

    # Step 4: Verify the chunks
    assert len(chunks) >= 1

    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert len(chunk.page_content) <= 500
        assert chunk.metadata["source"] == str(file_path.resolve())


def test_split_documents_empty_list():

    with pytest.raises(ValueError, match="docs cannot be empty"):
        split_documents([])


def test_split_documents_invalid_input_type():

    with pytest.raises(TypeError, match="docs must be a list"):
        split_documents("Hospital emergency guidelines")


def test_split_documents_invalid_document():

    with pytest.raises(TypeError, match="All items in docs must be Document objects"):
        split_documents(["Hospital emergency guidelines"])


def test_split_documents_zero_chunk_size():

    doc = Document(page_content="Hospital emergency guidelines")

    with pytest.raises(ValueError, match="chunk_size must be greater than zero"):
        split_documents([doc], chunk_size=0)


def test_split_documents_negative_chunk_size():

    doc = Document(page_content="Hospital emergency guidelines")

    with pytest.raises(ValueError, match="chunk_size must be greater than zero"):
        split_documents([doc], chunk_size=-100)

def test_split_documents_negative_chunk_overlap():

    doc = Document(page_content="Hospital emergency guidelines")

    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        split_documents([doc], chunk_overlap=-10)
def test_split_documents_overlap_equal_to_chunk_size():

    doc = Document(page_content="Hospital emergency guidelines")

    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size"
    ):
        split_documents(
            [doc],
            chunk_size=100,
            chunk_overlap=100
        )