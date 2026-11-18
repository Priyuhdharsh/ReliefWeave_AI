import pytest
from src.ingestion.loader import text_to_document, load_txt, load_pdf, load_docx
from langchain_core.documents import Document
from reportlab.pdfgen import canvas
from docx import Document as docsDocument


def test_text_to_document():
    doc = text_to_document(
        "Hospital A has 120 available beds.",
        "hospital_report.pdf"
    )

    assert isinstance(doc, Document)
    assert doc.page_content == "Hospital A has 120 available beds."
    assert doc.metadata['source'] == "hospital_report.pdf"


def test_load_txt(tmp_path):
    file_path = tmp_path / "hospital_guidelines.txt"

    file_path.write_text(
        "Hospital A has 120 available beds.",
        encoding="utf-8"
    )
    doc = load_txt(str(file_path))

    assert isinstance(doc, Document)
    assert doc.page_content == "Hospital A has 120 available beds."
    assert doc.metadata["source"] == str(file_path.resolve())


def test_load_txt_file_not_found(tmp_path):

    file_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_txt(str(file_path))


def test_load_txt_empty_file(tmp_path):

    file_path = tmp_path / "empty.txt"
    file_path.write_text("")

    with pytest.raises(ValueError):
        load_txt(str(file_path))


def create_test_pdf(file_path):
    pdf = canvas.Canvas(str(file_path))

    pdf.drawString(100, 750, "Hospital A has 120 available beds.")
    pdf.showPage()

    pdf.drawString(100, 750, "Hospital B has 45 available beds.")
    pdf.save()


def test_load_pdf(tmp_path):
    file_path = tmp_path / "hospital_report.pdf"

    create_test_pdf(file_path)

    docs = load_pdf(str(file_path))

    assert len(docs) == 2

    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.page_content.strip()
        assert doc.metadata["page"] in [1, 2]


def create_test_docx(file_path):
    doc = docsDocument()

    doc.add_paragraph("Chennai Emergency Hospital Report")
    doc.add_paragraph("Hospital A has 120 available beds.")
    doc.add_paragraph("Hospital B has 45 available beds.")

    doc.save(str(file_path))


def test_load_docx(tmp_path):
    file_path = tmp_path / "hospital_report.docx"

    create_test_docx(file_path)

    doc = load_docx(str(file_path))

    assert isinstance(doc, Document)
    assert "Hospital A has 120 available beds." in doc.page_content
    assert "Hospital B has 45 available beds." in doc.page_content
    assert doc.metadata["source"] == str(file_path.resolve())


def test_load_pdf_file_not_found(tmp_path):

    file_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        load_pdf(str(file_path))


def test_load_docx_file_not_found(tmp_path):

    file_path = tmp_path / "missing.docx"

    with pytest.raises(FileNotFoundError):
        load_docx(str(file_path))


def test_load_pdf_invalid_format(tmp_path):

    file_path = tmp_path / "hospital_report.txt"

    file_path.write_text(
        "Hospital A has 120 available beds.",
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Invalid file format"):
        load_pdf(str(file_path))


def test_load_docx_invalid_format(tmp_path):

    file_path = tmp_path / "hospital_report.pdf"

    file_path.write_text(
        "Hospital A has 120 available beds.",
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Invalid file format"):
        load_docx(str(file_path))

def test_load_pdf_corrupted_file(tmp_path):

    file_path = tmp_path / "corrupted.pdf"

    file_path.write_text(
        "This is not a valid PDF file.",
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Unable to read PDF file"):
        load_pdf(str(file_path))
        
def test_load_docx_corrupted_file(tmp_path):

    file_path = tmp_path / "corrupted.docx"

    file_path.write_text(
        "This is not a valid DOCX file.",
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Unable to read DOCX file"):
        load_docx(str(file_path))
        
def test_load_txt_invalid_encoding(tmp_path):

    file_path = tmp_path / "invalid_encoding.txt"

    file_path.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(ValueError, match="Unable to decode text file"):
        load_txt(str(file_path))