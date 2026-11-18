
from langchain_core.documents import Document
from pathlib import Path
from pypdf import PdfReader
from docx import Document as docsDocument


def text_to_document(text: str, source: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": source
        }
    )


def load_txt(file_path: str) -> Document:

    if not file_path or not file_path.strip():
        raise ValueError("File path cannot be empty")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        text = path.read_text(encoding="utf-8")

    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Unable to decode text file: {path}"
        ) from exc

    except OSError as exc:
        raise OSError(
            f"Unable to read text file: {path}"
        ) from exc

    if not text.strip():
        raise ValueError(f"Text file is empty: {path}")

    return text_to_document(
        text=text,
        source=str(path.resolve())
    )

def load_pdf(file_path: str) -> list[Document]:

    if not file_path or not file_path.strip():
        raise ValueError("File path cannot be empty")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Invalid file format: {path}")

    try:
        reader = PdfReader(str(path))

        if len(reader.pages) == 0:
            raise ValueError(f"PDF contains no pages: {path}")

        documents = []

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            if not text or not text.strip():
                continue

            doc = text_to_document(
                text=text,
                source=str(path.resolve())
            )

            doc.metadata["page"] = page_number

            documents.append(doc)

        if not documents:
            raise ValueError(f"No extractable text found in PDF: {path}")

        return documents
    except (ValueError, FileNotFoundError):
        raise

    except Exception as exc:
        raise ValueError(
            f"Unable to read PDF file: {path}"
        ) from exc


def load_docx(file_path: str) -> Document:

    if not file_path or not file_path.strip():
        raise ValueError("File path cannot be empty")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if path.suffix.lower() != ".docx":
        raise ValueError(f"Invalid file format: {path}")

    try:
        docx_file = docsDocument(str(path))

        text = "\n".join(
            paragraph.text
            for paragraph in docx_file.paragraphs
            if paragraph.text.strip()
        )

        if not text.strip():
            raise ValueError(f"DOCX file is empty: {path}")

        return text_to_document(
            text=text,
            source=str(path.resolve())
        )

    except (ValueError, FileNotFoundError):
        raise

    except Exception as exc:
        raise ValueError(
            f"Unable to read DOCX file: {path}"
        ) from exc
