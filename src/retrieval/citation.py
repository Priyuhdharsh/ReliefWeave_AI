
from langchain_core.documents import Document


def generate_citations(documents):

    cited_context = []
    references = {}

    source_mapping = {}

    citation_count = 1

    for doc in documents:

        source = doc.metadata.get(
            "source",
            "Unknown source"
        )

        # Reuse citation ID for the same source
        if source not in source_mapping:

            citation_id = citation_count

            source_mapping[source] = citation_id

            references[citation_id] = source

            citation_count += 1

        else:
            citation_id = source_mapping[source]

        # Attach citation to document content
        cited_text = (
            f"[{citation_id}] {doc.page_content}"
        )

        cited_context.append(cited_text)

    return {
        "context": "\n\n".join(cited_context),
        "references": references
    }
