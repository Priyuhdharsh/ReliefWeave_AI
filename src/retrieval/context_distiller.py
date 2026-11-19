
import re
from langchain_core.documents import Document


def tokenize(text):
    return set(re.findall(r"\b\w+\b", text.lower()))


def distill_context(query, documents, max_sentences=5):

    if not documents:
        return []

    query_tokens = tokenize(query)

    candidates = []

    for doc in documents:

        sentences = re.split(
            r"(?<=[.!?])\s+",
            doc.page_content
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_tokens = tokenize(sentence)

            # Calculate query-sentence relevance
            overlap = query_tokens.intersection(
                sentence_tokens
            )

            score = len(overlap) / max(len(query_tokens), 1)

            if score > 0:
                candidates.append({
                    "sentence": sentence,
                    "score": score,
                    "metadata": doc.metadata
                })

    # Sort by relevance
    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # Select top relevant sentences
    selected = candidates[:max_sentences]

    # Convert selected sentences into Documents
    distilled_docs = []

    for item in selected:

        distilled_docs.append(
            Document(
                page_content=item["sentence"],
                metadata=item["metadata"].copy()
            )
        )

    return distilled_docs
