def split_pages(pages, chunk_size=200, overlap=50):
    chunks = []

    for page in pages:
        text = page["text"]

        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "page": page["page"],
                    "text": chunk_text
                })

            if end >= len(text):
                break

            start = end - overlap

    return chunks