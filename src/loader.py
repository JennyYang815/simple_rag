from pypdf import PdfReader


def load_pdf(file_path, skip_pages=None):
    # 改进跳过逻辑，更加灵活
    if skip_pages is None:
        skip_pages = []

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        if page_number in skip_pages:
            continue

        text = page.extract_text()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages