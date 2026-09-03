def split_into_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    from read_pdf import extract_text_from_pdf

    pdf_text = extract_text_from_pdf("sample.pdf")
    chunks = split_into_chunks(pdf_text)

    
    print(f"Total chunks created: {len(chunks)}")
    print("\nfirst chunk:")
    print(chunks[0])