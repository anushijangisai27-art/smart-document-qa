from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings

if __name__ == "__main__":
    from read_pdf import extract_text_from_pdf
    from chunking import split_into_chunks

    pdf_text = extract_text_from_pdf("sample.pdf")
    chunks = split_into_chunks(pdf_text)
    embeddings = create_embeddings(chunks)

    print(f"Total chunks: {len(chunks)}")
    print(f"Size of each embedding: {embeddings[0].shape}")
    print(f"\first chunk  embedding (first 10 numbers):")
    print(embeddings[0][:10])