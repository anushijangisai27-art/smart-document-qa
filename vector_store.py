import faiss
import numpy as np

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def search_similar_chunks(query_embedding, index, chunks, top_k=4):
    query_vector = np.array([query_embedding])
    distances, indices = index.search(query_vector, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

if __name__ == "__main__":
    from read_pdf import extract_text_from_pdf
    from chunking import split_into_chunks
    from embeddings import create_embeddings, model

    pdf_text = extract_text_from_pdf("sample.pdf")
    chunks = split_into_chunks(pdf_text)
    embeddings = create_embeddings(chunks)

    index = build_faiss_index(embeddings)
    print(f"FAISS index built, {index.ntotal} embeddings stored")

    query = "What is machine learning?"
    query_embedding = model.encode([query])[0]

    results = search_similar_chunks(query_embedding, index, chunks)
    print(f"\nQuestion: {query}")
    print("\nMost relevant chunks found:")
    for i, r in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(r)