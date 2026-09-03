import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

from read_pdf import extract_text_from_pdf
from chunking import split_into_chunks
from embeddings import create_embeddings, model
from vector_store import build_faiss_index, search_similar_chunks

def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""Answer the question based on the context given below.
If the answer is not in the context, say "The answer was not found in the document."

Context:
{context}

Question: {query}

Answer:"""

    llm = genai.GenerativeModel("gemini-flash-latest")
    response = llm.generate_content(prompt)
    return response.text

def ask_question(query, index, chunks):
    query_embedding = model.encode([query])[0]
    relevant_chunks = search_similar_chunks(query_embedding, index, chunks)
    answer = generate_answer(query, relevant_chunks)
    return answer

if __name__ == "__main__":
    print("Reading PDF...")
    pdf_text = extract_text_from_pdf("sample.pdf")
    chunks = split_into_chunks(pdf_text)
    embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)
    print(f"Ready! {len(chunks)} chunks indexed.\n")

    while True:
        query = input("Ask your question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = ask_question(query, index, chunks)
        print(f"\nAnswer: {answer}\n")