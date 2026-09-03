import streamlit as st
from read_pdf import extract_text_from_pdf
from chunking import split_into_chunks
from embeddings import create_embeddings, model
from vector_store import build_faiss_index, search_similar_chunks
from main import generate_answer

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("📄 Smart Document Q&A")
st.write("Upload a PDF and ask questions about it!")

uploaded_files = st.file_uploader("Upload your PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    current_file_names = tuple(sorted(f.name for f in uploaded_files))

    if st.session_state.get("uploaded_file_names") != current_file_names:
        with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
            all_chunks = []

            for uploaded_file in uploaded_files:
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                pdf_text = extract_text_from_pdf("temp.pdf")

                if pdf_text.strip():
                    file_chunks = split_into_chunks(pdf_text)
                    all_chunks.extend(file_chunks)

            if not all_chunks:
                st.error("Couldn't extract any text from these PDFs. They might be scanned images — please try PDFs with selectable text.")
                st.stop()

            embeddings = create_embeddings(all_chunks)
            index = build_faiss_index(embeddings)

            st.session_state.index = index
            st.session_state.chunks = all_chunks
            st.session_state.uploaded_file_names = current_file_names

        st.success(f"{len(uploaded_files)} document(s) ready! Total {len(st.session_state.chunks)} chunks indexed. Ask your questions below.")
    query = st.text_input("Ask a question about your document:")

    if query:
        with st.spinner("Finding your answer..."):
            query_embedding = model.encode([query])[0]
            relevant_chunks = search_similar_chunks(
                query_embedding, st.session_state.index, st.session_state.chunks
            )
            answer = generate_answer(query, relevant_chunks)

        st.subheader("Answer:")
        st.write(answer)