# import streamlit as st
# from PyPDF2 import PdfReader
# import os
# from dotenv import load_dotenv

# from langchain_groq import ChatGroq
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# # ========= LangChain Compatibility Imports =========
# try: 
#     from langchain.text_splitter import RecursiveCharacterTextSplitter
# except: 
#     from langchain_text_splitters import RecursiveCharacterTextSplitter

# try:
#     from langchain.chains. question_answering import load_qa_chain
# except:
#     from langchain.chains import load_qa_chain

# try:
#     from langchain.prompts import PromptTemplate
# except:
#     from langchain.prompts. prompt import PromptTemplate
# # ==================================================

# load_dotenv()


# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader. pages:
#             content = page.extract_text()
#             if content:
#                 text += content
#     return text


# def get_text_chunks(text):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=10000,
#         chunk_overlap=1000
#     )
#     return splitter.split_text(text)


# def get_vector_store(text_chunks):
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("Faiss_index")


# def get_conversational_chain():
#     prompt_template = """
# Answer the question as detailed as possible from the provided context. 
# If the answer is not in the context, say: 
# "Answer is not present in the given PDF."

# Context:
# {context}

# Question: 
# {question}

# Answer:
# """

#     prompt = PromptTemplate(
#         template=prompt_template,
#         input_variables=["context", "question"]
#     )

#     model = ChatGroq(
#         model="llama-3.3-70b-versatile",
#         temperature=0.3
#     )

#     chain = load_qa_chain(
#         llm=model,
#         chain_type="stuff",
#         prompt=prompt
#     )
#     return chain


# def generate_quiz_from_pdf(pdf_text:  str, num_questions: int) -> str:
#     prompt = f"""
# Generate {num_questions} multiple-choice questions from this PDF content.
# Format:
# Q1. [Question]
# A) [Option]
# B) [Option]
# C) [Option]
# D) [Option]
# **Answer: [Correct]**

# Content:
# {pdf_text[:8000]}
# """

#     model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
#     response = model.invoke(prompt)
#     return str(response.content).strip()


# def user_input(user_question):
#     try:
#         embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2"
#         )

#         db = FAISS.load_local(
#             "Faiss_index",
#             embeddings,
#             allow_dangerous_deserialization=True
#         )

#         docs = db.similarity_search(user_question)
#         chain = get_conversational_chain()

#         response = chain(
#             {"input_documents": docs, "question": user_question},
#             return_only_outputs=True
#         )

#         st.markdown("### 💡 Answer:")
#         st.write(response["output_text"])

#     except FileNotFoundError:
#         st. warning("⚠️ Please upload and process PDFs first.")
#     except Exception as e: 
#         st.error(f"❌ Error: {str(e)}")


# def run_app():
#     st.markdown("## 💬 PDF Q&A + Quiz Generator")

#     pdf_docs = st.file_uploader(
#         "📂 Upload your PDF files",
#         accept_multiple_files=True,
#         type=["pdf"]
#     )

#     if st.button("Submit & Process PDFs", use_container_width=True):
#         if not pdf_docs:
#             st. warning("Please upload at least one PDF.")
#         else:
#             with st.spinner("Processing PDFs..."):
#                 raw_text = get_pdf_text(pdf_docs)

#                 if not raw_text.strip():
#                     st.error("No text extracted.")
#                     return

#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
                
#                 # Store PDF text in session state for quiz generation
#                 st.session_state['pdf_text'] = raw_text
                
#                 st.success("✅ Done! Ask questions or generate a quiz.")

#     st.markdown("---")

#     # Mode selection
#     mode = st.radio("Choose mode:", ["💬 Ask Questions", "🎯 Generate Quiz"], horizontal=True)

#     if mode == "💬 Ask Questions":
#         user_question = st.text_input("🗣️ Ask a question about your PDF:")
#         if user_question:
#             user_input(user_question)
    
#     else:  # Quiz mode
#         if 'pdf_text' in st.session_state:
#             num_questions = st.slider("Number of quiz questions:", 5, 20, 10)
            
#             if st.button("🎯 Generate Quiz", use_container_width=True):
#                 with st.spinner("Creating quiz from your PDF..."):
#                     quiz = generate_quiz_from_pdf(st.session_state['pdf_text'], num_questions)
                    
#                     st.markdown("### 🎯 Your Quiz")
#                     st.markdown(quiz)
                    
#                     st.download_button("📄 Download Quiz", quiz, file_name="pdf_quiz.md")
#         else:
#             st.info("👆 Upload and process a PDF first to generate a quiz")


# if __name__ == "__main__":
#     run_app()