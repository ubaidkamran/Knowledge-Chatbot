from langchain_huggingface import HuggingFaceEmbeddings
import re
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document
import streamlit as st

def get_answer(question):

    retriever = loaded_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
    result=retriever.get_relevant_documents(question)

    return result[0].page_content

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

with open("blogs/blog 2.txt", "r", encoding="utf-8") as file:
    text = file.read()

sentences=[s.strip() + '.' for s in re.split(r'\.\s*', text) if s]

chunks = []
current_chunk = []
current_word_count = 0
for sentence in sentences:
       word_count = len(sentence.split())
       if current_word_count + word_count <= 50:
           current_chunk.append(sentence)
           current_word_count += word_count
       else:
           # Commit current chunk and start a new one
           chunks.append(' '.join(current_chunk))
           current_chunk = [sentence]
           current_word_count = word_count

if current_chunk:
    chunks.append(' '.join(current_chunk))

documents = [None] * len(chunks)  # Create an empty list of the correct size

for i in range(len(chunks)):
   documents[i] = Document(page_content=chunks[i])

uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)

vector_store.save_local("faiss_index_dir")

loaded_vector_store = FAISS.load_local(
"faiss_index_dir",
embeddings, allow_dangerous_deserialization=True
)

st.set_page_config(page_title="Knowledge Chatbot")

st.title("📚 Your Personal Knowledge Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []


query = st.text_input("Ask something from your content:")
if query.strip():
    response = get_answer(query)
    st.session_state.history.append((query, response))

for q, r in reversed(st.session_state.history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {r}")