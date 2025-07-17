from langchain_huggingface import HuggingFaceEmbeddings
import re
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document
import streamlit as st

from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def get_answer(question):
    result = rag_chain({"query": question})
    # Extract the generated answer
    return result["result"].split("Answer:")[1].strip()

#Creating embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

#Making a FAISS Vector store
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

#Reading the blog .txt file
with open("blogs/blog 2.txt", "r", encoding="utf-8") as file:
    text = file.read()

#Cleaning up the read file
sentences=[s.strip() + '.' for s in re.split(r'\.\s*', text) if s]

chunks = []
current_chunk = []
current_word_count = 0
for sentence in sentences:
       word_count = len(sentence.split())
       if current_word_count + word_count <= 200:
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

#Converts the list into a dictionary
for i in range(len(chunks)):
   documents[i] = Document(page_content=chunks[i])

uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)

#Saves vector store
vector_store.save_local("faiss_index_dir")

#Loads the vector store
loaded_vector_store = FAISS.load_local(
"faiss_index_dir",
embeddings, allow_dangerous_deserialization=True
)

retriever = loaded_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})

model_version = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_version)
model = AutoModelForCausalLM.from_pretrained(model_version)

# Create a Hugging Face pipeline for text generation
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7
)
# Wrap the pipeline for LangChain compatibility
llm = HuggingFacePipeline(pipeline=pipe)
# Define the Prompt Template
template = """You are a helpful assistant. Use the context below to answer the user's question.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
# Define the RAG Chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
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
