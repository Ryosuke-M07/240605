import streamlit as st
import uuid
import hashlib
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_PATH = "./local_qdrant"

def init_page():
    st.set_page_config(page_title="Ask My PDF(s)", page_icon="🤗")
    st.session_state.costs = []

def hash_string(input_string):
    return hashlib.sha256(input_string.encode()).hexdigest()

def generate_session_id(username, password):
    user_hash = hash_string(username + password)
    st.session_state.session_id = user_hash

def generate_unique_collection_name(username, password):
    user_hash = hash_string(username + password)
    collection_name = f"collection_{user_hash}"
    st.session_state.collection_name = collection_name
    return collection_name

def select_model(openai_api_key):
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-3.5-16k", "GPT-4"))
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo"
    elif model == "GPT-3.5-16k":
        st.session_state.model_name = "gpt-3.5-turbo-16k"
    else:
        st.session_state.model_name = "gpt-4"
    st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
    return ChatOpenAI(temperature=0, model_name=st.session_state.model_name, api_key=openai_api_key)

def get_pdf_text():
    uploaded_file = st.file_uploader(label='Upload your PDF here😇', type='pdf')
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="text-embedding-ada-002", chunk_size=500, chunk_overlap=0)
        return text_splitter.split_text(text)
    else:
        return None

def load_qdrant(openai_api_key, collection_name):
    client = QdrantClient(path=QDRANT_PATH)
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    if collection_name not in collection_names:
        client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=1536, distance=Distance.COSINE))
        print('collection created')
    return Qdrant(client=client, collection_name=collection_name, embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key))

def build_vector_store(pdf_text, openai_api_key, collection_name):
    qdrant = load_qdrant(openai_api_key, collection_name)
    qdrant.add_texts(pdf_text)

def build_qa_model(llm, openai_api_key, collection_name):
    qdrant = load_qdrant(openai_api_key, collection_name)
    retriever = qdrant.as_retriever(search_type="similarity", search_kwargs={"k":10})
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, verbose=True)

def page_pdf_upload_and_build_vector_db(openai_api_key, collection_name):
    st.title("PDF Upload")
    container = st.container()
    with container:
        pdf_text = get_pdf_text()
        if pdf_text:
            with st.spinner("Loading PDF ..."):
                build_vector_store(pdf_text, openai_api_key, collection_name)

def ask(qa, query):
    with get_openai_callback() as cb:
        answer = qa(query)
    return answer, cb.total_cost

def page_ask_my_pdf(openai_api_key, collection_name):
    st.title("Ask My PDF(s)")
    llm = select_model(openai_api_key)
    container = st.container()
    response_container = st.container()
    with container:
        query = st.text_input("Query: ", key="input")
        if not query:
            answer = None
        else:
            qa = build_qa_model(llm, openai_api_key, collection_name)
            if qa:
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost = ask(qa, query)
                st.session_state.costs.append(cost)
            else:
                answer = None
        if answer:
            with response_container:
                st.markdown("## Answer")
                st.write(answer)

def main():
    init_page()

    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username and password:
            st.session_state['username'] = username
            st.session_state['password'] = password
            st.session_state['logged_in'] = True
        else:
            st.sidebar.error("Please enter both username and password")
    
    if st.session_state.get('logged_in'):
        st.success(f"Welcome {st.session_state['username']}")
        generate_session_id(st.session_state['username'], st.session_state['password'])
        st.sidebar.text(f"Session ID: {st.session_state.session_id}")

        openai_api_key = st.sidebar.text_input("OpenAI API Key", key="file_qa_api_key", type="password")
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key to proceed.")
            return
        
        collection_name = generate_unique_collection_name(st.session_state['username'], st.session_state['password'])

        selection = st.sidebar.radio("Go to", ["PDF Upload", "Ask My PDF(s)"])
        if selection == "PDF Upload":
            page_pdf_upload_and_build_vector_db(openai_api_key, collection_name)
        elif selection == "Ask My PDF(s)":
            page_ask_my_pdf(openai_api_key, collection_name)
        
        costs = st.session_state.get('costs', [])
        st.sidebar.markdown("## Costs")
        st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
        for cost in costs:
            st.sidebar.markdown(f"- ${cost:.5f}")
    else:
        st.warning("Please login to continue")

if __name__ == '__main__':
    main()
