import streamlit as st
#
from langchain_community.llms import Ollama
#from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
#
import time
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
# GROQ
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# Заменить OpenAIEmbeddings
#model_id = 'sentence-transformers/all-MiniLM-l6-v2'
model_id = 'BAAI/bge-small-en-v1.5'

#model_args = {'device': 'cuda'}
model_args = {'device': 'cpu'}

huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_id,
    model_kwargs=model_args,
    encode_kwargs={'normalize_embeddings': True}
)

if 'vectors' not in st.session_state:
    #st.session_state.embeddings = OllamaEmbeddings(model='sutyrin/saiga_mistral_7b')
    st.session_state.embeddings = huggingface_embeddings
    st.session_state.loader = WebBaseLoader('https://education.yandex.ru/handbook/ml/article/mashinnoye-obucheniye')
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    #st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

llm = Ollama(model='sutyrin/saiga_mistral_7b', base_url='http://localhost:11434')
#llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

prompt_template = """
Нужно ответить на вопрос, основываясь только на предоставленном контексте.
Подумай шаг за шагом, прежде чем предоставить подробный ответ.
<context>
{context}
</context>
Questions: {input}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = st.session_state.vectors.as_retriever()

# Цепочка поиска: эта цепочка принимает запрос пользователя,
# который передается извлекателю для извлечения релевантных документов.
# Эти документы (и исходные входные данные) затем передаются в LLM для генерации ответа

retrieval_chain = create_retrieval_chain(retriever, document_chain)

st.title("ChatGroq Demo")

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        .stAppDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
    """,
    unsafe_allow_html=True
)

input_text = st.text_input('Задайте вопрос.')

if input_text:
    start = time.process_time()
    response = retrieval_chain.invoke({'input': input_text})
    print('Response time:', time.process_time() - start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander('Document Similarity Search'):
        # Find the relevant chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("--------------------------------")