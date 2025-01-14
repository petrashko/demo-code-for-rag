import os
from dotenv import load_dotenv
#
from langchain_ollama import ChatOllama #, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
#
import streamlit as st
#
#from vectorize_documents import get_embeddings, vectordb_directory

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
# LangSmith tracking
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

working_dir = os.path.dirname(os.path.abspath(__file__))
vectordb_directory = 'vectordb_dir'
print(f'{working_dir}{os.sep}{vectordb_directory}')


def get_embeddings():
    #model_id = 'sentence-transformers/all-MiniLM-l6-v2'
    model_id = 'BAAI/bge-m3'

    #model_args = {'device': 'cuda'}
    model_args = {'device': 'cpu'}

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_id,
        model_kwargs=model_args,
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings


def get_vectorstore():
    huggingface_embeddings = get_embeddings()
    
    vectorstore = Chroma(
        persist_directory=f'{working_dir}{os.sep}{vectordb_directory}',
        embedding_function=huggingface_embeddings
    )
    return vectorstore


## streamlit framework

st.title('Demo Ollama With SAIGA MISTRAL')
#user_input = st.text_input('Задайте вопрос.')
user_input = st.chat_input('Задайте вопрос.')

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

# Ollama SAIGA_MISTRAL LLM

promt_template = '''
Ты - Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.
Используй следующие фрагменты релевантного контекста, чтобы ответить на вопрос.
Если ты не знаешь ответ, просто скажи, что не знаю.
Используй максимум двадцать предложений и сделай ответ кратким.
Question: {question}
Context: {context}
Answer:
'''

if 'llm' not in st.session_state:
    #st.session_state.llm = OllamaLLM(model='llama2', base_url='http://localhost:11434')
    #st.session_state.llm = OllamaLLM(model='sutyrin/saiga_mistral_7b', temperature=0.1, base_url='http://localhost:11434')
    st.session_state.llm = ChatOllama(model='sutyrin/saiga_mistral_7b', temperature=0.1, base_url='http://localhost:11434')

if 'rag_chain' not in st.session_state:
    prompt = ChatPromptTemplate.from_template(promt_template)
    output_parser = StrOutputParser()
    st.session_state.rag_chain = (prompt | st.session_state.llm | output_parser)

if 'retriever' not in st.session_state:
    vectorstore = get_vectorstore()
    #st.session_state.retriever = vectorstore.as_retriever()
    st.session_state.retriever = 'Моего сына зовут Марк, ему 12 лет. Он ходит в школу №40'

if user_input:
    st.write(st.session_state.rag_chain.invoke(
        {'context': st.session_state.retriever,  'question': user_input}
    ))


# if __name__ == '__main__':
#     print()
#     print('END!!!')

