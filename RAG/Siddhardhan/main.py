import os
#import json
from dotenv import load_dotenv
#
from langchain_ollama import ChatOllama #, OllamaLLM
#from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#
import streamlit as st
#
from vectorize_documents import get_embeddings, vectordb_directory

working_dir = os.path.dirname(os.path.abspath(__file__))

#config_data = json.load(open(f"{working_dir}/config.json"))
#GROQ_API_KEY = config_data["GROQ_API_KEY"]
#os.environ["GROQ_API_KEY"] = GROQ_API_KEY

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
# LangSmith tracking
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
#
#os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


def get_vectorstore():
    huggingface_embeddings = get_embeddings()
    
    vectorstore = Chroma(
        persist_directory=f'{working_dir}{os.sep}{vectordb_directory}',
        embedding_function=huggingface_embeddings
    )
    return vectorstore


def chat_chain(vectorstore):
    #llm = ChatGroq(model='llama-3.1-70b-versatile', temperature=0)
    llm = ChatOllama(model='sutyrin/saiga_mistral_7b', temperature=0.1, base_url='http://localhost:11434')
    
    retriever = vectorstore.as_retriever()
    
    memory = ConversationBufferMemory(
        llm=llm,
        output_key='answer',
        memory_key='chat_history',
        return_messages=True
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type='stuff',
        memory=memory,
        verbose=True,
        return_source_documents=True
    )

    return chain


## streamlit framework

# st.set_page_config(
#     page_title='Multi Doc Chat',
#     page_icon = '📚',
#     layout='centered'
# )

#st.title('📚 Multi Documents Chatbot')
st.title('Demo Ollama With SAIGA MISTRAL')

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

user_input = st.chat_input('Задайте вопрос.')

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = get_vectorstore()

if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = chat_chain(st.session_state.vectorstore)

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if user_input:
    st.session_state.chat_history.append({'role': 'user', 'content': user_input})

    with st.chat_message('user'):
        st.markdown(user_input)
 
    with st.chat_message('assistant'):
        response = st.session_state.rag_chain({'question': user_input})
        assistant_response = response['answer']
        st.session_state.chat_history.append({'role': 'assistant', 'content': assistant_response})
        st.markdown(assistant_response)

