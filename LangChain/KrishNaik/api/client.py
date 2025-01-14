import requests
import streamlit as st

# Откуда появился 'ask' см. файл server.py
def get_response_from_ollama(input_text):
    response = requests.post(
        url='http://localhost:8000/ask/invoke',
        verify=False,
        json={'input': {'topic': input_text}}
    )

    return response.json()['output']

# Откуда появился 'test' см. файл server.py
def get_response_from_ollama_2(input_text):
    response = requests.post(
        url='http://localhost:8000/test/invoke',
        verify=False,
        json={'input': {'topic': input_text}}
    )

    return response.json()['output']

# streamlit framework

st.title('Langchain Demo With SAIGA MISTRAL')

input_text = st.text_input('Задайте вопрос для LLM1.')
input_text2 = st.text_input('Задайте вопрос для LLM2.')

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

if input_text:
    st.write(get_response_from_ollama(input_text))

if input_text2:
    st.write(get_response_from_ollama_2(input_text2))

