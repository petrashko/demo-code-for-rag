from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
# LangSmith tracking
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

prompt=ChatPromptTemplate.from_messages(
    [
        ('system', 'Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.'),
        ('user', 'Question:{question}')
    ]
)

## streamlit framework

# st.title('Langchain Demo With SAIGA MISTRAL')
# input_text = st.text_input('Задайте вопрос.')
# hide_menu_style = '<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>'
# st.markdown(hide_menu_style, unsafe_allow_html=True)

st.title('Langchain Demo With SAIGA MISTRAL')
input_text = st.text_input('Задайте вопрос.')

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

# ollama LLAma2 LLM

#llm = Ollama(model='llama2', base_url='http://localhost:11434')
llm = Ollama(model='sutyrin/saiga_mistral_7b', base_url='http://localhost:11434')

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(
        chain.invoke({
            'question': input_text
        })
    )


# if __name__ == '__main__':
#     print()
#     print('END!!!')
