from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
#
from fastapi import FastAPI
import uvicorn
#
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
# LangSmith tracking
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

app = FastAPI(
    title='Langchain Server',
    version='1.0',
    decsription='A simple API Server'
)

# ollama LLAma2 LLM

#llm = Ollama(model='llama2', base_url='http://localhost:11434')
llm = Ollama(model='sutyrin/saiga_mistral_7b', base_url='http://localhost:11434')

prompt = ChatPromptTemplate.from_template('Напишите мне ответ на {topic} из 200 слов')
prompt2 = ChatPromptTemplate.from_template('Напишите мне стих на {topic} для ребенка 5 лет, состоящее из 100 слов.')

add_routes(
    app,
    prompt | llm,
    path='/ask'
)

# По сути здесь должна быть другая модель llm
add_routes(
    app,
    prompt2 | llm,
    path='/test'
)

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)

