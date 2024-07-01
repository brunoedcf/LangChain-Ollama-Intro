from typing import List

from fastapi import FastAPI
from langserve import add_routes

# Langsmith tracing
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("langsmith_api")

from langchain_community.chat_models import ChatOllama

local_llm = "llama3"
model = ChatOllama(model=local_llm, format="json", temperature=0)
# Utilize format="json" quando a saída esperada do LLM é um JSON.

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Você é um assistente de identificação e tradução de idiomas.
    Identifique qual é o idioma da frase de entrada do usuário.
    Depois disso deve traduzir para o idioma Italiano.
    Provenha um JSON com duas chaves, sendo elas 'idioma' com o idioma identificado escrito em português e em lowercase, e 'tradução' com a tradução para o Italiano.
    Retorne somente o JSON sem nenhuma explicação adicional.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Aqui está a frase do usuário: {sentence} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["sentence"],
)

from langchain_core.output_parsers import JsonOutputParser
# Use JsonOutputParser quando a saída esperada do LLM é um JSON.

chain = prompt | model | JsonOutputParser()

# App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# Adding chain route

add_routes(
    app,
    chain,
    path="/chain",
)

from langserve import RemoteRunnable

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
    # Go to http://localhost:8000/chain/playground/