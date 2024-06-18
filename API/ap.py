from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key is None:
    raise ValueError("The OPENAI_API_KEY environment variable is not set")

os.environ["OPENAI_API_KEY"] = openai_api_key

app=FastAPI(
    title="Langchain Server",
    verssion="1.0",
    description="A learning server"
)


add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)

# LLM-1
gpt=ChatOpenAI()


# LLM-2
llama=Ollama(model="llama2")

#  Prompts for different use cases
prompt1=ChatPromptTemplate.from_template("Write an essay about {topic} of 100 words")
prompt2=ChatPromptTemplate.from_template("Write a poem about {topic} of 100 words")


add_routes(
    app,
    prompt1|gpt,
    path="/essay"
)

add_routes(
    app,
    prompt2|llama,
    path="/poem"
)


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)


