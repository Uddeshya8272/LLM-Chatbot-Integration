# LLM-Chatbot-Integration
Chatbot Application Using LLMs with Langchain, FastAPI, Langserve, Uvicorn, and Langsmith Tracking.

                    This project aims to practice and showcase my work on the following key points:
--	Chatbot Development with Python: Create a chatbot using Python and connect it to Large Language Models (LLMs). The primary goal is to select the most appropriate LLM for each query.
--	Call Tracking with Langserve: Utilize Langserve to track calls made to LLM models. This involves monitoring costs, optimizing records, and efficiently storing outputs.
-- API Development: Develop an API to understand the concepts of API creation and handling requests. This involves learning how to design, implement, and interact with APIs effectively.
-- Efficient Pipeline Creation: Establish a robust pipeline for API calls and the chatbot, ensuring that all interactions are recorded and tracked seamlessly. This includes optimizing the entire workflow for efficiency and reliability.

1.	Chatbot with tracking the responses and optimising the environment


	Environment Setup:
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
if not openai_api_key or not langchain_api_key:
    raise ValueError("API keys are not set")
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
	Prompt Template:
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user query"),
    ("user", "Question: {question}")
])



	Combine and Execute Chain:
chain = prompt | llm | output_parser
if input_text:
    st.write(chain.invoke({'question': input_text}))

![pic1](https://github.com/Uddeshya8272/LLM-Chatbot-Integration/assets/118058617/ae3f47a5-5551-4ff6-9c38-25b85b950c3c)



2.	Making of the API 


	Initialize FastAPI:
from fastapi import FastAPI
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A learning server"
)

	Define LLM Models:
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
gpt = ChatOpenAI()
llama = Ollama(model="llama2")

	Define Prompts:
from langchain.prompts import ChatPromptTemplate
prompt1 = ChatPromptTemplate.from_template("Write an essay about {topic} of 100 words")
prompt2 = ChatPromptTemplate.from_template("Write a poem about {topic} of 100 words")




	Add Routes:


from langserve import add_routes
add_routes(app, gpt, path="/openai")
add_routes(app, prompt1 | gpt, path="/essay")
add_routes(app, prompt2 | llama, path="/poem")

	Run the FastAPI Server:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)


3.	Hitting the API-:


	API Response Functions:


import requests

def get_openai_response(input_text):
    response = requests.post(
        "http://localhost:8000/essay/invoke",
        json={'input': {'topic': input_text}}
    )
    return response.json()['output']

def get_llama_response(input_text):
    response = requests.post(
        "http://localhost:8000/poem/invoke",
        json={'input': {'topic': input_text}}
    )
    return response.json()['output']


	Streamlit Interface:


import streamlit as st

st.title("Langchain Demo with LLAMA2 API")
input_text1 = st.text_input("Write an essay on")
input_text2 = st.text_input("Write a poem on")

if input_text1:
    st.write(get_openai_response(input_text1))

if input_text2:
    st.write(get_llama_response(input_text2))

![WhatsApp Image 2024-06-17 at 22 21 28_b8b9d8e8](https://github.com/Uddeshya8272/LLM-Chatbot-Integration/assets/118058617/1851709b-a423-4688-94a8-798e9332a74c)



