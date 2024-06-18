from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# Access the environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key is None:
    raise ValueError("The OPENAI_API_KEY environment variable is not set")

os.environ["OPENAI_API_KEY"] = openai_api_key

langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
if langchain_api_key is None:
    raise ValueError("The LANGCHAIN_API_KEY environment variable is not set")

os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
# langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_API_KEY"]=os.getenv('LANGCHAIN_API_KEY')


## prompt template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful ssistant. Please response to the user query"),
        ("user","Question:{question}")
    ]
)

# streamlit format

st.title("Open ai chatbot")
input_text=st.text_input("Please ask question")


# LLM 1--------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------
# openai llm
# llm=ChatOpenAI(model="gpt-3.5-turbo-16k")

#---------------------------------------------------------------------------

# LLM 2---------------------------------------------------------------------------------------------------
# MAKING FREE LLM ENDPOINTS CALL
# keyy=os.getenv('HUGGINGFACE_API_KEY')
# from langchain_huggingface import HuggingFaceEndpoint

# repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# llm=HuggingFaceEndpoint(repo_id=repo_id,temperature=0.2, max_length=128, token=keyy)


# LLM 3----------------------------------------------------------------------------------------------------

llm=Ollama(model="llama2",temperature="0.2")


# ------------------For handling the outputs-----------who --------

output_parser=StrOutputParser()

# --------------------------------------------------------------

# making final chain
chain=prompt|llm|output_parser


if input_text:
    st.write(chain.invoke({'question':input_text}))
    



chain = prompt | llm | output_parser
if input_text:
    st.write(chain.invoke({'question': input_text}))
