import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN in environment variables.")

model = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=hf_token,
    temperature=0.7,
    max_new_tokens=1000,
)
llm = ChatHuggingFace(llm=model)

parser = StrOutputParser()


@tool
def multiply(a: int, b: int) -> int:
    """Given two numbers returns product of two numbers"""
    return a * b


llm_with_tool = llm.bind_tools([multiply])

messages = []

query = "Can you muliply 3 and 4 "

human_message = HumanMessage(query)

messages.append(human_message)

result = llm_with_tool.invoke("Can you muliply 4 and 20 ")

messages.append(result)

tool_result = multiply.invoke(result.tool_calls[0])

messages.append(tool_result)

print(llm_with_tool.invoke(messages).content)
