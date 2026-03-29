import json
import os
from typing import Annotated

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg, tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
exchange_rate_api_key = os.getenv("EXCHANGERATE_API_KEY")

if not hf_token:
    raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN in environment variables.")

if not exchange_rate_api_key:
    raise ValueError("Missing EXCHANGERATE_API_KEY in environment variables.")

model = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=hf_token,
    temperature=0.7,
    max_new_tokens=1000,
)
llm = ChatHuggingFace(llm=model)


@tool
def get_conversion_rate(base_currency: str, target_currency: str) -> float:
    """This function fetches the currency conversion factor between a given base currency and target currency"""
    url = f"https://v6.exchangerate-api.com/v6/{exchange_rate_api_key}/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    return response.json()


@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """This function fetches target cuurency value given conversion rate and base currency value"""
    return base_currency_value * conversion_rate


llm_with_tools = llm.bind_tools([get_conversion_rate, convert])

messages = []

query = "What is conversion rate factor between USD and INR and based on that convert 10 USD to INR"

messages.append(HumanMessage(query))

ai_message = llm_with_tools.invoke(messages)

messages.append(ai_message)

conversion_rate = 0

for tool_call in ai_message.tool_calls:
    if tool_call["name"] == "get_conversion_factor":
        tool_message1 = get_conversion_rate.invoke(tool_call)
        conversion_rate = json.loads(tool_message1.content)["conversion_rate"]
        messages.append(tool_message1)

    if tool_call["name"] == "convert":
        tool_call["args"]["conversion_rate"] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)

print(llm.invoke(messages).content)
