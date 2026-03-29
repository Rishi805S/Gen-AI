import os
from typing import Annotated

import requests
from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
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


search_tool = DuckDuckGoSearchRun()

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm, prompt=prompt, tools=[search_tool, get_conversion_rate, convert])

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=[search_tool, get_conversion_rate, convert],
    verbose=True,
)

result = agent_executor.invoke(
    {
        "input": "Find me the latest conversion rate from India to USA, and tell me the current INR for 10 USD"
    }
)

print(result["output"])
