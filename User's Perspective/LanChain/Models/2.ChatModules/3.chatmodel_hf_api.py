import os

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv(
    "HUGGINGFACEHUB_ACCESS_TOKEN"
)

if not hf_token:
    raise ValueError(
        "Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN in your .env file."
    )

llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)

chat_model = ChatHuggingFace(llm=llm)

result = chat_model.invoke("What is the capital of India?")

print(result.content)

