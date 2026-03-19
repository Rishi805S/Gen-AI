from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel
from typing import TypedDict
from dotenv import load_dotenv
from typing import cast
load_dotenv()

llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token="😹",
    temperature=0.7,
    max_new_tokens=100
)
model = ChatHuggingFace(llm=llm)

class Review(BaseModel):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)

result = cast(Review, structured_model.invoke(" The Hardware is great but the software feels bloated.There are too many pre installed apps which I can't remove. Also the UI looks outdated compared to other brands. Hope you will fix this soon in next software update"))

print(result)
print(result.summary)
print(result.sentiment)


