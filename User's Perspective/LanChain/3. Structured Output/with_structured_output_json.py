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

json_schema = {
    "title": "Student",
    "description": "Schema about Students",
    "type": "object",
    "properties": {
        "name": "string",
        "age": "integer"
    },
    "required": ['name']
}

structured_model = model.with_structured_output(json_schema)

print(structured_model)


