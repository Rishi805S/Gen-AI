from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="", dimensions=32)

result = embeddings.embed_query("what is ML")

print(str(result))