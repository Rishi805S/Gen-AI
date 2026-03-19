from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

text = "What is Machine Learning"
document = [
    "Hello Who are you "
    "I am Rishi"
    "What is Machine learning"
]

result = embeddings.embed_query(text)

print(str(result))