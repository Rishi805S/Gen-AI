import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

text = "What is Machine Learning"
document = [
    "machine learning is a complex technology",
    "English is a good language"
    "Gen AI is a Great revolution",
]

query = "English is very good language"

doc_embeddings = embeddings.embed_documents(document)
query_embeddings = embeddings.embed_query(query)

# converting list[list[float]] to cosine_similarity parameter data that is matrix so used np
np_doc_embeddings = np.array(doc_embeddings)
np_query_embeddings = np.array(query_embeddings).reshape(1, -1) # converts 1D array to 2D

similarity = cosine_similarity(np_doc_embeddings, np_query_embeddings)

best_doc = np.argmax(similarity)

print("Similarity scores:\n", similarity)
print("Best doc index:", best_doc)
print("Best doc:", document[best_doc])
# result = embeddings.embed_documents(document)

# print(str(result))