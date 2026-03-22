from  langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L12-v2")


docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

vector_retriever = vector_store.as_retriever(
    setattr="MMR", # Using Maximal Marginal Relevance
    kwargs={"k": 3, "lambda_mult": 1} # Lambda value for MMR which ranges from 0 - 1, 0 means only highest similarity without diversity among retrieved docs and 1 means only lowest similarity with diverstity among retrieved docs
)

query = "What is Langchain"
results = vector_retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n---- Result {i+1} ---- \n")
    print(doc.page_content)