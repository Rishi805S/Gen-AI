from langchain_core.documents import Document
from langchain_community.retrievers import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    huggingfacehub_api_token="hf_token",
)

chat_model = ChatHuggingFace(llm=llm)
# Relevant health & wellness documents
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L12-v2")

vector_store = FAISS.from_documents(
    documents=all_docs,
    embedding=embeddings
)

vector_retriever = vector_store.as_retriever(
    setattr="MMR", # Using Maximal Marginal Relevance
    kwargs={"k": 3, "lambda_mult": 1} # Lambda value for MMR which ranges from 0 - 1, 0 means only highest similarity without diversity among retrieved docs and 1 means only lowest similarity with diverstity among retrieved docs
)

multi_query_retriever = MultiQueryRetriever(
    retrievers=vector_retriever,
    llm=chat_model
)

query = "How to improve energy levels and maintain balance"

multi_query_results = multi_query_retriever.invoke(query)

for i, doc in enumerate(multi_query_results):
    print(f"\n---- Result {i+1} ---- \n")
    print(f"Document {i}: {doc.page_content}")



    