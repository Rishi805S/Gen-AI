from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(wiki_client=None, top_k_results=2, lang="en")

query = "What is Artificial Intelligence?"

result = retriever.invoke(query) # Using invoke function because it is a Runable

# print(result)

for i, doc in enumerate(result):
    print(f"\n---- Result {i+1} ---- \n")
    print(f"Document {i}: {doc.page_content}")