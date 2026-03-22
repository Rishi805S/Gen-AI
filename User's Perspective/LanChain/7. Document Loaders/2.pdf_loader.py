# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableSequence, RunnableParallel
# import os
from langchain_community.document_loaders import PyPDFLoader
# from dotenv import load_dotenv

# load_dotenv()

# llm = HuggingFaceEndpoint(
#    model='meta-llama/Llama-3.1-8B-Instruct',
#    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
#    temperature=0.7,
#    max_new_tokens=1000
# )
# model = ChatHuggingFace(llm=llm)

# parser = StrOutputParser()

loader = PyPDFLoader("python.pdf")

documents = loader.load()

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(documents[0].page_content)

# print(documents[0].page_content)
# print(documents[0].metadata)


