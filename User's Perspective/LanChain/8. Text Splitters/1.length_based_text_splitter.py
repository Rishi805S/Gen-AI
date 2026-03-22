from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Python.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=100,
    chunk_overlap=20,
    # length_function=len
)

result = splitter.split_documents(docs)

print(result[0])