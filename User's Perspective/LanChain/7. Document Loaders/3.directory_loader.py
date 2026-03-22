
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(
    path="Machine_learning_folder",
    glob="*.pdf",
    loader_cls=PyPDFLoader #type: ignore
)

# This is called Eager load, this loads all at once memory, not recommended when no.of files are too large or more
# documents = loader.load()

# This is called Lazy load, this loads only when required, recommended when no.of files are small
documents = loader.lazy_load()

for document in documents:
    print(document.metadata)

# print(len(documents))




