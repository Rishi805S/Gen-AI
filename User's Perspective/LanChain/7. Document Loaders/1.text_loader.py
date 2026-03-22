from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
   model='meta-llama/Llama-3.1-8B-Instruct',
   huggingfacehub_api_token='os.getenv("HUGGINGFACEHUB_API_TOKEN")',
   temperature=0.7,
   max_new_tokens=1000
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Generate summary on topic {topic}",
    input_variables=['topic']
)

chain = prompt | model | parser

# Create loader
loader = TextLoader("oops.txt", encoding="utf-8")

# Load documents
documents = loader.load()

print(chain.invoke({'topic': documents[0].page_content}))