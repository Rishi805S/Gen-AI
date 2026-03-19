from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
   model='meta-llama/Llama-3.1-8B-Instruct',
   huggingfacehub_api_token='😹',
   temperature=0.7,
   max_new_tokens=700
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Give me 5 facts about the topic. {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser 

result = chain.invoke({'topic': 'machine learning'})

chain.get_graph().print_ascii()

print(result)