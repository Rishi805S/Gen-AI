from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
   model='meta-llama/Llama-3.1-8B-Instruct',
   huggingfacehub_api_token='😹',
   temperature=0.7,
   max_new_tokens=1000
)
model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Write a joke on topic {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain this joke {topic}",
    input_variables=['topic']
)
parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
}))

print(chain.invoke({'topic': 'Movies'}))