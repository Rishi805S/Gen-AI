from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
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
    input_variables=['topic'],
    template="Generate a tweet on the topic {topic}"
)

prompt2 = PromptTemplate(
    input_variables=['topic'],
    template="Generate a LinkedIN post on the topic {topic}"
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})

print(parallel_chain.invoke({'topic': 'AI'}))