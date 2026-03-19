from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
   model='meta-llama/Llama-3.1-8B-Instruct',
   huggingfacehub_api_token='😹',
   temperature=0.7,
   max_new_tokens=1000
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a Detailed report on topic {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Summarize the following text {topic}",
    input_variables=['topic']
)

report_generation_chain = RunnableSequence(prompt1, model, parser)

def word_count(text):
    return len(text.split())

branch_chain = RunnableBranch(
    (lambda x: word_count(x) > 100, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_generation_chain, branch_chain)

print(final_chain.invoke({'topic': 'AI in Medical Field'}))