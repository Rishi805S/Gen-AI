from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
   model='meta-llama/Llama-3.1-8B-Instruct',
   huggingfacehub_api_token='😹',
   temperature=0.7,
   max_new_tokens=1000
)
model1 = ChatHuggingFace(llm=llm)

model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Generate short and simple notes from following text {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate 5 short quizzes from following text {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="merge provided notes and quiz into single documentation {notes} and {quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chains = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})

merge_chains = prompt3 | model1 | parser

chain = parallel_chains | merge_chains

text = "Artificial Intelligence is transforming the way humans interact with technology. It enables machines to learn from data, identify patterns, and make decisions with minimal human intervention. one of the most widely used applications of AI is in recommendation systems, where platforms like streaming servies and e-commere websites suggest content or products based on user behavior. machine learning, a subset of AI, focuses on building mdoels that improve thier performance as theya re exposed to more data. These models can be supervised, unsupervised, or reinforcement based, depending on how they learn from the data."

result = chain.invoke({'text': text})

print(result)