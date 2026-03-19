from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
   model='meta-llama/Llama-3.1-8B-Instruct',
   huggingfacehub_api_token='😹',
   temperature=0.7,
   max_new_tokens=1000
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Write a joke on topic {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

def count_word(text):
    return len(text.split())

chain = RunnableSequence(prompt, model, parser, RunnableParallel({
    'original_text': RunnablePassthrough(),
    'count_of_text': RunnableLambda(count_word),
    # 'text': RunnableLambda(lambda x: len(x.split()))
}))

result = chain.invoke({'topic': 'Movies'})

final_result = """{} \n word-count {}""".format(result['original_text'], result['count_of_text'])

print(final_result)