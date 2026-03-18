from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
   model='meta-llama/Llama-3.1-8B-Instruct',
   huggingfacehub_api_token='😹',
   temperature=0.7,
   max_new_tokens=1000
)
model = ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] # Currently couldn't figure out how to give description

parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instructions}",
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback {feedback}",
    input_variables=['feedback']
)

# This version didn't worked because pylance couldn't figure whether there exists .sentiment or not so we add type hint

# branch_chain = RunnableBranch(
#     (lambda x: x.sentiment == 'positive', prompt2 | model | parser1),
#     (lambda x: x.sentiment == 'negative', prompt3 | model | parser1),
#     RunnableLambda(lambda x : "could not find sentiment")
# )

# This version is recommended
def is_positive(x: Feedback) -> bool:
    return x.sentiment == 'positive'

def is_negative(x: Feedback) -> bool:
    return x.sentiment == 'negative'

branch_chain = RunnableBranch(
    (is_positive, prompt2 | model | parser1),
    (is_negative, prompt3 | model | parser1),
    RunnableLambda(lambda x: "could not find sentiment")
)

# This is Type hint in Lambda
branch_chain = RunnableBranch(
    (lambda x: isinstance(x, Feedback) and x.sentiment == 'positive', prompt2 | model | parser1),
    (lambda x: isinstance(x, Feedback) and x.sentiment == 'negative', prompt3 | model | parser1),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': "This is best smartphone"}))

# To generate flow graph
chain.get_graph().print_ascii()