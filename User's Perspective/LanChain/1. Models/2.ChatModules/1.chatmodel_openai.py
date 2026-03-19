from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

llm = ChatOpenAI(model='gpt-5')

result = llm.invoke("what is the name of Rishi")

print(result)