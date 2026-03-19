from langchain_openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

llm = OpenAI(model='gpt-5')

result = llm.invoke("what is the name of Rishi")

print(result)