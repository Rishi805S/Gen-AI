from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv


load_dotenv()

llm = ChatAnthropic(model_name='claude-3-5-sonnet', timeout=1, stop=None)

result = llm.invoke("what is the name of Rishi")

print(result)