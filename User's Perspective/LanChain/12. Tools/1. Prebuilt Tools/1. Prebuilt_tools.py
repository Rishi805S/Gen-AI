from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools import ShellTool

search = DuckDuckGoSearchResults()
shell = ShellTool()

result = shell.invoke("whoami")

print(result)