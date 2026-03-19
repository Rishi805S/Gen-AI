from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'You are an helpful AI assistant'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []

with open('Chat_history.txt', 'r') as file:
    chat_history.extend(file.readlines())
print(chat_history)
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'Hello'})
print(prompt)
