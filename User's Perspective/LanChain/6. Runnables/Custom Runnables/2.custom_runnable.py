import random
from abc import ABC, abstractmethod
from typing import Any


class Runnable(ABC):
    @abstractmethod
    def invoke(self, input_data) -> Any:
        pass
class CustomLLM(Runnable):

    def __init__(self):
        print('LLM created')

    def invoke(self, input_data):
        print('LLM invoked')
        response_list = [
            'Rishi is a good BOSS',
            'He is very good at communication skills ( Dont laugh serious matter 🤫)',
            'He is a good ..... ( lets discuss this after )'
        ]
        return {'response': random.choice(response_list)}

    def predict(self, prompt):

        response_list = [
            'Delhi is the capital of India',
            'IPL is a cricket league',
            'AI stands for Artificial Intelligence'
        ]

        return {'response': random.choice(response_list)}
        
class CustomPromptTemplate(Runnable):

    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
        print('Prompt created')
    
    def invoke(self, input_data, config=None):
        print('Prompt invoked')
        if isinstance(input_data, str):
            input_data = {"response": input_data}
        return self.template.format(**input_data)

    def format(self, input_dict):
        print('Prompt formatted')
        return self.template.format(**input_dict)
    
class CustomStrOutputParser(Runnable):

    def __init__(self):
        print('Parser created')
        pass

    def invoke(self, input_data):
        print('Parser invoked')
        return input_data['response']
  
class RunnableConnector(Runnable):

    def __init__(self, runnable_list):
        self.runnable_list = runnable_list
        print('Runnable Connector connected')

    def invoke(self, input_data):
        print('Runnable Connector invoked')

        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)

        return input_data
    
# RUN 
template = CustomPromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['length', 'topic']
)

llm = CustomLLM()

parser = CustomStrOutputParser()

chain = RunnableConnector([template, llm, parser])

result = chain.invoke({'length': 'short', 'topic': 'Machine learning'})

template1 = CustomPromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

template2 = CustomPromptTemplate(
    template='Explain the following joke {response}',
    input_variables=['response']
)

chain1 = RunnableConnector([template1, llm, parser])

chain2 = RunnableConnector([template2, llm, parser])

final_chain = RunnableConnector([chain1, chain2])

result = final_chain.invoke({'length': 'short', 'topic': 'Machine learning'})

print(result)
