from langchain_community.tools import tool

@tool # Tool Decorator
def multiply(a: int, b: int) -> int: # Type Hinting
    """Multiply two numbers""" # Doc  string
    return a*b

result = multiply.invoke({"a": 3, "b": 2})

print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)