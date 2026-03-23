from pydantic import BaseModel, Field
from langchain_community.tools import StructuredTool
class MultiplyTwoNumbers(BaseModel):
    a: int = Field(description="The first number to multiply")
    b: int = Field(description="The second number to multiply")

def multiply_numbers(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool(
    name="Multiply Two Numbers",
    description="Multiplies two numbers",
    func=multiply_numbers,
    args_schema=MultiplyTwoNumbers,
)

result = multiply_tool.invoke({"a": 3, "b": 2})

print(result)
print(multiply_tool.name)
print(multiply_tool.description)