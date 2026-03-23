from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field


class MySchema(BaseModel):
    a: int = Field(description="The first number to multiply")
    b: int = Field(description="The second number to multiply")


class MultiplyTool(BaseTool):
    name: str = "Multiply Two numbers"
    description: str = "Multiplies two numbers"
    args_schema: ArgsSchema | None = MySchema

    def _run(self, a: int, b: int) -> int:
        return a * b
    
multiply_tool = MultiplyTool()

result = multiply_tool.invoke({"a": 3, "b": 2}) 

print(result)
