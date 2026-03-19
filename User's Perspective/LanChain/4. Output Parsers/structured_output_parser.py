from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token="😹",
    temperature=0.7,
    max_new_tokens=100
)
model = ChatHuggingFace(llm=llm)


# Not working in current langchain version

# schema = [
#     ResponseSchema(name = 'fact1',description = ' Fact1 about the topic')
#     ResponseSchema(name = 'fact1',description = ' Fact1 about the topic')
#     ResponseSchema(name = 'fact1',description = ' Fact1 about the topic')
# ]

# parser = StructuredOutputParser.from_response_schemas(schema)

# template = PromptTemplate(
#     template="Give 3 points about topic {topic}. \n {format_instruction}",
#     input_variables=['topic'],
#     partial_variables={'format_instruction': parser.get_format_instructions()}
# )

# chain = template | model | parser

# result = chain.invoke({'topic': 'Black Hole'})

# final_results = parser.parse(result.content)

# print(final_results)