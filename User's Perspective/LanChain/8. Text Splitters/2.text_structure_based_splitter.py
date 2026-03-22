from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Python learning roadmap
Hey what is the roadmap to learn Python completely in order to get started with Machine 
Learning and what are all the topics I need to cover in order to actually keep Python in my 
resume and get a conﬁdence that "I completed or Know Python"
Excellent question, Rishi — this is the right mindset.
If your goal is to learn Python completely enough to conﬁdently add it to your resume and to move
toward Machine Learning (ML), then you need a structured roadmap that takes you from Python
basics →  intermediate →  advanced →  libraries →  ML foundation.
Let’s break it down into 5 clear stages so that by the end, you’ll actually feel conﬁdent saying: “I know
Python.”
Stage 1: Python Core Fundamentals (Foundation)
Goal: Be comfortable writing clean, correct Python code.
Topics to Learn
Python Basics
Installing Python, using IDLE / VS Code / Jupyter Notebook
Input/output, variables, data types (int, float, str, bool)
Operators: arithmetic, comparison, logical, bitwise
Control Flow
if, elif, else
for, while, break, continue
Functions
Deﬁning and calling functions
Default parameters, return statements
Lambda functions
Data Structures
Lists, Tuples, Sets, Dictionaries — CRUD operations, nesting, comprehensions
String manipulation (slicing, methods)
Loops + Comprehensions
List / Set / Dict Comprehensions (important for ML data preprocessing)
Mini Projects
Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API. 1/4
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", " ", ""]
)

result = splitter.split_text(text)

print(result)