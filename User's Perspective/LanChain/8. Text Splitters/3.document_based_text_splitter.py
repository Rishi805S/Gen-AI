from langchain_text_splitters import MarkdownTextSplitter

text = """
# 🚀 The Rise of Artificial Intelligence

Artificial Intelligence (AI) is transforming industries across the globe. From healthcare to finance, AI is enabling smarter decision-making and automation.

---

## 🧠 What is AI?

AI refers to the simulation of human intelligence in machines that are programmed to think and learn.

### Key Concepts

- Machine Learning (ML)
- Deep Learning (DL)
- Natural Language Processing (NLP)
- Computer Vision

---

## 📚 Types of AI

### 1. Narrow AI
AI systems designed to perform a specific task.

Example:
- Voice assistants like Siri and Alexa
- Recommendation systems

### 2. General AI
A theoretical form of AI that can perform any intellectual task that a human can do.

---

## 🏗️ Applications of AI

### Healthcare
AI is used for:
- Disease diagnosis
- Drug discovery
- Personalized treatment

### Finance
- Fraud detection
- Algorithmic trading
- Risk assessment

---

## ⚙️ Sample Code

Here is a simple Python example using AI:

```python
def predict(data):
    model = load_model("model.pkl")
    return model.predict(data)
"""

splitter = MarkdownTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

result = splitter.split_text(text)

print(result[1])