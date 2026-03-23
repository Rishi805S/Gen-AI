from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

model = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token="😹",
    temperature=0.7,
    max_new_tokens=100
)
llm = ChatHuggingFace(llm=model)

embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")


video_id = "xEmrFePGjEg"

transcript = ""

try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id) # This returns Object
    raw_data = transcript_list.to_raw_data() # This converts objects into raw data that is Json data
    transcript = " ".join(chunk['text'] for chunk in raw_data) # Remove all durations and get only content

except TranscriptsDisabled:
    print("No Captions available")

splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 60)

chunks = splitter.split_text(transcript)

docs = [Document(page_content=chunk) for chunk in chunks]

vector_store = Chroma.from_documents(embedding=embeddings, documents=docs) # Stores Documents, Meta data, unique id for docs

retriever = vector_store.as_retriever(type="similarity", kwargs={"k": 4})


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an AI assistant..
    You are given the following extracted parts of a long document and a question. Provide a conversational answer with respect to question .
    Context : {context}
    Question : {question}
    """
)

def format_docs(retriever):
    return "\n\n".join(doc.page_content for doc in retriever)

question = "What is Data Mining?"

retrieved_docs = retriever.invoke(question)

final_prompt = prompt.invoke({"context": format_docs(retrieved_docs), "question": question})

answer = llm.invoke(final_prompt)

print(answer.content)


parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough(),
})

main_chain = parallel_chain | prompt | llm | StrOutputParser()

result = main_chain.invoke(question)

print(result)
