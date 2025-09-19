from langchain_openai import ChatOpenAI
#from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
import os
from chroma_utils import vectorstore
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Now, the OpenAI library will automatically pick up OPENAI_API_KEY
# You can also access it explicitly if needed:
# api_key = os.getenv("OPENAI_API_KEY")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

output_parser = StrOutputParser()



contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def get_rag_chain(model="llama-3.1-8b-instant"):
    # Replace "YOUR_GROQ_API_KEY" with your actual Groq API key
    #groq_api_key = "YOUR_GROQ_API_KEY"

    # Initialize ChatGroq by passing the API key directly
    llm = ChatOpenAI(model=model)
    ##llm = ChatGroq(api_key=groq_api_key, model_name=model)
    #llm = ChatGroq(model=model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)    
    return rag_chain