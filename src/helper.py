import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv  # Make sure python-dotenv is installed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import google.generativeai as palm

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks



def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


# def get_conversational_chain(vector_store):
#     llm=GooglePalm()
#     memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
#     return conversation_chain




def get_conversational_chain(vector_store):
    # Set your API key as an environment variable or replace with actual key
    palm.configure(api_key=GOOGLE_API_KEY)

    # Get the model instance
    models = list(palm.list_models())  # Convert generator to list
    chat_models = [m for m in models if "chat" in m.name.lower()]  
    if not chat_models:
        raise ValueError("No suitable chat model found")
    model = palm.get_model(chat_models[0].name)

    # Initialize conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create the conversational chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vector_store.as_retriever(),
        memory=memory,
        verbose=True  
    )

    return conversation_chain

