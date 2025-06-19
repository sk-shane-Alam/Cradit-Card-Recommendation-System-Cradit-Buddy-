from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from dotenv import load_dotenv
import os
import pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings  # Or HuggingFace
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.helper import load_pdf, text_split
from pinecone import Pinecone as PineconeClient, ServerlessSpec

load_dotenv()


# 1️⃣ Initialize Pinecone client and index
pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"),
                   environment=os.getenv("PINECONE_ENV"))
index_name = "credit-index"

pinecone_index = pc.Index(index_name)

# Load and split documents
data = load_pdf("data/")
text_chunks = text_split(data)

# Use a 1024-dim embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cpu'}
)

# Connect vector store
vectorstore = PineconeVectorStore(
    index=pinecone_index,
    embedding=embedding_model
)
vectorstore.add_documents(text_chunks)

print("Documents indexed to Pinecone!")