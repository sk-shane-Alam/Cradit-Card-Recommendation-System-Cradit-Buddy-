from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents


def text_split(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=300)
    text_chunks = text_splitter.split_documents(data)
    return text_chunks




