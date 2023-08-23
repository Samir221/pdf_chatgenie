import numpy as np
import threading
import time
from azure.storage.blob import BlobServiceClient, ContentSettings
import os
import hashlib
from transformers import AutoModel, AutoTokenizer
import PyPDF2
from io import BytesIO
from langchain import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_env_variables(env_file_path=None):
    """Load environment variables from a file."""
    if env_file_path is None:
        # Default to the .env file in the same directory as this script
        env_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    
    with open(env_file_path, "r") as f:
        for line in f.readlines():
            line = line.strip()  # Remove newline characters and spaces
            if line and not line.startswith("#"):  # Ignore comments
                key, value = line.split("=", 1)
                os.environ[key] = value.strip('"')  # Remove quotes if present

# Load the environment variables
load_env_variables()

# Global dictionary to track uploaded files and their timestamps
uploaded_files = {}

SESSION_TIMEOUT = 3600  # File lifetime in seconds, e.g., 1 hour

class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata if metadata else {}

class FilePreparer:
    def __init__(self):
        self.accepted_extensions = [".pdf"]
        # Start the cleanup thread when an instance of FilePreparer is created
        self.cleanup_thread = threading.Thread(target=self.cleanup_files, daemon=True)
        self.cleanup_thread.start()

    def get_content_type(self, filename):
        if filename.endswith(".pdf"):
            return "application/pdf"
        elif filename.endswith(".doc"):
            return "application/msword"
        else:
            return "application/octet-stream"

    def upload_to_azure(self, file):
        connection_string = os.environ.get("AZURE_CONNECTION_STRING")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client("userfiles")

        if file.size > 50 * 1024 * 1024:
            raise ValueError("File size exceeds 50 MB limit.")

        file_bytes = file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        blob_name = f"{file.name}"

        content_settings = ContentSettings(content_type=self.get_content_type(file.name))
        container_client.upload_blob(name=blob_name, data=file_bytes, content_settings=content_settings, metadata={'file_hash': file_hash})
        
        # Track the upload timestamp for cleanup
        uploaded_files[blob_name] = time.time()

        return blob_name

    def fetch_from_azure(self, blob_name):
        connection_string = os.environ.get("AZURE_CONNECTION_STRING")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client("userfiles")
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob()
        return blob_data.readall()

    def convert_PDFfile_to_text(self, blob_name):
        binary_content = self.fetch_from_azure(blob_name)
        
        if blob_name.endswith('.pdf'):
            with BytesIO(binary_content) as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text_content = ""
                for page in reader.pages:
                    text_content += page.extract_text()
        else:
            text_content = binary_content.decode("utf-8")

        return text_content

    def split_the_text(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
        documents = [Document(text)]
        split_up_texts = text_splitter.split_documents(documents)
        return split_up_texts

    def vectorize_text(self, split_up_texts, apikey):
        tokenizer = OpenAIEmbeddings(openai_api_key=apikey)
        docsearch = FAISS.from_documents(split_up_texts, tokenizer)        
        return docsearch

    def delete_from_azure(self, blob_name):
        connection_string = os.environ.get("AZURE_CONNECTION_STRING")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client("userfiles")
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.delete_blob()

    def cleanup_files(self):
        while True:
            time.sleep(60)  # Check every minute
            current_time = time.time()
            
            files_to_delete = []
            
            for filename, timestamp in uploaded_files.items():
                if current_time - timestamp > SESSION_TIMEOUT:
                    files_to_delete.append(filename)
            
            for filename in files_to_delete:
                try:
                    self.delete_from_azure(filename)
                    del uploaded_files[filename]
                except Exception as e:
                    print(f"Error deleting {filename}: {e}")

class ChatBot:
    def __init__(self, apikey):
        self.LLM = OpenAI(temperature=0, openai_api_key=apikey)

    def initialize_retrieval_qa(self, docsearch):
        qa = RetrievalQA.from_chain_type(llm=self.LLM, chain_type="stuff", retriever=docsearch.as_retriever())
        return qa
