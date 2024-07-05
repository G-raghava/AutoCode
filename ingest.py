#!/usr/bin/env python3
import os
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
 
#from langchain.document_loaders import (
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    PythonLoader,
    DirectoryLoader,
)
 
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS

KEY = ''
# Load environment variables
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-MiniLM-L6-v2')
chunk_size = 600
chunk_overlap = 50
 
#source_documents
if KEY=='events':
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'events_source_documents')
    persist_directory = os.environ.get('PERSIST_DIRECTORY', 'db_events')
elif KEY=='host':
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'host_source_documents')
    persist_directory = os.environ.get('PERSIST_DIRECTORY', 'db_host')
elif KEY=='show':
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'show_commands_source_documents')
    persist_directory = os.environ.get('PERSIST_DIRECTORY', 'db_show')
elif KEY=='config':
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'config_source_documents')
    persist_directory = os.environ.get('PERSIST_DIRECTORY', 'db_config')
elif KEY == 'default':
    source_directory = os.environ.get('SOURCE_DIRECTORY', '/Users/rohanbadiger/Desktop/GenAi/FT_ST_LIB_FILES_ALL')
    persist_directory = os.environ.get('PERSIST_DIRECTORY', 'db_all_ft_st_lib_files')
else:
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
    persist_directory = os.environ.get('PERSIST_DIRECTORY', 'db_vsc')
 
# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""
 
    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e
 
        return doc
 
# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".py": (PythonLoader, {}),
    # Add more mappings for other file extensions and loaders as needed
}
 
 
def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()
 
    raise ValueError(f"Unsupported file extension '{ext}'")
 
 
def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
 
    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()
 
    return results
 
 
def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    # documents = load_documents(source_directory, ignored_files)
    source_documents_path = r'C:\Users\gudiwada\Desktop\Working_Model\source_documents'
    loader = DirectoryLoader(source_documents_path, glob="**/*.py", use_multithreading=True, loader_cls=PythonLoader)
    documents = loader.load()
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
 
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
 
    """
    print(f"Inside func...")
    for i, text in enumerate(texts):
        if i<4:
            print(type(text))
            print(f"Text {i+1}: {text}\n")
    print(f"Documents...")
    print(type(documents))
    for i, document in enumerate(documents):
        if i<4:
            print(type(document))
            print(f"Document {i+1}: {document}\n")
        #print(document)
    """
 
    def chunk_csv_rows(file_path):
        df = pd.read_csv(file_path)
        chunks = df.to_dict(orient='records')
        return chunks
    
    doccs = []
    for filename in os.listdir(source_directory):
        if os.path.isfile(os.path.join(source_directory, filename)) and filename.endswith(".csv"):
            # print(filename)
            file_path = source_directory+"/"+filename
            # print(file_path)
            chunks = chunk_csv_rows(file_path)
 
            for i, chunk in enumerate(chunks):
                page_content_str = str(chunk)
                # Create a new Document object and assign values
                docc = Document(
                    page_content=page_content_str,
                    metadata={'source': file_path, 'row': 3, **chunk}
                )
                doccs.append(docc)
                if i<4:
                    # print(type(docc))
                    print(f"Chunk {i+1}: {docc}\n")
 
    return texts
    # return doccs
 
def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False
 
def main():
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
 
    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        #print(texts)
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    db.persist()
    db = None
 
    print(f"Ingestion complete! You can now run AutoCodeGenerator.py to query your documents")
 
 
if __name__ == "__main__":
    main()