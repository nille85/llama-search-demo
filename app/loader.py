from llama_index.readers.notion import NotionPageReader

from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader
from infrastructure import get_configuration, get_vector_store, get_vector_database, get_configuration, get_embed_model, get_sbert_embed_model, get_translator
from storer import Storer
from langdetect import detect
import os
from typing import List
from llama_index.core.schema import Document



def load_documents(folder: str):
    #Read configuration
    config = get_configuration("dev_config.toml")

    #Setup integrations
    embed_model = get_sbert_embed_model()
    vector_database = get_vector_database(config["qdrant"]["url"])
    

    
    storer = Storer(embed_model)
    pdf_reader = PDFReader()

    for filename in os.listdir(folder):
        
        if filename.endswith(".pdf"):
            vector_store_name = filename
            vector_store = get_vector_store(vector_database,  vector_store_name)
            #Load documents
            pdf_documents = pdf_reader.load_data(folder + "/" + filename)
            storer.store(vector_store, pdf_documents)

load_documents("files/GenAI")