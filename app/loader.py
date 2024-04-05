from llama_index.readers.notion import NotionPageReader

from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader
from infrastructure import get_configuration, get_vector_store, get_vector_database, get_configuration, get_embed_model, get_sbert_embed_model, get_translator
from storer import Storer
from langdetect import detect





def load_documents(vector_store):
    #Read configuration
    config = get_configuration("dev_config.toml")

    #Setup integrations
    embed_model = get_sbert_embed_model()
    vector_database = get_vector_database(config["qdrant"]["url"])
    vector_store = get_vector_store(vector_database,  vector_store)

    
    storer = Storer(embed_model)
    pdf_reader = PDFReader()

    #Load documents
    pdf_documents = pdf_reader.load_data("files/GenAI.pdf")


    for document in pdf_documents:
        print(f"File_Name: {document.metadata}, Doc Id: {document.get_doc_id()}")
        try:
            lang = detect(document.get_text())
            print(f"Language: {lang}")
        except:
            print("language could not be detected")
   
    

    storer.store(vector_store, pdf_documents)

load_documents("genai")