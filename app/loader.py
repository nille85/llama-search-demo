from llama_index.readers.notion import NotionPageReader

from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader
from infrastructure import get_notion_reader, get_configuration, get_vector_store
from storer import Storer
from app.infrastructure import get_vector_database, get_configuration, get_embed_model, get_llm_model, get_notion_reader, get_vector_store



#Read configuration
config = get_configuration("dev_config.toml")

#Setup integrations
embed_model = get_embed_model()
vector_database = get_vector_database(config["qdrant"]["url"])
vector_store = get_vector_store(vector_database, "genai")

storer = Storer(embed_model)


#notion_reader : NotionPageReader = get_notion_reader(config["notion"]["api_key"])
#directory_reader = SimpleDirectoryReader(input_dir="./files")
pdf_reader = PDFReader()

#Load documents
pdf_documents = pdf_reader.load_data("files/GenAI for software engineering.pdf")
#notion_documents = notion_reader.load_data(database_id="edb01a5d96264617851d70abd329f6c8")


for document in pdf_documents:
    print(f"File_Name: {document.metadata}, Doc Id: {document.get_doc_id()}")

storer.store(vector_store, pdf_documents)