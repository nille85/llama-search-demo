import qdrant_client
import tomlkit
from llama_index.core import  VectorStoreIndex
from llama_index.readers.notion import NotionPageReader
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from app.translator import Translator


def get_translator():
    pretrained_model = "facebook/nllb-200-distilled-600M"
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    return Translator(model, tokenizer)

def get_configuration(config_file_path: str):
    with open(config_file_path, "r") as file:
            return tomlkit.load(file)


def get_embed_model():
    return FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")

def get_sbert_embed_model():
    return HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    

def get_llm_model():
    return Ollama(model="mistral", request_timeout=120.0)

def get_vector_database(url):
    db =  qdrant_client.QdrantClient(url)
    return db

def get_notion_reader(integration_token): 
    return NotionPageReader(integration_token)

def get_vector_store(db, collection_name):
   return QdrantVectorStore(client=db, collection_name=collection_name)

def get_vector_store_index(vector_store, embed_model):
     storage_context = StorageContext.from_defaults(vector_store=vector_store)
     index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model
     )
     return index

