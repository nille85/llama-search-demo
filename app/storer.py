from llama_index.core import  VectorStoreIndex
from llama_index.core import StorageContext

class Storer:

    def __init__(self, embed_model):
        self.embed_model = embed_model


    def store(self, vector_store, documents):
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        return index
   
