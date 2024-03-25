from llama_index.core import  VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode
class Retriever:

    def __init__(self, llm):
        self.llm = llm

    def query(self, index: VectorStoreIndex, query):
        query_engine = index.as_query_engine(llm=self.llm)
        response = query_engine.query(query)
        return response

    def chat(self, index: VectorStoreIndex, query):
        chat_engine = index.as_chat_engine(llm=self.llm, chat_mode=ChatMode.CONDENSE_QUESTION)
        response = chat_engine.chat(query)
        return response

    
