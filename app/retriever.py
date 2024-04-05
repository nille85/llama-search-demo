from llama_index.core import  VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers.type import ResponseMode


class Retriever:

    def __init__(self, llm):
        self.llm = llm

    def query(self, index: VectorStoreIndex, query):

        # configure retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5
        )

        # configure response synthesizer
        response_synthesizer = get_response_synthesizer(llm=self.llm)

   
        # assemble query engine
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            response_mode=ResponseMode.SIMPLE_SUMMARIZE,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
            llm=self.llm
        )        
       
        response = query_engine.query(query)
        return response

    def chat(self, index: VectorStoreIndex, query):
        chat_engine = index.as_chat_engine(llm=self.llm, chat_mode=ChatMode.CONDENSE_QUESTION)
        response = chat_engine.chat(query)
        return response

    
