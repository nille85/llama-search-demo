from app.infrastructure import get_vector_database, get_configuration, get_vector_store_index, get_embed_model, get_llm_model, get_notion_reader, get_vector_store
from app.retriever import Retriever
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)




if __name__ == "__main__":
    config = get_configuration("dev_config.toml")
    vector_db = get_vector_database(config["qdrant"]["url"])
    notion_reader = get_notion_reader(config["notion"]["api_key"])
    embed_model = get_embed_model()
    #specify collection name from vector store, here it is 'genai'
    retriever_vector_store = get_vector_store(vector_db, "genai")
    index = get_vector_store_index(retriever_vector_store,embed_model)
    llm_model = get_llm_model()

    retriever = Retriever(llm_model)
    response = retriever.query(index, "Could you summarize the given context? Return your response which covers the key points of the text and does not miss anything important, please.")

    print(response)

    

