from app.infrastructure import get_vector_database, get_configuration, get_vector_store_index, get_embed_model, get_llm_model, get_notion_reader, get_vector_store
from app.retriever import Retriever
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
console = Console()


def reply(response):
    md = Markdown(response.response)
    console.print(md)
    table = Table(title=" Sources used")
    table.add_column("Score", style="cyan")
    table.add_column("Metadata", style="magenta")
    for source_node in response.source_nodes:
        table.add_row(str(source_node.score), str(source_node.node.metadata))
    console.print(table)

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
    #We retrieve a document, and summarize it.
    response = retriever.query(index, "Can you summarize the given context? The response must covers the key points of the text in bullet point format and it must not miss anything important. The response must be returned in markdown format.")
    reply(response)

    while True:
        console.print("Starting chat mode")
        user_input = input("Enter your input: ")
        if user_input == "\\bye":
            break
        else:
        # Process the user input here
            response = retriever.chat(index, user_input)
            reply(response)


    

