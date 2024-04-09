from app.infrastructure import get_vector_database, get_configuration, get_vector_store_index, get_sbert_embed_model, get_embed_model, get_llm_model, get_notion_reader, get_vector_store, get_translator
from app.retriever import Retriever
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
console = Console()

def query(retriever, index, prompt):
        #translated_prompt = translator.translate(prompt, "nld_Latn", "eng_Latn")
        response = retriever.query(index, prompt)
        #response.response= translator.translate(response.response, "eng_Latn", "nld_Latn")
        return response

def chat(retriever, index, prompt):
        #translated_prompt = translator.translate(prompt, "nld_Latn", "eng_Latn")
        response = retriever.chat(index, prompt)
        #response.response = translator.translate(response.response, "eng_Latn", "nld_Latn")
        return response

def print_query(prompt):
    console.print(prompt, style="green") 

def print_response(response):
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
    translator = get_translator()
    
    embed_model = get_sbert_embed_model()
    
    #specify collection name from vector store, here it is 'genai'
    retriever_vector_store = get_vector_store(vector_db, "genai_topic")


    index = get_vector_store_index(retriever_vector_store,embed_model)
    
    llm_model = get_llm_model()
    retriever = Retriever(llm_model)
    #We retrieve a document, and summarize it.
    #prompt = """Kan je de huidige context samenvatten? Het antwoord moet de belangrijkste punten bevatten. Het antwoord mag niets belangrijks missen."""
    topic = "GenAI"
    role = f"{topic} specialist"
    specialization = f"distilling large documents about {topic} and explaining them in plain language without the use of technical jargon "
    target_audience_role = f"business leaders with a lack of {topic} technical skills and knowledge"
    background = f"""With a deep understanding of the {topic} landscape and the ability to quickly absorb complex information, you excel at boiling down large documents into actionable insights.
    You leverage your knowledge of {topic}, research expertise, and communication skills to create summaries that are both informative and engaging."""
    goal = f"help business leaders make informed decisions about their {topic} initiatives by providing them with a clear understanding of the topic and its potential impact on their organization."
    prompt = f"""
    As a {role}, you are an expert in {specialization}. Your target audience are {target_audience_role}.
    {background}
    Your goal is to {goal}.
    Summarize the paper titled "Attention is All you need?".
    """
    print_query(prompt)
    response = query(retriever, index, prompt)
    print_response(response)
   
    
    """ 
    while True:
        user_input = input("Stel je vraag: ")
       
        if user_input == "\\bye":
            break
        else:
            response = chat(retriever, index, user_input)
            print_response(response)
 """
    


