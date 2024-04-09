import streamlit as st
from app.infrastructure import get_vector_database, get_configuration, get_vector_store_index, get_sbert_embed_model, get_llm_model, get_vector_store
from app.retriever import Retriever
from streamlit_pdf_viewer import pdf_viewer

config = get_configuration("dev_config.toml")
vector_db = get_vector_database(config["qdrant"]["url"])


embed_model = get_sbert_embed_model()



st.set_page_config(layout="wide")
st.title('RAG Prototype')



# Create Prompt
def create_prompt(topic, target_audience_role, target_audience_interests, question, response_format):
    role = f"{topic} specialist"
    prompt = f"""You are a {role} bot. Your goal is to help {target_audience_role}. Your target audience cares about {target_audience_interests}.
    Answer the following question in a concise and clear manner so that the target audience understand this. Don't use {topic} terminology that the target audience doesn't understand.
    Question: {question}
    Answer in {response_format}.
    """
    return prompt




configuration, document  = st.columns([0.3, 0.7])

with document:
    ## Select Document
    document_val = st.selectbox("Select document:", ("paper_3.pdf", "paper_2.pdf", "paper_1.pdf"))

    #specify collection name from vector store, here it is 'genai'
    retriever_vector_store = get_vector_store(vector_db, document_val)
    index = get_vector_store_index(retriever_vector_store,embed_model)
    llm_model = get_llm_model()
    
    st.markdown("## PDF Content")
    

    pdf_viewer(f"files/GenAI/{document_val}",height=800)


with configuration:
    st.markdown("## Bot Configuration")
    st.markdown("### Prompt")
    topic_val = st.text_input("Topic", value="GenAI")
    target_audience_role_val = st.text_input("Target Audience Role", value=f"business leaders with a lack of {topic_val} technological skills and knowledge.")
    target_audience_interests_val = st.text_input("Target Audience Interests", value=f"the ROI of their digital transformation projects and how {topic_val} technology can help them to achieve their business goals")
    response_format_val = st.selectbox("Select format:", ("plain text", "bullet point fomat"))
    st.markdown("### Search")
    top_k_val = st.slider('Top K', 2, 20)
    similarity_cut_off_val = st.slider('Similarity Cut Off', 0.0, 1.0, 0.5)
    retriever = Retriever(llm_model,top_k_val, similarity_cut_off_val)
    st.markdown("### Synthesis")

st.markdown("## Ask a Question")
with st.form("question"):
    question_val = st.text_input("Question", value="Summarize the given context. Don't leave out anything important")
    submitted = st.form_submit_button("Submit")

if submitted:
    prompt = create_prompt(topic_val, target_audience_role_val, target_audience_interests_val, question_val, response_format_val)
    st.markdown(f"## Prompt Used")
    st.write(f"{prompt}")
    response = retriever.query(index, prompt)
    st.markdown('# Result')
    st.markdown(f':green[{response.response}]')
    
    st.markdown('# Sources used')
    for source_node in response.source_nodes:

        col1, col2 = st.columns([0.2, 0.8])
        with col1:
            st.markdown(f'**Score:** {source_node.score}')
            st.markdown(f'**Filename:** {source_node.node.metadata["file_name"]}')
            st.markdown(f'**Page:** {source_node.node.metadata["page_label"]}')

        with col2:
            st.markdown(f'{source_node.node.text}')
        st.divider()

st.divider()



st.header('Debug Area')
"st.session_state object:", st.session_state