

from langchain.chat_models import ChatOllama
model = ChatOllama(model="mistral", base_url = "http://localhost:11434")


def prompt(user_question, knowledge):
    tech_topic = "GenAI"
    role = f"{tech_topic} specialist"
    target_audience_role = f"""business leaders with a lack of {tech_topic} technological skills and knowledge."""
    target_audience_interests = f"the ROI of their digital transformation projects and how {tech_topic} technology can help them to achieve their business goals"
    prompt = f"""You are a {role} bot. Your goal is to help {target_audience_role} . 
    Your target audience cares about {target_audience_interests}.
    
    Answer the following question in a concise and clear manner so that the target audience understand this? Don't use {tech_topic} terminology that the target audience doesn't understand.
    
    Question: {user_question}
    Use the following knowledge to help if relevant: {knowledge}
    If If a question is irrelevant to {tech_topic}, just say "I don't know".
    """ 
    return prompt



response = model.invoke(prompt("Summarize the document in the given context", "my knowledge"))
print(response)