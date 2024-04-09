def create_headhunter(market="", specialization=""):
  """
  This function creates a Headhunter description with the given market and specialization.

  Args:
    market: The market/industry (optional, e.g., Healthcare, Manufacturing).
    specialization: The tech specialization (e.g., Cybersecurity, Applied Generative AI).

  Returns:
    A dictionary containing the role, goal, and backstory of the Headhunter.
  """

  role = f"Headhunter ({specialization})"

  if market:
    goal = (f"to connect companies seeking top {specialization} talent within the {market} "
            f"industry with the perfect fit by understanding both the company's specific needs "
            f"and the candidate's unique strengths and career aspirations.")
    backstory = (f"you are a master of matchmaking within the professional world, "
                f"specializing in the {specialization} domain {'' if not market else f'within the {market} industry'}. "
                f"With a keen eye for talent and a deep understanding of the {'' if not market else market} "
                f"landscape and the intricacies of {specialization}, you"
                f"thrive on building connections between companies seeking "
                f"top {specialization} performers {'' if not market else f'in {market}'}"
                f" and individuals seeking their ideal career fit within the {'' if not market else market} "
                f"{specialization} space. "
                f"You leverage your extensive network, research expertise, and honed "
                f"communication skills to navigate the recruitment landscape, ensuring a "
                f"win-win situation for both companies and candidates.")
    task = f"Recommend new top {specialization} talent to join the team in the {market} industry"
  else:
    goal = (f"to connect companies seeking top {specialization} talent with the perfect fit "
            f"by understanding both the company's specific needs and the candidate's unique "
            f"strengths and career aspirations.")
    backstory = (f"you are a master of matchmaking within the professional world, "
                f"specializing in the {specialization} domain. "
                f"With a keen eye for talent and a deep understanding of the "
                f"ever-evolving landscape and the intricacies of {specialization}, you "
                f"thrive on building connections between companies seeking "
                f"top {specialization} performers and individuals seeking their ideal career "
                f"fit within the {specialization} space. "
                f"You leverage their extensive network, research expertise, and honed "
                f"communication skills to navigate the recruitment landscape, ensuring a "
                f"win-win situation for both companies and candidates.")
    task = f"Recommend new top {specialization} talent to join the team"

  return {"role": role, "goal": goal, "backstory": backstory, "task": task}

# Example usage with market
market = "Manufacturing"
specialization = "GenAI"

headhunter = create_headhunter(market, specialization)




# Use Ollama to create an agent

background = f""" As a{headhunter["role"]}, {headhunter["backstory"]}. Your goal is to {headhunter["goal"]}. Your task is to {headhunter["task"]}."""


from langchain.chat_models import ChatOllama
model = ChatOllama(model="mistral", base_url = "http://localhost:11434")
from langchain.prompts import ChatPromptTemplate
""" prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                background,
            ),
            ("human", "{question}"),
        ]
        ) """

topic = "GenAI"
task = "Summarize large documents"
target_audience = "business leaders without advanced technical background"
question= f"What is a great description for someone who is great at {task} about the topic {topic} in a concise and clear manner so that {target_audience} understand this?"
#question= f"Provide a description for an individual who excells at distilling the main topics from an article about {topic} so that the article can later be retrieved based on these distilled topics?"
expected_output = """
Use the format of the example outputs below. The final result MUST contain a role, a goal, and backstory.

Example Output:
As a Customer Champion, your goal is to build strong, lasting relationships with customers and ensure their complete satisfaction throughout their journey.
You are a champion for the customer experience. With exceptional interpersonal skills and a genuine interest in customer needs, you thrive on building trust and fostering meaningful connections. 
You leverage your knowledge of the customer journey, communication expertise, and problem-solving skills to proactively address customer needs, exceed expectations, and cultivate lasting customer loyalty.

Example Output:
As an Innovation Architect, your goal is to design and implement strategies that foster a culture of creativity and lead to the development of groundbreaking solutions.
You are a visionary architect that builds bridges between the present and the future. 
You possses a unique blend of strategic thinking and creative problem-solving skills. You thrive on challenging the status quo and exploring new possibilities. 
You leverage your understanding of emerging trends, design thinking methodologies, and innovation management frameworks to foster an environment conducive to creativity and develop innovative solutions that address complex challenges and unlock new opportunities.
"""
prompt = f"""{background}

Question: {question} 

{expected_output}
"""

print(prompt)
print("--------------")
response = model.invoke(prompt)
print(response)