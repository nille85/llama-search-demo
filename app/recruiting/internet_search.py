import requests
import json
import tomlkit
def get_configuration(config_file_path: str):
    with open(config_file_path, "r") as file:
            return tomlkit.load(file)

config = get_configuration("dev_config.toml")
print(config["serper"]["api_key"])

def search_internet(query):
    """Useful to search the internet
    about a a given topic and return relevant results"""
    top_result_to_return = 4
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': config["serper"]["api_key"],
        'content-type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    # check if there is an organic key
    if 'organic' not in response.json():
      return "Sorry, I couldn't find anything about that, there could be an error with you serper api key."
    else:
      print(response.json())
      results = response.json()['organic']
      string = []
      for result in results[:top_result_to_return]:
        try:
          string.append('\n'.join([
              f"Title: {result['title']}", f"Link: {result['link']}",
              f"Snippet: {result['snippet']}", "\n-----------------"
          ]))
        except KeyError:
          next

      return '\n'.join(string)


response = search_internet("GenAI specialist role")
print(response)