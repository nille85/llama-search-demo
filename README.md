## Install using Poetry
Dependendencies for this project are managed using Poetry

`poetry install` installs all the required dependencies for this project

`poetry shell` opens up a shell where you can then run your python commands. 

## Configuration
This uses `tomlkit``

Add a file e.g. dev_config.toml with the following content. If you want to integrate with Notion, you can do so by passing your secret in the file.
```toml
[notion]
api_key="NOTION_API_KEY"


[qdrant]
url = "QDRANT_URL"

```

## Vector Store
`docker compose up` will spin up a container which runs the qdrant vector store locally on the configured port 6333.

You can access the qdrant dashboard [here](http://localhost:6333/dashboard)

`docker compose down` stops containers and removes containers, networks created by up

## Loading Documents
There are some examples on how to load:
*  a single PDF into a vector store
* how to load all files from a directory
* how you can load databases and pages from Notion.

`python app\loader.py`

## Querying Document
look into main.py to update your query targeted at a specified collection within the vector store.
run using `python main.py`