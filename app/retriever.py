class Retriever:

    def __init__(self, llm):
        self.llm = llm

    def query(self, index, query):
        query_engine = index.as_query_engine(llm=self.llm)
        response = query_engine.query(query)
        return response
