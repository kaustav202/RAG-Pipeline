import os
from typing import Dict
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, connections
from langchain_elasticsearch import ElasticsearchRetriever


class ElasticRetriever:
    def __init__(self):
        self.es_url = os.getenv('ELASTIC_URL')
        self.es_user = os.getenv('ELASTIC_USER')
        self.es_password = os.getenv('ELASTIC_PASSWORD')
        self.client = Elasticsearch(
            [self.es_url],
            basic_auth=(self.es_user, self.es_password)
        )

    def bm25_query(self, search_query: str) -> Dict:
        return {
            "query": {
                "match": {
                    'content': search_query,
                },
            },
        }

    def config_retriever(self, index, content_field):
        bm25_retriever = ElasticsearchRetriever.from_es_params(
        index_name=index,
        body_func=self.bm25_query,
        content_field=content_field,
        url=self.es_url,
        username=self.es_user,
        password=self.es_password
    )
        return bm25_retriever


    def search(self, search_query: str) -> Dict:
        query_body = self.bm25_query(search_query)
        response = self.client.search(
            index=self.index_name,
            body=query_body
        )
        return response



# Initialize the Elasticsearch wrapper
# es_wrapper = ElasticRetriever()

# Perform a search
# search_query = "example search text"
# response = es_wrapper.search(search_query)

# Print or process the response
# print(response)
