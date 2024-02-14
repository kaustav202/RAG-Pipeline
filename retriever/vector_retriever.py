import os
from typing import Dict
from langchain_pinecone import PineconeVectorStore  
from langchain_voyageai import VoyageAIEmbeddings
from pinecone import Pinecone, ServerlessSpec


class VectorRetriever:

    def __init__(self):
        self.pinecone_api_key = os.getenv('PINECONE_KEY')
        self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)
        self.voyage_api_key = os.getenv('VOYAGE_KEY')
        self.embeddings = VoyageAIEmbeddings(voyage_api_key=self.voyage_api_key, model=os.getenv('EMBEDDING_MODEL'))
    
    def config_retriever(self, index, search_type="similarity_score_threshold", search_kwargs=None):
        self.index = self.pinecone_client.Index(index)
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)

        if search_kwargs is None:
            search_kwargs = {"k": 10, "score_threshold": 0.5}

        vector_retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        return vector_retriever



# vector_store_wrapper = VectorStoreWrapper()

# retriever = vector_store_wrapper.create_retriever()


