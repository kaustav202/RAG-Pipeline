from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore  
from langchain_voyageai import VoyageAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import os
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Dict, Iterable

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_elasticsearch import ElasticsearchRetriever


from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


pinecone_api_key = os.getenv('PINECONE_KEY')


# pinecone_wrapper.create_index("vec-store", dimension=1024)


pc = Pinecone(api_key=pinecone_api_key)

index = pc.Index("prod-l2")


emb_obj = VoyageAIEmbeddings(voyage_api_key=os.getenv('VOYAGE_KEY'), model="voyage-law-2")


vector_store = PineconeVectorStore(index=index, embedding=emb_obj)




pinecone_retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 10, "score_threshold": 0.5}
)


       
es_client = Elasticsearch(['localhost:9200'], basic_auth=('elastic', 'changeme'))
 
es_url = "http://localhost:9200"

def bm25_query(search_query: str) -> Dict:
    return {
        "query": {
            "match": {
                'content': search_query,
            },
        },
    }


bm25_retriever = ElasticsearchRetriever.from_es_params(
    index_name="prod_l2",
    body_func=bm25_query,
    content_field='content',
    url=es_url,
    username="elastic",
    password="changeme"
)



from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

compressor = CrossEncoderReranker(model=model, top_n=5)
pinecone_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=pinecone_retriever
)


bm25_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=bm25_retriever
)


ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, pinecone_retriever], weights=[0.7, 0.3]
)




os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_KEY')

from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)


