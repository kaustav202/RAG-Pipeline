# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
import numpy as np
import os
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Dict, Iterable

from retriever.vector_retriever import VectorRetriever
from retriever.bm25_retriever import ElasticRetriever
from modelconf.rerank_model import ReRanker
from retriever.ensemble_retriever import HybridRetriever

vec_retriever = VectorRetriever()
es_retriever = ElasticRetriever()

params = {"k": 10, "score_threshold": 0.5}

semantic_retriever = vec_retriever.config_retriever(index="test",search_kwargs=params)
bm25_retriever = es_retriever.config_retriever(index="test", content_field="content")

contextual_compressor = ReRanker()
semantic_retriever_reranked = contextual_compressor.apply_compression(semantic_retriever)

bm25_retriever_reranked = contextual_compressor.apply_compression(bm25_retriever)
