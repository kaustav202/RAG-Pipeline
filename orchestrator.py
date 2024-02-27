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
from models.rerank_model import ReRanker
from retriever.ensemble_retriever import HybridRetriever

vec_retriever = VectorRetriever()
es_retriever = ElasticRetriever()

params = {"k": 10, "score_threshold": 0.5}

semantic_retriever = vec_retriever.config_retriever(index="test",search_kwargs=params)
bm25_retriever = es_retriever.config_retriever(index="test", content_field="content")

contextual_compressor = ReRanker()
semantic_retriever_reranked = contextual_compressor.apply_compression(semantic_retriever)

bm25_retriever_reranked = contextual_compressor.apply_compression(bm25_retriever)

hybrid = HybridRetriever(bm25_retriever,semantic_retriever_reranked)

ensemble_retriever = hybrid.create_ensemble_retriever(weights=[0.5,0.5])


# docs = bm25_retriever.invoke("Hi")

# print(docs)


template = """Answer the question in detail and elaborately based only on the following context :


{context}

Question: {question}
"""


prompt = ChatPromptTemplate.from_template(template)



from models.chat_model import AnthropicModel
anthropic_client = AnthropicModel()

llm = anthropic_client.get_llm()


from chain import Chain

chains = Chain(llm=llm, retriever=ensemble_retriever)

base_qna_chain = chains.qna_chain(prompt=prompt)

inference = base_qna_chain.run("input")


qna_sources_chain = chains.qna_sources_chain(prompt=prompt)

inference_s = qna_sources_chain("input")

#Meta class :

# control params

# invoke

# inference chain



