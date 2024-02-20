from transformers import CrossEncoder
from typing import List, Dict
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder



class ReRanker:

    def __init__(self, model_name="BAAI/bge-reranker-base"):

        self.model_name = model_name
        # self.model = CrossEncoder(model_name)
        self.model_hf = HuggingFaceCrossEncoder(model_name=model_name)
        print(f"ReRanker initialized with model: {model_name}")


    def rerank(self, query: str, documents: List[str]) -> Dict[str, float]:

        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        ranked_docs = dict(zip(documents, scores))
        ranked_docs = dict(sorted(ranked_docs.items(), key=lambda item: item[1], reverse=True))
        return ranked_docs

    def apply_compression(self, retriever):
        compressor = CrossEncoderReranker(model=self.model_hf, top_n=5)
        reranked_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        return reranked_retriever


    def print_ranking(self, ranked_docs: Dict[str, float]):
        """
        Print the re-ranked documents along with their scores.
        
        :param ranked_docs: Dictionary containing documents and their relevance scores.
        """
        for doc, score in ranked_docs.items():
            print(f"Document: {doc}")
            print(f"Relevance Score: {score}\n")



if __name__ == "__main__":
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker = ReRanker(model_name=model_name)
    
    query = ""
    documents = []
    
    ranked_docs = reranker.rerank(query=query, documents=documents)

