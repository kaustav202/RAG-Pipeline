from langchain.retrievers import EnsembleRetriever
from typing import List, Dict, Union

class HybridRetriever:
    def __init__(self, bm25_retriever, semantic_retriever):
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
    
    def create_ensemble_retriever(self, weights: List[float] = [0, 1]) -> EnsembleRetriever:
        """
        Create an EnsembleRetriever with the given retrievers and weights.
        
        :param weights: List of weights for the retrievers. Must match the number of retrievers.
        :return: An instance of EnsembleRetriever.
        """
        if len(weights) != 2:
            raise ValueError("Weights list must match the number of retrievers.")

        if sum(weights) != 1:
            raise ValueError("Weights are not complementary.")
        
        return EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.semantic_retriever],
            weights=weights
        )




# retriever_manager = RetrieverManager(bm25_retriever=bm25_retriever, semantic_retriever_reranked=semantic_retriever_reranked)
# ensemble_retriever = retriever_manager.create_ensemble_retriever(weights=[0, 1])
# print(ensemble_retriever)
