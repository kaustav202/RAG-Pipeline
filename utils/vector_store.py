from pinecone import Pinecone, ServerlessSpec
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Tuple

class PineconeWrapper:
    def __init__(self, api_key: str = '', environment: str = "us-west1-gcp"):

        pc = Pinecone(
            api_key = api_key
        )
        self.pc = pc


    def create_index(self, index_name: str, dimension: int, metric: str = "cosine"):

        if index_name in self.pc.list_indexes():
            raise ValueError(f"Index '{index_name}' already exists.")
        self.pc.create_index(
        name=index_name, dimension=dimension, metric=metric , spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
        ),
        deletion_protection="disabled"
  )
        self.index = self.pc.Index(index_name)
        print(f"Index '{index_name}' created with dimension {dimension}.")



    def connect_index(self, index_name: str):

        indexes = self.pc.list_indexes()

        is_there = False

        for ind in indexes:
            if index_name == ind.name:
                is_there = True

        if not is_there:
            raise ValueError(f"Index '{index_name}' does not exist.")
        self.index = self.pc.Index(index_name)
        print(f"Connected to index '{index_name}'.")



    def upsert_vectors(self, vectors: List[Tuple[str, List[float]]]):
        """
        :param vectors: List of tuples where each tuple contains an ID and a vector (list of floats).
        """

        if not self.index:
            raise RuntimeError("No index connected. Please connect or create an index first.")
        self.index.upsert(vectors=vectors)
        print(f"Upserted {len(vectors)} vectors.")


if __name__ == "__main__":
    pc = PineconeWrapper()
    pc.create_index("test", 1024)
    pc.connect_index(index_name="test")
    vectors = np.random.randint(0, 10, (100, 1024))
    pc.upsert_vectors(vectors.tolist())
