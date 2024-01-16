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



