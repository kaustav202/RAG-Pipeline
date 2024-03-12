from langchain_pinecone import PineconeVectorStore  
from langchain_voyageai import VoyageAIEmbeddings
from utils.vector_store import PineconeWrapper
from pinecone import Pinecone, ServerlessSpec
import numpy as np

from preprocess.tokenizer import TextProcessor
from models.embedding_model import EmbeddingProcessor
from utils.elastic_store import ElasticsearchWrapper

import os
from dotenv import load_dotenv
load_dotenv()

from uuid import uuid4
from langchain_core.documents import Document




# text_processor = TextProcessor()



def load_file_to_string(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


# file_path = './lex.uz_ru_docs_-6396140_03-09-2024.txt'
# text = load_file_to_string(file_path)


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=700,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
