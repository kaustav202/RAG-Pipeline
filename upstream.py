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
# text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=4)
# docs = text_splitter.split_documents(documents)

# docs = text_splitter.create_documents([text])

# final_texts = []

# for i in docs:
#     c = str(i)
#     final_texts.append(c)



def ret_docs(ft, id="raw"):
    doc_list = []

    for text in ft:
        cid = id + "_" + str(uuid4())
        document = Document( page_content=text, metadata={"source": cid})
        doc_list.append(document)
    
    return doc_list



voyage_api_key = os.getenv('VOYAGE_KEY')
embedding_model = "voyage-law-2"
# processor = EmbeddingProcessor(api_key=voyage_api_key, model=embedding_model)


# embeddings = processor.embed_documents(final_texts)


def assign_ids_to_lists(nested_lists):
    result = []
    for index, nested_list in enumerate(nested_lists, start=1):
        serial_id = f"id{index}"
        metadata = {"created_at" : "04-09-2024"}
        result.append((serial_id, nested_list, metadata))
    
    return result


# vectors = assign_ids_to_lists(embeddings)



es_wrapper = ElasticsearchWrapper(index_name='prod_l1')

# es_wrapper.bulk_index_documents_from_dir(final_texts)

# search_results = es_wrapper.search('Brazil')





pinecone_api_key = os.getenv('PINECONE_KEY')

# pinecone_wrapper = PineconeWrapper(api_key=pinecone_api_key)

# pinecone_wrapper.create_index("vec-store", dimension=1024)


pc = Pinecone(api_key=pinecone_api_key)

index = pc.Index("prod-l1")

emb_obj = VoyageAIEmbeddings(voyage_api_key=os.getenv('VOYAGE_KEY'), model="voyage-law-2")


vector_store = PineconeVectorStore(index=index, embedding=emb_obj)


# vec_docs = ret_docs(final_texts)

# uuids = [str(uuid4()) for _ in range(len(vec_docs))]

# vector_store.add_documents(docs)



import boto3

s3_client = boto3.client('s3', 
         aws_access_key_id="",
         aws_secret_access_key= "")


def download_and_process_files(bucket_name, prefix, local_dir):
    """
    Downloads all text files from an S3 bucket and processes them with a custom function.

    :param bucket_name: Name of the S3 bucket.
    :param prefix: Prefix (folder) in the S3 bucket where the files are located.
    :param local_dir: Local directory to save the downloaded files.
    """
    # Ensure local directory exists
    os.makedirs(local_dir, exist_ok=True)

    # List objects within the specified bucket and prefix
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    if 'Contents' not in response:
        print("No files found.")
        return
    
    ctr= 1
    for obj in response['Contents']:
        key = obj['Key']
        
        if key.endswith('.txt'):  # Ensure the file is a text file
            file_name = os.path.basename(key)
            local_path = os.path.join(local_dir, file_name)

            if os.path.isfile(local_path):
                print(f"File '{file_name}' exists in directory '{local_dir}'. Skipping ! ")
                print("count ",ctr)
                ctr = ctr +1
                continue

            s3_client.download_file(bucket_name, key, local_path)
            print(f"Downloaded {file_name} to {local_path}")

            with open(local_path, 'r') as file:
                text = file.read()
                print("\n Size File : ", len(text))

                docs = text_splitter.create_documents([text])
                final_texts = []
                print("Len Docs : ", len(docs))

                for i in docs:
                    c = str(i)
                    final_texts.append(c)

                es_wrapper.bulk_index_documents_from_dir(final_texts)
                
                # id_file = str(file_name)
                # vec_docs = ret_docs(final_texts, id_file)

                # uuids = [str(uuid4()) for _ in range(len(vec_docs))]

                vector_store.add_documents(docs)

            print("count ",ctr)
            ctr = ctr +1





bucket_name = 'my-bucket-123'
prefix = 'source/'
local_dir = './downloaded'


download_and_process_files(bucket_name, prefix, local_dir)
