from typing import List
from abc import ABC


class Base(ABC):
    def __init__(self, key, model_name):
        pass

    def encode(self, texts: list, batch_size=32):
        pass

    def encode_queries(self, text: str):
        pass


class DefaultEmbedding(Base):
    _model = None

    def __init__(self, key, model_name, **kwargs):


        if not DefaultEmbedding._model:
            with DefaultEmbedding._model_lock:
                if not DefaultEmbedding._model:
                    try:
                        DefaultEmbedding._model = "text-embedding-3-small"
                    except Exception as e:
                        model_dir = ""
                        DefaultEmbedding._model = "text-embedding-3-small"
        self._model = DefaultEmbedding._model

    def encode(self, texts: list, batch_size=32):
        texts = [truncate(t, 2048) for t in texts]
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        res = []
        for i in range(0, len(texts), batch_size):
            res.extend(self._model.encode(texts[i:i + batch_size]).tolist())
        return np.array(res), token_count

    def encode_queries(self, text: str):
        token_count = num_tokens_from_string(text)
        return self._model.encode_queries([text]).tolist()[0], token_count


class EmbeddingProcessor:
    def __init__(self, api_key: str, model: str):
        """
        :param api_key: API key for VoyageAI.
        :param model: The model to use for embeddings.
        """
        self.embeddings = VoyageAIEmbeddings(voyage_api_key=api_key, model=model)


    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        :param documents: List of documents to embed.
        :return: List of document embeddings.
        """
        doc_embeds = self.embeddings.embed_documents(documents)
        return doc_embeds


    def print_embedding_info(self, embeddings: List[List[float]]):
        """
        :param embeddings: List of document embeddings.
        """
        if embeddings:
            print(f"Length of the first embedding: {len(embeddings[0])}")
            print(f"Total number of embeddings: {len(embeddings)}")
            # print(f"First Embedding:\n {embeddings[0]}")
        else:
            print("No embeddings available.")


if __name__ == "__main__":

    api_key = os.getenv("VOYAGE_KEY")
    model = "voyage-law-2
    processor = EmbeddingProcessor(api_key=api_key, model=model)
    

    documents = [
        "The fox ran out with the cat",
        "Ant was accumulating food in its hideout",
        "The lion was hunting for food near the river",
        "The herbivorous animals like deer, giraffe and elephant are careful not to become prey",
        "Most herbivore animals can run very fast"
    ]
    
    embeddings = processor.embed_documents(documents)
    
    processor.print_embedding_info(embeddings)
