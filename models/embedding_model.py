from typing import List
# from  rag-core/utils import num_tokens_from_string, truncate
from abc import ABC


class Base(ABC):
    def __init__(self, key, model_name):
        pass

    def encode(self, texts: list, batch_size=32):
        raise NotImplementedError("Please implement encode method!")

    def encode_queries(self, text: str):
        raise NotImplementedError("Please implement encode method!")


class DefaultEmbedding(Base):
    _model = None

    def __init__(self, key, model_name, **kwargs):


        if not DefaultEmbedding._model:
            with DefaultEmbedding._model_lock:
                if not DefaultEmbedding._model:
                    try:
                        DefaultEmbedding._model = ""
                    except Exception as e:
                        model_dir = ""
                        DefaultEmbedding._model = ""
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

