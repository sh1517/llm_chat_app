from langchain_community.embeddings import HuggingFaceEmbeddings

def create_embedding(embedding_model, model_name: str):
    embeddings = embedding_model(model_name=model_name)
    return embeddings


class EmbeddingManager:
    embeddings = {}
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if 'BAAI/bge-m3' not in EmbeddingManager.embeddings:
            EmbeddingManager.embeddings['BAAI/bge-m3'] = create_embedding(HuggingFaceEmbeddings, 'BAAI/bge-m3')

    def get_embeddings(self, model_name: str):
        return self.embeddings[model_name]
