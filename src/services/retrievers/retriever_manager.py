from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from langchain.docstore.document import Document

from services.embeddings.embedding_manager import EmbeddingManager

# embedding = EmbeddingManager()
sample_contents = ['바퀴에 구멍이 나면 지게차 바퀴 교체가 필요합니다.', 
                   '지게차에서 탄 냄새가 나면 엔진 오일을 갈아야 합니다.', 
                   '브레이크가 동작하지 않아서 브레이크 패드를 교체해야 합니다.', 
                   '필터에 먼지가 보여 연료 필터를 청소했습니다.']

def create_retrives(documents: list = []):
    docs = []
    for doc in documents:
        docs.append(Document(page_content=doc))

    # vectorstore = FAISS.from_documents(docs, embedding=embedding)
    vectorstore = BM25Retriever.from_documents(docs)
    # retriever = vectorstore.as_retriever()

    return vectorstore


class RetrieverManager:
    retrievers = {}
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.retrievers['manual'] = create_retrives(sample_contents)