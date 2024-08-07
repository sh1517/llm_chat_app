from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import TextLoader

from langchain.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter

from services.embeddings.embedding_manager import EmbeddingManager

# loader = TextLoader("../../databases/toyota_manual.txt", encoding='utf-8')
# toyota_manual_content = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


def create_retrives(contents: str):
    documents = text_splitter.split_documents(contents)

    # vectorstore = FAISS.from_documents(documents, embedding=EmbeddingManager().get_embeddings('BAAI/bge-m3'))
    # vectorstore.save_local("./toyota_manual")
    # retriever = vectorstore.as_retriever()

    retriever = BM25Retriever.from_documents(documents)

    return retriever


class RetrieverManager:
    retrievers = {}
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # self.retrievers['manual'] = create_retrives(toyota_manual_content)
        pass