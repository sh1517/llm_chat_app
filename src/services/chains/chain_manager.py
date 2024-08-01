from services.templates import chat_template
from services.apis import bedrock_api_connector
from services.retrievers.retriever_manager import RetrieverManager

from models.basic_chat_request import BasicChatRequest

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.globals import set_debug
set_debug(True)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def chain_maker(memory=False, prompt=None, model_params=None, vector_db=False, retrievers=False):
    prompt = prompt if prompt is not None else chat_template.simple_chat_template()

    haiku = bedrock_api_connector.get_bedrock_llm_model(model_name='Anthropic Claude v3 Haiku', model_params=model_params)
    sonnet = bedrock_api_connector.get_bedrock_llm_model(model_name='Anthropic Claude v3 Sonnet', model_params=model_params)
    opus = bedrock_api_connector.get_bedrock_llm_model(model_name='Anthropic Claude v3 Opus', model_params=model_params)

    llm = haiku.configurable_alternatives(
        ConfigurableField(id="llm"),
        default_key="haiku",
        sonnet=sonnet,
        opus=opus,
    )

    if memory:
        chain_base = prompt | llm | StrOutputParser()
        chain = RunnableWithMessageHistory(
            chain_base,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
    elif vector_db:
        chain = (
            RunnablePassthrough.assign(source_documents=(lambda x: x["source_documents"])) 
            | prompt
            | llm
            | StrOutputParser()
        )
    elif retrievers:
        chain = (
            {"source_documents": RetrieverManager().retrievers["manual"], "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    else:
        chain = (
            {"input": RunnablePassthrough()} 
            | prompt
            | llm
            | StrOutputParser()
        )

    return chain

class ChainManager:
    chains = {}
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.chains['basic'] = chain_maker(memory=False, prompt=chat_template.simple_chat_template())
        self.chains['memory'] = chain_maker(memory=True, prompt=chat_template.memory_chat_template())
        self.chains['knowledge'] = chain_maker(memory=True, prompt=chat_template.rag_chat_template(), vector_db=True)
        self.chains['retrievers'] = chain_maker(memory=False, prompt=chat_template.rag_chat_template(), retrievers=True)
        self.chains['multi_turn_extractor'] = chain_maker(memory=False, prompt=chat_template.multi_turn_chat_template())

    def simple_chat_chain(self, message: str):
        return self.chains['basic'].invoke(message)
        # return basic_chat_request.message

    def memory_chat_chain(self, message: str, session_id: str):
        return self.chains['memory'].invoke({'input': message},  
                                            config={"configurable": {"session_id": session_id}})

    def multi_turn_extractor_chain(self, message: str):
        return self.chains['multi_turn_extractor'].invoke(message)

    def knowledge_chat_chain(self, basic_chat_request: BasicChatRequest):
        return self.chains['knowledge'].invoke({'input': basic_chat_request.message, 
                                                "source_documents": basic_chat_request.multi_turn},  
                                                config={"configurable": {"session_id": basic_chat_request.session_id}})

    def stream_chat_chain(self, message: str):
        chunks = self.chains['basic'].stream(message)

        for chunk in chunks:
            print(chunk)
            yield chunk

    def retriever_chat_chain(self, message: str):
        return self.chains['retrievers'].invoke(message)