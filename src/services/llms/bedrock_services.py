from fastapi.responses import StreamingResponse

from langchain.output_parsers.json import parse_and_check_json_markdown

from services.chains.chain_manager import ChainManager

from langchain_core.runnables import RunnableParallel, RunnableAssign, RunnableBinding
from langchain_core.runnables.configurable import RunnableConfigurableAlternatives
from models.basic_chat_request import BasicChatRequest

class BedrockService:
    chain_manager = ChainManager
    
    def __init__(self):
        self.chain_manager = ChainManager()

    def simple_chat(self, basic_chat_request: BasicChatRequest):
        # return self.chain_manager.simple_chat_chain(basic_chat_request.message)
        return self.chain_manager.simple_chat_chain(basic_chat_request)
    
    def memory_chat(self, basic_chat_request: BasicChatRequest):
        return self.chain_manager.memory_chat_chain(basic_chat_request.message, basic_chat_request.session_id)
    
    def multiturn_chat(self, basic_chat_request: BasicChatRequest):

        if basic_chat_request.multi_turn:
            return self.chain_manager.knowledge_chat_chain(basic_chat_request)

        extracted_response = self.chain_manager.multi_turn_extractor_chain(basic_chat_request.message)
        print(extracted_response)
        keys = ["model_name", "abnormal_part", "abnormal_symptom"]
        try:
            multi_turn_filter = parse_and_check_json_markdown(extracted_response, keys)
        except:
            multi_turn_filter = {'model_name': None, 
                                 'abnormal_part': None, 
                                 'abnormal_symptom': basic_chat_request.message}

        key_checker = [key for key, value in multi_turn_filter.items() if value is None]
        # return multi_turn_filter
        if key_checker:
            return multi_turn_filter
        else:
            basic_chat_request.multi_turn = multi_turn_filter
            return self.chain_manager.knowledge_chat_chain(basic_chat_request)
    
    def stream_chat(self, basic_chat_request: BasicChatRequest):
        return StreamingResponse(self.chain_manager.stream_chat_chain(basic_chat_request.message),
                                 media_type='text/event-stream')
    
    def retriever_chat(self, basic_chat_request: BasicChatRequest):
        return self.chain_manager.retriever_chat_chain(basic_chat_request.message)
    
    def chain_information(self, basic_chat_request: BasicChatRequest):
        res = self.chain_manager.chain_information_response(basic_chat_request)
        return res.last.mapper.steps__['answer'].last