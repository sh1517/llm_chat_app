from fastapi import APIRouter

from models.basic_chat_request import BasicChatRequest

from services.llms.bedrock_services import BedrockService

router = APIRouter()
bedrock_service = BedrockService()

@router.post('/chat/basic')
async def basic_chat(basic_chat_request: BasicChatRequest):
    return bedrock_service.simple_chat(basic_chat_request)

@router.post('/chat/memory')
async def memory_chat(basic_chat_request: BasicChatRequest):
    return bedrock_service.memory_chat(basic_chat_request)

@router.post('/chat/multi-turn')
async def multi_turn_chat(basic_chat_request: BasicChatRequest):
    return bedrock_service.multiturn_chat(basic_chat_request)

@router.post('/chat/stream')
async def stream_chat(basic_chat_request: BasicChatRequest):
    return bedrock_service.stream_chat(basic_chat_request)

@router.post('/chat/retriever')
async def retriever_chat(basic_chat_request: BasicChatRequest):
    return bedrock_service.retriever_chat(basic_chat_request)