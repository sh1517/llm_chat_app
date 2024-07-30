from fastapi import APIRouter

from models.basic_chat_request import BasicChatRequest

from services.llms.bedrock_services import BedrockService

router = APIRouter()
bedrock_service = BedrockService()

@router.post('/chat')
async def basic_chat(basic_chat_request: BasicChatRequest):
    return bedrock_service.simple_chat(basic_chat_request)

@router.post('/chat/memory')
async def basic_chat(basic_chat_request: BasicChatRequest):
    return bedrock_service.memory_chat(basic_chat_request)

@router.post('/chat/multi-turn')
async def basic_chat(basic_chat_request: BasicChatRequest):
    return bedrock_service.multiturn_chat(basic_chat_request)