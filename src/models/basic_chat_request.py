from pydantic import BaseModel, validator
from typing import Optional

LLM_MODEL_LIST = {
    "Claude v3 Haiku": 'haiku',
    "Claude v3 Sonnet": 'sonnet',
    "Claude v3 Opus": 'opus'
}

class MultiTurn(BaseModel):
    model_name: Optional[str] = None
    abnormal_part: Optional[str] = None
    abnormal_symptom: Optional[str] = None


class BasicChatRequest(BaseModel):
    connection_id: str = "01F8MECHZX3TBDSZ7N4J8B0F4X"
    message_id: str = "01F8MECHZX3TBDSZ7N4J8B0F4Y"
    session_id: str = "01F8MECHZX3TBDSZ7N4J8B0F4Z"
    conversation_type: Optional[str] = None
    knowledge_type: Optional[str] = None
    message: Optional[str] = None
    llm: Optional[str] = None
    multi_turn: Optional[MultiTurn] = None

    @validator("llm", pre=True, always=True)
    def set_llm_modle(cls, input_llm_value):
        return LLM_MODEL_LIST.get(input_llm_value, 'haiku')