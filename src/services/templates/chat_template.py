from langchain.prompts import PromptTemplate

def simple_chat_template(template:bool = True) -> any:
    prompt = """ 
        The following is a friendly conversation between a human and an AI. 
        The AI is talkative and provides lots of specific details from its context. 
        If the AI does not know the answer to a question, it truthfully says it does not know.
        {input}
    """

    return PromptTemplate.from_template(template=prompt) if template else prompt

def memory_chat_template(template:bool = True) -> any:
    prompt = """ 
        ###Instruction
        The following is a friendly conversation between a human and an AI. 
        The AI is talkative and provides lots of specific details from its context. 
        If the AI does not know the answer to a question, it truthfully says it does not know.
        ###History conversation:
        {chat_history}
        ###Current conversation:
        {input}
    """

    return PromptTemplate.from_template(template=prompt) if template else prompt

def multi_turn_chat_template(template:bool = True) -> any:
    prompt = """ 
        ###Instruction
        1. 제시하는 지게차 관련 텍스트 문장에서 model_name, abnormal_part, abnormal_symptom 정보를 추출해주세요.
        2. 추출된 정보는 아래의 << FORMATTING >> 형식에 맞춰서 JSON 형태로 출력해주세요.

        << FORMATTING >>
        Return a markdown code snippet with a JSON object formatted to look like:
        ```json
            {{{{
            \"model_name\": str \\ 중장비 모델명 (예: D30S-3, D30S-7, D30S-10)
            \"abnormal_part\": str \\ 이상이 발생한 부품 (예: 엔진, 유압시스템, 드라이브트레인, 조향 시스템, 브레이크 시스템, 전기 시스템, 마스트 & 포크, 차체)
            \"abnormal_symptom\": str \\ 이상 증상 (예: 흔들림, 소음, 발열, 시동꺼짐,  에러코드)
            }}}}
        ```

        REMEMBER: "model_name" MUST be a name of model, model name are D30S-3, D30S-7, D30S-10, If no value can be found return null.
        REMEMBER: "abnormal_symptom" MUST be sure to include the abnormal symptoms entered. Even if the model name is null. If no value can be found return null.
        REMEMBER: "abnormal_symptom" Do not include model name.

        ###Input
        << INPUT >> {input} \n\n << OUTPUT (remember to include the ```json)>>

        REMEMBER: Read request again {input}
    """

    return PromptTemplate.from_template(template=prompt) if template else prompt


def rag_chat_template(template=True) -> any:
    prompt = """The following is a friendly conversation between a human and an AI. 
                The AI is talkative and provides lots of specific details from its context. 
                If the AI does not know the answer to a question, it truthfully says it does not know.
                {input}
                
                {source_documents}
                """

    return PromptTemplate.from_template(template=prompt) if template else prompt