import os
import boto3

from collections import OrderedDict

from langchain_community.chat_models import BedrockChat

def get_bedrock_client():
    return boto3.client(
        service_name='bedrock-runtime',
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
        endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", "https://bedrock-runtime.us-west-2.amazonaws.com"),
        verify=False,
    )

def get_bedrock_llm_model(model_name='Anthropic Claude v3 Haiku', model_params=None, stream=False) -> BedrockChat:
    model_info = model_info_selector(model_name)

    model_id = model_info[0]
    model_parameters = model_params if model_params is not None else model_info[1]

    bedrock_client = get_bedrock_client()

    return BedrockChat(
        model_id=model_id,
        model_kwargs=model_parameters,
        client = bedrock_client,
        streaming=stream
    )

def model_info_selector(model=None, temperature=0.0, topP=0.9, max_tokens=1024):
    
    model_id = {
        "amazon_titan_tg1": "amazon.titan-tg1-large",
        "mistral_mixtral_8_7": "mistral.mixtral-8x7b-instruct-v0:1",
        "ai21_j2_mid": "ai21.j2-mid",
        "ai21_j2_jumbo_instruct": "ai21.j2-jumbo-instruct",
        "anthropic_claude_instance_v1" : "anthropic.claude-instant-v1",
        "anthropic_claude_v1" : "anthropic.claude-v1",
        "anthropic_claude_v2" : "anthropic.claude-v2",
        "anthropic_claude_v2_1" : "anthropic.claude-v2:1",
        "anthropic_claude_v3_sonnet" : "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic_claude_v3_haiku" : "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic_claude_v3_opus" : "anthropic.claude-3-opus-20240229-v1:0"
    }

    model_kwargs = {
        "amazon_titan_model_kwargs": {
            "maxTokenCount": max_tokens,
            "stopSequences": [],
            "temperature": temperature,
            "topP": topP
        },

        "claude_model_kwargs": {
            "max_tokens": max_tokens,
            "stop_sequences": ["\\n\\nHuman:", "\\nHuman:", "Human:"],
            "temperature": temperature,
            "top_k": 100,
            "top_p": topP,
            "anthropic_version": "bedrock-2023-05-31"
        },

        "ai21_model_kwargs": {
            "maxTokens": max_tokens,
            "stopSequences": ['\nHuman:'],
            "temperature": temperature,
            "topP": topP
        },

        "mixtral_8_7_model_kwargs": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": 100,
            "top_p": topP,
        }
    }

    amazon_titan_tg1_large = (model_id["amazon_titan_tg1"], model_kwargs["amazon_titan_model_kwargs"])
    mistral_mixtral_8_7 = (model_id["mistral_mixtral_8_7"], model_kwargs["mixtral_8_7_model_kwargs"])
    ai21_j2_mid = (model_id["ai21_j2_mid"], model_kwargs["ai21_model_kwargs"])
    ai21_j2_jumbo_instruct = (model_id["ai21_j2_jumbo_instruct"], model_kwargs["ai21_model_kwargs"])
    anthropic_claude_instance_v1 = (model_id["anthropic_claude_instance_v1"], model_kwargs["claude_model_kwargs"])
    anthropic_claude_v1 = (model_id["anthropic_claude_v1"], model_kwargs["claude_model_kwargs"])
    anthropic_claude_v2 = (model_id["anthropic_claude_v2"], model_kwargs["claude_model_kwargs"])
    anthropic_claude_v2_1 = (model_id["anthropic_claude_v2_1"], model_kwargs["claude_model_kwargs"])
    anthropic_claude_v3_sonnet = (model_id["anthropic_claude_v3_sonnet"], model_kwargs["claude_model_kwargs"])
    anthropic_claude_v3_haiku = (model_id["anthropic_claude_v3_haiku"], model_kwargs["claude_model_kwargs"])
    anthropic_claude_v3_opus = (model_id["anthropic_claude_v3_opus"], model_kwargs["claude_model_kwargs"])

    model_list = OrderedDict({
        'Amazon Titan Large': amazon_titan_tg1_large,
        'Mistral Mixtral 8x7 Instruct': mistral_mixtral_8_7,
        'AI21 Jurrasic-2 Mid': ai21_j2_mid,
        'AI21 Jurrasic-2 Ultra': ai21_j2_jumbo_instruct,
        'Anthropic Claude Instant v1': anthropic_claude_instance_v1,
        'Anthropic Claude v1': anthropic_claude_v1,
        'Anthropic Claude v2': anthropic_claude_v2,
        'Anthropic Claude v2.1': anthropic_claude_v2_1,
        'Anthropic Claude v3 Sonnet': anthropic_claude_v3_sonnet,
        'Anthropic Claude v3 Haiku': anthropic_claude_v3_haiku,
        'Anthropic Claude v3 Opus': anthropic_claude_v3_opus,
    })

    return model_list.get(model, model_list)