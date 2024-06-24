# Use the native inference API to send a text message to Meta Llama 3
# and print the response stream.

from typing import List
import boto3
import json
import os
import logging
from typing import Iterator, Optional
from concurrent.futures import Executor
from dbgpt.core import MessageConverter, ModelOutput, ModelRequest, ModelRequestContext
from dbgpt.model.proxy.llms.proxy_model import ProxyModel
from dbgpt.core.interface.message import ModelMessage, ModelMessageRoleType
from dbgpt.model.parameter import ProxyModelParameters
from dbgpt.model.proxy.base import ProxyLLMClient


logger = logging.getLogger(__name__)

# BEDROCK_DEFAULT_MODEL = "meta.llama3-8b-instruct-v1:0"
BEDROCK_DEFAULT_MODEL = "Claude 3 Sonnet"


def _to_llama_message(messages: List[ModelMessage]):
    history = []
    # Add history conversation
    history.append("<|begin_of_text|>")
    for message in messages:
        if message.role == ModelMessageRoleType.HUMAN:
            history.append("<|start_header_id|>user<|end_header_id|>")
        elif message.role == ModelMessageRoleType.SYSTEM:
            # As of today, system message is not supported.
            history.append("<|start_header_id|>user<|end_header_id|>")
        elif message.role == ModelMessageRoleType.AI:
            history.append("<|start_header_id|>assistant<|end_header_id|>")
        else:
            pass
        history.append(message.content)
        history.append("<|eot_id|>")

    history.append("<|start_header_id|>assistant<|end_header_id|>")
    prompt ="\n".join(history)

    native_request = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.9,
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)
    return request

def _to_llama_response(streaming_response)->str:
    texts = list()
    for event in streaming_response["body"]:
        chunk = json.loads(event["chunk"]["bytes"])
        if "generation" in chunk:
            texts.append(chunk["generation"])
    text = "".join(texts)
    return text


def _to_claude_message(messages: List[ModelMessage]):
    msg = list()

    pre_item = None
    for message in messages:
        item = dict()
        if message.role == ModelMessageRoleType.AI:
            item['role'] = "assistant"
        else:
            item['role'] = "user"
        item['content'] = [{"type": "text", "text": message.content}]

        if pre_item:
            if pre_item['role'] == item['role']:
                pre_item['content'].extend( item['content'])
                continue

        pre_item = item
        msg.append(item)

    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.9,
        "messages":msg
    }
    logger.warn(native_request)
     # Convert the native request to JSON.
    request = json.dumps(native_request)
    return request


def _to_claude_response(streaming_response)->str:
    texts = list()
    for event in streaming_response["body"]:
        chunk = json.loads(event["chunk"]["bytes"])
        if chunk["type"] == "content_block_delta":
            texts.append(chunk["delta"].get("text", ""))
    text = "".join(texts)
    return text

def bedrock_generate_stream(
    model: ProxyModel, tokenizer=None, params=None, device=None, context_len=4096
):
    # Create a Bedrock Runtime client in the AWS Region of your choice.

    model_name = model.get_params().model_name
    client: BedrockLLMClient = model.proxy_llm_client
    context = ModelRequestContext(stream=True, user_name=params.get("user_name"))
    request = ModelRequest.build_request(
        model_name,
        messages=params["messages"],
        temperature=params.get("temperature"),
        context=context,
        max_new_tokens=params.get("max_new_tokens"),
    )
    for r in client.sync_generate_stream(request):
        yield r
    


class BedrockLLMClient(ProxyLLMClient):
    def __init__(
        self,
        model: Optional[str] = None,
        model_alias: Optional[str] = None,
        context_length: Optional[int] = 4096
    ):
        self._model = model
        super().__init__(
            model_names=[model, model_alias],
            context_length=context_length,
        )

        region = os.getenv("AWS_REGION")
        access_key_id=os.getenv("AWS_ACCESS_KEY", None)
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", None)

        if access_key_id and secret_access_key:
            self.client = boto3.client("bedrock-runtime", 
                                region_name=region, 
                                aws_access_key_id=access_key_id,
                                aws_secret_access_key=secret_access_key
            )
            meta_client = boto3.client("bedrock", 
                                region_name=region, 
                                aws_access_key_id=access_key_id,
                                aws_secret_access_key=secret_access_key
            )
        else:
            self.client = boto3.client("bedrock-runtime", 
                                region_name=region
            )
            meta_client = boto3.client("bedrock", 
                                region_name=region, 
                                aws_access_key_id=access_key_id,
                                aws_secret_access_key=secret_access_key
            )

        response = meta_client.list_foundation_models()
        self.model_info = {model['modelName']:model['modelId'] for model in response["modelSummaries"]}



    @classmethod
    def new_client(
        cls,
        model_params: ProxyModelParameters,
        default_executor: Optional[Executor] = None,
    ) -> "BedrockLLMClient":
        return cls(
            model=model_params.proxyllm_backend,
            model_alias=model_params.model_name,
            context_length=model_params.max_context_size,
        )

    @property
    def default_model(self) -> str:
        return self._model

    def sync_generate_stream(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> Iterator[ModelOutput]:
        request = self.local_covert_message(request, message_converter)
        model_key = request.model.strip()
        if model_key not in self.model_info:
            raise ValueError(f"does not support model {request.model} due to {self.model_info}")
        model_id = self.model_info[model_key]
        messages: List[ModelMessage] = request.get_messages()

        if model_key.startswith("Llama"):
            # TODO: add other model
            msg = _to_llama_message(messages)
        else:
            msg = _to_claude_message(messages)


        # Invoke the model with the request.
        streaming_response = self.client.invoke_model_with_response_stream(
            modelId=model_id, body=msg
        )
        # Extract and print the response text in real-time.

        if model_key.startswith("Llama"):
            # TODO: add other model
            text = _to_llama_response(streaming_response)
        else:
            text = _to_claude_response(streaming_response)


        yield ModelOutput(text=text, error_code=0)      


def main():
    model_params = ProxyModelParameters(
        model_name=BEDROCK_DEFAULT_MODEL,
        proxyllm_backend="not-used",
        model_path="not-used",
        proxy_server_url="not-used",
        proxy_api_key="not-used",
    )
    claude_client = BedrockLLMClient(
        model="bedrock"
    )
    model=ProxyModel(model_params=model_params,
                     proxy_llm_client=claude_client
                     )
    params={
        "messages": [
            ModelMessage(role=ModelMessageRoleType.HUMAN, content="用中文背诵《论语》第一章")
        ]
    }

    for part in bedrock_generate_stream(model=model, params=params):
        print(part, end="")

if __name__ == "__main__":
    main()
