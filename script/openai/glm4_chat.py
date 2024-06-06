import os
import string
import time
import logging
from asyncio.log import logger
import uuid
import random

import uvicorn
import gc
import json
import torch

from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, LogitsProcessor
from sse_starlette.sse import EventSourceResponse

EventSourceResponse.DEFAULT_PING_INTERVAL = 1000
MODEL_PATH = 'THUDM/glm-4-9b-chat'


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_id(prefix: str) -> str:
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=24))
    return f"{prefix}-{suffix}"


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class FunctionCallResponse(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ToolCallResponse(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: generate_id('call'))
    index: Optional[int] = 0
    type: Optional[str] = 'function'
    function: FunctionCallResponse = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCallResponse]] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCallResponse]] = None


class EmbeddingRequest(BaseModel):
    input: Union[List[str], str]
    model: str


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: CompletionUsage


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[Union[dict, List[dict]]] = None
    tool_choice: Optional[Union[str, dict]] = None
    repetition_penalty: Optional[float] = 1.1


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call", "tool_calls"]


class ChatCompletionResponseStreamChoice(BaseModel):
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "function_call", "tool_calls"]]
    index: int


class ChatCompletionResponse(BaseModel):
    model: str

    id: str = Field(default_factory=lambda: generate_id('chatcmpl'))
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


def process_response(output: str) -> Union[str, List[dict]]:
    content = ""
    function_call_arr = []
    for response in output.split("<|assistant|>"):
        if "\n" in response:
            metadata, content = response.split("\n", maxsplit=1)
        else:
            metadata, content = "", response
        if not metadata.strip():
            content = content.strip()
        else:
            parameters = eval(content.strip())
            function_call_arr.append({
                "name": metadata.strip(),
                "arguments": json.dumps(parameters, ensure_ascii=False)
            })
    return function_call_arr if len(function_call_arr) > 0 else content


@torch.inference_mode()
async def generate_stream_glm4(params):
    messages = params["messages"]
    tools = params["tools"]
    tool_choice = params["tool_choice"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 8192))
    messages = process_messages(messages, tools=tools, tool_choice=tool_choice)
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    params_dict = {
        "n": 1,
        "best_of": 1,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.0,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": -1,
        "repetition_penalty": repetition_penalty,
        "use_beam_search": False,
        "length_penalty": 1,
        "early_stopping": False,
        "stop_token_ids": [151329, 151336, 151338],
        "ignore_eos": False,
        "max_tokens": max_new_tokens,
        "logprobs": None,
        "prompt_logprobs": None,
        "skip_special_tokens": True,
    }
    sampling_params = SamplingParams(**params_dict)
    async for output in engine.generate(inputs=inputs, sampling_params=sampling_params, request_id="glm-4-9b"):
        output_len = len(output.outputs[0].token_ids)
        input_len = len(output.prompt_token_ids)
        ret = {
            "text": output.outputs[0].text,
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": output_len,
                "total_tokens": output_len + input_len
            },
            "finish_reason": output.outputs[0].finish_reason,
        }
        yield ret
    gc.collect()
    torch.cuda.empty_cache()


def process_messages(messages, tools=None, tool_choice=None):
    _messages = messages
    messages = []
    msg_has_sys = False

    def filter_tools(tool_choice, tools):
        function_name = tool_choice.get('function', {}).get('name', None)
        if not function_name:
            return []
        filtered_tools = [
            tool for tool in tools
            if tool.get('function', {}).get('name') == function_name
        ]
        return filtered_tools

    if tool_choice and tool_choice != "none":
        if isinstance(tool_choice, dict):
            tools = filter_tools(tool_choice, tools)
        if tools:
            messages.append(
                {
                    "role": "system",
                    "content": None,
                    "tools": tools
                }
            )
        msg_has_sys = True
    # add to metadata
    if isinstance(tool_choice, dict) and tools:
        messages.append(
            {
                "role": "assistant",
                "metadata": tool_choice["function"]["name"],
                "content": ""
            }
        )

    for m in _messages:
        role, content, tool_calls = m.role, m.content, m.tool_calls
        if role == "function":
            messages.append(
                {
                    "role": "observation",
                    "content": content
                }
            )
        elif role == "assistant" and tool_calls is not None:
            for response in content.split("<|assistant|>"):
                first_tool_call = tool_calls[0]
                messages.append(
                    {
                        "role": role,
                        "metadata": first_tool_call.function.name,
                        "content": first_tool_call.function.arguments
                    }
                )
        else:
            if role == "system" and msg_has_sys:
                msg_has_sys = False
                continue
            messages.append({"role": role, "content": content})

    return messages


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="glm-4")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")
    if request.tool_choice is None:
        request.tool_choice = "auto" if request.tools else "none"
    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,
        tools=request.tools,
        tool_choice=request.tool_choice,
    )
    logger.debug(f"==== request ====\n{gen_params}")

    if request.stream:
        predict_stream_generator = predict_stream(request.model, gen_params)
        output = await anext(predict_stream_generator)
        # output would be empty if it is tool call or response reading done
        if not output:
            return EventSourceResponse(predict_stream_generator, media_type="text/event-stream", sep="\n")
        # response done
        logger.debug(f"First result output: \n{output}")
        generate = parse_output_text(request.model, request.tools, output)
        return EventSourceResponse(generate, media_type="text/event-stream", sep="\n")

    response = ""
    async for response in generate_stream_glm4(gen_params):
        pass
    if response["text"].startswith("\n"):
        response["text"] = response["text"][1:]
    response["text"] = response["text"].strip()

    usage = UsageInfo()
    choice, finish_reason = create_choice_data(response["text"], request.tools, False)
    task_usage = UsageInfo.model_validate(response["usage"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(
        model=request.model,
        choices=[choice],
        object="chat.completion",
        usage=usage
    )


def calc_max_tool_name_len(tools: Optional[List[dict]]) -> int:
    max_tool_name_len = 0
    if not tools:
        return max_tool_name_len
    tool_names = [tool['function']['name'] for tool in tools if 'function' in tool and 'name' in tool['function']]
    max_tool_name_len = max(len(tool_name) for tool_name in tool_names)
    return max_tool_name_len


def is_return_tool_call(output: str, tools: Optional[List[dict]]) -> bool:
    if not tools:
        return False
    if output.startswith("\n"):
        output = output[1:]
    tool_names = [tool['function']['name'] for tool in tools if 'function' in tool and 'name' in tool['function']]
    return any(output.startswith(name) for name in tool_names)


async def predict_stream(model_id, gen_params):
    output = ""
    is_function_call = False
    has_send_first_chunk = False
    tools = gen_params.get("tools")
    max_tool_name_len = calc_max_tool_name_len(tools)
    async for new_response in generate_stream_glm4(gen_params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(output):]
        output = decoded_unicode

        if not is_function_call and len(output) > max_tool_name_len:
            is_function_call = is_return_tool_call(output, tools)
            if is_function_call:
                continue

            finish_reason = new_response["finish_reason"]
            if not has_send_first_chunk:
                message = DeltaMessage(
                    content="",
                    role="assistant",
                    function_call=None,
                )
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=message,
                    finish_reason=finish_reason
                )
                chunk = ChatCompletionResponse(
                    model=model_id,
                    choices=[choice_data],
                    created=int(time.time()),
                    object="chat.completion.chunk"
                )
                yield "{}".format(chunk.model_dump_json(exclude_unset=True))
            send_msg = delta_text if has_send_first_chunk else output[1:] if output.startswith("\n") else output
            has_send_first_chunk = True
            message = DeltaMessage(
                content=send_msg,
                role="assistant",
                function_call=None,
            )
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=message,
                finish_reason=finish_reason
            )
            chunk = ChatCompletionResponse(
                model=model_id,
                choices=[choice_data],
                created=int(time.time()),
                object="chat.completion.chunk"
            )
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    # if the total output length less than the max tool name length, has_send_first_chunk = False
    if is_function_call or not has_send_first_chunk:
        has_send_first_chunk = True
        yield output
    else:
        yield '[DONE]'


def create_choice_data(output: str, tools: Optional[List[dict]], stream: bool):
    # parse output to function format
    is_tool_call = is_return_tool_call(output, tools)
    if is_tool_call:
        try:
            output = process_response(output)
        except:
            logger.warning("Failed to parse tool call")

    # create choice response
    finish_reason = "stop"
    if isinstance(output, list):
        finish_reason = "tool_calls"
        tool_calls = []
        for obj in output:
            function = FunctionCallResponse(**obj)
            tool_calls.append(ToolCallResponse(function=function))
        if stream:
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant", tool_calls=tool_calls),
                finish_reason=None
            )
        else:
            choice_data = ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", tool_calls=tool_calls),
                finish_reason=finish_reason
            )
    else:
        if stream:
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content=output),
                finish_reason=None
            )
        else:
            choice_data = ChatCompletionResponseChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content=output),
                finish_reason=finish_reason
            )
    return choice_data, finish_reason


async def parse_output_text(model_id: str, tools: Optional[List[dict]], output: str):
    choice, finish_reason = create_choice_data(output, tools, True)
    chunk = ChatCompletionResponse(model=model_id, choices=[choice], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason=finish_reason
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, revision="269b8bad3c42cc639306b20dffd0b74ff7c12eac")
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        revision="269b8bad3c42cc639306b20dffd0b74ff7c12eac",
        tokenizer=MODEL_PATH,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        worker_use_ray=True,
        engine_use_ray=False,
        disable_log_requests=True
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    uvicorn.run(app, host='0.0.0.0', port=80, workers=1)
