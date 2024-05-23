import time
import uuid

from fastapi import FastAPI
import uvicorn
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
)
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from modelscope import AutoModelForCausalLM, AutoTokenizer

from _types import ChatCompletionCreateParams

app = FastAPI()
# load model

device = "cuda" # the device to load the model onto
model = AutoModelForCausalLM.from_pretrained(
    "qwen/Qwen1.5-7B-Chat",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen1.5-7B-Chat")

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionCreateParams):
    request_id = f"cmpl-{str(uuid.uuid4().hex)}"
    created_time = int(time.time())
    request.max_tokens = request.max_tokens or 1024
    if request.stream:
        pass
    else:
        choices = []
        message = ChatCompletionMessage(
            role="assistant",
            content="What can I help you today?",
        )
        choice = Choice(
            index=0,
            message=message,
            finish_reason="stop",
            logprobs=None,
        )
        choices.append(choice)
        usage = CompletionUsage(
            prompt_tokens=10,
            completion_tokens=23,
            total_tokens=33,
        )
        return ChatCompletion(
            id=request_id,
            choices=choices,
            model=request.model,
            created=created_time,
            object="chat.completion",
            usage=usage,
        )

if __name__ == "__main__":
    uvicorn.run("qwen1_5_7b_chat:app", host="0.0.0.0", port=8000, reload=True)
