from fastapi import APIRouter
from _types import ChatCompletionCreateParams

chat_completion_router = APIRouter(prefix="/v1")


@chat_completion_router.post("/chat/completions")
async def chat_completion(request: ChatCompletionCreateParams):
    request.max_tokens = request.max_tokens or 1024
    return request
