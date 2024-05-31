import aiohttp
import asyncio
import json
import logging
import time
from typing import List, Tuple

import numpy as np

from util import sample_requests, get_tokenizer

logger = logging.getLogger(__name__)
# Tuple[prompt_len, completion_len, request_time_in_milliseconds]
REQUEST_LATENCY: List[Tuple[int, int, float]] = []

# 替换为你的API密钥和端点
API_KEY = 'your_api_key'
API_URL = 'http://localhost:80/v1/chat/completions'
MODEL_UID = 'qwen1.5-chat-7b'

HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}'
}


async def send_request(session, payload, prompt_len):
    request_start_time = time.time()
    async with session.post(API_URL, json=payload, headers=HEADERS) as response:
        if response.status == 200:
            result = await response.json()
            completion_tokens = result["usage"]["completion_tokens"]
            request_end_time = time.time()
            request_latency = request_end_time - request_start_time
            REQUEST_LATENCY.append((prompt_len, completion_tokens, request_latency))
            return result
        else:
            return {'error': response.status, 'message': await response.text()}


async def benchmark(
    input_requests: List[Tuple[str, int, int]],
) -> None:
    async with aiohttp.ClientSession() as session:
        for idx, request in enumerate(input_requests):
            prompt, prompt_len, output_len = request
            payload = {
                'model': MODEL_UID,
                "n": 1,
                "temperature": 0,
                "top_p": 1.0,
                'messages': [{"role": "user", "content": prompt}],
                'max_tokens': 8192
            }
            response = await send_request(session, payload, prompt_len)
            print(f"Response {idx + 1}: {json.dumps(response, ensure_ascii=False, indent=2)}")


def main():
    logger.info("Preparing for benchmark.")
    dataset_path = r'ShareGPT_V3_unfiltered_cleaned_split.json'
    tokenizer_name_or_path = 'qwen/Qwen1.5-7B-Chat'
    num_request = 100
    tokenizer = get_tokenizer(tokenizer_name_or_path)
    input_requests = sample_requests(dataset_path, num_request, tokenizer)

    logger.info("Benchmark starts.")
    benchmark_start_time = time.time()
    asyncio.run(benchmark(input_requests))
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time

    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {len(REQUEST_LATENCY) / benchmark_time:.2f} requests/s")
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean(
        [
            latency / (prompt_len + output_len)
            for prompt_len, output_len, latency in REQUEST_LATENCY
        ]
    )
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency in REQUEST_LATENCY]
    )
    print("Average latency per output token: " f"{avg_per_output_token_latency:.2f} s")


if __name__ == '__main__':
    main()
