import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
# device = "cuda"  # the device to load the model onto
device = "cpu"

checkpoint_path = os.getenv('MODELSCOPE_CACHE') + "/qwen/Qwen1___5-7B-Chat"
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(device)
streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)

generation_kwargs = dict(input_ids=input_ids, streamer=streamer, max_new_tokens=512)
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

for new_text in streamer:
    print(new_text, end='', flush=True)
