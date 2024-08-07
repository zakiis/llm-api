import os
import logging.config
from typing import Optional, Dict, List, Union

import dotenv
from pydantic import BaseModel, Field

DEFAULTS = {
    'LLM_ENGIN': 'huggingface',  # transformers, vllm
    'LOG_LEVEL': 'INFO',
    'LOG_FILE': '',
    'LOG_FORMAT': '%(asctime)s.%(msecs)03d %(levelname)s [%(threadName)s] [%(filename)s:%(lineno)d] - %(message)s',
    'ACCESS_LOG_FORMAT': '%(asctime)s.%(msecs)03d %(levelname)s [%(threadName)s] [%(filename)s:%(lineno)d] - %('
                         'client_addr)s "%(request_line)s" %(status_code)s',
    'LOG_DATEFORMAT': '%Y-%m-%d %H:%M:%S',
}
dotenv.load_dotenv()


def get_env(key: str) -> str:
    return os.environ.get(key, DEFAULTS.get(key))


def get_bool_env(key) -> bool:
    value = get_env(key)
    return value.lower() == 'true' if value is not None else False


# 日志配置
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': get_env('LOG_FORMAT'),
            'datefmt': get_env('LOG_DATEFORMAT'),
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": get_env('ACCESS_LOG_FORMAT'),
            'datefmt': get_env('LOG_DATEFORMAT'),
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default'
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
    'root': {
        'level': get_env('LOG_LEVEL'),
        'handlers': ['console'],
    }
}
logging_inited = False


def __initialize_logging():
    global logging_inited
    if logging_inited:
        return
    log_file = get_env('LOG_FILE')
    if log_file:
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)
        LOGGING_CONFIG['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': log_file,
            'maxBytes': 1024 * 1024 * 1024,
            'backupCount': 5,
            'formatter': 'default'
        }
        LOGGING_CONFIG['handlers']['file_access'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': log_file,
            'maxBytes': 1024 * 1024 * 1024,
            'backupCount': 5,
            'formatter': 'access'
        }
        LOGGING_CONFIG['root']['handlers'] = ['console', 'file']
        LOGGING_CONFIG['loggers']['uvicorn.access']['handlers'] = ['access', 'file_access']
    logging.config.dictConfig(LOGGING_CONFIG)
    logging_inited = True


__initialize_logging()

# 模型设置
ENGINE = get_env("ENGINE", "huggingface").lower()
TASKS = get_env("TASKS", "llm").lower().split(",")  # llm, rag

class BaseSettings(BaseModel):
    """ Settings class. """
    host: Optional[str] = Field(
        default=get_env("HOST", "0.0.0.0"),
        description="Listen address.",
    )
    port: Optional[int] = Field(
        default=int(get_env("PORT", 8000)),
        description="Listen port.",
    )
    api_prefix: Optional[str] = Field(
        default=get_env("API_PREFIX", "/v1"),
        description="API prefix.",
    )
    engine: Optional[str] = Field(
        default=ENGINE,
        description="Choices are ['default', 'vllm', 'llama.cpp', 'tgi'].",
    )
    tasks: Optional[List[str]] = Field(
        default=list(TASKS),
        description="Choices are ['llm', 'rag'].",
    )
    # device related
    device_map: Optional[Union[str, Dict]] = Field(
        default=get_env("DEVICE_MAP", "auto"),
        description="Device map to load the model."
    )
    gpus: Optional[str] = Field(
        default=get_env("GPUS", None),
        description="Specify which gpus to load the model."
    )
    num_gpus: Optional[int] = Field(
        default=int(get_env("NUM_GPUs", 1)),
        ge=0,
        description="How many gpus to load the model."
    )
    activate_inference: Optional[bool] = Field(
        default=get_bool_env("ACTIVATE_INFERENCE", "true"),
        description="Whether to activate inference."
    )
    model_names: Optional[List] = Field(
        default_factory=list,
        description="All available model names"
    )
    # support for api key check
    api_keys: Optional[List[str]] = Field(
        default=get_env("API_KEYS", "").split(",") if get_env("API_KEYS", "") else None,
        description="Support for api key check."
    )


class LLMSettings(BaseModel):
    # model related
    model_name: Optional[str] = Field(
        default=get_env("MODEL_NAME", None),
        description="The name of the model to use for generating completions."
    )
    model_path: Optional[str] = Field(
        default=get_env("MODEL_PATH", None),
        description="The path to the model to use for generating completions."
    )
    dtype: Optional[str] = Field(
        default=get_env("DTYPE", "half"),
        description="Precision dtype."
    )

    # quantize related
    load_in_8bit: Optional[bool] = Field(
        default=get_bool_env("LOAD_IN_8BIT"),
        description="Whether to load the model in 8 bit."
    )
    load_in_4bit: Optional[bool] = Field(
        default=get_bool_env("LOAD_IN_4BIT"),
        description="Whether to load the model in 4 bit."
    )

    # context related
    context_length: Optional[int] = Field(
        default=int(get_env("CONTEXT_LEN", -1)),
        ge=-1,
        description="Context length for generating completions."
    )
    chat_template: Optional[str] = Field(
        default=get_env("PROMPT_NAME", None),
        description="Chat template for generating completions."
    )

    rope_scaling: Optional[str] = Field(
        default=get_env("ROPE_SCALING", None),
        description="RoPE Scaling."
    )
    flash_attn: Optional[bool] = Field(
        default=get_bool_env("FLASH_ATTN", "auto"),
        description="Use flash attention."
    )

    # support for transformers.TextIteratorStreamer
    use_streamer_v2: Optional[bool] = Field(
        default=get_bool_env("USE_STREAMER_V2", "true"),
        description="Support for transformers.TextIteratorStreamer."
    )

    interrupt_requests: Optional[bool] = Field(
        default=get_bool_env("INTERRUPT_REQUESTS", "true"),
        description="Whether to interrupt requests when a new request is received.",
    )


PARENT_CLASSES = [BaseSettings]
if "llm" in TASKS:
    if ENGINE == "default":
        PARENT_CLASSES.append(LLMSettings)
    elif ENGINE == "vllm":
        PARENT_CLASSES.extend([LLMSettings, VLLMSetting])
    elif ENGINE == "llama.cpp":
        PARENT_CLASSES.extend([LLMSettings, LlamaCppSetting])
    elif ENGINE == "tgi":
        PARENT_CLASSES.extend([LLMSettings, TGISetting])