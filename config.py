import os
import logging.config
import dotenv

DEFAULTS = {
    'LLM_ENGIN': 'huggingface',  # huggingface, modelscope, vllm
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
    logging.config.dictConfig(LOGGING_CONFIG)
    logging_inited = True


__initialize_logging()
