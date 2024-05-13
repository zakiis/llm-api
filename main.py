import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI
import uvicorn
from routes import chat_completion_router
import config


def create_app() -> FastAPI:
    fast_api = FastAPI()
    fast_api.include_router(chat_completion_router)
    return fast_api


# def initialize_logging():
#     log_handlers = None
#     log_file = config.get_env('LOG_FILE')
#     if log_file:
#         log_dir = os.path.dirname(log_file)
#         os.makedirs(log_dir, exist_ok=True)
#         log_handlers = [
#             logging.handlers.RotatingFileHandler(
#                 filename=log_file,
#                 maxBytes=1024 * 1024 * 1024,
#                 backupCount=5
#             ),
#             logging.StreamHandler(sys.stdout)
#         ]
#     logging.basicConfig(
#         level=config.get_env('LOG_LEVEL'),
#         format=config.get_env('LOG_FORMAT'),
#         datefmt=config.get_env('LOG_DATEFORMAT'),
#         handlers=log_handlers
#     )

app = create_app()


@app.get("/")
def status():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_config=config.LOGGING_CONFIG, reload=True)
