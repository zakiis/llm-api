import config


def create_hf_engine():
    pass


LLM_ENGINE = None

if config.get_env('LLM_ENGINE') == 'huggingface':
    LLM_ENGINE = create_hf_engine()


def get_llm_engine():
    return LLM_ENGINE
