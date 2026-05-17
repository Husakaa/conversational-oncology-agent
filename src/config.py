OLLAMA_MODELS = {
    "qwen2.5-7B": "qwen2.5-7B",
    "qwen3-0.6B": "qwen3-0.6B",
    "gemma4-nano-e2b": "gemma4-nano-e2b",
    "biomistral-7B": "biomistral-oncologo"
}

QWEN25_OPTIONS = {
    "temperature": 0.3,
    "top_k": 20,
    "top_p": 0.9,
    "num_ctx": 2048,
    "num_predict": 512
}

QWEN3_OPTIONS = {
    "temperature": 0.3,
    "top_k": 20,
    "top_p": 0.8,
    "min_p": 0,
    "num_ctx": 4096,
    "num_predict": 2024,
    "repeat_penalty": 1.2
}

GEMMA4_OPTIONS = {
    "temperature": 0.8,
    "top_k": 20,
    "top_p": 0.9,
    "num_ctx": 2048,
    "num_predict": 2024
}



BIOMISTRAL_OPTIONS = {
    "temperature": 0.4,
    "top_k": 30,
    "top_p": 0.85,
    "num_ctx": 2048,
    "num_predict": 512
}