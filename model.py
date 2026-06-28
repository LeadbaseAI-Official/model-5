from pathlib import Path
from typing import Optional
from llama_cpp import Llama
from chat_template import format_chat_prompt

_llm_instance: Optional[Llama] = None

def find_gguf_file() -> Path:
    # Check current directory
    for path in Path(".").glob("*.gguf"):
        return path
    # Check model/ directory
    model_dir: Path = Path("model")
    if model_dir.exists():
        for path in model_dir.glob("*.gguf"):
            return path
    # Fallback default
    return Path("gemma-4-12b-it-UD-Q4_K_XL.gguf")

def get_llm() -> Llama:
    global _llm_instance
    if _llm_instance is None:
        model_path: Path = find_gguf_file()
        if not model_path.exists():
            raise FileNotFoundError(f"No GGUF model file found. Expected one in root or model/ directory.")
        
        _llm_instance = Llama(
            model_path=str(model_path),
            n_threads=4,
            n_ctx=2048,
        )
    return _llm_instance

def run_model_query(prompt: str) -> str:
    try:
        llm: Llama = get_llm()
        formatted_prompt: str = format_chat_prompt(prompt)
        response = llm(
            formatted_prompt,
            max_tokens=256,
        )
        text_result: str = response["choices"][0]["text"]
        return text_result
    except Exception as e:
        return f"Exception raised while running llama-cpp: {e}"