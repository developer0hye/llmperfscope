"""vLLM as a library, without its server:

    llmperfscope run --label vllm-offline -- python examples/vllm_offline.py Qwen/Qwen3-0.6B
"""

import sys

import llmperfscope
from vllm import LLM, SamplingParams

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
NEW_TOKENS = 128

if __name__ == "__main__":
    with llmperfscope.phase("load"):
        llm = LLM(model=MODEL, gpu_memory_utilization=0.6, max_model_len=4096)

    params = SamplingParams(temperature=0.0, max_tokens=NEW_TOKENS, ignore_eos=True)
    prompts = ["The history of measurement is"] * 8
    with llmperfscope.phase("decode", tokens=NEW_TOKENS * len(prompts)):
        llm.generate(prompts, params)
