"""transformers as a library: phases and token times from inside the script.

    llmprof run --label transformers -- python examples/transformers_generate.py Qwen/Qwen3-0.6B

The script runs the same without llmprof: `mark` and `phase` do nothing when no profiler is listening.
"""

import sys

import llmprof
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
NEW_TOKENS = 128

with llmprof.phase("load"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")

inputs = tokenizer("The history of measurement is", return_tensors="pt").to(model.device)
with llmprof.phase("warmup"):
    model.generate(**inputs, max_new_tokens=8, do_sample=False)

for i in range(3):
    timer = llmprof.TokenTimer()
    with llmprof.phase("decode", tokens=NEW_TOKENS):
        model.generate(**inputs, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS, do_sample=False, streamer=timer)
    print(timer.summary(), flush=True)
