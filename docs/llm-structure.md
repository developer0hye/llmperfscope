# What the model's structure tells you

A speed number means little without the structure behind it. Attention reads the KV cache up to a token's position, a Mixture of Experts model reads only the experts a token is routed to, tensor split shares all of that over the GPUs and adds collectives. llmperfscope reads the model's own header, so a report can say which part a number is about and what the number could be at best.

This works for a GGUF file, and for a Hugging Face directory or a model id found in the local cache (`--model-file` names it when the command does not). Nothing about the structure is needed for a run to work: without a header these sections drop out.

## The model section

Every report that found the model starts with what it is made of:

```
deepseek41, 8 layers, embedding 5120, 16 experts per layer, 6 used per token: 4.31 GiB in 222 tensors.

| part | MiB | share | a token reads, MiB | types |
|---|---:|---:|---:|---:|
| moe_experts | 2,295 | 52% | 861 | MXFP4 |
| attention | 1,381 | 31% | 1,381 | F32, Q8_0 |
| moe_shared | 287 | 6% | 287 | Q8_0 |
| token_embedding | 170 | 4% | 0 | Q8_0 |
| output_head | 170 | 4% | 170 | Q8_0 |
| lookup_tables | 98 | 2% | 0 | Q8_0 |
| mixing | 8 | 0% | 8 | Q8_0, F32 |
| moe_router | 2 | 0% | 2 | F32 |
```

`a token reads` is what one token touches: all of a dense weight, `used / total` of the experts, nothing to speak of from a table it reads one row of. A model with no output head of its own multiplies by its token embedding, so then a token reads all of that too, and the report says so. Compare the two columns. The experts are often most of a file and far less of a token, so a smaller quantization of them gains less speed than memory.

## What a number is about

| Structure or technique | Where to look | What it tells you |
|---|---|---|
| **Weights in memory** | `decode_against_memory_bound` | The time a token needs just to read its weights at the memory's peak speed. Far below decode: the time goes somewhere else, and a faster kernel layout has room. Close to it: only fewer bytes can help |
| **Mixture of Experts** | `moe_share_of_a_token`, the part table, `kernels` | How much of a token the experts are. Kernel names cannot tell an expert's matrix multiply from a dense one, so `kernels` shows both as matrix multiplies |
| **KV cache and attention** | `between_tokens_fit`, `kv_cache_cost_grows_with_context`, the memory table's `context` | What each 1000 tokens of context add between tokens. Flat for attention with a window or a compressed cache. Needs replies at many prompt lengths: `--prompts mix` has them |
| **Prompt cache, context checkpoints** | `--turns N`, `first_token_by_turn`, `prompt_cached_share` | First token as a conversation grows, and how much of each prompt the cache held. A working cache keeps it flat. vLLM and SGLang only report the cached share with `--enable-prompt-tokens-details` and `--enable-cache-report`, which `--turns` and `--reuse-prompts` add. A server that does not report it shows no share, not 0% |
| **Routing by content** | `by_content`, `speed_depends_on_content` | Speed per language and domain, with a prompt set that says them. Experts are routed by content, so a difference here can be routing. Context length differs too, so check that first |
| **Work per request** | `first_token_fit`, `first_token_fixed_part` | What a prompt costs before its tokens count: rebuilding graphs, checkpoints, the first batch |
| **Graph reuse, llama.cpp** | `engine_counters`, `graphs_reserved_per_request` | How often the server reused a graph and how often it reserved memory for a new one |
| **Batching** | `first_token_fit.per_token_ms`, `prefill_tera_params_per_s` | The cost of one more token in a batch, and how many parameters per second that reaches. The second number compares runs and hardware |
| **Speculative decoding** | `speculative`, `speculative_decoding` | Tokens drafted and accepted. Accepted tokens arrive together, so judge it by output tokens per second, not by the time between tokens |
| **Tensor split** | `gpu_time_split`, `kernels` | Time in collectives, and how much of it is waiting for other GPUs. The floor above is divided by the GPUs |
| **Layer split, pipeline** | per-GPU memory and busy tables | A token still goes through the GPUs one after the other, so the floor is not divided |
| **Waiting for the GPU** | `host_waits_for_gpu` | Which host code waits, see [deep-dives.md](deep-dives.md) |
| **Weights on the host** | `weights_on_host`, the data movement table | What stays in host buffers, and whether the page cache or the disk serves it |

## Time by what the kernels do

`llmperfscope kernels` also sums kernel time by what kernels do for the model: attention, matrix multiplies, expert routing, normalization, table lookups, KV cache writes. It goes by kernel names, which are known for llama.cpp's CUDA backend, and another engine's kernels land in `other`. An expert's matrix multiply and a dense one run the same kernel, so they stay one row. To split that row, compare with the part table: what a token reads of each part says how the reading divides. With the model's header it also says how close matrix multiplies are to the limit of the memory:

```
| matrix multiply, dense or expert | 7.30 | 91% |
| attention | 0.20 | 3% |
| normalization | 0.19 | 2% |

Matrix multiplies take 7.30 ms per token. Reading the weights a token reads takes at least 6.34 ms at the memory's peak speed, so they run at 87% of what the memory allows. Close to 100%, only fewer bytes can help, not a faster kernel.
```

## How the floor is worked out

```
decode floor = weights a token reads / (memory GB/s x GPUs sharing them)
```

The memory speed comes from the driver: bus width x max memory clock x 2, which matches the GDDR and HBM data sheets. `--memory-gb-per-s` gives a figure by hand, for a GPU whose driver does not say. A report always names the source.

It is a floor, not a forecast. Kernels get less than the peak: on real work with 8 RTX A6000 the largest matrix multiplies reached 420-490 GB/s of 768. So read the floor as a question. If decode is many times the floor, reading weights is not what limits it, and the time goes to kernels' other work, collectives, the KV cache or the host.

The KV cache is left out of the floor, because what attention reads of it depends on the kind of attention. `between_tokens_fit` measures that part instead.

## A worked example: where the first token goes

The same model and server, with prompts of 2000 tokens, with and without a fix to llama.cpp's scheduler:

```
before: waiting for the GPU: 87% of the samples, in server_context_impl::pre_decode 43%, ggml_backend_sched_alloc_graph 30%, llama_context::synchronize 10%
after:  waiting for the GPU: 55% of the samples, in server_context_impl::pre_decode 28%, ggml_backend_sched_copy_input 18%, llama_context::synchronize 7%
```

Waiting in `alloc_graph` comes before the host work of the next batch, so the GPU sits idle during that work. After the fix the wait comes at `copy_input`, after that work. The finding `graphs_reserved_per_request` gives the same lead from the server's own count: 22 reservations in 8 requests. A trace of `llama-bench` never shows this, because it keeps its first reservation.

## What this still cannot tell

- **Which expert ran.** A routing histogram needs the engine to report it. So do per-operation times: kernel names do not say which layer or role a kernel was launched for.
- **A prefill floor.** The compute a GPU can do depends on the operation, so a report gives the rate reached, not how far it is from a limit.
- **Mirrored tensors.** The floor treats all weights as shared over the GPUs of a tensor split. Tensors the engine mirrors are read by every GPU, so the true floor is a little higher.
- **Tensors the engine skips.** Parts that only run on some tokens, such as a compressor that fires per block, count as read by every token.
