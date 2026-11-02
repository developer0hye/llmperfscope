"""What a model is made of, and what that predicts.

A speed number means little without the structure behind it. A token reads every dense weight once but only the experts it was routed to, its
attention reads the KV cache up to its position, and tensor split shares all of that over the GPUs. So the weights in a file predict how much memory
a token has to read, and that gives a floor no engine can beat. How far a measurement is from the floor says whether reading weights is the problem
at all. All numbers here come from the model's own header and the GPU's own report, so a new architecture needs a line in a table, not new logic.
"""

from __future__ import annotations

import glob
import json
import os
import re
import struct
from pathlib import Path

from . import gguf

# Which part of the model a tensor belongs to. The first match wins, so the specific names go first.
# GGUF names are llama.cpp's; experts are one tensor per layer with the expert number as the last dimension.
GGUF_PARTS = [
    ("vision_audio", r"^(v\.|a\.|mm\.|vision|audio)"),
    ("moe_router", r"\.(ffn_gate_inp(_shexp)?|exp_probs_b(_vl)?|ffn_gate_tid2eid)\b"),
    ("moe_experts", r"\.(ffn_\w+_(exps|chexps)|attn_v_exps)\b"),
    ("moe_shared", r"\.ffn_\w+_shexp\b"),
    ("norm", r"norm"),
    ("mixing", r"\.hc_"),
    ("lookup_tables", r"^per_layer_token_embd|\.ple_"),
    ("token_embedding", r"^(token_embd|tok_|pos_embd|token_types)"),
    ("output_head", r"^(output|cls)(\.|$)"),
    ("attention", r"\.(attn_|indexer\.|indexer_|engram_|nextn\.)"),
    ("feed_forward", r"\.ffn_"),
]
# Hugging Face names, where each expert has its own tensors.
HF_PARTS = [
    ("vision_audio", r"vision|visual|audio"),
    ("moe_router", r"mlp\.gate\b"),
    ("moe_shared", r"shared_experts?"),
    ("moe_experts", r"\.experts\."),
    ("norm", r"norm"),
    ("token_embedding", r"embed_tokens|\bwte\b|word_embeddings|embedding\.weight"),
    ("output_head", r"lm_head|embed_out"),
    ("attention", r"self_attn|\.attn\.|attention"),
    ("feed_forward", r"\.mlp\.|feed_forward"),
]
# Keys of a Hugging Face config.json, as different models spell them.
HF_KEYS = {
    "layers": ("num_hidden_layers", "n_layer", "num_layers"),
    "embedding": ("hidden_size", "n_embd", "d_model"),
    "heads": ("num_attention_heads", "n_head"),
    "heads_kv": ("num_key_value_heads",),
    "vocab": ("vocab_size",),
    "context_trained": ("max_position_embeddings",),
    "experts": ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts", "ffn_expert_count"),
    "experts_used": ("num_experts_per_tok", "num_experts_per_topk", "moe_top_k"),
    "experts_shared": ("n_shared_experts", "num_shared_experts"),
}
# Read for one row each, not as a whole: what a token reads of them is too small to count.
LOOKED_UP = {"token_embedding", "lookup_tables"}
MIB = 1024.0 * 1024.0


def _part_of(name, table):
    for part, pattern in table:
        if re.search(pattern, name):
            return part
    return "other"


def _elements(dims):
    count = 1
    for dim in dims:
        count *= dim
    return count


def _summarize(tensors, table, experts, experts_used):
    """Per part: size, types, and how much of it one token reads."""
    parts = {}
    for tensor in tensors:
        part = _part_of(tensor["name"], table)
        entry = parts.setdefault(part, {"part": part, "mib": 0.0, "tensors": 0, "types": {}, "read_mib": 0.0, "active_params": 0.0})
        mib = tensor["bytes"] / MIB
        entry["mib"] += mib
        entry["tensors"] += 1
        entry["types"][tensor["type"]] = round(entry["types"].get(tensor["type"], 0.0) + mib, 1)
        share = 1.0
        if part == "moe_experts" and experts and experts_used:
            share = experts_used / experts
        elif part in LOOKED_UP:
            share = 0.0
        entry["read_mib"] += mib * share
        entry["active_params"] += _elements(tensor["dims"]) * share
    rows = sorted(parts.values(), key=lambda row: -row["mib"])
    total = sum(row["mib"] for row in rows) or 1.0
    for row in rows:
        row["share"] = round(row["mib"] / total, 4)
        row["mib"] = round(row["mib"], 1)
        row["read_mib"] = round(row["read_mib"], 2)
        row["active_params"] = int(row["active_params"])
    return rows


def _with_totals(out):
    parts = out["parts"]
    out["weights_read_per_token_mib"] = round(sum(row["read_mib"] for row in parts), 2)
    out["active_params_per_token"] = sum(row["active_params"] for row in parts)
    return out


def from_gguf(path):
    model = gguf.read(path)
    metadata = model["metadata"]
    arch = metadata.get("general.architecture", "")

    def key(name):
        return metadata.get(f"{arch}.{name}")

    def number(value):
        # Some models have one value per layer here, as an array: the largest says what the keys are about.
        if isinstance(value, list):
            value = max(value) if value else None
        return value if isinstance(value, (int, float)) else None

    vocab = key("vocab_size")
    tokens = metadata.get("tokenizer.ggml.tokens")
    if vocab is None and isinstance(tokens, dict):
        vocab = tokens.get("array_length")
    experts = number(key("expert_count"))
    used = number(key("expert_used_count"))
    return _with_totals({
        "source": "gguf",
        "name": metadata.get("general.name") or os.path.basename(str(path)),
        "architecture": arch,
        "file_mib": round(model["file_bytes"] / MIB, 1),
        "tensors": len(model["tensors"]),
        "layers": number(key("block_count")),
        "embedding": number(key("embedding_length")),
        "heads": number(key("attention.head_count")),
        "heads_kv": number(key("attention.head_count_kv")),
        "vocab": vocab,
        "context_trained": number(key("context_length")),
        "experts": experts,
        "experts_used": used,
        "experts_shared": number(key("expert_shared_count")),
        "parts": _summarize(model["tensors"], GGUF_PARTS, experts, used),
    })


def _safetensors_header(path):
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
    return [{"name": name, "dims": entry["shape"], "type": entry["dtype"], "bytes": entry["data_offsets"][1] - entry["data_offsets"][0]}
            for name, entry in header.items() if name != "__metadata__"]


def from_directory(directory):
    """A Hugging Face model: config.json for the shape, the safetensors headers for the tensors. Quantized formats that keep their weights
    elsewhere show the config only."""
    directory = Path(directory)
    with open(directory / "config.json") as f:
        config = json.load(f)
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}

    def key(name):
        for spelled in HF_KEYS[name]:
            for source in (config, text):
                if isinstance(source.get(spelled), (int, float)):
                    return source[spelled]
        return None

    tensors = []
    for path in sorted(glob.glob(str(directory / "*.safetensors"))):
        tensors += _safetensors_header(path)
    experts, used = key("experts"), key("experts_used")
    out = {"source": "huggingface", "name": directory.name, "architecture": config.get("model_type"),
           "file_mib": round(sum(t["bytes"] for t in tensors) / MIB, 1), "tensors": len(tensors),
           **{name: key(name) for name in ("layers", "embedding", "heads", "heads_kv", "vocab", "context_trained", "experts_shared")},
           "experts": experts, "experts_used": used, "parts": _summarize(tensors, HF_PARTS, experts, used)}
    return _with_totals(out)


def _hub_snapshot(model_id, environ):
    """The local copy of a model named like org/name, if it has one."""
    if "/" not in model_id or model_id.startswith(("/", ".")):
        return None
    home = environ.get("HF_HOME") or os.path.join(environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "huggingface")
    hub = environ.get("HF_HUB_CACHE") or os.path.join(home, "hub")
    snapshots = glob.glob(os.path.join(hub, "models--" + model_id.replace("/", "--"), "snapshots", "*", "config.json"))
    if not snapshots:
        return None
    return os.path.dirname(max(snapshots, key=os.path.getmtime))


def describe(model, environ=None):
    """The structure of a model given as a GGUF file, a directory or a Hugging Face name. None when it cannot be found: it is a help to the
    numbers, and a run must not depend on it."""
    if not model:
        return None
    environ = environ if environ is not None else os.environ
    try:
        if str(model).endswith(".gguf") and os.path.isfile(model):
            return from_gguf(model)
        if os.path.isfile(os.path.join(str(model), "config.json")):
            return from_directory(model)
        snapshot = _hub_snapshot(str(model), environ)
        return from_directory(snapshot) if snapshot else None
    except (OSError, ValueError, KeyError, TypeError, struct.error) as e:
        return {"error": f"could not read the model's structure: {e!r}"[:200]}


def expectations(run):
    """What the structure predicts for this run, next to what was measured. None without the numbers to work it out."""
    model = run.get("model") or {}
    read = model.get("weights_read_per_token_mib")
    if not read:
        return None
    config = run.get("engine_config") or {}
    out = {"weights_read_per_token_mib": read, "active_params_per_token": model.get("active_params_per_token")}
    throughput = run.get("throughput") or {}
    decode = (throughput.get("decode") or {}).get("tokens_per_s")
    prefill = (throughput.get("prefill") or {}).get("tokens_per_s")
    if not decode:
        # A server without timers of its own: the time between tokens as the client saw it, unless accepted drafts arrive together.
        levels = sorted((t for t in throughput.values() if isinstance(t, dict) and "concurrency" in t), key=lambda t: t["concurrency"])
        gap = ((levels[0].get("itl_ms") or {}).get("p50") if levels and not levels[0].get("speculative") else None)
        decode = 1000.0 / gap if gap else None
    if prefill and model.get("active_params_per_token"):
        out["prefill_tera_params_per_s"] = round(model["active_params_per_token"] * prefill / 1e12, 3)
    bandwidth = memory_bandwidth(run)
    if not bandwidth:
        return out
    gpus = max(len(run.get("devices_used") or []), 1)
    # Tensor split shares the weights, each GPU reads its slice. With a layer split a token still goes through one GPU after the other.
    sharing = gpus if config.get("split_mode") == "tensor" else 1
    floor_ms = read * MIB / (bandwidth["gb_per_s"] * 1e9 * sharing) * 1000.0
    out.update({"memory_bandwidth": bandwidth, "gpus_sharing": sharing, "decode_floor_ms": round(floor_ms, 3)})
    if decode:
        measured = 1000.0 / decode
        out["decode_ms"] = round(measured, 3)
        out["decode_share_of_time_reading"] = round(floor_ms / measured, 3)
    return out


def memory_bandwidth(run):
    """What the run's GPUs can read per second at most: a figure given by hand, or what the driver says, which is not the same as what kernels get."""
    given = run.get("memory_gb_per_s_given")
    if given:
        return {"gb_per_s": float(given), "source": "given on the command line"}
    used = set(run.get("devices_used") or [])
    speeds = [d["memory_gb_per_s"] for d in run.get("devices") or [] if d.get("memory_gb_per_s") and (not used or d.get("index") in used)]
    return {"gb_per_s": min(speeds), "source": "driver: memory bus width x max clock"} if speeds else None
