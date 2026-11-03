"""Load driver for any OpenAI-compatible server (llama-server, vLLM, SGLang, TGI, Ollama, ...). The same client measures the
same things everywhere, which is what makes numbers comparable across engines: time to first token, the time between
tokens, and throughput under a given concurrency."""

from __future__ import annotations

import http.client
import json
import math
import os
import queue
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Public prompts in several languages, domains and lengths. Pinned, so a number measured with it today means the same next year. Each row
# carries its own license and source.
MIX_REVISION = "9460b6e9195f0ccb48562a945464f57fdac1b25b"
MIX_URL = "https://huggingface.co/datasets/developer0hye/llm-speed-prompt-mix/resolve/{revision}/data/{name}.jsonl"
MIX_FILES = ("belebele", "gsm8k_en", "hrm8k_ko", "humaneval", "humaneval_x", "long_code", "long_wiki_en", "long_wiki_ko", "mbpp", "mgsm", "mmlu_en", "mmmlu", "mt_bench")


def prompt_mix_path(directory=None, opener=None):
    """The prompt set as one local JSONL file, downloaded the first time."""
    root = Path(directory) if directory else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "llmperfscope"
    path = root / f"prompt-mix-{MIX_REVISION[:10]}.jsonl"
    if path.exists():
        return path
    root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp{os.getpid()}")
    print(f"[llmperfscope] downloading the prompt set to {path}", flush=True)
    try:
        with open(temporary, "wb") as out:
            for name in MIX_FILES:
                url = MIX_URL.format(revision=MIX_REVISION, name=name)
                with (opener or urllib.request.urlopen)(url, timeout=120) as response:
                    data = response.read()
                out.write(data if data.endswith(b"\n") else data + b"\n")
        os.replace(temporary, path)
    except OSError as e:
        if temporary.exists():
            temporary.unlink()
        raise SystemExit(f"could not download the prompt set: {e}. Pass your own file with --prompts.") from e
    return path


def percentile(values, q):
    """Nearest rank."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q / 100 * len(ordered)) - 1))
    return ordered[index]


def distribution(values, unit=1000.0, digits=2):
    """values in seconds -> mean and percentiles in ms by default."""
    if not values:
        return None
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values) * unit, digits),
        "p50": round(percentile(values, 50) * unit, digits),
        "p90": round(percentile(values, 90) * unit, digits),
        "p99": round(percentile(values, 99) * unit, digits),
        "max": round(max(values) * unit, digits),
    }


def load_prompts(path, filters=None):
    """JSONL with `prompt` (string) or `messages` (chat) per row; extra columns can be used in `filters`."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not ("prompt" in row or "messages" in row):
                raise ValueError(f"{path}: row without prompt or messages")
            if all(str(row.get(k)) == v for k, v in (filters or {}).items()):
                rows.append(row)
    return rows


# About 512 tokens of neutral English, for when no prompt file is given. Length scales by repetition.
_FILLER = ("The history of measurement is a history of agreeing on what a number means. A careful observer records what was "
           "done, under which conditions, and how much the result varies when nothing changes. ")


def synthetic_prompts(words, count):
    unit = _FILLER.split()
    return [{"prompt": f"[{i}] " + " ".join((unit * (words // len(unit) + 1))[:words]) + "\nSummarize the text above."}
            for i in range(count)]


def _text_of(choice):
    """The new text in a streamed choice. Reasoning models stream into a separate field; those are tokens too."""
    if "text" in choice:
        return choice["text"]
    delta = choice.get("delta") or {}
    return delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or ""


class Client:
    def __init__(self, base_url, model=None, max_tokens=64, extra_body=None, timeout_s=1800.0, ignore_eos=True, api_key=None):
        parsed = urllib.parse.urlsplit(base_url)
        self.host = parsed.hostname
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.https = parsed.scheme == "https"
        self.path = parsed.path.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self.timeout_s = timeout_s
        self.ignore_eos = ignore_eos
        self.api_key = api_key

    def _connection(self):
        kind = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        return kind(self.host, self.port, timeout=self.timeout_s)

    def discover_model(self):
        with urllib.request.urlopen(f"{'https' if self.https else 'http'}://{self.host}:{self.port}{self.path}/models",
                                    timeout=30) as response:
            data = json.load(response)
        self.model = data["data"][0]["id"]
        self.models_listed = len(data["data"])
        return self.model

    def request(self, row, nonce=""):
        """One streamed request. Returns timing per token as seen by the client. A nonce in front of the prompt keeps a server's prefix
        cache from answering, so prefill is really measured; without it the second load level would mostly hit the cache."""
        body = {"stream": True, "stream_options": {"include_usage": True}, "max_tokens": row.get("max_tokens", self.max_tokens),
                "temperature": 0}
        if self.model:
            body["model"] = self.model
        if self.ignore_eos:
            body["ignore_eos"] = True
        if "messages" in row:
            body["messages"] = _with_nonce(row["messages"], nonce)
            path = f"{self.path}/chat/completions"
        else:
            body["prompt"] = nonce + row["prompt"]
            path = f"{self.path}/completions"
        body.update(self.extra_body)
        body.update(row.get("extra_body") or {})
        payload = json.dumps(body).encode()
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        result = {"id": row.get("id"), "ok": False, "chunks": 0, "itl_s": []}
        for column in CONTENT_COLUMNS:
            if row.get(column):
                result[column] = row[column]
        connection = self._connection()
        t_send = time.monotonic()
        result["t_send"] = t_send
        try:
            connection.request("POST", path, body=payload, headers=headers)
            response = connection.getresponse()
            if response.status != 200:
                result["error"] = f"HTTP {response.status}: {response.read(300).decode('utf-8', 'replace')}"
                return result
            t_previous = None
            t_first_event = None
            finish = None
            while True:
                line = response.readline()
                if not line:
                    break
                t_now = time.monotonic()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    result["usage"] = event["usage"]
                    cached = (event["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
                    if cached is not None:
                        result["cached_tokens"] = cached
                if event.get("timings"):
                    result["server_timings"] = event["timings"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                finish = choices[0].get("finish_reason") or finish
                if t_first_event is None:
                    t_first_event = t_now
                text = _text_of(choices[0])
                if text:
                    if row.get("keep_text"):
                        result["text"] = result.get("text", "") + text
                    result["chunks"] += 1
                    if t_previous is None:
                        result["ttft_s"] = t_now - t_send
                    else:
                        result["itl_s"].append(t_now - t_previous)
                    t_previous = t_now
            result["t_done"] = time.monotonic()
            result["finish_reason"] = finish
            result["ok"] = result["chunks"] > 0
            if not result["ok"] and (result.get("usage") or {}).get("completion_tokens") and t_first_event is not None:
                # The tokens exist but are not shown, as with a model's hidden reasoning. No per-token times, but the request worked.
                result["ok"] = True
                result["hidden_tokens"] = True
                result["ttft_s"] = t_first_event - t_send
            if not result["ok"]:
                result["error"] = "the stream ended without a token"
        except (OSError, http.client.HTTPException, ValueError) as e:
            result["error"] = repr(e)
        finally:
            connection.close()
        return result


def _with_nonce(messages, nonce):
    if not nonce:
        return messages
    messages = [dict(m) for m in messages]
    for message in messages:
        if isinstance(message.get("content"), str):
            message["content"] = nonce + message["content"]
            break
    return messages


def run_load(client, rows, concurrency=1, repeat=1, seed=0, emit_each=None, level="", unique=True):
    """Closed loop: `concurrency` clients, each sending its next request when the previous one finishes."""
    work = [row for _ in range(repeat) for row in rows]
    random.Random(seed).shuffle(work)
    todo = queue.Queue()
    for i, row in enumerate(work):
        todo.put((row, f"[{level}.{i}] " if unique else ""))
    results = []
    lock = threading.Lock()

    def worker():
        while True:
            try:
                row, nonce = todo.get_nowait()
            except queue.Empty:
                return
            result = client.request(row, nonce)
            with lock:
                results.append(result)
                if emit_each:
                    emit_each(result)

    t0 = time.monotonic()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results, time.monotonic() - t0


def run_conversations(client, rows, turns, emit_each=None):
    """One client talks to the server: each row's question opens a turn, on top of the questions and replies before it, `turns` to a conversation.
    Every request repeats all of the conversation, which is what a server's prompt cache or context checkpoints are there for."""
    results = []
    t0 = time.monotonic()
    for start in range(0, len(rows) - turns + 1, turns):
        messages = []
        for turn, row in enumerate(rows[start:start + turns]):
            asked = row["messages"] if "messages" in row else [{"role": "user", "content": row["prompt"]}]
            messages = messages + [m for m in asked if m.get("role") != "assistant"]
            result = client.request({"id": row.get("id"), "messages": messages, "keep_text": True, "extra_body": {"cache_prompt": True}})
            result["turn"] = turn
            text = result.pop("text", "")
            results.append(result)
            if emit_each:
                emit_each(result)
            if not result.get("ok"):
                break
            messages = messages + [{"role": "assistant", "content": text}]
    return results, time.monotonic() - t0


def by_turn(results):
    """First token and what the cache held, per turn of a conversation."""
    turns = {}
    for r in results:
        if r.get("ok") and r.get("turn") is not None:
            turns.setdefault(r["turn"], []).append(r)
    rows = []
    for turn, items in sorted(turns.items()):
        prompt = sum((r.get("usage") or {}).get("prompt_tokens") or 0 for r in items)
        cached = sum(r.get("cached_tokens") or 0 for r in items)
        # A server that does not report it says nothing about its cache: that is not the same as nothing cached.
        known = prompt and any(r.get("cached_tokens") is not None for r in items)
        rows.append({"turn": turn + 1, "requests": len(items), "prompt_tokens_mean": round(prompt / len(items)),
                     "ttft_p50_ms": round(percentile([r["ttft_s"] for r in items], 50) * 1000.0, 2),
                     "cached_share": round(cached / prompt, 3) if known else None})
    return rows


# Columns of a prompt set that say what a prompt is about. What a model does can depend on it: experts get routed by content, table lookups miss
# by language.
CONTENT_COLUMNS = ("language", "domain")
CONTENT_MIN_REQUESTS = 5


def by_content(results):
    """Per language and domain: the time between tokens and per prompt token, for groups with enough requests to say something."""
    out = {}
    for column in CONTENT_COLUMNS:
        groups = {}
        for r in results:
            if r.get("ok") and r.get(column):
                groups.setdefault(r[column], []).append(r)
        rows = []
        for name, items in groups.items():
            gaps = [gap for r in items for gap in (r.get("itl_s") or [])[3:]]
            per_token = [r["ttft_s"] / tokens for r in items if (tokens := (r.get("usage") or {}).get("prompt_tokens")) and r.get("ttft_s") is not None]
            if len(items) >= CONTENT_MIN_REQUESTS and gaps and per_token:
                rows.append({"group": name, "requests": len(items), "itl_p50_ms": round(percentile(gaps, 50) * 1000.0, 3),
                             "first_token_ms_per_prompt_token": round(percentile(per_token, 50) * 1000.0, 3)})
        if len(rows) > 1:
            out[column] = sorted(rows, key=lambda row: row["itl_p50_ms"])
    return {"by_content": out} if out else {}


# Upper ends of the prompt length groups of `first_token_by_length`, in tokens.
LENGTH_GROUPS = (100, 300, 1000, 4000)
FIT_MIN_REQUESTS = 8
FIT_MAX_POINTS = 400


def _line_through(points):
    """(slope, intercept) by Theil-Sen: a median of slopes, so neither a few long prompts nor a few slow requests decide it."""
    sample = sorted(points)[::max(1, len(points) // FIT_MAX_POINTS)]
    slopes = [(b[1] - a[1]) / (b[0] - a[0]) for i, a in enumerate(sample) for b in sample[i + 1:] if b[0] != a[0]]
    slope = percentile(slopes, 50)
    return slope, percentile([y - slope * x for x, y in points], 50)


def _can_fit(points):
    lengths = [x for x, _ in points]
    return len(points) >= FIT_MIN_REQUESTS and max(lengths) >= 2 * min(lengths)


def _by_length(points):
    """Mean and median per group of lengths."""
    groups = []
    low = 0
    for high in LENGTH_GROUPS + (None,):
        values = [y for x, y in points if x >= low and (high is None or x < high)]
        if values:
            groups.append({"tokens": f"{low}+" if high is None else f"{low}-{high - 1}", "requests": len(values),
                           "mean_ms": round(sum(values) / len(values), 2), "p50_ms": round(percentile(values, 50), 2)})
        low = high
    return groups


def first_token_by_length(results):
    """First token against the tokens the server had to process for it, as a table and as a line. The line says what a prompt costs
    before its tokens count: work per request that a benchmark with long prompts hides, and most of what someone with a short question
    waits for."""
    points = []
    for r in results:
        if not r.get("ok") or r.get("ttft_s") is None:
            continue
        tokens = ((r.get("usage") or {}).get("prompt_tokens") or 0) - (r.get("cached_tokens") or 0)
        if tokens > 0:
            points.append((tokens, r["ttft_s"] * 1000.0))
    if not points:
        return {}
    out = {"first_token_by_length": _by_length(points)}
    if _can_fit(points):
        per_token, fixed = _line_through(points)
        typical = percentile([x for x, _ in points], 50)
        total = fixed + per_token * typical
        out["first_token_fit"] = {"fixed_ms": round(fixed, 2), "per_token_ms": round(per_token, 4), "requests": len(points),
                                  "median_prompt_tokens": round(typical),
                                  "fixed_share_at_median": round(max(fixed, 0.0) / total, 3) if total > 0 else None}
    return out


def between_tokens_by_context(results, skip=3):
    """The time between tokens against how much context a token has behind it, one point per reply. A token's attention reads the KV cache up
    to its position, so this line is what the cache costs, and it stays flat for attention with a window or a compressed cache. The first gaps
    of a reply are left out: they hold graph rebuilds and cache setup, not the token."""
    points = []
    for r in results:
        gaps = r.get("itl_s") or []
        tokens = (r.get("usage") or {}).get("prompt_tokens")
        if r.get("ok") and tokens and len(gaps) > skip:
            kept = gaps[skip:]
            points.append((tokens + skip + len(kept) / 2.0, percentile(kept, 50) * 1000.0))
    if not points:
        return {}
    out = {"between_tokens_by_context": _by_length(points)}
    if _can_fit(points):
        slope, base = _line_through(points)
        out["between_tokens_fit"] = {"base_ms": round(base, 3), "per_1000_context_tokens_ms": round(slope * 1000.0, 3), "requests": len(points),
                                     "median_context_tokens": round(percentile([x for x, _ in points], 50))}
    return out


def speculative_summary(results):
    """Tokens drafted and accepted, when the server reports them."""
    timings = [r["server_timings"] for r in results if r.get("server_timings")]
    drafted = sum(t.get("draft_n") or 0 for t in timings)
    if not drafted:
        return {}
    accepted = sum(t.get("draft_n_accepted") or 0 for t in timings)
    return {"speculative": {"drafted": drafted, "accepted": accepted, "accepted_share": round(accepted / drafted, 3)}}


def summarize_load(results, wall_s):
    """What a client saw. Token counts come from the server's usage when it sends one, otherwise from stream chunks."""
    ok = [r for r in results if r["ok"]]
    out = {"requests": len(results), "failed": len(results) - len(ok), "wall_s": round(wall_s, 3)}
    errors = [r["error"] for r in results if r.get("error")]
    if errors:
        out["first_error"] = errors[0][:300]
    if not ok:
        return out
    completion = prompt = cached = 0
    exact = True
    per_request_decode = []
    per_request_prefill = []
    itl = []
    for r in ok:
        usage = r.get("usage") or {}
        tokens = usage.get("completion_tokens")
        if tokens is None:
            tokens = r["chunks"]
            exact = False
        completion += tokens
        cached += r.get("cached_tokens") or 0
        prompt_tokens = usage.get("prompt_tokens")
        if prompt_tokens:
            prompt += prompt_tokens
            per_request_prefill.append(prompt_tokens / r["ttft_s"])
        itl += r["itl_s"]
        if r["itl_s"]:
            per_request_decode.append(len(r["itl_s"]) / sum(r["itl_s"]))
    timings = [r["server_timings"] for r in ok if r.get("server_timings")]
    for key, count, spent in (("server_prefill_tokens_per_s", "prompt_n", "prompt_ms"), ("server_decode_tokens_per_s", "predicted_n", "predicted_ms")):
        milliseconds = sum(t.get(spent) or 0.0 for t in timings)
        if milliseconds:
            out[key] = round(sum(t.get(count) or 0 for t in timings) / milliseconds * 1000, 3)
    hidden = sum(1 for r in ok if r.get("hidden_tokens"))
    if hidden:
        out["requests_with_hidden_tokens"] = hidden
    out.update({
        "completion_tokens": completion,
        "prompt_tokens": prompt,
        "prompt_tokens_cached": cached,
        "tokens_exact": exact,
        "output_tokens_per_s": round(completion / wall_s, 3) if wall_s > 0 else None,
        "ttft_ms": distribution([r["ttft_s"] for r in ok]),
        "itl_ms": distribution(itl),
        "decode_tokens_per_s_mean": round(sum(per_request_decode) / len(per_request_decode), 3) if per_request_decode else None,
        "prefill_tokens_per_s_mean": round(sum(per_request_prefill) / len(per_request_prefill), 3) if per_request_prefill else None,
    })
    out.update(first_token_by_length(ok))
    out.update(between_tokens_by_context(ok))
    out.update(speculative_summary(ok))
    if prompt:
        out["prompt_cached_share"] = round(cached / prompt, 3)
    turns = by_turn(ok)
    if turns:
        out["by_turn"] = turns
    out.update(by_content(ok))
    return out


def wait_ready(url, is_alive, timeout_s=3600.0, poll_s=0.2):
    """Polls a health URL until it answers 200. Returns the time it took."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if not is_alive():
            raise RuntimeError("the server exited while loading")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return time.monotonic() - t0
        except (urllib.error.URLError, OSError, http.client.HTTPException):
            pass
        time.sleep(poll_s)
    raise TimeoutError(f"{url} did not become ready in {timeout_s:.0f} s")


def fetch_metrics(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        return None


def pull_ollama_model(base, model):
    """Ollama fetches models over its own API; the stream of progress ends when the download is done."""
    request = urllib.request.Request(f"{base}/api/pull", data=json.dumps({"model": model}).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        for line in response:
            event = json.loads(line)
            if event.get("error"):
                raise RuntimeError(f"ollama pull: {event['error']}")


class GaugeWatcher(threading.Thread):
    """Polls the server's metrics during a load and keeps the peak of every usage gauge. Counters can be read at the end, but how
    full a cache pool got can only be seen while it happens."""

    WORDS = {"usage", "utilization", "perc", "percent", "ratio", "waiting", "running", "processing", "queue", "pending", "deferred"}
    NOT_GAUGES = ("_total", "_sum", "_count", "_bucket", "_created", "_info")

    def __init__(self, url, period_s=1.0):
        super().__init__(daemon=True, name="llmperfscope-gauges")
        self.url = url
        self.period_s = period_s
        self.peaks = {}
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()
        self.join(timeout=10)

    def run(self):
        while not self._stop_event.is_set():
            text = fetch_metrics(self.url)
            if text:
                for name, value in parse_prometheus(text).items():
                    if self.is_usage_gauge(name):
                        self.peaks[name] = max(self.peaks.get(name, value), value)
            self._stop_event.wait(self.period_s)

    @classmethod
    def is_usage_gauge(cls, name):
        """Whole words only: `ratio` must not match `generation`, and timestamps must not pass for gauges."""
        base = name.split("{")[0].lower()
        if base.endswith(cls.NOT_GAUGES):
            return False
        return bool(cls.WORDS & set(re.split(r"[_:.]", base)))


def parse_prometheus(text):
    """name{labels} -> value."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out


def metrics_delta(before, after):
    """Counters that moved during the run. Engine-specific names pass through untouched."""
    a, b = parse_prometheus(before or ""), parse_prometheus(after or "")
    return {k: round(b[k] - a.get(k, 0.0), 6) for k in b if b[k] != a.get(k, 0.0)}
