#!/usr/bin/env python3
"""Benchmark HiCache ``layer_first`` vs ``page_first_kv_split``: TTFT + throughput.

Three-round scenario driven by the caller's server layout (the script is
layout-agnostic; the layout is only echoed in the output labels)::

  Round1 (fill)   : ``num_prompts`` unique prompts fill the L1 device pool.
  Round2 (evict)  : ``num_prompts`` different prompts overflow L1, LRU-evicting
                    Round1's KV into the L2 host pool (D2H).
  Round3 (replay) : re-issue Round1's identical prompts; their prefix is now an
                    L1 miss + L2 hit, so the pool restores it (H2D).

Each round sends the ``num_prompts`` requests concurrently (``ThreadPoolExecutor``)
with ``stream=True`` and measures:

  - TTFT (time-to-first-token): request send -> first delta chunk, a proxy for
    the prefill cost (which includes D2H on write_through / H2D on replay).
  - decode throughput: total output tokens / round wall-clock.

Round3 output is checked against the Round1 needles as a soft data-integrity
signal (not a hard pass/fail gate, since this is a perf benchmark).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

try:
    from transformers import AutoTokenizer
except Exception:  # noqa: BLE001
    AutoTokenizer = None

_QUERY = "Question: what is the secret passcode?\nAnswer:"
_ROUND2_QUERY = "Question: what is 2 + 3?\nAnswer:"


def _filler_to_tokens(tok, sentence: str, target_tokens: int) -> str:
    """Repeat ``sentence`` until the concatenation reaches ``target_tokens``."""
    per = len(tok.encode(sentence, add_special_tokens=False))
    n = max(1, target_tokens // per)
    text = sentence * n
    while len(tok.encode(text, add_special_tokens=False)) < target_tokens:
        text += sentence
    return text


def _build_prompt(tok, needle_sentence: str, filler_sentence: str, prompt_tokens: int, query: str) -> str:
    needle_tokens = len(tok.encode(needle_sentence, add_special_tokens=False))
    doc = needle_sentence + _filler_to_tokens(
        tok, filler_sentence, max(1, prompt_tokens - needle_tokens)
    )
    return doc + "\n\n" + query


def stream_completion(base_url: str, prompt: str, max_tokens: int, timeout: int = 1200):
    """Stream one completion; return (text, ttft_s, total_s)."""
    payload = json.dumps(
        {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stop": ["Question:"],
        }
    ).encode("utf-8")
    url = base_url.rstrip("/") + "/v1/completions"
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.monotonic()
    ttft = None
    parts = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if ttft is None:
                ttft = time.monotonic() - t0
            choices = obj.get("choices") or []
            if choices:
                parts.append(choices[0].get("text") or "")
    total = time.monotonic() - t0
    text = "".join(parts)
    return text, (ttft if ttft is not None else total), total


def run_round(base_url, prompts, max_tokens, num_prompts, label, tok):
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=num_prompts) as ex:
        futs = [ex.submit(stream_completion, base_url, p, max_tokens) for p in prompts]
        results = [f.result() for f in futs]
    wall = time.monotonic() - t0

    texts = [r[0] for r in results]
    ttfts_ms = [r[1] * 1000.0 for r in results]
    comp_tokens = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
    total_tokens = sum(comp_tokens)
    mean_ttft = sum(ttfts_ms) / len(ttfts_ms)
    median_ttft = sorted(ttfts_ms)[len(ttfts_ms) // 2]
    throughput = total_tokens / wall if wall > 0 else 0.0

    print(
        f"[{label:18s}] wall={wall:6.3f}s  mean_ttft={mean_ttft:7.1f}ms  "
        f"median_ttft={median_ttft:7.1f}ms  tokens={total_tokens:4d}  "
        f"throughput={throughput:7.2f} tok/s",
        flush=True,
    )
    return {
        "wall": wall,
        "mean_ttft_ms": mean_ttft,
        "median_ttft_ms": median_ttft,
        "total_tokens": total_tokens,
        "throughput": throughput,
        "texts": texts,
    }


def build_prompts(tok, num, prompt_tokens, filler_a, filler_b, seed):
    """Unique prompts per ``seed`` so each round is a fresh L1/L2 set."""
    prompts_a = [
        _build_prompt(tok, f"The secret passcode is {seed + i}. ", filler_a, prompt_tokens, _QUERY)
        for i in range(num)
    ]
    prompts_b = [
        _build_prompt(
            tok, f"The unrelated record number is {seed + 100000 + i}. ", filler_b, prompt_tokens, _ROUND2_QUERY
        )
        for i in range(num)
    ]
    return prompts_a, prompts_b


def run_scenario(base_url, tok, prompts_a, prompts_b, max_tokens, num, seed, tag):
    """One full fill -> evict -> replay cycle; returns the three round stats."""
    r1 = run_round(base_url, prompts_a, max_tokens, num, f"{tag} fill", tok)
    r2 = run_round(base_url, prompts_b, max_tokens, num, f"{tag} evict", tok)
    r3 = run_round(base_url, prompts_a, max_tokens, num, f"{tag} replay", tok)
    # Soft integrity: replay must reproduce each fill needle.
    matched = sum(str(seed + i) in r3["texts"][i] for i in range(num))
    return r1, r2, r3, matched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:32345")
    ap.add_argument("--layout", default="layer_first", help="echoed label only")
    ap.add_argument("--server-log", default=None, help="optional, prints D2H/H2D counts")
    ap.add_argument(
        "--model-path", default="/mnt/paas/weights/DeepSeek-Coder-V2-Lite-Instruct"
    )
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--prompt-tokens", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--warmup-rounds", type=int, default=1)
    ap.add_argument("--measure-rounds", type=int, default=3)
    args = ap.parse_args()

    if AutoTokenizer is None:
        print("ERROR: transformers.AutoTokenizer is required.", file=sys.stderr)
        return 2

    tok = AutoTokenizer.from_pretrained(args.model_path)

    filler_a = (
        "The wind carried dry leaves across the empty field while a distant "
        "train passed under the grey evening sky. "
    )
    # Different filler so Round2 never shares/hits Round1's L2 entries.
    filler_b = (
        "A small boat drifted along the quiet river as fishermen pulled in "
        "their nets beneath the rising morning sun. "
    )

    num = args.num_prompts

    print(
        f"bench: layout={args.layout} num_prompts={num} prompt_tokens~={args.prompt_tokens} "
        f"max_tokens={args.max_tokens} warmup_rounds={args.warmup_rounds} "
        f"measure_rounds={args.measure_rounds}",
        flush=True,
    )

    def scenario(seed, tag):
        prompts_a, prompts_b = build_prompts(
            tok, num, args.prompt_tokens, filler_a, filler_b, seed
        )
        return run_scenario(
            args.base_url, tok, prompts_a, prompts_b, args.max_tokens, num, seed, tag
        )

    # Warmup (results discarded): flush cold-start / first-request kernels.
    for w in range(args.warmup_rounds):
        print(f"warmup round {w + 1}/{args.warmup_rounds} (discarded)", flush=True)
        scenario(w * 1000000, f"warmup{w}")

    # Measured rounds, each with unique needles so replay is a true L2 hit.
    rounds = []
    for m in range(args.measure_rounds):
        seed = 1000 * (m + 1)
        rounds.append(scenario(seed, f"m{m + 1}"))

    phases = ("fill", "evict", "replay")
    stat = {p: [] for p in phases}
    matched_total = 0
    for r1, r2, r3, matched in rounds:
        stat["fill"].append(r1)
        stat["evict"].append(r2)
        stat["replay"].append(r3)
        matched_total += matched

    print("=" * 70)
    print(
        f"layout={args.layout}  warmup_rounds={args.warmup_rounds}  "
        f"measure_rounds={args.measure_rounds}  (averaged over {args.measure_rounds} runs)"
    )
    print("phase          mean_ttft_ms  median_ttft_ms  tokens  throughput(tok/s)")
    for p in phases:
        rows = stat[p]
        n = len(rows)
        mean_ttft = sum(r["mean_ttft_ms"] for r in rows) / n
        median_ttft = sum(r["median_ttft_ms"] for r in rows) / n
        total_tokens = sum(r["total_tokens"] for r in rows) / n
        throughput = sum(r["throughput"] for r in rows) / n
        print(
            f"{p:14s} {mean_ttft:12.1f} {median_ttft:15.1f} "
            f"{total_tokens:6.0f} {throughput:17.2f}"
        )
    print(f"Round3 needle matches: {matched_total}/{num * args.measure_rounds}")
    print("=" * 70)

    if args.server_log:
        try:
            with open(args.server_log, encoding="utf-8") as f:
                log = f.read()
        except OSError as e:
            log = ""
            print(f"WARNING: cannot read server log: {e}")
        d2h = log.count("kv_exchange_copy D2H")
        h2d = log.count("kv_exchange_copy H2D")
        print(f"server log: kv_exchange_copy D2H x{d2h}, kv_exchange_copy H2D x{h2d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())