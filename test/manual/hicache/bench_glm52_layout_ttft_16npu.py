#!/usr/bin/env python3
"""Measure L2 replay TTFT for GLM-5.2 HiCache layouts, then compare results."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FILLER_A = (
    "The wind carried dry leaves across the empty field while a distant train "
    "passed under the grey evening sky. "
)
FILLER_B = (
    "A small boat drifted along the quiet river as fishermen pulled in their "
    "nets beneath the rising morning sun. "
)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def make_prompt(tokenizer, prefix: str, filler: str, target_tokens: int, question: str) -> str:
    filler_tokens = max(1, len(tokenizer.encode(filler, add_special_tokens=False)))
    repeats = max(1, (target_tokens + filler_tokens - 1) // filler_tokens)
    while True:
        prompt = prefix + filler * repeats + "\n\nQuestion: " + question + "\nAnswer:"
        if len(tokenizer.encode(prompt, add_special_tokens=False)) >= target_tokens:
            return prompt
        repeats += 1


def build_prompts(tokenizer, count: int, prompt_tokens: int, seed: int):
    prompts_a = [
        make_prompt(
            tokenizer,
            f"The secret passcode is {seed + i}. ",
            FILLER_A,
            prompt_tokens,
            "What is the secret passcode?",
        )
        for i in range(count)
    ]
    prompts_b = [
        make_prompt(
            tokenizer,
            f"The unrelated record number is {seed + 100000 + i}. ",
            FILLER_B,
            prompt_tokens,
            "What is 2 + 3?",
        )
        for i in range(count)
    ]
    return prompts_a, prompts_b


def stream_one(base_url: str, prompt: str, max_tokens: int, timeout: int):
    payload = json.dumps(
        {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stop": ["Question:"],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_text_ms = None
    pieces = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            choices = event.get("choices") or []
            piece = (choices[0].get("text") or "") if choices else ""
            if piece:
                if first_text_ms is None:
                    first_text_ms = (time.perf_counter() - start) * 1000
                pieces.append(piece)
    if first_text_ms is None:
        raise RuntimeError("stream ended without a generated text token")
    return first_text_ms, "".join(pieces)


def run_phase(base_url: str, prompts: list[str], max_tokens: int, timeout: int):
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [
            pool.submit(stream_one, base_url, prompt, max_tokens, timeout)
            for prompt in prompts
        ]
        return [future.result() for future in futures]


def marker_count(path: Path, marker: str) -> int:
    with path.open(encoding="utf-8", errors="replace") as stream:
        return sum(marker in line for line in stream)


def run_cycle(args, tokenizer, seed: int, label: str):
    prompts_a, prompts_b = build_prompts(
        tokenizer, args.num_prompts, args.prompt_tokens, seed
    )
    actual_lengths = [
        len(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in prompts_a + prompts_b
    ]
    fill = run_phase(args.base_url, prompts_a, args.max_tokens, args.timeout)
    run_phase(args.base_url, prompts_b, args.max_tokens, args.timeout)
    h2d_before = marker_count(args.server_log, "kv_exchange_copy H2D")
    replay = run_phase(args.base_url, prompts_a, args.max_tokens, args.timeout)
    h2d_after = marker_count(args.server_log, "kv_exchange_copy H2D")
    h2d_delta = h2d_after - h2d_before
    if h2d_delta <= 0:
        raise RuntimeError(f"{label}: no H2D marker during replay; TTFT is not an L2 hit")
    ttfts = [item[0] for item in replay]
    text_matches = sum(a[1] == b[1] for a, b in zip(fill, replay))
    needle_matches = sum(str(seed + i) in replay[i][1] for i in range(args.num_prompts))
    print(
        f"{label}: replay TTFT={','.join(f'{x:.1f}' for x in ttfts)} ms; "
        f"H2D markers={h2d_delta}; exact replay={text_matches}/{args.num_prompts}; "
        f"needle={needle_matches}/{args.num_prompts}",
        flush=True,
    )
    return {
        "label": label,
        "ttft_ms": ttfts,
        "h2d_markers": h2d_delta,
        "exact_replay": text_matches,
        "needle_matches": needle_matches,
        "prompt_tokens": actual_lengths,
    }


def run_benchmark(args) -> int:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required for exact prompt sizing") from error

    estimated_one_set = args.num_prompts * (args.prompt_tokens + args.max_tokens)
    estimated_two_sets = 2 * estimated_one_set
    if not (estimated_one_set < args.l1_tokens < estimated_two_sets):
        raise ValueError("A must fit L1 while A+B must exceed L1 for eviction")
    if estimated_two_sets >= args.l1_tokens * args.l2_ratio * 0.9:
        raise ValueError("A+B must fit L2 with at least 10% headroom")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(
        f"layout={args.layout} prompts={args.num_prompts} prompt_tokens≈{args.prompt_tokens} "
        f"output_tokens={args.max_tokens} warmup={args.warmup_rounds} "
        f"measured={args.measure_rounds} L1={args.l1_tokens} L2_ratio={args.l2_ratio}",
        flush=True,
    )
    d2h_start = marker_count(args.server_log, "kv_exchange_copy D2H")
    for i in range(args.warmup_rounds):
        run_cycle(args, tokenizer, 900000000 + i * 1000, f"warmup{i + 1}")

    cycles = [
        run_cycle(args, tokenizer, 1000000 + i * 1000, f"measure{i + 1}")
        for i in range(args.measure_rounds)
    ]
    d2h_delta = marker_count(args.server_log, "kv_exchange_copy D2H") - d2h_start
    if d2h_delta <= 0:
        raise RuntimeError("no D2H marker; write-through transfer did not run")
    samples = [value for cycle in cycles for value in cycle["ttft_ms"]]
    result = {
        "layout": args.layout,
        "config": {
            "model_path": args.model_path,
            "num_prompts": args.num_prompts,
            "prompt_tokens": args.prompt_tokens,
            "max_tokens": args.max_tokens,
            "warmup_rounds": args.warmup_rounds,
            "measure_rounds": args.measure_rounds,
            "l1_tokens": args.l1_tokens,
            "l2_ratio": args.l2_ratio,
        },
        "ttft_ms": samples,
        "mean_ttft_ms": statistics.mean(samples),
        "median_ttft_ms": statistics.median(samples),
        "p95_ttft_ms": percentile(samples, 0.95),
        "d2h_markers": d2h_delta,
        "h2d_markers": sum(cycle["h2d_markers"] for cycle in cycles),
        "exact_replay": sum(cycle["exact_replay"] for cycle in cycles),
        "needle_matches": sum(cycle["needle_matches"] for cycle in cycles),
        "cycles": cycles,
    }
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.layout}: mean={result['mean_ttft_ms']:.1f} ms, "
        f"median={result['median_ttft_ms']:.1f} ms, "
        f"p95={result['p95_ttft_ms']:.1f} ms, "
        f"D2H={d2h_delta}, measured H2D={result['h2d_markers']}",
        flush=True,
    )
    return 0


def compare_results(layer_path: Path, page_path: Path) -> int:
    layer = json.loads(layer_path.read_text(encoding="utf-8"))
    page = json.loads(page_path.read_text(encoding="utf-8"))
    if layer["layout"] != "layer_first" or page["layout"] != "page_first_kv_split":
        raise ValueError("expected layer_first and page_first_kv_split result files")
    if layer["config"] != page["config"]:
        raise ValueError("benchmark configurations differ between layouts")

    print("TTFT for measured L2 replay requests (lower is better):")
    print("layout                mean_ms  median_ms    p95_ms")
    for result in (layer, page):
        print(
            f"{result['layout']:22s} {result['mean_ttft_ms']:8.1f} "
            f"{result['median_ttft_ms']:10.1f} {result['p95_ttft_ms']:9.1f}"
        )
    for key, label in (
        ("mean_ttft_ms", "mean"),
        ("median_ttft_ms", "median"),
        ("p95_ttft_ms", "p95"),
    ):
        change = (layer[key] / page[key] - 1.0) * 100
        print(f"layer_first vs page_first {label}: {change:+.1f}%")
    non_regressed = (
        layer["median_ttft_ms"] <= page["median_ttft_ms"]
        and layer["p95_ttft_ms"] <= page["p95_ttft_ms"]
    )
    print(f"TTFT non-regression (median and p95): {'PASS' if non_regressed else 'FAIL'}")
    return 0 if non_regressed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--layer-json", type=Path)
    parser.add_argument("--page-json", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--layout", choices=["layer_first", "page_first_kv_split"])
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=int, default=3000)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--measure-rounds", type=int, default=5)
    parser.add_argument("--l1-tokens", type=int, default=8192)
    parser.add_argument("--l2-ratio", type=float, default=2.5)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.compare:
        if not args.layer_json or not args.page_json:
            parser.error("--compare requires --layer-json and --page-json")
        return compare_results(args.layer_json, args.page_json)
    if not all(
        (args.base_url, args.layout, args.server_log, args.model_path, args.output_json)
    ):
        parser.error("run mode requires --base-url, --layout, --server-log, --model-path, --output-json")
    return run_benchmark(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"BENCHMARK ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
