#!/usr/bin/env python3
"""GSM8K few-shot accuracy evaluation against a running SGLang server.

Self-contained (stdlib only). Sends N questions as raw completions to the
OpenAI-compatible ``/v1/completions`` endpoint, extracts the final numeric
answer from each model output, and compares it against the ground-truth value
(the number following ``####`` in the dataset answer).

Usage:
    python3 e2e_gsm8k_eval.py \
        --base-url http://127.0.0.1:32345 \
        --data-path /home/l00889655/sglang/gsm8k_test.jsonl \
        --num-questions 100 --num-shots 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _numbers(text: str) -> list[str]:
    return _NUM_RE.findall(text.replace(",", ""))


def extract_pred_answer(text: str) -> str | None:
    """Final numeric answer of a generated completion = last number it emits."""
    nums = _numbers(text)
    return nums[-1] if nums else None


def extract_gt_answer(answer: str) -> str | None:
    """Ground truth is the value after the trailing ``####`` separator."""
    if "####" in answer:
        answer = answer.rsplit("####", 1)[-1]
    nums = _numbers(answer)
    return nums[-1] if nums else None


def _to_float(s: str | None) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def build_few_shot(examples: list[dict]) -> str:
    return "\n\n".join(
        f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in examples
    )


def get_one_prompt(few_shot: str, question: str) -> str:
    if few_shot:
        return few_shot + "\n\nQuestion: " + question + "\nAnswer:"
    return "Question: " + question + "\nAnswer:"


def post_completion(
    base_url: str, prompt: str, max_tokens: int, temperature: float, timeout: int
) -> str:
    payload = json.dumps(
        {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["Question:"],
        }
    ).encode("utf-8")
    url = base_url.rstrip("/") + "/v1/completions"
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["text"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:32345")
    parser.add_argument("--data-path", default="/home/l00889655/sglang/gsm8k_test.jsonl")
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    rows = read_jsonl(args.data_path)
    shot_rows = rows[: args.num_shots]
    eval_rows = rows[args.num_shots : args.num_shots + args.num_questions]
    if len(eval_rows) < args.num_questions:
        print(
            f"ERROR: only {len(eval_rows)} questions available "
            f"(need {args.num_questions}); data has {len(rows)} rows and "
            f"{args.num_shots} are used for few-shot.",
            file=sys.stderr,
        )
        return 2

    few_shot = build_few_shot(shot_rows)
    print(
        f"GSM8K eval: num_questions={len(eval_rows)} shots={args.num_shots} "
        f"base_url={args.base_url} temperature={args.temperature} "
        f"max_tokens={args.max_tokens}",
        flush=True,
    )

    prompts = [get_one_prompt(few_shot, r["question"]) for r in eval_rows]
    gt_answers = [extract_gt_answer(r["answer"]) for r in eval_rows]
    if any(g is None for g in gt_answers):
        print("ERROR: could not extract ground truth for some rows", file=sys.stderr)
        return 2

    # ordered results: pred answer string (None on request/parse error)
    preds: list[str | None] = [None] * len(eval_rows)
    error_msgs: list[str | None] = [None] * len(eval_rows)
    tic = time.perf_counter()

    def run(i: int) -> None:
        try:
            text = post_completion(
                args.base_url, prompts[i], args.max_tokens, args.temperature, args.timeout
            )
            preds[i] = extract_pred_answer(text)
        except Exception as e:  # noqa: BLE001 - report and continue
            error_msgs[i] = f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run, i) for i in range(len(eval_rows))]
        for fut in as_completed(futures):
            fut.result()  # propagate nothing; errors recorded in error_msgs

    latency = time.perf_counter() - tic

    correct = 0
    for pred, gt in zip(preds, gt_answers):
        pv, gv = _to_float(pred), _to_float(gt)
        if pv is not None and gv is not None and abs(pv - gv) < 1e-6:
            correct += 1

    n_invalid = sum(1 for p in preds if p is None)
    n_errors = sum(1 for e in error_msgs if e is not None)
    accuracy = correct / len(eval_rows)

    print(f"\n{'=' * 56}")
    print(f"GSM8K accuracy: {accuracy:.3f} ({correct}/{len(eval_rows)})")
    print(f"invalid (no answer extracted): {n_invalid}/{len(eval_rows)}")
    print(f"request errors: {n_errors}/{len(eval_rows)}")
    print(f"latency: {latency:.2f}s  ({len(eval_rows) / latency:.2f} q/s)")
    print(f"{'=' * 56}")

    mismatches = [
        i
        for i in range(len(eval_rows))
        if _to_float(preds[i]) is None
        or abs((_to_float(preds[i]) or 0) - (_to_float(gt_answers[i]) or 0)) >= 1e-6
    ]
    if mismatches:
        print("\nsample mismatches (pred | gt | question):")
        for i in mismatches[:10]:
            q_short = eval_rows[i]["question"][:90].replace("\n", " ")
            print(f"  [{i}] pred={preds[i]!r} gt={gt_answers[i]!r}  Q: {q_short}")
            if error_msgs[i]:
                print(f"       error: {error_msgs[i]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())