#!/usr/bin/env python3
"""layer_first DeepSeek-Coder（通用 MLA/AscendC）roundtrip 的 profiling 采集脚本（短/长两档 + D2H/H2D 分方向）。

背景
----
DeepSeek-Coder layer_first + hicache(write_through) 的 roundtrip 场景里：
  * W（write）：doc+query 的 prefill 会触发 write_through 的 D2H backup；
  * E（evict）：另一段 filler 把 W 的 doc 挤出 L1（LRU），L2 仍保留；
  * R（replay）：重发 doc+query，prefix 在 L1 miss + L2 hit，触发 H2D restore。

profiling 目标是把这两次传输抓进 trace，研究传输算子的耗时与设备利用率。

设计
----
  1. warmup：不 profile，先发 N 轮短 completion，过滤首次 kernel launch / 分配抖动；
  2. 短档 / 长档各跑一遍，每档拆两个方向独立采集：
       d2h 段：start_profile → 发 W（抓 D2H backup 的 prefill）→ 落盘
       h2d 段：start_profile → 发 R（抓 H2D restore 的 prefill）→ 落盘
     （E 请求在两段之间做、不 profile，只负责把 W doc 淘汰出 L1。）

之所以每档拆两段、而不是一次跑 W/E/R：
  服务端的 profile_by_stage 只在“第一段 prefill + 第一段 decode”各采 num_steps 步，
  计数器停后不重置；一次连发 W/E/R 只会覆盖到 W 的 prefill(decode)，R 的 H2D 抓不到。

依赖：transformers（按 token 长度构造 filler）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

try:
    from transformers import AutoTokenizer
except Exception:  # noqa: BLE001
    AutoTokenizer = None

_QUERY = "Question: what is the secret passcode?\nAnswer:"
_EVICT_QUERY = "Question: what is 2 + 3?\nAnswer:"
_SENTENCE = (
    "The wind carried dry leaves across the empty field while a distant "
    "train passed under the grey evening sky. "
)


def _http_json(base_url: str, path: str, data: dict | None = None, timeout: int = 120):
    """POST(默认)/GET 一个 JSON 接口，返回 body 字节串。"""
    url = base_url.rstrip("/") + path
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def post_completion(base_url: str, prompt: str, max_tokens: int, timeout: int = 600) -> str:
    body = json.loads(
        _http_json(
            base_url,
            "/v1/completions",
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=timeout,
        )
    )
    return body["choices"][0]["text"]


def start_profile(base_url: str, output_dir: str, prefix: str, num_steps: int) -> None:
    """触发 profile_by_stage + detailed_annotations 的一次采集（num_steps 自动停）。"""
    _http_json(
        base_url,
        "/start_profile",
        {
            "output_dir": output_dir,
            "num_steps": num_steps,
            "activities": ["CPU", "GPU"],
            "profile_by_stage": True,
            "detailed_annotations": True,
            "profile_prefix": prefix,
            "merge_profiles": False,
        },
        timeout=30,
    )


def flush_profile(base_url: str, seconds: float = 4.0) -> None:
    """等 num_steps 自动 stop 落盘后，best-effort 补一个 stop（已停则 no-op）。"""
    time.sleep(seconds)
    try:
        _http_json(base_url, "/stop_profile", timeout=30)
    except Exception:  # noqa: BLE001 —— 落盘已由 on_trace_ready 负责，此处仅兜底
        pass


def _filler_to_tokens(tok, sentence: str, target_tokens: int) -> str:
    """按 tokenizer 精确构造到 target_tokens 长度的填充文本。"""
    per = len(tok.encode(sentence, add_special_tokens=False))
    n = max(1, target_tokens // per)
    text = sentence * n
    while len(tok.encode(text, add_special_tokens=False)) < target_tokens:
        text += sentence
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:32346")
    ap.add_argument(
        "--output-dir", default="/home/l00889655/sglang/profiler"
    )
    ap.add_argument(
        "--model-path", default="/mnt/paas/weights/DeepSeek-Coder-V2-Lite-Instruct"
    )
    ap.add_argument("--needle", type=int, default=48219)
    ap.add_argument("--short-doc-tokens", type=int, default=2048)
    ap.add_argument("--long-doc-tokens", type=int, default=10000)
    # evict 必须足够把当前 L1 里的 W doc 挤出（> max-total-tokens - doc_tokens）。
    ap.add_argument("--evict-tokens", type=int, default=15000)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--warmup-rounds", type=int, default=2)
    args = ap.parse_args()

    if AutoTokenizer is None:
        print("ERROR: transformers.AutoTokenizer is required.", file=sys.stderr)
        return 2

    tok = AutoTokenizer.from_pretrained(args.model_path)
    needle_sentence = f"The secret passcode is {args.needle}. "

    def build_prompts(doc_tokens: int) -> tuple[str, str]:
        doc = needle_sentence + _filler_to_tokens(tok, _SENTENCE, doc_tokens)
        evict = _filler_to_tokens(tok, _SENTENCE, args.evict_tokens)
        return doc + "\n\n" + _QUERY, evict + "\n\n" + _EVICT_QUERY

    # ---- warmup（不 profile）----
    print(f"warmup x{args.warmup_rounds} ...", flush=True)
    for _ in range(args.warmup_rounds):
        post_completion(args.base_url, "What is 1 + 1?\nAnswer:", max_tokens=1)

    # ---- 短 / 长两档，各拆 D2H + H2D ----
    profiles = [
        ("short", args.short_doc_tokens),
        ("long", args.long_doc_tokens),
    ]
    for label, doc_tokens in profiles:
        doc_prompt, evict_prompt = build_prompts(doc_tokens)
        print(
            f"=== [{label}] doc_tokens={doc_tokens} evict_tokens={args.evict_tokens} ===",
            flush=True,
        )

        # -- D2H（W：write_through backup，prefill 里触发 D2H）--
        d2h_dir = os.path.join(args.output_dir, label, "d2h")
        os.makedirs(d2h_dir, exist_ok=True)
        start_profile(args.base_url, d2h_dir, f"{label}-d2h", args.num_steps)
        post_completion(args.base_url, doc_prompt, args.max_tokens)
        flush_profile(args.base_url)
        print(f"[{label}] D2H(W) profile -> {d2h_dir}", flush=True)

        # -- E（仅淘汰 W doc，不 profile）--
        post_completion(args.base_url, evict_prompt, 8)

        # -- H2D（R：L1 miss + L2 hit，prefill 里触发 H2D restore）--
        h2d_dir = os.path.join(args.output_dir, label, "h2d")
        os.makedirs(h2d_dir, exist_ok=True)
        start_profile(args.base_url, h2d_dir, f"{label}-h2d", args.num_steps)
        post_completion(args.base_url, doc_prompt, args.max_tokens)
        flush_profile(args.base_url)
        print(f"[{label}] H2D(R) profile -> {h2d_dir}", flush=True)

    print("done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())