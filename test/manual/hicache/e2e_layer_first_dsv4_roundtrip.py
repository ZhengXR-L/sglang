#!/usr/bin/env python3
"""L1<->L2 (layer_first DSv4) D2H + H2D roundtrip validation.

DeepSeek-V4 host pools (``DeepSeekV4PagedHostPool`` /
``DeepSeekV4StateHostPool``) mirror the device pools through the NPU AIV
``kv_exchange_copy`` path under layer_first_kv_split + Memfabric. This script
drives a W/E/R sequence and checks both transfer markers and deterministic
replay output.

  1. W (write):  ``doc + query`` where ``doc`` starts with a unique passcode
     needle followed by long filler (write_through mirrors it to L2 -> D2H).
  2. E (evict): a second, different long filler overflows the L1 device pool,
     LRU-evicting W's ``doc`` out of L1 while L2 retains it.
  3. R (replay): re-issue the identical ``doc + query``. W's prefix is now an
     L1 miss + L2 hit, so the pools restore it (H2D).

Pass criteria (hard): at least one D2H and one H2D AIV marker, plus a
nonempty identical W/R completion. Needle presence is informational because
the base DSpark model may not reliably retrieve it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

try:
    from transformers import AutoTokenizer
except Exception:  # noqa: BLE001
    AutoTokenizer = None

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _last_number(text: str) -> str | None:
    nums = _NUM_RE.findall(text.replace(",", ""))
    return nums[-1] if nums else None


def post_completion(base_url: str, prompt: str, max_tokens: int, timeout: int = 1200):
    payload = json.dumps(
        {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    url = base_url.rstrip("/") + "/v1/completions"
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        choice = body["choices"][0]
        return choice["text"], choice.get("finish_reason"), body.get("usage")


def _filler_to_tokens(tok, sentence: str, target_tokens: int) -> str:
    per = len(tok.encode(sentence, add_special_tokens=False))
    n = max(1, target_tokens // per)
    text = sentence * n
    while len(tok.encode(text, add_special_tokens=False)) < target_tokens:
        text += sentence
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:32345")
    ap.add_argument(
        "--server-log",
        default="/home/l00889655/sglang/e2e_layer_first_logs/roundtrip_dsv4_server.log",
    )
    ap.add_argument("--needle", type=int, default=48219)
    ap.add_argument(
        "--model-path",
        default="/mnt/paas/weights/DeepSeek-V4-Flash-0731-w8a8",
    )
    ap.add_argument("--doc-tokens", type=int, default=6000)
    ap.add_argument("--evict-tokens", type=int, default=16000)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    if AutoTokenizer is None:
        print(
            "ERROR: transformers.AutoTokenizer is required to size the prompt "
            "by token count.",
            file=sys.stderr,
        )
        return 2

    tok = AutoTokenizer.from_pretrained(args.model_path)

    sentence = (
        "The wind carried dry leaves across the empty field while a distant "
        "train passed under the grey evening sky. "
    )
    needle_sentence = f"The secret passcode is {args.needle}. "

    doc = needle_sentence + _filler_to_tokens(tok, sentence, args.doc_tokens)
    evict = _filler_to_tokens(tok, sentence, args.evict_tokens)

    query = "Question: what is the secret passcode?\nAnswer:"
    doc_prompt = doc + "\n\n" + query
    evict_prompt = evict + "\n\nQuestion: what is 2 + 3?\nAnswer:"

    print(
        f"roundtrip: needle={args.needle} doc_tokens~={args.doc_tokens} "
        f"evict_tokens~={args.evict_tokens}",
        flush=True,
    )

    ans_w, finish_w, usage_w = post_completion(args.base_url, doc_prompt, args.max_tokens)
    print(f"phase W done: finish={finish_w} usage={usage_w}", flush=True)
    if os.environ.get("SKIP_E", "0") == "1":
        ans_e = ""
        print("phase E skipped", flush=True)
    else:
        ans_e, finish_e, usage_e = post_completion(args.base_url, evict_prompt, 8)
        print(f"phase E done: finish={finish_e} usage={usage_e}", flush=True)
    ans_r, finish_r, usage_r = post_completion(args.base_url, doc_prompt, args.max_tokens)
    print(f"phase R done: finish={finish_r} usage={usage_r}", flush=True)

    needle_str = str(args.needle)
    needle_w = needle_str in ans_w
    needle_r = needle_str in ans_r
    same = ans_w.strip() == ans_r.strip()

    d2h = h2d = 0
    try:
        with open(args.server_log, encoding="utf-8") as f:
            log = f.read()
    except OSError as e:
        print(f"WARNING: cannot read server log {args.server_log}: {e}")
        log = ""
    expect_aiv = os.environ.get("EXPECT_AIV", "1") == "1"
    expect_transfer = os.environ.get("EXPECT_TRANSFER", "1") == "1"
    marker = "HiCache DSV4 AIV kv_exchange_copy " if expect_aiv else "HiCache DSV4 "
    d2h = log.count(marker + "D2H")
    h2d = log.count(marker + "H2D")

    print("=" * 62)
    print(f"W answer: {ans_w.strip()!r}  (last_num={_last_number(ans_w)})")
    print(f"E answer: {ans_e.strip()!r}  (last_num={_last_number(ans_e)})")
    print(f"R answer: {ans_r.strip()!r}  (last_num={_last_number(ans_r)})")
    print(f"server log: {marker}D2H x{d2h}, {marker}H2D x{h2d}")
    print(
        f"checks: W_contains_needle={needle_w} R_contains_needle={needle_r} "
        f"W==R={same} W_nonempty={bool(ans_w.strip())}"
    )
    print("=" * 62)

    ok_d2h = d2h > 0
    ok_h2d = h2d > 0
    ok = (not expect_transfer or (ok_d2h and ok_h2d)) and bool(ans_w.strip()) and same

    print("RESULT:", "PASS" if ok else "FAIL")
    if expect_transfer and not ok_d2h:
        print("  missing D2H marker (no device->host backup)")
    if expect_transfer and not ok_h2d:
        print("  missing H2D marker (no host->device restore)")
    if not same or not ans_w.strip():
        print("  W and R answers differ or are empty (replay data-integrity failure)")
    if not (needle_w and needle_r):
        print("  NOTE: needle not present in W/R (base model may not retrieve it)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
