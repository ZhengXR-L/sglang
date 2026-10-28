"""Coder V2 Lite serial streaming A/B/R TTFT benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import urllib.request
import uuid

LAYOUTS = {'layer_first': 'layer_first_kv_split', 'page_first': 'page_first_kv_split'}


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def http(base, route, body=None):
    req = urllib.request.Request(base+route, data=json.dumps(body).encode() if body is not None else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as response:
        data = response.read()
    return data.decode() if route == '/flush_cache' else json.loads(data)


def token_event(obj):
    # Ignore empty metadata chunks. Reject all-zero final outputs below.
    ids = obj.get('output_ids')
    return isinstance(ids, list) and len(ids) > 0


def stream(base, body, require_valid=True):
    encoded = json.dumps({**body, 'stream': True}).encode()
    req = urllib.request.Request(base+'/generate', data=encoded,
                                 headers={'Content-Type': 'application/json'})
    first = None
    first_text = None
    last = None
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as response:
        for raw in response:
            line = raw.decode('utf-8').strip()
            if not line.startswith('data:'):
                continue
            value = line[5:].strip()
            if value == '[DONE]':
                break
            obj = json.loads(value)
            if 'error' in obj:
                raise ValueError(f'Stream error: {obj}')
            if token_event(obj) and first is None:
                first = time.perf_counter()-start
            if obj.get('text') and first_text is None:
                first_text = time.perf_counter()-start
            last = obj
    elapsed = time.perf_counter()-start
    if last is None or (require_valid and first is None):
        raise ValueError(f'No generated token in stream; last={str(last)[:1000]}')
    ids = last.get('output_ids', [])
    if require_valid and (not ids or not any(ids) or not last.get('text', '').strip()):
        raise ValueError('Empty/all-zero output; sample is invalid')
    return {'ttft_ms': first*1000 if first is not None else None, 'first_text_ms': first_text*1000 if first_text is not None else None,
            'total_ms': elapsed*1000, 'response': last}


def prepare(source_plan, dest, repeats):
    """Keep identical natural-text token sequences in every server session."""
    source = json.loads(Path(source_plan).read_text())
    # Repeat natural few-shot text, then append an intact question suffix.
    cases = []
    for target in (1024, 4096, 8192):
        for i in range(repeats):
            q = source['questions'][i]
            base = q['input_ids']
            tail = base[-256:]
            count = target+128-len(tail)
            ids = (base*((count+len(base)-1)//len(base)))[:count]+tail
            cases.append({'id': f't{target}-i{i}', 'target_host_tokens': target,
                          'input_ids': ids, 'sha256': sha(ids), 'source_qid': q['qid']})
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(source['model_path'], trust_remote_code=True)
    excluded = set(tok.all_special_ids)
    rng = random.Random(60001)
    forbidden_first = excluded | {c['input_ids'][0] for c in cases}
    b = []
    while len(b) < 10000:
        v = rng.randrange(tok.vocab_size)
        if v not in excluded and (b or v not in forbidden_first):
            b.append(v)
    plan = {'model_path': source['model_path'], 'source_plan': str(source_plan), 'cases': cases,
            'eviction_input_ids': b, 'eviction_sha256': sha(b), 'repeats': repeats}
    save(dest, plan)


def preflight(info, plan, layout):
    if info.get('hicache_mem_layout') != LAYOUTS[layout] or info.get('hicache_io_backend') != 'kernel_ascend':
        raise ValueError('Resolved layout/backend does not match requested comparison')
    if info.get('tp_size') != 1:
        raise ValueError('Require TP1')
    if info.get('speculative_algorithm') is not None or info.get('hicache_write_policy') != 'write_through':
        raise ValueError('Require no MTP and write_through')
    if info.get('stream_interval') != 1 or info.get('model_path') != plan['model_path']:
        raise ValueError('Require streaming interval 1 and the planned model')
    caps = [x.get('memory_usage', {}).get('token_capacity') for x in info.get('internal_states', [])]
    caps = [x for x in caps if isinstance(x, int) and x > 0]
    if len(caps) != 1 or min(caps) < 10000+8:
        raise ValueError(f'Insufficient actual per-worker capacity: {caps}')
    return min(caps)


def evaluate(args):
    out = Path(args.output); out.mkdir(exist_ok=False)
    plan = json.loads(Path(args.plan).read_text())
    if any(sha(c['input_ids']) != c['sha256'] for c in plan['cases']) or sha(plan['eviction_input_ids']) != plan['eviction_sha256']:
        raise ValueError('Plan hash changed')
    info = http(args.base_url, '/server_info'); save(out/'server_info.json', info)
    capacity = preflight(info, plan, args.layout)
    save(out/'arguments.json', {**vars(args), 'actual_capacity': capacity})
    rows = []
    try:
        for case in plan['cases']:
            folder = out/case['id']; folder.mkdir()
            http(args.base_url, '/flush_cache', {})
            outputs = {}
            for phase, ids in [('A', case['input_ids']), ('B', plan['eviction_input_ids']), ('R', case['input_ids'])]:
                body = {'rid': uuid.uuid4().hex, 'input_ids': ids, 'routed_dp_rank': 0,
                        'sampling_params': {'temperature': 0, 'max_new_tokens': 8 if phase == 'B' else 16,
                                            'ignore_eos': True}}
                save(folder/f'{phase}_request.json', body)
                result = stream(args.base_url, body, require_valid=phase != 'B')
                outputs[phase] = result
                save(folder/f'{phase}_result.json', result)
                if result['response'].get('meta_info', {}).get('dp_rank') not in (None, 0):
                    raise ValueError('Unexpected DP rank for TP1')
                if phase == 'A':
                    if result['response']['meta_info'].get('cached_tokens') != 0:
                        raise ValueError('A was not a cold recompute')
                    # Same untimed grace for pending write-through work on both layouts.
                    time.sleep(0.5)
            tiers = outputs['R']['response']['meta_info'].get('cached_tokens_details', {})
            if tiers.get('device') != 0 or tiers.get('host') != case['target_host_tokens']:
                raise ValueError(f'Expected pure L2 {case["target_host_tokens"]}; got {tiers}')
            if outputs['A']['response']['output_ids'] != outputs['R']['response']['output_ids']:
                raise ValueError('A/R output token mismatch; performance sample is invalid')
            row = {'id': case['id'], 'host_tokens': case['target_host_tokens'], 'input_tokens': len(case['input_ids']),
                   'warmup': case['id'].endswith('-i0'), 'cached_tokens_details': tiers, 'token_match': True,
                   **{f'{p.lower()}_ttft_ms': outputs[p]['ttft_ms'] for p in ('A', 'B', 'R')},
                   'r_first_text_ms': outputs['R']['first_text_ms']}
            rows.append(row)
            with (out/'samples.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
    except Exception as exc:
        save(out/'error.json', {'error': repr(exc), 'completed': len(rows)})
        raise
    save(out/'summary.json', {'layout': args.layout, 'complete': True, 'rows': rows})


def summarize(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    blocks = []
    for block in manifest['blocks']:
        p = root/block['name']/'results/summary.json'
        if p.exists():
            blocks.append({**block, **json.loads(p.read_text())})
    lines = ['# DeepSeek-Coder-V2-Lite-Instruct HiCache 布局 TTFT 对比', '',
             '比较 kernel_ascend 的 layer_first_kv_split 与 page_first_kv_split。', '',
             '固定 TP1、并发 1；同一张 NPU，按 layer/page/page/layer 顺序。', '',
             '每场景输入为缓存前缀长度 +128 token；A 重算，B=10000 token 驱逐，R 原输入回迁。A/R 输出16 token，B输出8 token，temperature=0、ignore_eos=true。', '',
             '每种长度每次服务先预热一轮（i0，不计时统计），再测五轮。TTFT 从请求发送到首个含 output_ids 的事件；同时保存首个非空文本时间。', '',
             '强制 A 无缓存、R 的 L1=0 且 L2 命中指定长度、A/R 输出 IDs 全等；任一失败中止该次比较，不能剔除后继续报通过。', '',
             '不加载逐调用 event/JSONL audit hook。使用 info 日志；计时没有逐层日志 I/O。B 请求可能因填满 KV 容量而不生成 token，是否有效由后续 R 的 L1=0、L2 精确命中及 A/R 输出一致共同判定。缓存写入可能与计算重叠，A TTFT 不等于 D2H 总耗时，R TTFT 也不等于纯算子耗时。', '',
             '## 结果', '', '| L2 前缀 | layer 均值 ms | page 均值 ms | layer p50 | page p50 | layer p95 | page p95 | layer 相对变化 |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    stats = []
    for target in (1024, 4096, 8192):
        groups = {}
        for layout in LAYOUTS:
            vals = [r['r_ttft_ms'] for b in blocks if b['layout'] == layout for r in b['rows'] if r['host_tokens'] == target and not r['warmup']]
            if vals:
                groups[layout] = {'n': len(vals), 'mean_ms': statistics.mean(vals), 'p50_ms': statistics.median(vals),
                                  'p95_ms': sorted(vals)[max(0, __import__('math').ceil(.95*len(vals))-1)]}
        if len(groups) == 2:
            a, b = groups['layer_first'], groups['page_first']
            change = 100*(a['mean_ms']/b['mean_ms']-1)
            stats.append({'host_tokens': target, **groups, 'layer_change_pct': change})
            lines.append(f"| {target} | {a['mean_ms']:.2f} | {b['mean_ms']:.2f} | {a['p50_ms']:.2f} | {b['p50_ms']:.2f} | {a['p95_ms']:.2f} | {b['p95_ms']:.2f} | {change:+.2f}% |")
    complete = len(blocks) == 4 and all(len(b['rows']) == 18 for b in blocks)
    lines += ['', f'完整四次服务比较：{complete}。负的相对变化表示 layer 更快。', '',
              '## 各次服务结果', '', '| 次序 | 布局 | L2 前缀 | 样本数 | R TTFT 均值 ms |', '|---:|---|---:|---:|---:|']
    for b in blocks:
        for target in (1024,4096,8192):
            vals = [r['r_ttft_ms'] for r in b['rows'] if r['host_tokens'] == target and not r['warmup']]
            lines.append(f"| {b['name']} | {b['layout']} | {target} | {len(vals)} | {statistics.mean(vals):.2f} |")
    lines += ['', '## 参数与证据', '', '```json', json.dumps(manifest, ensure_ascii=False, indent=2), '```', '',
              '每次服务的 server_info.json、environment.json、server.log，以及每样本 A/B/R 完整请求/响应均在同目录保存。', '',
              '源配置和权重不变，比较两种 host 布局的实际 kernel_ascend 路径。这里只覆盖并发1；共享服务器负载变化、测量样本数限制了对其他工作负载的外推。', '',
              '本轮计时不逐调用观测算子；当前通路依据服务实际解析布局和源码版本确定。', '']
    save(root/'comparison.json', {'complete': complete, 'results': stats})
    (root/'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare'); q.add_argument('--source-plan', required=True); q.add_argument('--output', required=True)
    q = sub.add_parser('eval'); q.add_argument('--plan', required=True); q.add_argument('--output', required=True)
    q.add_argument('--base-url', default='http://127.0.0.1:32430'); q.add_argument('--layout', choices=LAYOUTS, required=True)
    q = sub.add_parser('report'); q.add_argument('--root', required=True)
    args = p.parse_args()
    if args.command == 'prepare': prepare(args.source_plan, args.output, 6)
    elif args.command == 'eval': evaluate(args)
    else: summarize(args.root)
