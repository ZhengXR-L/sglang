"""Summarize saved NPU evidence; never changes the server or test results."""
import json
from pathlib import Path
import shlex
import sys

root = Path(sys.argv[1])
read = lambda p: json.loads(p.read_text())
launch = read(root / 'audit/launch.json')
lines = ['# DSV4 GSM8K 长 few-shot AIV 往返验证', '',
         '测试日期：2026-10-27（Asia/Hong_Kong）。容器：`sglang-lzx-B103`。', '',
         f"代码提交：`{launch['git_commit']}`。本次未修改业务代码或 wheel。", '',
         '## 服务启动参数', '', '```bash', shlex.join(launch['argv']), '```', '',
         '观测通过本次服务私有 sitecustomize 包装真实 Memfabric kv_exchange_copy，返回成功且当前流 NPU event 完成后才记录覆盖。', '',
         '```json', json.dumps(launch['environment'], ensure_ascii=False, indent=2), '```', '']
launcher = Path(__file__).with_name('run_server.sh')
if launcher.exists():
    lines += ['完整启动脚本（含 Ascend 环境初始化和 export）：', '', '```bash', launcher.read_text(), '```', '']
for n in (2, 10, 100):
    result = root / f'results_{n}'
    if not (result / 'summary.json').exists():
        continue
    summary = read(result / 'summary.json')
    plan = read(root / f'plan_{n}.json')
    info = read(result / 'server_info.json')
    lines += [f'## {n} 题结果', '', '```json', json.dumps(summary, ensure_ascii=False, indent=2), '```', '',
              f"few-shot 示例数：{plan['num_shots']}；提示长度：{min(len(q['input_ids']) for q in plan['questions'])}～{max(len(q['input_ids']) for q in plan['questions'])} token。", '',
              f"实际布局：`{info.get('hicache_mem_layout')}`；I/O：`{info.get('hicache_io_backend')}`。", '',
              '| 题目 | 标准答案 | A | R | L1 命中 | L2 命中 | 输出 token 全等 | 路径覆盖 |',
              '|---|---|---|---|---:|---:|---|---|']
    rows = []
    for q in plan['questions']:
        case = result / q['qid']
        if not (case / 'verdict.json').exists():
            continue
        row = read(case / 'verdict.json')
        rows.append(row)
        tiers = row['cached_tokens_details']
        lines.append(f"| {row['qid']} | {row['ground_truth']} | {row['baseline_answer']} | {row['replay_answer']} | {tiers['device']} | {tiers['host']} | {row['token_match']} | {row['path_covered']} |")
    lines += ['', '每题依次 flush → A（重算）→ 等待 A 的七池各层 D2H → B（65536 token，不同首 token）→ R（原输入）→ 等待 R 的七池各层 H2D。', '',
              '请求固定 routed_dp_rank=0；A/R temperature=0、max_new_tokens=512、stop=Question:；B max_new_tokens=8、ignore_eos=true。', '',
              '完整 input_ids、sampling_params、RID 见每题 requests.jsonl；响应、缓存分项见 responses.jsonl。B 输入固定随机种子 60001，保存于 eviction_input_ids.json，不参与数学评分。', '',
              '### 逐池逐层完成证据', '',
              '| 池 | 每题物理层数 | A 阶段 D2H 完成调用合计 | R 阶段 H2D 完成调用合计 | 每题双向全层覆盖 |',
              '|---|---:|---:|---:|---|']
    if rows:
        for pool in rows[0]['d2h']:
            d = [r['d2h'][pool] for r in rows]
            h = [r['h2d'][pool] for r in rows]
            layers = sorted(set(len(v['layers']) for v in d + h))
            lines.append(f"| {pool} | {layers} | {sum(v['calls'] for v in d)} | {sum(v['calls'] for v in h)} | {all(v['complete'] for v in d+h)} |")
        evidence = read(result / rows[0]['qid'] / 'aiv_evidence.json')
        samples = []
        for direction in ('D2H', 'H2D'):
            samples.extend(next((e for e in evidence if e.get('pool') == pool and e.get('direction') == direction and e.get('event') == 'aiv_complete'), {}) for pool in rows[0]['d2h'])
        lines += ['', '首题各池双向关键完成记录：', '', '```json', json.dumps(samples, ensure_ascii=False, indent=2), '```', '']
lines += ['## 结论边界', '',
          '本评测用于判断真实 NPU 数据通路、样本精度及答案保持；不等同完整 GSM8K 测试集精度验收。长 few-shot 与此前短 5-shot 的输入协议不同，不能直接比较百分比。', '',
          'flush 建立的是 KV 缓存冷态，不能重置进程内 C4 状态历史。若同目录含两题和十题结果，则十题复用了两题预检后的服务；单独的一百题目录使用新启动服务，常规服务 warmup 后直接开始评测。', '',
          '观测增加 event 和 JSONL 开销，本次不评价 TTFT，也未验证其他模型、MTP、并发及其他布局。', '',
          '原始证据目录：`' + str(root) + '`。', '']
(root / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines[:8]))
