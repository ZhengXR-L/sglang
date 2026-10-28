"""Run a TP8 GLM-5.2 layer/page ABBA TTFT comparison on free NPUs."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

source = Path(__file__).resolve().parent
root = Path(sys.argv[1]).resolve()
devices = sys.argv[2] if len(sys.argv) > 2 else '4,5,6,7,8,9,14,15'
ids = [int(x) for x in devices.split(',')]
assert len(ids) == len(set(ids)) == 8 and all(0 <= x < 16 for x in ids)
assert not root.exists(), f'Run directory already exists: {root}'
root.mkdir()
scripts = root/'scripts'; scripts.mkdir()
for name in ('bench.py', 'run_server.sh', 'run_compare.py'):
    shutil.copy2(source/name, scripts/name)
repo = '/home/l00889655/sglang/sglang'
port = 32432
model = '/mnt/paas/weights/GLM-5.2-w4a8c8'
order = ('layer_first', 'page_first_kv_split', 'page_first_kv_split', 'layer_first')
stamp = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(name, value):
    (root/name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def capture(args, path, timeout=60):
    with (root/path).open('w') as log:
        return subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode


def busy_devices():
    output = subprocess.check_output(['npu-smi', 'info'], text=True, timeout=30)
    (root/'waiting_npu.log').write_text(output)
    rows = re.findall(r'^\|\s+(\d+)\s+(\d+)\s+\|\s+(\d+)\s+\|', output, re.M)
    return [{'device': 2*int(card)+int(chip), 'pid': int(pid)}
            for card, chip, pid in rows if 2*int(card)+int(chip) in ids]


def wait_devices(block):
    deadline = time.monotonic()+43200
    while True:
        busy = busy_devices()
        if not busy:
            time.sleep(10)
            if not busy_devices():
                return
        save('stage.json', {'time': stamp(), 'stage': 'waiting_free_devices',
                            'block': block, 'busy': busy})
        if time.monotonic() > deadline:
            raise TimeoutError('Selected devices did not become free within 12 hours')
        time.sleep(30)


def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered)-1)*p
    lo = int(position)
    hi = min(lo+1, len(ordered)-1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)


def report(manifest, status):
    blocks = []
    for entry in manifest['blocks']:
        path = root/entry['name']/'result.json'
        if path.exists():
            blocks.append({**entry, **json.loads(path.read_text())})
    groups = {}
    for layout in ('layer_first', 'page_first_kv_split'):
        vals = [x for block in blocks if block['layout'] == layout for x in block['ttft_ms']]
        if vals:
            groups[layout] = {'n': len(vals), 'mean_ms': statistics.mean(vals),
                              'median_ms': statistics.median(vals), 'p95_ms': percentile(vals, .95)}
    complete = (status['error'] is None and len(blocks) == 4
                and all(len(b['ttft_ms']) == 10
                        and b['needle_matches'] == 10 and b['d2h_markers'] > 0
                        and b['h2d_markers'] > 0 for b in blocks))
    comparison = {'complete': complete, 'groups': groups,
                  'blocks': [{k: b[k] for k in ('name', 'layout', 'mean_ttft_ms',
                                               'median_ttft_ms', 'p95_ttft_ms',
                                               'd2h_markers', 'h2d_markers',
                                               'exact_replay', 'needle_matches')}
                             for b in blocks]}
    if len(groups) == 2:
        comparison['layer_change_pct'] = 100*(groups['layer_first']['mean_ms'] /
                                               groups['page_first_kv_split']['mean_ms']-1)
    save('comparison.json', comparison)
    lines = ['# GLM-5.2 W4A8C8 kernel_ascend 布局 TTFT 对比', '',
             '比较 layer_first_kv_split 与 page_first_kv_split 的实际服务路径。', '',
             'TP8/EP8、NPU '+devices+'；并发 2，每请求约 1000 输入 token、最多 16 输出 token；'
             'KV 容量 3072 token，HiCache ratio 2.5，write_through。', '',
             '每条测试采用 A 填充、B 驱逐、R 重放；每次服务预热一轮，再测五轮，'
             '两条并发请求每次服务产生 10 条正式 TTFT 样本。四次服务顺序为 layer/page/page/layer。', '',
             'TTFT 从流式 /v1/completions 请求发送到首次非空文本。'
             '每轮要求 D2H/H2D 日志标记和预设答案一致；A/R 文本一致率单独报告。'
             '这些日志标记验证发生了传输，但不提供每请求精确的 L1/L2 命中 token 数。', '',
             '## 结果', '',
             '| 布局 | 样本数 | 平均 TTFT ms | 中位数 ms | p95 ms |',
             '|---|---:|---:|---:|---:|']
    for layout in ('layer_first', 'page_first_kv_split'):
        if layout in groups:
            g = groups[layout]
            lines.append(f"| {layout} | {g['n']} | {g['mean_ms']:.2f} | {g['median_ms']:.2f} | {g['p95_ms']:.2f} |")
    lines += ['', f'完整四轮比较：{complete}。']
    if 'layer_change_pct' in comparison:
        lines.append(f"layer 相对 page 的平均 TTFT 变化：{comparison['layer_change_pct']:+.2f}%（负数表示更快）。")
    lines += ['', '## 各轮结果', '',
              '| 轮次 | 布局 | 平均 ms | D2H 标记 | H2D 标记 | A/R 文本一致 | 答案一致 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for b in blocks:
        lines.append(f"| {b['name']} | {b['layout']} | {b['mean_ttft_ms']:.2f} | "
                     f"{b['d2h_markers']} | {b['h2d_markers']} | "
                     f"{b['exact_replay']}/10 | {b['needle_matches']}/10 |")
    lines += ['', '## 参数与证据', '', '```json',
              json.dumps(manifest, ensure_ascii=False, indent=2), '```', '',
              '逐轮请求结果、server_info、日志和 NPU 快照保存在源码目录外的运行目录：'+str(root)+'。', '',
              '本测试沿用早期 GLM 的双并发工作负载并按最新 TP8 精度测试配置缩短前缀；'
              '因此不能与此前 TP16、约 3000-token 前缀的绝对 TTFT 直接比较。'
              '共享服务器负载和有限样本数也限制了外推。', '']
    (root/'report.md').write_text('\n'.join(lines), encoding='utf-8')


manifest = {'start_utc': stamp(), 'devices': devices, 'model_path': model,
            'commit': subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip(),
            'git_status': subprocess.check_output(['git', '-C', repo, 'status', '--short'], text=True),
            'script_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts.iterdir()},
            'order': order, 'blocks': []}
save('manifest.json', manifest)
status = {'start_utc': stamp(), 'error': None, 'finished_blocks': 0}
server = None
try:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
    for index, layout in enumerate(order):
        name = f'{index+1}_{layout}'
        wait_devices(name)
        current = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
        if current != manifest['commit']:
            raise RuntimeError('Source HEAD changed during comparison')
        block = root/name; block.mkdir()
        manifest['blocks'].append({'name': name, 'layout': layout})
        save('manifest.json', manifest)
        save('stage.json', {'time': stamp(), 'stage': 'starting', 'block': name})
        capture(['npu-smi', 'info'], f'{name}/npu_before.log')
        env = {**os.environ, 'NPU_DEVICES': devices, 'LAYOUT': layout, 'PORT': str(port)}
        with (block/'server.log').open('w') as log:
            server = subprocess.Popen(['bash', str(scripts/'run_server.sh')], env=env,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (block/'server.pid').write_text(str(server.pid))
        deadline = time.monotonic()+900
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'{name}: server exited {server.returncode}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f'{name}: health timeout')
            time.sleep(5)
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/server_info', timeout=30) as resp:
            info = json.load(resp)
        save(f'{name}/server_info.json', info)
        expected = 'layer_first_kv_split' if layout == 'layer_first' else 'page_first_kv_split'
        if info.get('tp_size') != 8 or info.get('hicache_mem_layout') != expected \
                or info.get('hicache_io_backend') != 'kernel_ascend' or info.get('model_path') != model:
            raise ValueError(f'{name}: resolved server config differs from requested TP8/kernel_ascend/layout')
        save('stage.json', {'time': stamp(), 'stage': 'measuring', 'block': name})
        rc = capture([sys.executable, '-u', str(scripts/'bench.py'),
                      '--base-url', f'http://127.0.0.1:{port}', '--layout', layout,
                      '--server-log', str(block/'server.log'), '--model-path', model,
                      '--output-json', str(block/'result.json'), '--num-prompts', '2',
                      '--prompt-tokens', '1000', '--max-tokens', '16',
                      '--warmup-rounds', '1', '--measure-rounds', '5',
                      '--l1-tokens', '3072', '--l2-ratio', '2.5'], f'{name}/bench.log', 7200)
        if rc:
            raise RuntimeError(f'{name}: benchmark exited {rc}')
        result = json.loads((block/'result.json').read_text())
        if len(result['ttft_ms']) != 10 or result['needle_matches'] != 10:
            raise ValueError(f'{name}: incomplete or inconsistent replay')
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=15)
        server = None
        status['finished_blocks'] += 1
        capture(['npu-smi', 'info'], f'{name}/npu_after.log')
        save('run_status.json', status)
        time.sleep(5)
except Exception as exc:
    status['error'] = repr(exc)
finally:
    if server is not None and server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=15)
    status['end_utc'] = stamp()
    report(manifest, status)
    save('run_status.json', status)
    save('stage.json', {'stage': 'finished', **status})
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
sys.exit(0 if status['error'] is None else 1)
