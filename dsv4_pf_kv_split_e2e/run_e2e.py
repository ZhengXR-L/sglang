"""Run DeepSeek-V4 long-few-shot A/B/R transfer validation; stop own server."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

source = Path(__file__).resolve().parent
root = Path(sys.argv[1]).resolve()
devices = sys.argv[2]
count = int(sys.argv[3]) if len(sys.argv) > 3 else 3
if not 1 <= count <= 100:
    raise ValueError('Question count must be 1..100')
model_path = Path(os.environ.get('MODEL_PATH', '/mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp'))
train_path = Path(os.environ.get('TRAIN_PATH', '/home/l00889655/sglang/gsm8k_train.jsonl'))
test_path = Path(os.environ.get('TEST_PATH', '/home/l00889655/sglang/gsm8k_test.jsonl'))
sglang_root = Path(os.environ.get('SGLANG_ROOT', '/home/l00889655/sglang/sglang'))
kernel_root = Path(os.environ.get('KERNEL_ROOT', '/home/l00889655/sglang/sgl-kernel-npu'))
port = int(os.environ.get('PORT', '32420'))
dp_rank = int(os.environ.get('DP_RANK', '0'))
eviction_tokens = int(os.environ.get('EVICTION_TOKENS', '65536'))
startup_timeout = int(os.environ.get('STARTUP_TIMEOUT', '3600'))

weight_index_path = next((model_path / name for name in (
    'quant_model_weights.safetensors.index.json', 'model.safetensors.index.json')
    if (model_path / name).is_file()), None)
if weight_index_path is None:
    raise FileNotFoundError(f'No safetensors index in {model_path}')
for path in (model_path / 'config.json', model_path / 'tokenizer.json',
             weight_index_path, train_path, test_path):
    if not path.is_file():
        raise FileNotFoundError(path)
weight_index = json.loads(weight_index_path.read_text())
missing_shards = [name for name in set(weight_index['weight_map'].values())
                  if not (model_path / name).is_file()]
if missing_shards:
    raise FileNotFoundError(f'Missing model shards: {missing_shards[:5]}')
root.mkdir(exist_ok=False)
scripts = root / 'scripts'
scripts.mkdir()
for name in ('gsm8k_aiv.py', 'aiv_audit_hook.py', 'run_server.sh', 'report_npu.py',
             'run_e2e.py'):
    shutil.copy2(source / name, scripts / name)


def save(name, data):
    (root / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(args, logfile, timeout):
    save('stage.json', {'time': stamp(), 'command': args, 'log': logfile})
    with (root / logfile).open('w') as log:
        return subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode


server = None
status = {'start_utc': stamp(), 'evaluation_exit': None, 'error': None}
try:
    files = [*scripts.iterdir(), train_path, test_path, model_path / 'config.json']
    save('sha256.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    rc = run([sys.executable, str(scripts / 'gsm8k_aiv.py'), 'plan',
              '--train', str(train_path),
              '--test', str(test_path),
              '--test-offset', '5', '--num-questions', str(count),
              '--model-path', str(model_path),
              '--min-prompt-tokens', '8192', '--max-shots', '128', '--seed', '0',
              '--output', str(root / f'plan_{count}.json')], 'plan.log', 180)
    if rc:
        raise RuntimeError(f'Plan failed: {rc}')
    env = {**os.environ, 'AUDIT_DIR': str(root / 'audit'), 'NPU_DEVICES': devices,
           'PORT': str(port), 'MODEL_PATH': str(model_path), 'SGLANG_ROOT': str(sglang_root),
           'KERNEL_ROOT': str(kernel_root)}
    with (root / 'server.log').open('w') as log:
        server = subprocess.Popen(['bash', str(scripts / 'run_server.sh')], env=env,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (root / 'server.pid').write_text(str(server.pid))
    save('stage.json', {'time': stamp(), 'stage': 'waiting_health', 'server_pid': server.pid})
    deadline = time.monotonic() + startup_timeout
    while True:
        if server.poll() is not None:
            raise RuntimeError(f'Server exited during startup: {server.returncode}')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3) as response:
                if response.status == 200:
                    break
        except Exception:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError('Server health timeout')
        time.sleep(5)
    status['evaluation_exit'] = run([sys.executable, '-u', str(scripts / 'gsm8k_aiv.py'), 'eval',
        '--plan', str(root / f'plan_{count}.json'), '--audit-dir', str(root / 'audit'),
        '--base-url', f'http://127.0.0.1:{port}', '--dp-rank', str(dp_rank),
        '--eviction-tokens', str(eviction_tokens),
        '--max-new-tokens', '512', '--output', str(root / f'results_{count}')], 'eval.log', 21600)
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
    status['server_exit'] = server.returncode if server is not None else None
    if (root / f'results_{count}/summary.json').exists():
        status['summary'] = json.loads((root / f'results_{count}/summary.json').read_text())
        try:
            status['report_exit'] = run([sys.executable, str(scripts / 'report_npu.py'), str(root)], 'report.log', 60)
        except Exception as exc:
            status['report_error'] = repr(exc)
    save('run_status.json', status)
    save('stage.json', {'time': stamp(), 'stage': 'finished', **status})
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
sys.exit(0 if status['evaluation_exit'] == 0 and status['error'] is None
         and status.get('report_exit') == 0 else 1)
