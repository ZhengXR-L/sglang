"""Run the unchanged serial protocol on 100 cases; save evidence and stop own server."""
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
root.mkdir(exist_ok=False)
scripts = root / 'scripts'
scripts.mkdir()
for name in ('gsm8k_aiv.py', 'aiv_audit_hook.py', 'run_server.sh', 'report_npu.py', 'run_100.py'):
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
    files = [*scripts.iterdir(), Path('/home/l00889655/sglang/gsm8k_train.jsonl'),
             Path('/home/l00889655/sglang/gsm8k_test.jsonl')]
    save('sha256.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    rc = run([sys.executable, str(scripts / 'gsm8k_aiv.py'), 'plan',
              '--train', '/home/l00889655/sglang/gsm8k_train.jsonl',
              '--test', '/home/l00889655/sglang/gsm8k_test.jsonl',
              '--test-offset', '5', '--num-questions', '100',
              '--model-path', '/mnt/paas/weights/DeepSeek-V4-Flash-0731-w8a8',
              '--min-prompt-tokens', '8192', '--max-shots', '128', '--seed', '0',
              '--output', str(root / 'plan_100.json')], 'plan_100.log', 180)
    if rc:
        raise RuntimeError(f'Plan failed: {rc}')
    env = {**os.environ, 'AUDIT_DIR': str(root / 'audit'), 'NPU_DEVICES': devices, 'PORT': '32420'}
    with (root / 'server.log').open('w') as log:
        server = subprocess.Popen(['bash', str(scripts / 'run_server.sh')], env=env,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (root / 'server.pid').write_text(str(server.pid))
    save('stage.json', {'time': stamp(), 'stage': 'waiting_health', 'server_pid': server.pid})
    deadline = time.monotonic() + 900
    while True:
        if server.poll() is not None:
            raise RuntimeError(f'Server exited during startup: {server.returncode}')
        try:
            with urllib.request.urlopen('http://127.0.0.1:32420/health', timeout=3) as response:
                if response.status == 200:
                    break
        except Exception:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError('Server health timeout')
        time.sleep(5)
    status['evaluation_exit'] = run([sys.executable, '-u', str(scripts / 'gsm8k_aiv.py'), 'eval',
        '--plan', str(root / 'plan_100.json'), '--audit-dir', str(root / 'audit'),
        '--base-url', 'http://127.0.0.1:32420', '--dp-rank', '0', '--eviction-tokens', '65536',
        '--max-new-tokens', '512', '--output', str(root / 'results_100')], 'eval_100.log', 21600)
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
    if (root / 'results_100/summary.json').exists():
        status['summary'] = json.loads((root / 'results_100/summary.json').read_text())
        try:
            status['report_exit'] = run([sys.executable, str(scripts / 'report_npu.py'), str(root)], 'report.log', 60)
        except Exception as exc:
            status['report_error'] = repr(exc)
    save('run_status.json', status)
    save('stage.json', {'time': stamp(), 'stage': 'finished', **status})
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
sys.exit(0 if status['evaluation_exit'] == 0 and status['error'] is None else 1)
