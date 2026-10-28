"""Run audited GLM-5.2 MLA/Indexer GSM8K cases; save evidence and stop own server."""
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
count = int(sys.argv[3]) if len(sys.argv) > 3 else 100
model_path = os.environ.get('MODEL_PATH', '/mnt/paas/weights/GLM-5.2-w4a8c8')
data_path = os.environ.get('DATA_PATH', '/home/l00889655/sglang/gsm8k_test.jsonl')
port = int(os.environ.get('PORT', '32348'))
sglang_root = os.environ.get('SGLANG_ROOT', '/home/l00889655/sglang/sglang')
if count < 1 or count > 100:
    raise ValueError('count must be 1..100')
if not Path(data_path).is_file() or not (Path(model_path) / 'config.json').is_file():
    raise ValueError('DATA_PATH and MODEL_PATH/config.json must exist')
if not (Path(sglang_root) / 'python' / 'sglang').is_dir():
    raise ValueError('SGLANG_ROOT must point to the sglang checkout')
root.mkdir(exist_ok=False)
scripts = root / 'scripts'
scripts.mkdir()
for name in ('gsm8k_aiv.py', 'aiv_audit_hook.py', 'run_server.sh', 'report_npu.py', 'verify_saved_run.py', 'run_100.py'):
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
    files = [*scripts.iterdir(), Path(data_path)]
    save('sha256.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    rc = run([sys.executable, str(scripts / 'gsm8k_aiv.py'), 'plan',
              '--test', data_path,
              '--num-shots', '5', '--num-questions', str(count),
              '--model-path', model_path,
              '--output', str(root / 'plan.json')], 'plan.log', 180)
    if rc:
        raise RuntimeError(f'Plan failed: {rc}')
    env = {**os.environ, 'AUDIT_DIR': str(root / 'audit'), 'NPU_DEVICES': devices,
           'PORT': str(port), 'MODEL_PATH': model_path, 'SGLANG_ROOT': sglang_root}
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
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3) as response:
                if response.status == 200:
                    break
        except Exception:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError('Server health timeout')
        time.sleep(5)
    status['evaluation_exit'] = run([sys.executable, '-u', str(scripts / 'gsm8k_aiv.py'), 'eval',
        '--plan', str(root / 'plan.json'), '--audit-dir', str(root / 'audit'),
        '--base-url', f'http://127.0.0.1:{port}', '--eviction-tokens', '0',
        '--max-new-tokens', '512',
        '--output', str(root / 'results')], 'eval.log', 21600)
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
    if (root / 'results/summary.json').exists():
        status['summary'] = json.loads((root / 'results/summary.json').read_text())
        if count == 100:
            try:
                status['independent_audit_exit'] = run(
                    [sys.executable, str(scripts / 'verify_saved_run.py'), str(root)],
                    'independent_audit.log', 180)
            except Exception as exc:
                status['independent_audit_error'] = repr(exc)
        save('run_status.json', status)
        try:
            status['report_exit'] = run([sys.executable, str(scripts / 'report_npu.py'), str(root)], 'report.log', 60)
        except Exception as exc:
            status['report_error'] = repr(exc)
    save('run_status.json', status)
    save('stage.json', {'time': stamp(), 'stage': 'finished', **status})
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
sys.exit(0 if status['evaluation_exit'] == 0 and status['error'] is None
         and status.get('report_exit') == 0
         and (count != 100 or status.get('independent_audit_exit') == 0) else 1)
