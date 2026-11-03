"""Own four server lifetimes, run ABBA benchmark, save report, clean up own groups."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

src = Path(__file__).resolve().parent
root = Path(sys.argv[1]).resolve()
devices = sys.argv[2]
assert len(set(devices.split(','))) == 8 and all(x.isdigit() for x in devices.split(','))
root.mkdir(exist_ok=False)
scripts = root/'scripts'; scripts.mkdir()
for name in ('run_server.sh', 'bench.py', 'run_compare.py'):
    shutil.copy2(src/name, scripts/name)
repo = os.environ.get('SGLANG_ROOT', '/home/l00889655/sglang/sglang')
kernel_repo = os.environ.get('KERNEL_ROOT', '/home/l00889655/sglang/sgl-kernel-npu')
port = int(os.environ.get('PORT', '32432'))
startup_timeout = int(os.environ.get('STARTUP_TIMEOUT', '2400'))
model_path = os.environ.get('MODEL_PATH', '/mnt/paas/weights/GLM-5.2-w4a8c8')
if not (Path(model_path) / 'tokenizer.json').is_file():
    raise FileNotFoundError(f'Missing tokenizer.json in {model_path}')
stamp = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(name, value):
    (root/name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def capture(args, path, timeout=60):
    with (root/path).open('w') as log:
        return subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode


def wait_devices(block):
    if os.environ.get('GLM52_PERF_START_IMMEDIATELY') == '1':
        return
    wanted = {int(x) for x in devices.split(',')}
    deadline = time.monotonic()+43200
    while True:
        output = subprocess.check_output(['npu-smi','info'], text=True, timeout=30)
        (root/'waiting_npu.log').write_text(output)
        rows = re.findall(r'^\|\s+(\d+)\s+(\d+)\s+\|\s+(\d+)\s+\|', output, re.M)
        busy = [{'device':2*int(card)+int(chip),'pid':int(pid)} for card,chip,pid in rows
                if 2*int(card)+int(chip) in wanted]
        if not busy:
            # Avoid capturing the gap between a just-launched process's context and memory setup.
            time.sleep(10)
            output2 = subprocess.check_output(['npu-smi','info'], text=True, timeout=30)
            rows2 = re.findall(r'^\|\s+(\d+)\s+(\d+)\s+\|\s+(\d+)\s+\|', output2, re.M)
            if not any(2*int(a)+int(b) in wanted for a,b,_ in rows2):
                return
        save('stage.json', {'time':stamp(),'stage':'waiting_free_devices','block':block,'busy':busy})
        if time.monotonic() > deadline:
            raise TimeoutError('Eight devices did not become free within 12 hours')
        time.sleep(30)


manifest = {'start_utc': stamp(), 'devices': devices,
    'plan_generator': 'builtin_arithmetic_v1',
    'commit': subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'], text=True).strip(),
    'kernel_commit': subprocess.check_output(['git','-C',kernel_repo,'rev-parse','HEAD'], text=True).strip(),
    'model_path': model_path, 'port': port,
    'git_status': subprocess.check_output(['git','-C',repo,'status','--short'], text=True),
    'kernel_git_status': subprocess.check_output(['git','-C',kernel_repo,'status','--short'], text=True),
    'script_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts.iterdir()},
    'order': ['layer_first','page_first','page_first','layer_first'], 'blocks': []}
save('manifest.json', manifest)
status = {'start_utc': stamp(), 'error': None, 'finished_blocks': 0}
server = None
try:
    with socket.socket() as sock: sock.bind(('127.0.0.1',port))
    if capture([sys.executable,str(scripts/'bench.py'),'prepare','--model-path',model_path,
                '--output',str(root/'plan.json')], 'prepare.log',180):
        raise RuntimeError('Plan preparation failed')
    manifest['plan_sha256'] = hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()
    save('manifest.json', manifest)
    for index, layout in enumerate(manifest['order']):
        name = f'{index+1}_{layout}'
        wait_devices(name)
        current = subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()
        kernel_current = subprocess.check_output(['git','-C',kernel_repo,'rev-parse','HEAD'],text=True).strip()
        if current != manifest['commit'] or kernel_current != manifest['kernel_commit']:
            raise RuntimeError('Source HEAD changed while waiting; stop rather than compare different versions')
        block = root/name; block.mkdir()
        manifest['blocks'].append({'name': name, 'layout': layout})
        save('manifest.json', manifest)
        save('stage.json', {'time': stamp(), 'stage':'starting', 'block':name})
        capture(['npu-smi','info'], f'{name}/npu_before.log')
        capture(['bash','-lc','uptime; free -h'], f'{name}/host_before.log')
        env = {**os.environ, 'NPU_DEVICES':devices, 'LAYOUT':layout, 'PORT':str(port),
               'SGLANG_ROOT':repo, 'KERNEL_ROOT':kernel_repo, 'MODEL_PATH':model_path}
        with (block/'server.log').open('w') as log:
            server = subprocess.Popen(['bash',str(scripts/'run_server.sh')], env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        save(f'{name}/environment.json', {k:v for k,v in env.items() if k in ('NPU_DEVICES','LAYOUT','PORT','SGLANG_ROOT','KERNEL_ROOT','MODEL_PATH')})
        (block/'server.pid').write_text(str(server.pid))
        deadline = time.monotonic()+startup_timeout
        while True:
            if server.poll() is not None: raise RuntimeError(f'{name}: server exited {server.returncode}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3) as resp:
                    if resp.status == 200: break
            except Exception: pass
            if time.monotonic() > deadline: raise TimeoutError(f'{name}: health timeout')
            time.sleep(5)
        save('stage.json', {'time':stamp(),'stage':'measuring','block':name})
        rc = capture([sys.executable,'-u',str(scripts/'bench.py'),'eval','--plan',str(root/'plan.json'),
                      '--output',str(block/'results'),'--layout',layout,
                      '--base-url',f'http://127.0.0.1:{port}'], f'{name}/bench.log',7200)
        if rc: raise RuntimeError(f'{name}: benchmark exited {rc}')
        os.killpg(server.pid,signal.SIGTERM)
        try: server.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid,signal.SIGKILL); server.wait(timeout=15)
        server = None
        status['finished_blocks'] += 1
        capture(['npu-smi','info'], f'{name}/npu_after.log')
        save('run_status.json',status)
        time.sleep(5)
except Exception as exc:
    status['error'] = repr(exc)
finally:
    if server is not None and server.poll() is None:
        os.killpg(server.pid,signal.SIGTERM)
        try: server.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid,signal.SIGKILL); server.wait(timeout=15)
    status['end_utc'] = stamp()
    status['report_exit'] = capture([sys.executable,str(scripts/'bench.py'),'report','--root',str(root)], 'report.log',60)
    save('run_status.json',status)
    save('stage.json', {'stage':'finished',**status})
    print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)
comparison = json.loads((root/'comparison.json').read_text()) if (root/'comparison.json').exists() else {}
sys.exit(0 if status['error'] is None and status['report_exit'] == 0
         and status['finished_blocks'] == 4 and comparison.get('complete') else 1)
