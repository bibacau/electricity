"""Offline entry point for the assembled portable Windows distribution.

No downloads, pip, model installers, administrator elevation, or system changes.
"""
import hashlib
import json
import os
import secrets
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def verify_assets(root):
    manifest=json.loads((root/'bundle.json').read_text(encoding='utf-8'))
    if manifest.get('complete') is not True:
        raise RuntimeError('This build is incomplete; the publisher must finish bundling the local model. Nothing will be downloaded or installed on your computer.')
    if manifest.get('platform')!='windows-x64':
        raise RuntimeError('This is not a Windows x64 bundle')
    for relative,expected in manifest['assets'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256(path)!=expected:
            raise RuntimeError(f'Bundled file is missing or changed: {relative}. No repair/download will be attempted.')
    return manifest


def wait_for_engine(port,process,seconds=120):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if process.poll() is not None:
            raise RuntimeError('The bundled engine exited. See data/engine.log; CPU compatibility or available RAM may be the cause.')
        try:
            with opener.open(f'http://127.0.0.1:{port}/health',timeout=1) as r:
                if r.status==200:
                    return
        except (OSError,urllib.error.URLError):
            pass
        time.sleep(.25)
    raise RuntimeError('Bundled engine startup timed out; see data/engine.log')


def open_when_ready(port,cancel):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(100):
        if cancel.is_set():
            return
        try:
            with opener.open(f'http://127.0.0.1:{port}/api/health',timeout=1) as r:
                if r.status==200:
                    webbrowser.open(f'http://127.0.0.1:{port}')
                    return
        except (OSError,urllib.error.URLError):
            pass
        cancel.wait(.1)


def main():
    root=Path(__file__).resolve().parent
    if sys.platform!='win32':
        raise RuntimeError('This portable distribution targets Windows x64')
    manifest=verify_assets(root)
    sys.path[:0]=[str(root/'vendor'),str(root/'app')]
    data=root/'data'
    data.mkdir(exist_ok=True)
    engine_port,app_port=free_port(),free_port()
    while app_port==engine_port:
        app_port=free_port()
    os.environ['PERSONAL_AI_BACKEND']='llama.cpp'
    os.environ['PERSONAL_AI_DATA']=str(data)
    os.environ['OLLAMA_URL']=f'http://127.0.0.1:{engine_port}'
    # Per-launch loopback authentication; never stored in the bundle or database.
    engine_key=secrets.token_urlsafe(32)
    os.environ['PERSONAL_AI_ENGINE_KEY']=engine_key
    from personal_ai.store import Store
    store=Store(data)
    if not store.settings()['model']:
        settings=store.settings()
        settings['model']=manifest['model_alias']
        store.execute('UPDATE settings SET value=? WHERE id=1',(json.dumps(settings),))
    cancel=threading.Event()
    with (data/'engine.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen([
            str(root/'engine'/'llama-server.exe'),'-m',str(root/manifest['model_path']),
            '--host','127.0.0.1','--port',str(engine_port),'--alias',manifest['model_alias'],
            '--no-webui','-c','4096','-np','1','-ngl','0','-t',str(max(1,min(4,(os.cpu_count() or 2)-1))),
        ],cwd=root,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,
            env={**os.environ,'LLAMA_API_KEY':engine_key})
        try:
            print('Loading the bundled local model. No installation or Internet is required.')
            wait_for_engine(engine_port,process)
            threading.Thread(target=open_when_ready,args=(app_port,cancel),daemon=True).start()
            import uvicorn
            from personal_ai.api import create_app
            print('Atlas is running. Keep this window open; press Ctrl+C here to stop.')
            uvicorn.run(create_app(),host='127.0.0.1',port=app_port)
        finally:
            cancel.set()
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(f'Atlas could not start: {exc}',file=sys.stderr)
        sys.exit(1)
