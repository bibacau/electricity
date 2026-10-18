"""Explicit pretrained integration check. Requires existing verified engine/model files.

No downloads or model substitutions. Uses an isolated temporary workbench.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import time

import httpx
from fastapi.testclient import TestClient
from personal_ai.api import create_app


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    args=parser.parse_args()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port=sock.getsockname()[1]
    key=secrets.token_urlsafe(32)
    with tempfile.TemporaryDirectory(prefix='atlas-pretrained-test-') as directory:
        with (Path(directory)/'engine.log').open('w') as log:
            engine=args.engine.resolve()
            process=subprocess.Popen([str(engine),'-m',str(args.model.resolve()),'--host','127.0.0.1','--port',str(port),'--alias','atlas-local','--no-webui','-c','4096','-np','1','-ngl','0','-t','4'],cwd=engine.parent,env={**os.environ,'LLAMA_API_KEY':key},stdout=log,stderr=subprocess.STDOUT)
            try:
                base=f'http://127.0.0.1:{port}'
                with httpx.Client(base_url=base,trust_env=False,timeout=2) as native:
                    for _ in range(120):
                        if process.poll() is not None:
                            raise RuntimeError('Engine exited during startup')
                        try:
                            if native.get('/health').status_code==200:
                                break
                        except httpx.HTTPError:
                            pass
                        time.sleep(.25)
                    else:
                        raise RuntimeError('Engine readiness timed out')
                    assert native.get('/v1/models').status_code==401,'Native model endpoint must reject missing authentication'
                    assert native.get('/v1/models',headers={'Authorization':f'Bearer {key}'}).status_code==200
                os.environ['PERSONAL_AI_BACKEND']='llama.cpp'
                os.environ['OLLAMA_URL']=base
                os.environ['PERSONAL_AI_ENGINE_KEY']=key
                with TestClient(create_app(Path(directory)/'data'),base_url='http://127.0.0.1',headers={'X-Workbench':'1'}) as client:
                    models=client.get('/api/models');models.raise_for_status()
                    assert models.json()['models'][0]['name']=='atlas-local'
                    settings=client.get('/api/settings').json()
                    settings.update(model='atlas-local',name='Atlas')
                    client.put('/api/settings',json=settings).raise_for_status()
                    client.post('/api/memories',json={'kind':'preference','text':'My favorite programming language is Python.','verified':True}).raise_for_status()
                    first=client.post('/api/chat',json={'text':'What is my favorite programming language? Reply in one short sentence.'})
                    first.raise_for_status()
                    answer=first.json()
                    assert 'python' in answer['content'].lower(),answer['content']
                    second=client.post('/api/chat',json={'conversation_id':answer['conversation_id'],'text':'What language did you just mention? Reply with the language name only.'})
                    second.raise_for_status()
                    assert 'python' in second.json()['content'].lower(),second.json()
                    assert len(client.get('/api/conversations/'+str(answer['conversation_id'])).json())==4
                    print(json.dumps({'result':'passed','native_authentication':'enforced','first_response':answer['content'],'followup':second.json()['content'],'persisted_messages':4}))
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__=='__main__':
    main()
