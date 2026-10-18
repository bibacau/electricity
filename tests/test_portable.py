import asyncio
import importlib.util
import json
from pathlib import Path
import zipfile

import httpx
import pytest

from personal_ai.inference import LlamaCpp, ModelUnavailable


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_portable_integrity_and_incomplete_gate(tmp_path):
    launcher=load_file('portable_launcher',Path('portable/launcher.py'))
    asset=tmp_path/'model.gguf'
    asset.write_bytes(b'checksum fixture only')
    manifest={'complete':False,'platform':'windows-x64','assets':{'model.gguf':launcher.sha256(asset)}}
    (tmp_path/'bundle.json').write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError,match='incomplete'):
        launcher.verify_assets(tmp_path)
    manifest['complete']=True
    (tmp_path/'bundle.json').write_text(json.dumps(manifest))
    assert launcher.verify_assets(tmp_path)['complete']
    asset.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='changed'):
        launcher.verify_assets(tmp_path)
    manifest['assets']={'../outside':'0'*64}
    (tmp_path/'bundle.json').write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError):
        launcher.verify_assets(tmp_path)


def test_archive_traversal_is_rejected(tmp_path):
    builder=load_file('portable_builder',Path('scripts/build_portable.py'))
    for name in ['../outside.txt',r'..\outside.txt','C:/outside.txt']:
        archive=tmp_path/'unsafe.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr(name,'not permitted')
        with pytest.raises(ValueError):
            builder.extract(archive,tmp_path/'destination')


def test_llama_server_protocol(monkeypatch):
    original=httpx.AsyncClient
    def handler(request):
        assert request.headers['Authorization']=='Bearer contract-fixture-key'
        if request.url.path=='/v1/models':
            return httpx.Response(200,json={'data':[{'id':'atlas-local'}]})
        assert request.url.path=='/v1/chat/completions'
        body=json.loads(request.content)
        assert body['model']=='atlas-local'
        assert body['stream'] is False
        return httpx.Response(200,json={'choices':[{'message':{'content':'protocol fixture'}}]})
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:original(**kwargs,transport=httpx.MockTransport(handler)))
    async def exercise():
        model=LlamaCpp('http://127.0.0.1:12345',api_key='contract-fixture-key')
        assert await model.models()==[{'name':'atlas-local'}]
        assert await model.chat('atlas-local',[{'role':'user','content':'hello'}])=='protocol fixture'
        with pytest.raises(ModelUnavailable,match='keyword recall'):
            await model.embed('unsupported','hello')
    asyncio.run(exercise())
