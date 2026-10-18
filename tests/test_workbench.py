import asyncio
import json
import sqlite3
import threading
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from personal_ai.api import create_app
from personal_ai.inference import ModelUnavailable, Ollama, retrieve
from personal_ai.learning import GridWorld, train_grid
from personal_ai.store import Store
from personal_ai.tools import ToolBox, ToolDenied
from personal_ai.training import encoded, fingerprint, promote, read_dataset, rollback

HEADERS={"X-Workbench":"1"}


class ContractModel:
    """Deterministic protocol fixture; never used as application inference."""
    def __init__(self):
        self.calls=[]
        self.fail=False

    async def models(self):
        return [{"name":"contract-test-model"}]

    async def chat(self,model,messages):
        if self.fail:
            raise ModelUnavailable("service unavailable")
        self.calls.append((model,messages))
        return "Contract fixture response"

    async def embed(self,model,text):
        return [1.0,0.5,0.25]


@pytest.fixture
def client(tmp_path):
    app=create_app(tmp_path,inference=ContractModel())
    with TestClient(app,headers=HEADERS,base_url='http://127.0.0.1') as client:
        yield client


def test_chat_personality_memory_history_and_persistence(client,tmp_path):
    config=client.get('/api/settings').json()
    config.update(name='Nova',model='contract-test-model')
    assert client.put('/api/settings',json=config).status_code==200
    memory=client.post('/api/memories',json={"kind":"preference","text":"I prefer Python examples","verified":True}).json()
    result=client.post('/api/chat',json={"text":"Show examples"}).json()
    assert result['recalled_ids']==[memory['id']]
    assert result['message_id']
    call=client.app.state.model.calls[-1]
    assert 'Nova' in call[1][0]['content']
    assert 'Python' in call[1][0]['content']
    again=client.post('/api/chat',json={"text":"Continue","conversation_id":result['conversation_id']})
    assert again.status_code==200
    assert len(client.app.state.model.calls[-1][1])==4
    store=Store(tmp_path)
    assert len(store.rows('SELECT * FROM messages'))==4
    assert store.settings()['name']=='Nova'


def test_model_failure_does_not_persist_fake_messages(client):
    client.app.state.model.fail=True
    response=client.post('/api/chat',json={"text":"hello"})
    assert response.status_code==503
    assert client.get('/api/conversations').json()==[]
    assert client.app.state.store.rows('SELECT * FROM messages')==[]


def test_memory_edit_delete_semantic_vectors(client):
    settings=client.get('/api/settings').json()
    settings['embedding_model']='embed-fixture'
    client.put('/api/settings',json=settings)
    mid=client.post('/api/memories',json={"text":"felines are cats"}).json()['id']
    row=client.app.state.store.rows('SELECT * FROM memories')[0]
    assert json.loads(row['embedding'])==[1,0.5,0.25]
    recalled=retrieve([row],'unrelated words',[1,0.5,0.25],'embed-fixture')
    assert recalled[0]['id']==mid
    assert client.put(f'/api/memories/{mid}',json={"kind":"assumption","text":"a revised note"}).status_code==200
    assert client.get('/api/memories').json()[0]['text']=='a revised note'
    assert client.delete(f'/api/memories/{mid}').status_code==200
    assert client.get('/api/memories').json()==[]


def test_feedback_curates_latest_corrected_dataset(client):
    r=client.post('/api/chat',json={"text":"Question"}).json()
    assert client.post('/api/feedback',json={"message_id":r['message_id'],"rating":1}).status_code==200
    assert client.post('/api/feedback',json={"message_id":r['message_id'],"rating":-1,"correction":"Correct answer"}).status_code==200
    data=client.get('/api/dataset').json()
    assert data['count']==1
    assert data['samples'][0]['messages'][1]['content']=='Correct answer'
    assert client.get('/api/memories').json()[0]['kind']=='correction'
    client.delete('/api/conversations/'+str(r['conversation_id']))
    assert client.get('/api/dataset').json()['count']==0


def test_private_chat_does_not_persist(client):
    config=client.get('/api/settings').json()
    config['remember_chat']=False
    client.put('/api/settings',json=config)
    response=client.post('/api/chat',json={"text":"private"}).json()
    assert response['message_id'] is None
    assert client.get('/api/conversations').json()==[]


def test_local_request_boundaries(client):
    assert client.post('/api/stop',headers={'Origin':'https://evil.example'}).status_code==403
    assert client.get('/api/export',headers={'Host':'evil.example'}).status_code==403
    with TestClient(client.app,base_url='http://127.0.0.1') as unauth:
        assert unauth.post('/api/stop').status_code==403
    assert client.post('/api/memories',content='x'*1_100_001).status_code==413
    assert client.get('/').headers['content-security-policy'].startswith("default-src 'self'")


def test_tool_permissions_stop_and_document_search(client,tmp_path):
    assert client.post('/api/documents',json={'name':'note.txt','text':'local knowledge'}).status_code==200
    assert client.post('/api/documents',json={'name':'note.txt','text':'overwrite'}).status_code==409
    assert client.post('/api/documents',json={'name':'../escape.txt','text':'no'}).status_code==422
    assert client.get('/api/documents/search?q=knowledge').json()[0]['name']=='note.txt'
    tools=client.app.state.tools
    with pytest.raises(ToolDenied):
        tools.write('../escape.txt','no')
    outside=tmp_path/'outside.txt'
    outside.write_text('private')
    (tools.root/'link.txt').symlink_to(outside)
    with pytest.raises(ToolDenied):
        tools.path('link.txt')
    assert client.post('/api/stop').status_code==200
    assert client.post('/api/chat',json={'text':'no'}).status_code==409
    assert client.post('/api/experiments',json={}).status_code==409
    assert client.post('/api/documents',json={'name':'stopped.txt','text':'no'}).status_code==403
    assert client.post('/api/resume').status_code==200


def test_backup_is_consistent_database(client,tmp_path):
    client.post('/api/memories',json={'text':'persistent knowledge'})
    r=client.post('/api/backup')
    assert r.status_code==200
    path=tmp_path/'copy.sqlite3'
    path.write_bytes(r.content)
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert db.execute('SELECT text FROM memories').fetchone()[0]=='persistent knowledge'


def test_grid_environment_and_real_policy_improvement():
    env=GridWorld(3)
    assert env.reset(seed=42)==env.reset(seed=42)
    env.position=(1,2)
    assert env.step(1)[1:4]==(1.0,True,False)
    with pytest.raises(ValueError):
        env.step(9)
    result=train_grid(1500,5,42,threading.Event(),30)
    assert result['completed_episodes']==1500
    assert result['candidate']['episodes']==100
    assert result['candidate']['success_rate']>=0.95
    assert result['candidate']['mean_reward']>result['baseline']['mean_reward']
    assert result['weight_updates'] is False
    assert len(result['policy'])>0


def test_training_stop_and_stale_jobs(tmp_path):
    stop=threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        train_grid(1000,5,42,stop,1)
    store=Store(tmp_path)
    store.execute("INSERT INTO experiments(status,config) VALUES ('running','{}')")
    from personal_ai.learning import Experiments
    jobs=Experiments(store,threading.Event())
    assert store.rows('SELECT status FROM experiments')[0]['status']=='interrupted'
    jobs.close()


def test_experiment_api_records_result(client):
    r=client.post('/api/experiments',json={'episodes':100,'size':3,'seed':42,'seconds':10})
    assert r.status_code==200
    client.app.state.jobs.future.result(timeout=15)
    row=client.get('/api/experiments').json()[0]
    assert row['status']=='completed'
    assert row['result']['completed_episodes']==100


def test_ollama_contract_and_connection_failure(monkeypatch):
    original=httpx.AsyncClient
    requests=[]
    def handler(request):
        requests.append(request)
        if request.url.path=='/api/tags':
            return httpx.Response(200,json={'models':[{'name':'local'}]})
        if request.url.path=='/api/embed':
            return httpx.Response(200,json={'embeddings':[[0.1,0.2]]})
        return httpx.Response(200,json={'message':{'content':'protocol response'}})
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw: original(**kw,transport=httpx.MockTransport(handler)))
    async def exercise():
        model=Ollama()
        assert (await model.models())[0]['name']=='local'
        assert await model.chat('local',[{'role':'user','content':'hello'}])=='protocol response'
        assert await model.embed('local','hello')==[0.1,0.2]
        with pytest.raises(ModelUnavailable):
            await model.chat('',[])
    asyncio.run(exercise())
    payload=json.loads(requests[1].content)
    assert payload['stream'] is False
    assert payload['options']['num_ctx']==4096
    with pytest.raises(ValueError):
        Ollama('https://external.example')


def test_dataset_and_checkpoint_gate(tmp_path):
    from argparse import Namespace
    dataset=tmp_path/'dataset.jsonl'
    dataset.write_text(json.dumps({'messages':[{'role':'user','content':'q'},{'role':'assistant','content':'a'}]})+'\n')
    assert len(read_dataset(dataset))==1
    dataset.write_text('{}')
    with pytest.raises(ValueError):
        read_dataset(dataset)
    adapter=tmp_path/'adapter'
    adapter.mkdir()
    (adapter/'weights').write_text('fixture: registry gate only')
    report={'metric':'heldout_completion_nll_lower_is_better','examples':4,'passed':True,'baseline':2.0,'candidate':1.5,'adapter':str(adapter),'adapter_hash':fingerprint(adapter)}
    report_path=tmp_path/'report.json'
    report_path.write_text(json.dumps(report))
    args=Namespace(report=report_path,name='v1',data=tmp_path/'state')
    promote(args)
    assert Store(args.data).rows('SELECT * FROM checkpoints')[0]['active']==1
    (adapter/'weights').write_text('changed')
    with pytest.raises(ValueError,match='changed'):
        promote(args)
    report['candidate']=3.0
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='regressed'):
        promote(args)
    with pytest.raises(ValueError,match='Unknown'):
        rollback(Namespace(data=args.data,name='missing'))


def test_completion_masking_rejects_incompatible_template():
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            return [1,2,3,4] if len(messages)==2 else [1,2]
    rows=[{'messages':[{'role':'user','content':'q'},{'role':'assistant','content':'a'}]}]
    assert encoded(rows,Tokenizer(),10)[0]['labels']==[-100,-100,3,4]
    with pytest.raises(ValueError,match='exceeds'):
        encoded(rows,Tokenizer(),3)


def test_restore_preserves_previous_database(tmp_path):
    import subprocess
    import sys
    directory=tmp_path/'state'
    store=Store(directory)
    store.execute("INSERT INTO memories(kind,text) VALUES ('fact','original')")
    backup=store.backup()
    store.execute("UPDATE memories SET text='changed'")
    result=subprocess.run([sys.executable,'scripts/restore.py',str(backup),'--data',str(directory),'--server-stopped'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert Store(directory).rows('SELECT text FROM memories')==[{'text':'original'}]
    old=list(directory.glob('before-restore-*.sqlite3'))
    assert len(old)==1
    with sqlite3.connect(old[0]) as db:
        assert db.execute('SELECT text FROM memories').fetchone()[0]=='changed'


def test_stop_discards_inflight_inference(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    entered,release=threading.Event(),threading.Event()
    class SlowModel(ContractModel):
        async def chat(self,model,messages):
            entered.set()
            await asyncio.to_thread(release.wait,5)
            return 'This response must be discarded'
    app=create_app(tmp_path,inference=SlowModel())
    with TestClient(app,headers=HEADERS,base_url='http://127.0.0.1') as client:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(client.post,'/api/chat',json={'text':'start'})
            assert entered.wait(3)
            assert client.post('/api/stop').status_code==200
            assert client.post('/api/resume').status_code==409
            release.set()
            assert future.result(timeout=5).status_code==409
        assert client.get('/api/conversations').json()==[]
        assert client.post('/api/resume').status_code==200
