import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal

from .inference import ModelUnavailable, Ollama, LlamaCpp, retrieve, system_prompt
from .learning import Experiments
from .store import Store
from .tools import ToolBox, ToolDenied


class Settings(BaseModel):
    name: str = Field(default="Atlas",min_length=1,max_length=100)
    style: str = Field(default="Clear, thoughtful and candid",max_length=2000)
    language: str = Field(default="English",max_length=100)
    detail: Literal["brief","balanced","detailed"] = "balanced"
    model: str = Field(default="",max_length=200)
    embedding_model: str = Field(default="",max_length=200)
    remember_chat: bool = True


class Memory(BaseModel):
    kind: Literal["fact","preference","correction","episode","skill","assumption"] = "fact"
    text: str = Field(min_length=1,max_length=12000)
    verified: bool = False


class Chat(BaseModel):
    conversation_id: int | None = None
    text: str = Field(min_length=1,max_length=12000)


class Feedback(BaseModel):
    message_id: int
    rating: Literal[-1,1]
    correction: str = Field(default="",max_length=12000)


class Experiment(BaseModel):
    episodes: int = Field(default=1500,ge=50,le=10000)
    size: int = Field(default=5,ge=3,le=10)
    seed: int = Field(default=42,ge=0,le=99999)
    seconds: int = Field(default=30,ge=1,le=300)


class Document(BaseModel):
    name: str = Field(min_length=1,max_length=160,pattern=r"^[\w .-]+\.txt$")
    text: str = Field(max_length=262144)


def create_app(data_dir=None, inference=None):
    store = Store(Path(data_dir or os.environ.get("PERSONAL_AI_DATA","data")))
    backend = os.environ.get("PERSONAL_AI_BACKEND","ollama")
    base_url = os.environ.get("OLLAMA_URL","http://127.0.0.1:11434")
    model = inference or (LlamaCpp(base_url,os.environ.get("PERSONAL_AI_ENGINE_KEY")) if backend == "llama.cpp" else Ollama(base_url))
    toolbox = ToolBox(store.directory/"documents",store.event)
    jobs = Experiments(store,toolbox.stopped)
    chat_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        yield
        jobs.close()

    app = FastAPI(title="Personal AI Workbench",version="0.1.0",lifespan=lifespan)
    app.state.store, app.state.model, app.state.jobs = store, model, jobs
    app.state.tools = toolbox

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = request.url.hostname
        if host not in {"127.0.0.1","localhost","::1"}:
            return JSONResponse({"detail":"Local access only"},status_code=403)
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
            return JSONResponse({"detail":"Cross-origin requests are not permitted"},status_code=403)
        if request.method in {"POST","PUT","PATCH","DELETE"} and request.headers.get("x-workbench") != "1":
            return JSONResponse({"detail":"Missing local UI request header"},status_code=403)
        # Bound actual request bytes too, including chunked requests without Content-Length.
        if request.method in {"POST","PUT","PATCH"}:
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 1_100_000:
                    return JSONResponse({"detail":"Request exceeds 1.1 MB"},status_code=413)
            request._body = bytes(body)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ModelUnavailable)
    async def model_error(request, exc):
        store.event("model.error",{"message":str(exc)})
        return JSONResponse({"detail":str(exc)},status_code=503)

    @app.exception_handler(ToolDenied)
    async def tool_error(request, exc):
        return JSONResponse({"detail":str(exc)},status_code=403)

    @app.get("/api/health")
    async def health():
        return {"status":"ok", "database":bool(store.rows("SELECT id FROM settings")),"emergency_stop":toolbox.stopped.is_set()}

    @app.get("/api/models")
    async def models():
        return {"models":await model.models()}

    @app.get("/api/settings")
    async def settings():
        return store.settings()

    @app.put("/api/settings")
    async def save_settings(value: Settings):
        store.execute("UPDATE settings SET value=? WHERE id=1",(value.model_dump_json(),))
        store.event("settings.update",{"model":value.model})
        return value

    @app.get("/api/conversations")
    async def conversations():
        return store.rows("SELECT * FROM conversations ORDER BY id DESC")

    @app.get("/api/conversations/{cid}")
    async def history(cid:int):
        return store.rows("SELECT * FROM messages WHERE conversation_id=? ORDER BY id",(cid,))

    @app.delete("/api/conversations/{cid}")
    async def delete_conversation(cid:int):
        store.execute("DELETE FROM conversations WHERE id=?",(cid,))
        return {"deleted":True}

    @app.post("/api/chat")
    async def chat(value: Chat):
        async with chat_lock:
            if toolbox.stopped.is_set():
                raise HTTPException(409,"Emergency stop is active")
            config = store.settings()
            cid = value.conversation_id
            if cid is not None and not store.rows("SELECT id FROM conversations WHERE id=?",(cid,)):
                raise HTTPException(404,"Conversation not found")
            if not config["remember_chat"] and cid is not None:
                raise HTTPException(409,"Disable history selection for a private conversation")
            memories = store.rows("SELECT * FROM memories ORDER BY id DESC LIMIT 2000")
            vector = None
            retrieval_mode = "keyword"
            if config["embedding_model"]:
                vector = await model.embed(config["embedding_model"],value.text)
                retrieval_mode = "semantic_with_keyword_fallback"
            recalled = retrieve(memories,value.text,vector,config["embedding_model"])
            messages = store.rows("SELECT role,content FROM (SELECT id,role,content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 20) ORDER BY id",(cid,)) if cid else []
            # Keep context bounded; the model backend enforces the final token budget.
            context, budget = [], 16000
            for message in reversed(messages):
                if len(message["content"]) > budget:
                    break
                context.insert(0,message)
                budget -= len(message["content"])
            payload = [{"role":"system","content":system_prompt(config,recalled)}]+context+[{"role":"user","content":value.text}]
            answer = await model.chat(config["model"],payload)
            if toolbox.stopped.is_set():
                raise HTTPException(409,"Stopped; response discarded")
            mid = None
            if config["remember_chat"]:
                with store.connect() as db:
                    if cid is None:
                        cid = db.execute("INSERT INTO conversations(title) VALUES (?)",(value.text[:80],)).lastrowid
                    db.execute("INSERT INTO messages(conversation_id,role,content) VALUES (?,'user',?)",(cid,value.text))
                    mid = db.execute("INSERT INTO messages(conversation_id,role,content) VALUES (?,'assistant',?)",(cid,answer)).lastrowid
            return {"conversation_id":cid,"message_id":mid,"content":answer,"recalled_ids":[m["id"] for m in recalled],"retrieval":retrieval_mode}

    @app.get("/api/memories")
    async def memories():
        return store.rows("SELECT id,kind,text,verified,embedding_model,created FROM memories ORDER BY id DESC")

    async def memory_values(value):
        name = store.settings()["embedding_model"]
        embedding = json.dumps(await model.embed(name,value.text)) if name else None
        return value.kind,value.text,int(value.verified),embedding,name or None

    @app.post("/api/memories")
    async def add_memory(value: Memory):
        mid = store.execute("INSERT INTO memories(kind,text,verified,embedding,embedding_model) VALUES (?,?,?,?,?)",await memory_values(value))
        return {"id":mid,"mechanism":"memory_update","weight_updates":False}

    @app.put("/api/memories/{mid}")
    async def edit_memory(mid:int,value:Memory):
        if not store.rows("SELECT id FROM memories WHERE id=?",(mid,)):
            raise HTTPException(404,"Memory not found")
        store.execute("UPDATE memories SET kind=?,text=?,verified=?,embedding=?,embedding_model=? WHERE id=?",(*await memory_values(value),mid))
        return {"updated":True}

    @app.delete("/api/memories/{mid}")
    async def delete_memory(mid:int):
        store.execute("DELETE FROM memories WHERE id=?",(mid,))
        return {"deleted":True}

    @app.get("/api/export")
    async def export():
        return {"version":1,"settings":store.settings(),"memories":await memories(),"conversations":await conversations(),
                "messages":store.rows("SELECT * FROM messages"),"feedback":store.rows("SELECT * FROM feedback")}

    @app.post("/api/backup")
    async def backup():
        return FileResponse(store.backup(),filename="workbench.sqlite3",media_type="application/vnd.sqlite3")

    @app.post("/api/feedback")
    async def feedback(value:Feedback):
        if not store.rows("SELECT id FROM messages WHERE id=? AND role='assistant'",(value.message_id,)):
            raise HTTPException(404,"Assistant message not found")
        fid = store.execute("INSERT INTO feedback(message_id,rating,correction) VALUES (?,?,?)",(value.message_id,value.rating,value.correction))
        if value.correction:
            store.execute("INSERT INTO memories(kind,text,verified) VALUES ('correction',?,1)",(value.correction,))
        return {"id":fid,"mechanism":"feedback_and_memory","weight_updates":False}

    @app.get("/api/dataset")
    async def dataset():
        rows = store.rows("""SELECT m.id,m.conversation_id,m.content,f.rating,f.correction FROM messages m JOIN feedback f ON f.message_id=m.id
            WHERE f.id=(SELECT MAX(id) FROM feedback WHERE message_id=m.id) ORDER BY m.id""")
        samples=[]
        for row in rows:
            prompt=store.rows("SELECT content FROM messages WHERE conversation_id=? AND id<? AND role='user' ORDER BY id DESC LIMIT 1",(row["conversation_id"],row["id"]))
            answer=row["correction"] or (row["content"] if row["rating"]==1 else "")
            if prompt and answer:
                samples.append({"messages":[{"role":"user","content":prompt[0]["content"]},{"role":"assistant","content":answer}]})
        return {"samples":samples,"count":len(samples),"curation":"Latest explicit feedback only; review before training."}

    @app.post("/api/documents")
    async def upload(value:Document):
        try:
            return {"path":toolbox.write(value.name,value.text)}
        except FileExistsError:
            raise HTTPException(409,"Document already exists; choose a new filename")

    @app.get("/api/documents/search")
    async def search(q:str):
        if not q.strip() or len(q)>200:
            raise HTTPException(422,"Search requires 1–200 characters")
        return toolbox.search(q)

    @app.post("/api/experiments")
    async def start_experiment(value:Experiment):
        try:
            return {"id":jobs.start(value.model_dump())}
        except ValueError as exc:
            raise HTTPException(409,str(exc))

    @app.get("/api/experiments")
    async def experiments():
        return [{**r,"config":json.loads(r["config"]),"result":json.loads(r["result"]) if r["result"] else None} for r in store.rows("SELECT * FROM experiments ORDER BY id DESC LIMIT 30")]

    @app.post("/api/stop")
    async def stop():
        toolbox.stopped.set()
        store.event("emergency.stop",{})
        return {"stopped":True,"note":"Active Ollama compute may continue until its request ends; responses are discarded. Game bridge must be stopped separately."}

    @app.post("/api/resume")
    async def resume():
        if jobs.future and not jobs.future.done():
            raise HTTPException(409,"Wait for the experiment to stop before resuming")
        if chat_lock.locked():
            raise HTTPException(409,"Wait for the in-flight model request before resuming")
        toolbox.stopped.clear()
        store.event("emergency.resume",{})
        return {"stopped":False}

    @app.get("/api/system")
    async def system():
        ram=psutil.virtual_memory()
        return {"cpu_percent":psutil.cpu_percent(),"ram_used_gib":round(ram.used/2**30,2),"ram_total_gib":round(ram.total/2**30,2),
                "process_rss_mib":round(psutil.Process().memory_info().rss/2**20,1),"gpu":None,"gpu_note":"GPU telemetry is not configured",
                "events":store.rows("SELECT * FROM events ORDER BY id DESC LIMIT 50"),"stopped":toolbox.stopped.is_set(),
                "capabilities":{"model_backend":backend+" (availability checked separately)","code_execution":False,"browser":False,"minecraft":"standalone manual bridge","visual_control":False}}

    @app.get("/api/checkpoints")
    async def checkpoints():
        return store.rows("SELECT * FROM checkpoints ORDER BY id DESC")

    static = Path(__file__).parent/"static"
    app.mount("/static",StaticFiles(directory=static),name="static")

    @app.get("/")
    async def index():
        return FileResponse(static/"index.html")

    return app
