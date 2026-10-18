# Atlas · Personal AI Workbench

A locally controlled foundation for a personal AI: Python/FastAPI orchestration, Ollama inference, SQLite memory, an offline web interface, real tabular reinforcement learning, an optional LoRA pipeline, and a separate Minecraft Java bridge.

**Status:** Phase 1 application and early learning modules are implemented. This is an extensible initial version, not a generally autonomous or self-improving intelligence. The portable package includes a local model; source-development mode needs a separately available local engine/model. See [validation](docs/VALIDATION.md) for precisely what has been exercised here.

## Portable Windows delivery (no installation)

The user-facing target is a self-contained ZIP: extract it and double-click **Start Atlas.cmd**. It bundles CPython, the Windows dependency wheels, llama.cpp, and a small licensed GGUF model. It uses your existing browser, stores data inside its own folder, and performs no downloads, pip operations, administrator elevation, or system installation at launch. Python, Ollama, and Node.js are not prerequisites for the portable core.

**Current packaging status:** the complete portable ZIP has been built at `/workspace/deliverables/atlas-windows-portable.zip`, including the verified pretrained model. Real model inference, memory recall, multi-turn persistence, and private engine authentication were tested using the corresponding Linux engine. The launcher refuses incomplete or altered bundles. Windows execution remains unverified on this Linux builder. The separate source ZIP is a developer archive, not the portable application.

Publisher-side build command (not an end-user setup step):

```bash
.venv/bin/python scripts/build_portable.py
```

It verifies the Python package against the Python Software Foundation NuGet SHA-512 metadata, the llama.cpp release against its published SHA-256, and the model against its publisher's LFS digest and Apache-2.0 license metadata. It only writes the final ZIP after every required artifact is present. The target is 64-bit Windows 10/11; optional Minecraft and training modules are not bundled into this core distribution.

## Contributor setup on Windows

These source-development instructions are for contributors. They are not requirements for users of a completed portable bundle.

Requirements: Windows 10/11, Python **3.11–3.13**, and [Ollama](https://ollama.com/download/windows). Node.js 22+ is needed only for the optional Minecraft bridge. Python dependencies and model downloads require Internet during installation; runtime has no mandatory cloud dependency.

From PowerShell in this repository:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
ollama pull qwen2.5:1.5b
powershell -ExecutionPolicy Bypass -File scripts\start.ps1
```

The process-scoped PowerShell flag runs these local scripts without changing your machine's execution policy. Review scripts first. If Python's `py` launcher is unavailable, create `.venv` using `python -m venv .venv`, then rerun setup. Do not run as administrator.

Open the web app in your browser on local port **8765**. In **Personality**, refresh installed models, enter `qwen2.5:1.5b`, and save. Ollama's Windows app normally runs the inference service; if it is stopped, start it or run `ollama serve` in another terminal. Use `ollama list` to check your installed tags.

Qwen2.5 **1.5B** and **7B** are example Apache-2.0 open-weight choices; check the actual model card and distribution license before downloading/replacing models. Model licenses are independent of this application's source license. The 1.5B quantized model is a lightweight starting point; its reasoning and instruction-following are limited. A 7B quantized model generally needs substantially more RAM/VRAM. CPU inference works with Ollama but can be slow. Plan roughly 8 GB system RAM for the small-model workflow and 16+ GB for larger quantized models; actual memory depends on context length, quantization, and other workloads. Fine-tuning has separate, higher requirements.

For semantic recall, optionally install an embedding model:

```powershell
ollama pull nomic-embed-text
```

Set `embedding_model` to `nomic-embed-text` in Personality. Save each existing memory again to embed it. With no embedding model, retrieval uses keyword overlap and preference/correction priority; it is **not semantic search**. Documents are searched explicitly and are not automatically inserted into prompts.

### Linux/macOS development

```bash
bash scripts/setup.sh
bash scripts/start.sh
```

Use an existing checkout; each cloud task is already isolated, so no extra Git worktree is needed. The app listens only on loopback. `python -m personal_ai --data <directory> --port <port>` selects another data directory or port. `OLLAMA_URL` may select another **loopback HTTP** Ollama port, not a remote endpoint. The runtime refuses cross-origin browser calls and unrecognized hostnames; do not expose it through a public reverse proxy.

## What you can use

- **Chat:** select an installed local model, save conversations, revisit/delete history, or disable history for new private chats. Private chat still reads your explicit memories. Nothing is automatically remembered as a fact. Failed inference does not create fabricated messages in the database.
- **Memory library:** add/edit/delete verified or unverified facts, preferences, corrections, episodes, and skills; export workspace JSON. Skill entries are reusable text procedures recalled in context, not executable downloaded programs.
- **Personalization:** identity, communication style, language, detail, model, embedding model, and chat retention. Changing these preserves existing memories.
- **Feedback:** mark replies useful or supply corrections. Corrections become verified user memory and supervised examples. Export the latest feedback for each reply as JSONL; review it before training. Rejected responses without corrections are excluded.
- **Learning lab:** run bounded CPU grid experiments, inspect rewards, random-policy baseline, learned policy performance, and job history. A curriculum increases board size. One worker, 50–10,000 episodes, 1–300 seconds, 60 steps per episode. The objective/reward function is deliberately fixed in this first environment. There is no open-ended autonomous browsing or self-modification loop.
- **Tools:** manually upload new `.txt` documents up to 256 KiB and search that dedicated directory. Model text cannot invoke tools. No shell, arbitrary program launching, code execution, browser automation, or desktop input is exposed.
- **System:** real CPU/RAM readings, process memory, action/error metadata, emergency stop, and SQLite backup. GPU telemetry is not implemented; absent values are never invented.

## Learning is three different things

| Mechanism | What changes | What does not change |
|---|---|---|
| Memory and corrections | SQLite data and future retrieved context | Neural-network weights |
| Grid Q-learning | A saved action-value table and policy | Language-model weights |
| Optional LoRA | Separately saved neural adapter parameters | Your original/base inference model |

Grid evaluation uses 100 held-out random **start seeds** in the same deterministic dynamics, also used by the random-policy baseline. It is a narrow measurement, not proof of game transfer or novel-world generalization. Policies, rewards, and evaluations are persisted in experiment results. Completed jobs survive restarts; interrupted jobs are marked accordingly. See [training](docs/TRAINING.md).

## Backup, restore, and privacy

All workbench state lives in `data/` by default. The database is `data/workbench.sqlite3`; uploads are under `data/documents/`. The database is not encrypted. Use Windows disk encryption and protect the folder if your data is sensitive. Local model inference is subject to the inference engine's own settings and logging.

Use **System → Download memory backup** for a transactionally consistent SQLite backup. It includes conversations, memories, feedback, preferences, experiments, events, and checkpoint registry. JSON export is for inspection/portability; JSON re-import is not implemented. Neither export includes document files or model weights.

To restore, shut down the workbench first:

```powershell
.venv\Scripts\python.exe scripts\restore.py path\to\workbench.sqlite3 --data data --server-stopped
```

The restore utility checks database integrity/table names and preserves the previous database under a unique name. Restore only your own trusted backups. Copy `data/documents/`, training outputs, original model directories, and Ollama's model store separately for a complete machine migration. Checkpoint paths are absolute; after migration, evaluate/register the adapters at their new paths. Copying only the registry does not migrate weights.

For an offline reinstall on another machine, download matching Windows wheels ahead of time (`pip download -r requirements.lock -d wheelhouse` on the target Python/platform), install them with `--no-index --find-links wheelhouse`, and install/build the application with local build dependencies. Also retain the model files and installer. Offline **operation** after setup is supported; the repository itself does not bundle multi-gigabyte models or installers.

## Minecraft (experimental, standalone)

The bridge supports **Minecraft Java**, not Bedrock. It connects exclusively to a local server at `127.0.0.1:25565` as `AtlasLocal` using offline authentication. Use a private development server bound to loopback with offline authentication deliberately enabled. Never expose an offline-mode server publicly. Online Microsoft authentication is not implemented.

```powershell
cd adapters\minecraft
npm ci --ignore-scripts
npm test
node bridge.cjs --authorized
```

Only pass `--authorized` for a server you own or are permitted to automate. After the `ready` observation, type one JSON command per line:

```json
{"action":"observe"}
{"action":"move","direction":"forward","ms":500}
{"action":"dig","position":[1,64,2]}
{"action":"craft","item":"oak_planks","count":1}
{"action":"pause"}
{"action":"resume"}
{"action":"stop"}
```

Movement is limited to two seconds; digging is limited to a small block allowlist within four blocks and a five-second timeout. Crafting uses inventory-only recipes and a five-second timeout that disconnects on expiry. Each process allows 100 actions; it never queues simultaneous actions. Pause clears movement and stops digging; an already submitted crafting action may complete. Stop disconnects. Ctrl+C also stops the bridge. The workbench emergency button does **not** control this separate process.

Bridge output supplies position, health, hunger, inventory, and action history as JSON lines. Save commands/output yourself to collect demonstrations; no demonstration learning pipeline is implemented. The bridge is not wired into the dashboard or an autonomous planner. Live gameplay, Minecraft RL, enemy handling, complex pathfinding/building, screenshot interpretation, and Windows mouse/keyboard control remain future work. No anti-cheat bypass is included.

## Validation and troubleshooting

```powershell
.venv\Scripts\python.exe -m pytest
cd adapters\minecraft
npm test
```

- **Disconnected model:** verify `ollama list`, service port 11434, and exact model tag. The app never downloads models silently. A backend 503 indicates inference is unavailable, not an AI response.
- **Embedding errors:** remove the embedding-model setting to use keyword recall, or install the selected local embedding model. Changing embedding models requires re-saving memories.
- **Emergency stop:** blocks new chats, tools, and experiments, and signals running grid work to stop. An active Ollama computation may finish in Ollama, but its response is discarded. Resume waits for in-flight work to finish. Stop each standalone bridge separately.
- **Interrupted training:** LoRA training runs as a separate CLI; stop it with Ctrl+C. Previously completed epoch checkpoints remain candidates. It does not respond to the web emergency button.
- **Slow model / out of memory:** select a smaller quantized model, close other applications, reduce training context, or train on suitable GPU hardware. CPU resources in the web panel are host readings, not measurements of model intelligence.
- **Portable model:** the licensed model is already bundled; no model download is needed when launching the portable application. The cloud download policy has been updated and real pretrained inference has passed a Linux integration test. Windows execution is still a separate target-platform validation step.

## Project map

```text
personal_ai/
  api.py           Local HTTP routes, lifecycle, request boundaries
  inference.py     Replaceable Ollama adapter, recall, prompt assembly
  store.py         SQLite persistence and backups
  tools.py         Dedicated-document capabilities and stop control
  learning.py      Environment, Q-learning, job runner, evaluation
  training.py      Optional offline LoRA train/evaluate/registry CLI
  static/          Local HTML/CSS/JS, no CDN or build-time Node requirement
adapters/minecraft/ Standalone bounded Mineflayer bridge
scripts/            Windows/Linux setup, launch, database restore
tests/              API, persistence, boundaries, learning, training gates
docs/               Architecture, training, evidence, future milestones
```

See [architecture and roadmap](docs/ARCHITECTURE.md), [training instructions](docs/TRAINING.md), and [validation evidence](docs/VALIDATION.md).
