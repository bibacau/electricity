# Validation evidence

## Portable packaging update

The requested user delivery now requires **no installation**: a self-contained Windows x64 ZIP with a bundled Python runtime, dependencies, llama.cpp engine and pretrained GGUF model. Source-development installation instructions remain optional contributor documentation.

The Windows Python 3.12.10 NuGet package was downloaded and verified against the Python Software Foundation's published SHA-512. Windows CPython 3.12 dependency wheels were downloaded through pip's package-index hash checks. The official llama.cpp `b11669` Windows CPU release was downloaded and verified against its published SHA-256 (`37b3efcf962847ca869745fd4ccf7f5e7e8dacfeff9fa3523b4ca8cdc66d632b`). Python and llama-server executables were checked as Windows x64 PE files; they could not be executed on this Linux host.

The cloud model-download policy is now active. The required Qwen2.5-0.5B-Instruct Q4_K_M GGUF model was downloaded and verified against its publisher LFS SHA-256 and Apache-2.0 metadata. A complete Windows portable ZIP has been assembled; this supersedes the earlier download blocker. The model is included in the ZIP, so launch performs no downloads. The launcher creates an ephemeral per-run authentication key shared only with its loopback model engine; the engine rejects requests without that key. The command launcher disables bytecode writes so immutable packaged files retain their verification hashes.
After adding the portable path, **19 core Python tests passed**, including llama-server protocol handling, incomplete-bundle and altered-file rejection, and archive traversal rejection. These extend the 16-test baseline below. The separate LoRA and Minecraft test results below remain valid; their code was not changed in this packaging update. The launcher contains no downloader, dependency installer or elevation mechanism, and will not attempt to repair missing assets on the user's machine. Actual Windows launch remains outstanding. Pretrained inference was tested successfully with the same verified model and the matching Linux engine release: the app recalled a saved Python language preference, answered a second turn consistently, and persisted four messages. The authenticated integration check also verified that unauthenticated native model requests return HTTP 401. This proves the model/backend path on Linux; it does not certify the Windows binaries under Windows.

Validation performed on Debian Linux, Python 3.12.14, Node.js 24.19.0, CPU PyTorch 2.14.1, Transformers 4.57.6, PEFT 0.21.2, and Accelerate 1.15.0. Windows launch scripts were reviewed but could not be executed on this Linux host. Core Python dependency versions are captured in `requirements.lock`; Mineflayer dependencies are captured in its `package-lock.json`.

| Check | Outcome | Scope |
|---|---|---|
| `bash scripts/setup.sh` | Passed, including **16 Python tests** | Pinned dependency installation, editable packaging, SQLite persistence, personality, recall, chat protocol, errors, feedback, privacy, tools, request boundaries, backup/restore, stop, RL, checkpoint gates |
| `python -m pytest tests/training_optional.py -q -s` | **1 passed** | Actual CPU LoRA gradients, nonzero updated adapter parameters, unchanged base files, completion loss evaluation, registration, rollback, rejection of training data as evaluation |
| `npm ci --ignore-scripts` and `npm test` in `adapters/minecraft` | Install succeeded; **2 tests passed** | Mineflayer import, bounded command validation and emergency-control command acceptance |
| JavaScript syntax checks | Passed | Browser app and standalone bridge |
| `python tests/browser_smoke.py --python .venv/bin/python --browser /usr/bin/chromium` | Passed | Real Chromium: memory add/edit/reload, personality, real grid job/result, honest missing-model error, mobile overflow check; no JavaScript page errors |
| Live HTTP health and experiment request | Passed | Uvicorn service, SQLite access, actual saved experiment result |
| `pip check` | Passed | No broken installed Python dependency requirements |
| Real pretrained local conversation | **Passed with llama.cpp on Linux** | Same bundled model; memory recall, two turns, persistence, native engine authentication. Ollama-specific real inference remains untested. |
| Real semantic model embeddings | **Not run** | Protocol/vector matching tested with fixtures; pretrained embedding download blocked |
| Minecraft live server actions | **Not run** | No authorized local Minecraft Java server available |
| Windows desktop visual control | **Not implemented** | Roadmap only |

## Measured grid result

One persisted live API experiment used seed 42, 1,500 episodes, a final 5×5 board, a 30-second limit, and 100 held-out start seeds:

- Completed training episodes: **1,500**.
- Random baseline success: **54/100**, mean reward **−0.3108**.
- Learned-policy success: **100/100**, mean reward **0.9344**.
- Mechanism: tabular Q-learning; **no neural-network weight updates**.

The evaluation board/dynamics remain the training task. Held-out starting seeds do not establish transfer to unseen dynamics, Minecraft, or general reasoning.

## Measured LoRA mechanics result

The optional integration test created a tiny, randomly initialized one-layer GPT-2 model entirely locally, used eight artificial training pairs and four separate evaluation pairs, and ran 32 CPU optimizer steps. It verified actual adapter parameter changes and that base model files retained their original fingerprint. Held-out completion NLL changed from **3.0024** to **2.6990**.

This fixture validates training mechanics. It is **not a pretrained assistant**, not a useful conversational model, and not evidence of personalized language-model quality. It is never used by the app, dashboard, or Ollama. The test's temporary files are isolated from production state. Real-model training, GPU/CUDA behavior, and adapter conversion to Ollama remain unverified.

## Known notices and limitations

- The installed Starlette emits a deprecation warning for its HTTPX-backed TestClient. Tests still execute successfully; moving to its newer test transport is future maintenance.
- PEFT adjusts GPT-2 Conv1D LoRA orientation in the synthetic training test; this is a documented warning, not a disabled assertion.
- Transformers warns about a deprecated dtype keyword with the selected version; model loading and real training succeeded.
- Initial Playwright browser download was blocked. The smoke test succeeded using the preinstalled system Chromium instead; no TLS or artifact verification was disabled.
- Earlier probes to model hosts received proxy HTTP 403. The saved network update subsequently took effect and the required GGUF model download succeeded. No external model-service credentials were required. The portable launcher generates its own short-lived local engine authentication token at runtime and never saves it in the bundle or database.
- Ollama protocol handling was tested against fixtures. The llama.cpp path was additionally tested with the actual pretrained model; backend-specific outcomes are distinguished above.
- The backend intentionally exposes no arbitrary shell/code execution, browser control, sensitive-file tool, or desktop input. Approval workflows for such future tools have not been implemented because those tools are not available.

## Reproduce optional browser checks

Install Playwright separately, then use an installed supported Chromium executable. The smoke script starts its own isolated server and temporary database, runs UI workflows, and stops that server afterward. It does not reuse personal data. On Windows, pass the actual browser executable path or install Playwright Chromium and omit `--browser`. Screenshots are saved to the OS temporary directory.

## Remaining target-machine validation

Install Ollama and a licensed model on the Windows laptop, select its tag in Personality, and exchange two chat turns. Save a preference, restart the workbench, and confirm it appears in recalled context. If using embeddings, install the selected embedding model and re-save memories, then verify a semantically related query. Test the Minecraft bridge only against an authorized private loopback server, including stop/pause behavior. Validate any fine-tuned model on representative held-out tasks before manually switching the production model tag.

## Reproduce the pretrained check

With the retained verified Linux engine and model files, run:

```bash
.venv/bin/python tests/local_inference_smoke.py --engine /workspace/deliverables/portable-build-cache/linux-engine/llama-b11669/llama-server --model /workspace/deliverables/atlas-portable-stage/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

This starts its own authenticated loopback engine, uses a temporary database, verifies real responses and persistence, and stops the engine afterward. It does not install or download anything.
