# Architecture and development plan

## Core boundaries

The first version uses a single local FastAPI process, SQLite transactions, and a static browser UI. This avoids a second Node build/runtime for the core application while leaving the HTTP API usable by a future React or desktop client. There is no CDN, remote analytics, mandatory subscription, or paid inference API.

Ollama owns model loading, quantization, and execution. `inference.Ollama` is a small async protocol adapter (`models`, `chat`, `embed`); it can be replaced without changing persistence or learning modules. Model selection is explicit. Ollama itself must be installed and configured locally. This repository does not package an open-weight model. Only loopback inference endpoints are accepted.

SQLite stores explicit memories with type and verification status, conversations, feedback, episodic experiment results, and checkpoint metadata. Embeddings are stored with their originating model; cosine comparison only uses matching model/dimension pairs. Blank embeddings use lexical retrieval. Retrieval currently scans at most 2,000 recent memories and selects six entries; larger deployments should introduce paginated browsing, a dedicated vector index, migrations, and memory aging. Short-term context is limited to 20 messages and an approximate character budget, with Ollama's final token budget enforced by its runtime. The application does not promise unlimited conversation recall.

`ToolBox` grants only new document creation and text search in its own directory. The model cannot call it automatically. File traversal, symlinks escaping that directory, overwrites, and emergency-stopped requests are rejected. This is a narrow tool boundary, **not an operating-system sandbox**. Arbitrary untrusted code execution is intentionally not implemented. Future code tools need a real isolated worker (for example a disposable restricted container/VM), explicit approvals scoped to exact operations, resource limits, no inherited credentials, and tested cancellation. A Python subprocess alone would not supply that isolation.

Browser calls require a custom header and exact same origin; hostile Host headers are rejected to reduce DNS rebinding risk. The server binds loopback, sets a restrictive CSP, and renders model/memory text with `textContent`. It has no LAN authentication or encryption. These controls are intended for one trusted local user; they do not protect against malware or another local process running as that user. Do not publish the server on the Internet. Model prompts are advisory, not a security boundary; structural tool isolation is what limits model actions.

Grid experiments execute on one CPU thread with seed, episode, step, and wall-clock bounds. Results distinguish interrupted, failed, and completed work. Jobs that were running at process loss become interrupted on startup. SQLite survives process restarts. Live threads, servers, and model connections do not.

LoRA training is a separate explicit operator command with optional dependencies. It uses local model files, disables remote model code, masks prompt tokens out of supervised loss, bounds steps, saves adapters per epoch, and evaluates completion negative log-likelihood on held-out examples. A registry points to evaluated candidates; it never modifies the base model or automatically switches Ollama. Loss evaluation alone does not establish real-world capability or safety. QLoRA, distributed training, reward models, preference optimization, and automatic Ollama import are not implemented.

Minecraft lives in a separate Node process to avoid granting game-control permissions to the chat application. Its allowlist is deliberately small. The grid environment uses the Gymnasium reset/step shape without depending on Gymnasium; it is not advertised as a registered Gymnasium `Env`. A future adapter can subclass it and provide action/observation spaces. No Minecraft training or visual control is claimed.

## Extension contracts

- Inference: async `models()`, `chat(model, messages)`, `embed(model, text)`; failures raise `ModelUnavailable`.
- Environment: `reset(seed) -> (observation, info)` and `step(action) -> (observation, reward, terminated, truncated, info)`.
- Tools: explicitly scoped paths/capabilities, audit event, bounded input, stop check; no model-provided Python evaluation.
- Persistence: transactional `Store.connect`, JSON versioned exports; add schema migrations before altering existing tables in future releases.
- Checkpoints: immutable saved adapter files, dataset identity, measured baseline/candidate report, explicit registration and rollback.

## Milestones

1. **Implemented core:** local inference adapter, real UI, persistent chat, memory CRUD/export/backup, optional embeddings, personality, feedback. Pretrained inference must be validated against an installed model on the target laptop.
2. **Implemented first learning modules:** bounded grid experiments, curriculum, held-out start-seed evaluation, dataset curation/export, optional offline LoRA pipeline and registry. Larger-model training remains hardware-dependent.
3. **Experimental structured games:** manual local Mineflayer bridge. Next: repeatable private-server scenario reset, reward instrumentation, recorded demonstration schema, navigation baseline, and integration tests against a pinned Minecraft server version.
4. **Planned permissioned autonomy:** objective/job abstractions, plans reviewed before sensitive operations, trusted tool executor, explicit approval tokens, sandboxed software tasks, per-tool budgets, and human progress reports. Do not expose OS tools until the isolation is implemented and tested.
5. **Planned visual interaction:** Windows-specific capture of an explicitly selected window, screenshot model adapter, foreground/window identity checks, bounded input actions, panic hotkey, integration tests. Do not implement global uncontrolled mouse/keyboard hooks as a shortcut.
6. **Planned deeper learning:** dataset versioning/splits, diverse regression benchmarks, multi-seed uncertainty estimates, LoRA deployment import, preference optimization, QLoRA on supported GPUs, staged rollout with task-based regression and rollback.

The foundation deliberately scales through replaceable modules rather than promising general intelligence from a small model. None of the planned milestones is represented as working in the dashboards.
