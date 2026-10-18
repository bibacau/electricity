# Training and checkpoint operations

There are no neural weight updates during ordinary chat, feedback, or grid experiments. LoRA training is an explicit offline CLI operation. It is optional; ordinary chat does not require PyTorch.

## Dataset preparation

Rate or correct assistant replies, then export JSONL from the Learning lab. Each example contains exactly one user/assistant pair. Review text for correctness, sensitive data, duplication, and licensing. Prefer diverse independent examples over many paraphrases of one answer. The application does not automatically convert raw personal files into training data.

Split examples into `train.jsonl` and `heldout.jsonl` before training. Each line has this shape:

```json
{"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Reviewed answer"}]}
```

At least four held-out examples are required to exercise the evaluation code, but four is not a meaningful production benchmark. Use a much larger, representative held-out set plus reasoning, factuality, privacy, and tool-regression tasks. Exact training/evaluation duplicate pairs are rejected. Related/paraphrased leakage is not detected automatically.

## Dependencies and models

Install the **appropriate PyTorch build for your platform first**, following [PyTorch's official selector](https://pytorch.org/get-started/locally/). CPU is supported for small experiments; CUDA needs a compatible NVIDIA GPU/driver. Then install the training extra:

```powershell
.venv\Scripts\python.exe -m pip install -e '.[training]'
```

Obtain a trusted Hugging Face causal-language-model checkpoint in a local directory, including tokenizer/chat-template files. Verify the exact model license and file integrity using the publisher's metadata. Ollama GGUF files cannot be passed directly to Transformers training. The training CLI sets `local_files_only=True` and `trust_remote_code=False`; custom remote model code is not executed and missing files are an error. Downloading models is a separate, explicit task. No API key is mandatory for public permissively licensed weights.

## Train a separate adapter

```powershell
.venv\Scripts\python.exe -m personal_ai.training train --model C:\models\base-model --dataset train.jsonl --output data\training\run-001 --device cpu --epochs 1 --max-steps 100 --max-length 512
```

The output directory must not exist, so an earlier run is never silently overwritten. AdamW optimizes rank-8 LoRA adapter weights; the original base weights are frozen by PEFT. Each completed epoch saves `epoch-N/` plus a training metadata file in the run directory. Prompt tokens are masked out of loss; incompatible chat templates or examples exceeding the context bound fail with a clear error. Training is real gradient-based optimization, not a memory-update demonstration. There is no automatic training schedule or auto-activation. Ctrl+C stops the CLI; the web emergency button governs only the web process.

This first implementation loads float32 base parameters. It is suitable for testing the pipeline on small models, and expensive on large ones. Roughly four bytes per base parameter are needed just for the base parameters, plus adapters, activations, gradients, framework overhead, and tokenizer/context memory. A 7B base alone needs about 28 GB before those extras. Smaller models, CPU RAM, or a suitably large GPU are needed. QLoRA/4-bit training is a future extension and is not an implemented CLI option.

## Evaluate, register, and roll back

```powershell
.venv\Scripts\python.exe -m personal_ai.training evaluate --model C:\models\base-model --adapter data\training\run-001\epoch-1 --dataset heldout.jsonl --report data\training\run-001-evaluation.json --device cpu
.venv\Scripts\python.exe -m personal_ai.training promote --report data\training\run-001-evaluation.json --name run-001 --data data
.venv\Scripts\python.exe -m personal_ai.training rollback --name earlier-passing-run --data data
```

Evaluation computes token-weighted completion negative log-likelihood with and without the adapter on the same held-out examples. Promotion refuses non-finite or regressed loss, absent evidence, fewer than four examples, or an adapter changed since evaluation. The evaluation JSON is a local operator artifact, not a cryptographic attestation; do not trust third-party forged reports. Lower held-out loss does not guarantee better reasoning, alignment, or task success. Run manual and task-based regression checks before using a candidate.

**Registry activation is not inference deployment.** Chat continues to use the model selected in Personality. To use a trained adapter in Ollama, follow the current Ollama import documentation for your exact architecture and supported adapter format, verify a new imported model tag, and select that tag explicitly. This conversion/import is not automated or tested by this project. Rollback changes the registry; to roll back chat, select the previous known-good installed tag. Original base files and earlier adapters are retained.

## Grid learning

The web Learning lab runs tabular Q-learning without neural dependencies. It records measured episode rewards, a random-policy baseline and greedy learned policy on the same held-out starts, the action-value table, seed, board size, and limits. Progress is displayed after experiment completion, never invented while a job is running. Interrupted runs do not display a completed candidate evaluation. The grid is a deliberate toy environment, not Minecraft or proof of general AI.
