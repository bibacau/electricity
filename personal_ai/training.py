"""Optional offline LoRA training CLI. Heavy dependencies are imported only on use."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from .store import Store


def read_dataset(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages", [])
        if not isinstance(messages,list) or len(messages) != 2 or any(not isinstance(m,dict) for m in messages) or [m.get("role") for m in messages] != ["user","assistant"] or any(not isinstance(m.get("content"),str) or not m["content"].strip() for m in messages):
            raise ValueError("Each JSONL row needs one nonempty user/assistant pair")
        rows.append(row)
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def fingerprint(path):
    root = Path(path)
    h = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
    if not files:
        raise ValueError("Cannot fingerprint an empty directory")
    for p in files:
        h.update((str(p.relative_to(root)) if root.is_dir() else p.name).encode())
        with p.open("rb") as f:
            for chunk in iter(lambda:f.read(1048576),b""):
                h.update(chunk)
    return h.hexdigest()


def example_hash(row):
    return hashlib.sha256(json.dumps(row["messages"],sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def encoded(rows, tokenizer, max_length):
    samples=[]
    for row in rows:
        full=tokenizer.apply_chat_template(row["messages"],tokenize=True,add_generation_prompt=False)
        prefix=tokenizer.apply_chat_template(row["messages"][:1],tokenize=True,add_generation_prompt=True)
        if full[:len(prefix)] != prefix or len(full)<=len(prefix):
            raise ValueError("Model chat template does not support reliable completion masking")
        if len(full)>max_length:
            raise ValueError("Example exceeds max length; curate shorter examples or increase --max-length")
        samples.append({"input_ids":full,"labels":[-100]*len(prefix)+full[len(prefix):]})
    return samples


def load_base(path, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32)
    model.to(device)
    return model,tokenizer


def tensors(sample,device):
    import torch
    return {k:torch.tensor([v],dtype=torch.long,device=device) for k,v in sample.items()}


def train(args):
    import torch
    from peft import LoraConfig, get_peft_model
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows=read_dataset(args.dataset)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    model,tokenizer=load_base(args.model,args.device)
    samples=encoded(rows,tokenizer,args.max_length)
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,target_modules="all-linear",task_type="CAUSAL_LM"))
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.learning_rate)
    step=0
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(samples)
        losses=[]
        for sample in samples:
            optimizer.zero_grad()
            loss=model(**tensors(sample,args.device)).loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; candidate will not be activated")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            step+=1
            losses.append(loss.item())
            if step>=args.max_steps:
                break
        checkpoint=output/f"epoch-{epoch+1}"
        model.save_pretrained(checkpoint,safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        (output/"training.json").write_text(json.dumps({"base_model":str(Path(args.model).resolve()),"dataset_hash":fingerprint(args.dataset),"example_hashes":[example_hash(row) for row in rows],"seed":args.seed,"steps":step,"mechanism":"LoRA weight updates","auto_activation":False},indent=2))
        print(json.dumps({"epoch":epoch+1,"steps":step,"training_loss":sum(losses)/len(losses),"checkpoint":str(checkpoint)}),flush=True)
        if step>=args.max_steps:
            break


def evaluate(args):
    import torch
    from peft import PeftModel
    rows=read_dataset(args.dataset)
    if len(rows)<4:
        raise ValueError("Evaluation requires at least four held-out examples")
    metadata=Path(args.adapter).parent/"training.json"
    if not metadata.is_file():
        raise ValueError("Adapter requires training.json from this pipeline")
    training=json.loads(metadata.read_text())
    if training["dataset_hash"]==fingerprint(args.dataset):
        raise ValueError("Use a separate held-out evaluation dataset")
    if set(training["example_hashes"]) & {example_hash(row) for row in rows}:
        raise ValueError("Evaluation examples overlap the training dataset")
    model,tokenizer=load_base(args.model,args.device)
    if str(Path(args.model).resolve())!=training["base_model"]:
        raise ValueError("Evaluation base model must match training")
    samples=encoded(rows,tokenizer,args.max_length)
    def score(net):
        net.eval()
        total,tokens=0.0,0
        with torch.no_grad():
            for sample in samples:
                n=sum(x!=-100 for x in sample["labels"][1:])
                loss=net(**tensors(sample,args.device)).loss.item()
                if not math.isfinite(loss):
                    raise ValueError("Non-finite evaluation loss")
                total+=loss*n
                tokens+=n
        return total/tokens
    baseline=score(model)
    candidate=score(PeftModel.from_pretrained(model,args.adapter,is_trainable=False,local_files_only=True))
    report={"metric":"heldout_completion_nll_lower_is_better","baseline":baseline,"candidate":candidate,
            "examples":len(samples),"adapter":str(Path(args.adapter).resolve()),"adapter_hash":fingerprint(args.adapter),
            "base_model":str(Path(args.model).resolve()),"dataset_hash":fingerprint(args.dataset),"passed":candidate<=baseline,
            "limitations":"Loss is a limited regression gate, not a reasoning, safety or gaming benchmark. Review examples manually."}
    with Path(args.report).open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


def promote(args):
    report=json.loads(Path(args.report).read_text())
    if report.get("metric")!="heldout_completion_nll_lower_is_better" or report.get("examples",0)<4 or not report.get("passed"):
        raise ValueError("Candidate needs a passing held-out evaluation")
    if any(not isinstance(report.get(k),(int,float)) or not math.isfinite(report[k]) for k in ["baseline","candidate"]) or report["candidate"]>report["baseline"]:
        raise ValueError("Invalid or regressed evaluation")
    if fingerprint(report["adapter"])!=report["adapter_hash"]:
        raise ValueError("Adapter changed since evaluation")
    store=Store(Path(args.data))
    with store.connect() as db:
        db.execute("UPDATE checkpoints SET active=0")
        db.execute("INSERT INTO checkpoints(name,path,baseline,candidate,evidence,active) VALUES (?,?,?,?,?,1)",(args.name,report["adapter"],report["baseline"],report["candidate"],json.dumps(report)))
    store.event("checkpoint.promote",{"name":args.name,"inference_switched":False})
    print("Evaluated candidate registered. Ollama inference remains unchanged; conversion/import is a separate manual step.")


def rollback(args):
    store=Store(Path(args.data))
    with store.connect() as db:
        if not db.execute("SELECT id FROM checkpoints WHERE name=?",(args.name,)).fetchone():
            raise ValueError("Unknown checkpoint")
        db.execute("UPDATE checkpoints SET active=(name=?)",(args.name,))
    store.event("checkpoint.rollback",{"name":args.name,"inference_switched":False})
    print("Registry rolled back. Select the corresponding imported model in Personality to change inference.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ["train","evaluate"]:
        p=sub.add_parser(name)
        p.add_argument("--model",required=True,help="Local Hugging Face model directory")
        p.add_argument("--dataset",required=True)
        p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
        p.add_argument("--max-length",type=int,default=512)
        if name=="train":
            p.add_argument("--output",required=True)
            p.add_argument("--epochs",type=int,choices=range(1,101),default=1)
            p.add_argument("--max-steps",type=int,default=100)
            p.add_argument("--seed",type=int,default=42)
            p.add_argument("--learning-rate",type=float,default=0.0002)
        else:
            p.add_argument("--adapter",required=True)
            p.add_argument("--report",required=True)
    p=sub.add_parser("promote")
    p.add_argument("--report",required=True)
    p.add_argument("--name",required=True)
    p.add_argument("--data",default="data")
    p=sub.add_parser("rollback")
    p.add_argument("--name",required=True)
    p.add_argument("--data",default="data")
    args=parser.parse_args()
    if getattr(args,"max_steps",1)<1 or getattr(args,"max_length",1)<1 or not math.isfinite(getattr(args,"learning_rate",1)) or getattr(args,"learning_rate",1)<=0:
        parser.error("Step limit, context length and learning rate must be positive and finite")
    globals()[args.command](args)


if __name__=="__main__":
    main()
