"""Explicit heavy integration test. Random tiny weights test mechanics, not intelligence.

Run: python -m pytest tests/training_optional.py -q
"""
import json
from argparse import Namespace

import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('transformers')
pytest.importorskip('peft')

from personal_ai.training import train, evaluate, fingerprint, promote, rollback
from personal_ai.store import Store


def test_actual_lora_training_evaluation_and_rollback(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from safetensors.torch import load_file
    torch.set_num_threads(1)
    torch.manual_seed(42)
    base=tmp_path/'base'
    base.mkdir()
    vocab={word:i for i,word in enumerate(['<pad>','<eos>','<unk>','user','assistant',':','Answer']+[f'Q{i}' for i in range(12)])}
    backend=Tokenizer(WordLevel(vocab,unk_token='<unk>'))
    backend.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='<pad>',eos_token='<eos>',unk_token='<unk>')
    tokenizer.chat_template="{% for m in messages %}{{ m['role'] + ': ' + m['content'] + ' ' }}{% if m['role'] == 'assistant' %}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    tokenizer.save_pretrained(base)
    GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab),n_positions=64,n_ctx=64,n_embd=16,n_layer=1,n_head=2,eos_token_id=1,pad_token_id=0)).save_pretrained(base)
    base_hash=fingerprint(base)
    for name,indices in [('train',range(8)),('heldout',range(8,12))]:
        (tmp_path/f'{name}.jsonl').write_text('\n'.join(json.dumps({'messages':[{'role':'user','content':f'Q{i}'},{'role':'assistant','content':'Answer'}]}) for i in indices))
    output=tmp_path/'run'
    train(Namespace(model=str(base),dataset=str(tmp_path/'train.jsonl'),output=str(output),device='cpu',seed=42,max_length=64,learning_rate=0.02,epochs=4,max_steps=32))
    assert fingerprint(base)==base_hash,'Base model must not change'
    adapter=output/'epoch-4'
    weights=load_file(adapter/'adapter_model.safetensors')
    assert any(t.abs().sum().item()>0 for k,t in weights.items() if 'lora_B' in k),'LoRA B starts at zero and must receive real updates'
    report=tmp_path/'evaluation.json'
    evaluate(Namespace(model=str(base),adapter=str(adapter),dataset=str(tmp_path/'heldout.jsonl'),report=str(report),device='cpu',max_length=64))
    evidence=json.loads(report.read_text())
    assert evidence['candidate']<evidence['baseline']
    assert evidence['passed']
    state=tmp_path/'data'
    promote(Namespace(report=str(report),name='tiny-mechanics-v1',data=str(state)))
    promote(Namespace(report=str(report),name='tiny-mechanics-v2',data=str(state)))
    rollback(Namespace(name='tiny-mechanics-v1',data=str(state)))
    rows=Store(state).rows('SELECT name FROM checkpoints WHERE active=1')
    assert rows==[{'name':'tiny-mechanics-v1'}]
    with pytest.raises(ValueError,match='separate held-out'):
        evaluate(Namespace(model=str(base),adapter=str(adapter),dataset=str(tmp_path/'train.jsonl'),report=str(tmp_path/'invalid.json'),device='cpu',max_length=64))
