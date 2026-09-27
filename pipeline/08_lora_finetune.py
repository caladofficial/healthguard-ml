"""Stage 8 — LoRA / QLoRA fine-tuning path (GPU machine).

NOT EXECUTED HERE. This box is 2 vCPU / 2 GB RAM with no GPU, so the
parameter-efficient path cannot run in this environment. It is included as the
runnable artefact for a GPU host, because the brief asks the fine-tuning method to
be chosen by compute budget, and this is the branch chosen when a GPU exists.

Compute-budget rule actually applied on this box
------------------------------------------------
    budget = 2 vCPU, 2 GB RAM, no GPU
    -> tree ensemble, full fit.  Measured: 400-tree LightGBM trains in 15 s on
       165k rows and scores macro-F1 0.9274.
    budget = 1 GPU (>=8 GB)
    -> QLoRA on a 4-bit base model, config below.

Run on a GPU host:
    python3 pipeline/08_lora_finetune.py --base Qwen/Qwen2.5-0.5B-Instruct \
        --train data/pipeline/train_augmented.parquet --epochs 2 --quant 4bit
"""
import os, sys, json, argparse, time, resource

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pipeline_cfg as C          # noqa: E402

DEFAULT_BASE = os.environ.get('HG_LLM_MODEL', 'Qwen/Qwen2.5-0.5B-Instruct')
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']


def render(row):
    """One triage record as a short clinical sentence (the model's input text)."""
    return (f"{int(row['age'])} year old, "
            f"HR {row['hr']:.0f}, SBP {row['sbp']:.0f}, RR {row['rr']:.0f}, "
            f"SpO2 {row['spo2']:.0f}, temp {row['temp_c']:.1f} C. Triage level:")


def build_bnb_config(bits):
    from transformers import BitsAndBytesConfig
    import torch
    if bits == 4:
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                  bnb_4bit_compute_dtype=torch.float16,
                                  bnb_4bit_use_double_quant=True)
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=DEFAULT_BASE)
    ap.add_argument('--train', default=os.path.join(C.PIPE, 'train_augmented.parquet'))
    ap.add_argument('--epochs', type=float, default=2.0)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--bsz', type=int, default=16)
    ap.add_argument('--quant', default='4bit', choices=['4bit', '8bit', 'none'])
    ap.add_argument('--max_rows', type=int, default=60000)
    a = ap.parse_args()

    import torch
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset

    df = pd.read_parquet(a.train).head(a.max_rows)
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = [render(r) + ' T' + str(int(r[C.LABEL])) for _, r in df.iterrows()]
    enc = tok(texts, truncation=True, max_length=128, padding='max_length',
              return_tensors='pt')
    enc['labels'] = enc['input_ids'].clone()
    ds = Dataset.from_dict({k: v for k, v in enc.items()}).train_test_split(
        test_size=0.1, seed=C.SEED)

    kw = dict(device_map='auto')
    bnb = build_bnb_config(a.quant)
    if bnb is not None:
        kw['quantization_config'] = bnb
    model = AutoModelForCausalLM.from_pretrained(a.base, **kw)
    if bnb is not None:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, task_type='CAUSAL_LM'))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    args = TrainingArguments(
        output_dir=os.path.join(ROOT, 'models', 'lora_out'),
        num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.bsz, logging_steps=25,
        save_strategy='epoch', report_to=[], bf16=torch.cuda.is_available())
    tr = Trainer(model=model, args=args, train_dataset=ds['train'],
                 eval_dataset=ds['test'])
    out = tr.train()
    r = resource.getrusage(resource.RUSAGE_SELF)
    C.log_json(f'08_lora_{a.quant}.json', {
        'stage': '08_lora_finetune', 'base': a.base, 'quantisation': a.quant,
        'lora_r': LORA_R, 'lora_alpha': LORA_ALPHA, 'target_modules': TARGET_MODULES,
        'epochs': a.epochs, 'lr': a.lr, 'batch_size': a.bsz, 'rows': int(len(df)),
        'trainable_params': int(trainable), 'total_params': int(total),
        'trainable_pct': round(100 * trainable / max(total, 1), 3),
        'train_loss': round(float(out.metrics.get('train_loss', -1)), 4),
        'eval_loss': round(float(out.metrics.get('eval_loss', -1)), 4),
        'wall_seconds': round(time.time() - t0, 1),
        'cpu_seconds': round(r.ru_utime + r.ru_stime, 1),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',
    })
    print('saved 08_lora_%s.json' % a.quant)


if __name__ == '__main__':
    main()
