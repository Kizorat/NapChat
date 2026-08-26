#!/usr/bin/env python3
"""
STADIO A — continued pretraining sul napoletano, prima dell'SFT sui task.

Perche' serve
-------------
L'SFT sui task addestra coppie prompt->target: l'adapter impara a condizionare i
PRIMI token della risposta, ma la distribuzione linguistica sottostante resta
quella del modello di base, cioe' italiana. Effetto osservato nei log: la
generazione parte in napoletano e scivola in italiano man mano che si allontana
dal prompt ("pero' poi ce sta sempre l'incognita che puo' essere un veleno per
la"). Il continued pretraining sposta la DISTRIBUZIONE, non solo il
condizionamento iniziale.

Inoltre usa tutto il testo napoletano disponibile invece delle sole parole di
target: sul pool di train sono ~11.000 parole contro le 6.005 che T3 usa come
supervisione.

Si esegue UNA VOLTA per modello e serve tutti e tre i task:

    python pretrain_dialect.py --model minerva --split-dir /kaggle/working/split
    python finetune_t1_traduzione.py --model minerva --split-dir ... \\
        --init-adapter /kaggle/working/cpt/minerva-7b-instruct-v1.0/adapter_final
    (idem per T2 e T3)

Anti-leakage
------------
Il testo viene ricostruito ESCLUSIVAMENTE dai file train.json dei tre layout.
I pool di dev e test non vengono mai letti per il training: i turni di test
comparirebbero come testo di pretraining, che e' leakage puro e invaliderebbe
tutta la valutazione. Il dev viene usato solo per la perplexity di monitoraggio.

Formato
-------
Righe "PARLANTE: turno napoletano", identiche al formato dei contesti di
T1/T2/T3, cosi' fra stadio A e stadio B non c'e' disallineamento di formato.
Nessun chat template, nessun masking: language modeling puro, loss su tutto.
I turni vengono concatenati e impacchettati in blocchi di --block-size token.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

from Minerva7B.T1_Traduzione.training.common import (LAYOUT_DIRS, MODEL_REGISTRY, find_lora_target_modules,
                   load_backbone, load_hf_token, load_split, resolve_model, slug)


def ricostruisci_turni(split_dir, split_name):
    """Ricostruisce i turni napoletani dai file di UN solo split.

    Ogni layout espone il turno in modo diverso, quindi si prende l'unione
    tenendo la ricostruzione piu' lunga per ciascun (conversazione, turn_index):
      T1: target = turno intero
      T2: prefisso (dentro il prompt) + target = turno intero
      T3: target = turno intero
    """
    turni = {}
    for layout in ("T1", "T2", "T3"):
        try:
            rows = load_split(split_dir, layout, split_name)
        except SystemExit:
            continue
        for r in rows:
            chiave = (r["conversazione"], int(r["turn_index"]))
            if layout == "T2":
                marker = "Continua il turno in napoletano: "
                pref = r["prompt"].split(marker)[-1].strip() if marker in r["prompt"] else ""
                testo = (pref + " " + r["target"]).strip()
            else:
                testo = r["target"].strip()
            spk = r.get("speaker", "?")
            if len(testo) > len(turni.get(chiave, ("", ""))[1]):
                turni[chiave] = (spk, testo)
    ordinati = [turni[k] for k in sorted(turni)]
    return ordinati


def costruisci_testo(turni):
    """Una riga per turno, nello stesso formato dei contesti dei tre layout."""
    return "\n".join(f"{spk}: {testo}" for spk, testo in turni)


class BlocchiDataset:
    """Language modeling puro: input_ids == labels, nessun masking."""

    def __init__(self, testo, tok, block_size):
        ids = tok(testo, add_special_tokens=False)["input_ids"]
        n = (len(ids) // block_size) * block_size
        if n == 0:                                  # testo piu' corto di un blocco
            n, block_size = len(ids), len(ids)
        self.blocchi = [ids[i:i + block_size] for i in range(0, n, block_size)]
        self.n_token = len(ids)

    def __len__(self):
        return len(self.blocchi)

    def __getitem__(self, i):
        b = self.blocchi[i]
        return {"input_ids": b, "attention_mask": [1] * len(b), "labels": list(b)}


def collate(batch):
    import torch
    return {k: torch.tensor([b[k] for b in batch], dtype=torch.long) for k in batch[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=f"alias ({'/'.join(MODEL_REGISTRY)}) o repo_id")
    ap.add_argument("--split-dir", default="/kaggle/working/split")
    ap.add_argument("--out-dir", default="/kaggle/working/cpt")
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="piu' basso dell'SFT: qui si sposta la distribuzione, non si "
                         "impara un mapping, e con 11k parole e' facile degradare il modello")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--dump-text", action="store_true",
                    help="salva il testo di pretraining su file, per ispezionarlo")
    a = ap.parse_args()

    import torch
    from transformers import (AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback,
                              Trainer, TrainingArguments, set_seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    set_seed(a.seed)
    repo_id = resolve_model(a.model)
    if not torch.cuda.is_available():
        sys.exit("Nessuna GPU CUDA.")
    cc = torch.cuda.get_device_capability(0)
    bf16_ok = cc[0] >= 8
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    print(f"=== STADIO A — continued pretraining | {repo_id} ===")
    print(f"GPU {torch.cuda.get_device_name(0)} (cc {cc[0]}.{cc[1]}) -> "
          f"{'bf16' if bf16_ok else 'fp16'}")

    token = load_hf_token(a.hf_token)
    tok = AutoTokenizer.from_pretrained(repo_id, token=token, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- testo: SOLO dal pool di train -------------------------------------
    turni_tr = ricostruisci_turni(a.split_dir, "train")
    turni_dv = ricostruisci_turni(a.split_dir, "dev")
    if not turni_tr:
        sys.exit("Nessun turno ricostruito dal train: controlla --split-dir.")
    testo_tr, testo_dv = costruisci_testo(turni_tr), costruisci_testo(turni_dv)
    parole_tr = len(testo_tr.split())
    print(f"Turni ricostruiti: train {len(turni_tr)} ({parole_tr} parole) | "
          f"dev {len(turni_dv)} ({len(testo_dv.split())} parole)")
    print("  (dev e test non entrano mai nel training: sarebbe leakage)")
    # trasparenza: alcune formule brevissime ricorrono naturalmente in piu' punti
    # del corpus, quindi compaiono sia in train sia in test. Non e' leakage - e'
    # lo stesso fenomeno per cui "sì sì sì" appare in mezza conversazione - ma il
    # numero va riportato invece di essere taciuto.
    turni_te = ricostruisci_turni(a.split_dir, "test")
    comuni = {t for _, t in turni_tr} & {t for _, t in turni_te}
    if comuni:
        print(f"  {len(comuni)} turni identici presenti sia in train sia in test "
              f"(formule ricorrenti, mediana {sorted(len(c.split()) for c in comuni)[len(comuni)//2]} "
              f"parole): non e' leakage di split, e' ricorrenza lessicale nel parlato")

    outdir = Path(a.out_dir) / slug(repo_id)
    outdir.mkdir(parents=True, exist_ok=True)
    if a.dump_text:
        (outdir / "testo_pretraining.txt").write_text(testo_tr, encoding="utf-8")
        print(f"  testo salvato in {outdir / 'testo_pretraining.txt'}")

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model, arch = load_backbone(repo_id, dtype, quantization_config=quant,
                               device_map={"": 0}, token=token, low_cpu_mem_usage=True,
                               attn_implementation="eager")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    target_modules = find_lora_target_modules(model)
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    ds_tr = BlocchiDataset(testo_tr, tok, a.block_size)
    ds_dv = BlocchiDataset(testo_dv, tok, a.block_size) if turni_dv else None
    print(f"Blocchi da {a.block_size} token: train {len(ds_tr)} "
          f"({ds_tr.n_token} token totali)" + (f" | dev {len(ds_dv)}" if ds_dv else ""))
    if len(ds_tr) < 4:
        print("  ! pochissimi blocchi: abbassa --block-size per averne di piu'")

    eff = a.batch_size * a.grad_accum
    spe = max(1, math.ceil(len(ds_tr) / eff))
    max_steps = max(1, math.ceil(a.epochs * spe))
    print(f"Batch efficace {eff} | {spe} step/epoca | {max_steps} step totali")

    args = TrainingArguments(
        output_dir=str(outdir),
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=0.3,
        # NEFTune e' pensato per l'instruction tuning: qui si modella testo, non
        # si segue un'istruzione, quindi resta disattivato
        eval_strategy="epoch" if ds_dv else "no",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=bool(ds_dv),
        metric_for_best_model="loss" if ds_dv else None,
        greater_is_better=False if ds_dv else None,
        logging_steps=max(1, spe // 4),
        report_to="none",
        seed=a.seed,
        fp16=not bf16_ok, bf16=bf16_ok,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_dv,
                      data_collator=collate,
                      callbacks=[EarlyStoppingCallback(2)] if ds_dv else None)
    trainer.train()

    final = outdir / "adapter_final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))

    log = trainer.state.log_history
    for r in log:
        if "eval_loss" in r:
            r["eval_perplexity"] = math.exp(min(r["eval_loss"], 20))
    ppl = [r["eval_perplexity"] for r in log if "eval_perplexity" in r]
    riepilogo = {
        "stadio": "A — continued pretraining",
        "repo_id": repo_id, "architettura": arch,
        "gpu": torch.cuda.get_device_name(0), "bf16": bf16_ok,
        "lora": {"r": a.lora_r, "alpha": a.lora_alpha, "dropout": a.lora_dropout,
                 "n_moduli_target": len(target_modules)},
        "hyperparams": {"lr": a.lr, "epochs": a.epochs, "block_size": a.block_size,
                        "effective_batch_size": eff, "max_steps": max_steps},
        "dati": {"turni_train": len(turni_tr), "parole_train": parole_tr,
                 "token_train": ds_tr.n_token, "blocchi_train": len(ds_tr),
                 "turni_dev": len(turni_dv),
                 "fonte": "solo i file train.json dei tre layout — dev e test esclusi"},
        "perplexity_dev": {"iniziale": round(ppl[0], 2) if ppl else None,
                           "finale": round(ppl[-1], 2) if ppl else None,
                           "migliore": round(min(ppl), 2) if ppl else None},
        "best_metric": trainer.state.best_metric,
        "adapter_final": str(final),
        "_uso": "passalo ai tre script di SFT con --init-adapter",
    }
    (outdir / "summary_cpt.json").write_text(
        json.dumps(riepilogo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(riepilogo["perplexity_dev"], indent=2))
    print(f"\nAdapter stadio A: {final}")
    print(f"Usalo con:  --init-adapter {final}")


if __name__ == "__main__":
    main()
