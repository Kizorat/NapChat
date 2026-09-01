#!/usr/bin/env python3
"""
train_t2.py — SFT QLoRA per T2 (completamento di turno in napoletano).

Tarato su Kaggle T4 (16 GB, compute capability 7.5): niente bf16, niente
FlashAttention, 4-bit nf4 + gradient checkpointing + ottimizzatore paginato.

    python train_t2.py --model minerva --split-dir /kaggle/working/split \\
        --out-dir /kaggle/working/runs/t2 --epochs 4 --lr 1e-4

Selezione del checkpoint: --metric eval_loss (default) oppure ctx_acc.
"""

import argparse
import csv as _csv
import json
import math
import os
import random
import time
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, EarlyStoppingCallback, Trainer,
                          TrainerCallback, TrainingArguments)

import t2_common as C

VISION = ("vision_tower", "vision_model", "multi_modal_projector", "visual",
          "image_encoder", "patch_embed")
TARGET = {"q_proj", "k_proj", "v_proj", "o_proj",
          "gate_proj", "up_proj", "down_proj"}


# La scelta di precisione vive in t2_common (C.scegli_dtype): train ed eval
# devono usare la stessa, altrimenti si valuta un modello caricato in un dtype
# diverso da quello in cui e' stato addestrato.


def moduli_lora(model):
    """Target LoRA scelti per NOME COMPLETO, non per suffisso: i modelli
    multimodali (Gemma-3) hanno q_proj/k_proj anche nel vision tower e
    attaccarci un adapter significa allenare parametri che questo task non usa."""
    nomi = set()
    for n, m in model.named_modules():
        if any(v in n for v in VISION):
            continue
        corto = n.split(".")[-1]
        if corto in TARGET and hasattr(m, "weight"):
            nomi.add(corto)
    return sorted(nomi) or ["q_proj", "v_proj"]


class TrainerPesato(Trainer):
    """Trainer con loss pesata per token.

    Serve a una cosa sola: dare al prefisso del turno un peso diverso da
    quello del target (vedi DatasetT2.peso_prefisso). Con pesi 0/1 il risultato
    coincide con la cross-entropy standard, quindi eval_loss resta confrontabile
    con i run precedenti.
    """

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None, **kw):
        inputs = dict(inputs)                 # prediction_step riusa il dict
        pesi = inputs.pop("pesi", None)
        labels = inputs.pop("labels")
        inputs.update(C.extra_forward(model, inputs["input_ids"]))
        out = model(**inputs)
        logits = out.logits[:, :-1]
        tgt = labels[:, 1:]
        maschera = tgt != -100
        if not maschera.any():
            perdita = logits.sum() * 0.0
            return (perdita, out) if return_outputs else perdita
        w = (pesi[:, 1:] if pesi is not None
             else maschera.float())[maschera]
        ce = F.cross_entropy(logits[maschera].float(), tgt[maschera],
                             reduction="none")
        somma = (ce * w).sum()

        # Normalizzazione: NON e' un dettaglio cosmetico.
        #
        # Da transformers 4.46, quando il modello accetta i loss kwargs (tutti
        # i CausalLM recenti), Trainer.training_step NON divide piu' la loss
        # per gradient_accumulation_steps: si aspetta che compute_loss
        # normalizzi da se' su num_items_in_batch, cioe' sul totale dei token
        # supervisionati dell'INTERA finestra di accumulo. Restituire una media
        # per micro-batch fa sommare GRAD_ACCUM medie fra loro: con accum=8 la
        # loss loggata parte da ~40 invece di ~5, e con essa i gradienti, che
        # e' come moltiplicare per 8 il learning rate senza saperlo.
        #
        # In valutazione num_items_in_batch non arriva e i pesi valgono 0 o 1,
        # quindi w.sum() coincide con il numero di token e le due strade danno
        # lo stesso numero.
        if num_items_in_batch is not None:
            perdita = somma / num_items_in_batch
        else:
            perdita = somma / w.sum().clamp(min=1e-6)
        return (perdita, out) if return_outputs else perdita


def campiona_stratificato(records, n, seed=0):
    """Sottoinsieme di dev per la callback, con le conversazioni nelle stesse
    proporzioni dell'insieme completo.

    Prima era `records[:n]`, cioe' i primi n in ordine di file. Sugli split
    originali quei 64 item erano 64 su 64 della sola KPN001: l'andamento di
    ctx_acc che si leggeva durante il training descriveva UNA conversazione,
    non il dev. E' il numero su cui si decide quando fermarsi, quindi conviene
    che sia rappresentativo. Il valore riportato da eval_t2.py e' sempre stato
    calcolato su tutto il dev: questo tocca solo la lettura intra-training."""
    if n >= len(records):
        return list(records)
    rng = random.Random(seed)
    gruppi = defaultdict(list)
    for r in records:
        gruppi[r.get("conversazione", "?")].append(r)
    fuori = []
    for c, v in sorted(gruppi.items()):
        k = min(len(v), max(1, round(n * len(v) / len(records))))
        fuori += rng.sample(v, k)
    rng.shuffle(fuori)
    return fuori[:n]


class StopSuNaN(TrainerCallback):
    """Ferma il run alla prima loss non finita.

    Serve soprattutto ai modelli della famiglia Gemma: sono addestrati in
    bf16 e su una T4 (che il bf16 non ce l'ha in hardware) si gira in fp16,
    dove alcune attivazioni escono dall'intervallo rappresentabile. Il sintomo
    e' una loss che diventa nan e non torna piu' indietro; senza questo
    controllo il run prosegue per ore producendo un adapter inutile."""

    def on_log(self, args, state, control, logs=None, **kw):
        v = (logs or {}).get("loss")
        if v is not None and not math.isfinite(v):
            print(f"\n! loss non finita ({v}) allo step {state.global_step}: "
                  "interrompo.\n"
                  "  Su T4 la causa quasi sempre e' fp16 su un modello nato in "
                  "bf16 (Gemma).\n"
                  "  Non si risolve abbassando il learning rate: e' overflow "
                  "di rappresentazione,\n  non instabilita' di ottimizzazione. "
                  "Rilancia con --dtype fp32 (circa il doppio\n  del tempo, ma "
                  "finito) oppure passa a una GPU Ampere, dove parte il bf16.",
                  flush=True)
            control.should_training_stop = True


class CtxAccCallback(TrainerCallback):
    """Aggiunge ctx_acc / ppl_target / acc_token a ogni valutazione.
    Due forward per item su un sottoinsieme: costa pochi secondi e rende
    visibile la cosa che il eval_loss da solo nasconde, cioe' se il modello sta
    imparando a produrre napoletano ignorando la conversazione."""

    def __init__(self, model, tok, records, stile, max_seq_len, n=64, seed=0):
        self.model, self.tok = model, tok
        self.records = campiona_stratificato(records, n, seed)
        self.composizione = dict(Counter(r.get("conversazione", "?")
                                         for r in self.records))
        self.stile, self.max_seq_len = stile, max_seq_len
        self.storia = []

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        try:
            m = C.metriche_teacher_forcing(self.model, self.tok, self.records,
                                           self.stile, self.max_seq_len, batch=2)
        except Exception as e:
            print(f"  [ctx] non calcolato: {e}")
            return
        m["step"] = state.global_step
        self.storia.append(m)
        if metrics is not None:
            for k in ("ctx_acc", "acc_token", "ppl_target"):
                if m[k] is not None:
                    metrics[f"eval_{k}"] = m[k]
        print(f"  [ctx] step {state.global_step}: ctx_acc={m['ctx_acc']} "
              f"acc_token={m['acc_token']} ppl_target={m['ppl_target']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="minerva")
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "bf16", "fp16", "fp32"],
                    help="auto: bf16 se la GPU lo supporta, fp32 per Gemma su "
                         "T4, fp16 altrove")
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out-dir", default="/kaggle/working/runs/t2")
    ap.add_argument("--stile", default="prefill", choices=["prefill", "chat"])
    ap.add_argument("--epochs", type=float, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--neftune", type=float, default=5.0,
                    help="rumore NEFTune sugli embedding. Pensato per l'SFT su "
                         "poche centinaia di esempi; 0 = spento")
    ap.add_argument("--peso-prefisso", type=float, default=0.3,
                    help="peso della loss sul prefisso del turno. 0 = solo "
                         "target, come nella versione precedente")
    ap.add_argument("--metric", default="eval_loss",
                    choices=["eval_loss", "eval_ctx_acc", "eval_acc_token"])
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init-adapter", default="",
                    help="adapter da cui CONTINUARE (es. uno stadio precedente)")
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-dev-samples", type=int, default=0)
    ap.add_argument("--solo-golden", action="store_true",
                    help="scarta dal train i target sintetici (fonte != golden)")
    a = ap.parse_args()

    from peft import (LoraConfig, PeftModel, get_peft_model,
                      prepare_model_for_kbit_training)
    from transformers import set_seed
    set_seed(a.seed)

    repo = C.MODEL_REGISTRY.get(a.model, a.model)
    nome_prec, dtype = C.scegli_dtype(repo, a.dtype)
    print(f"Modello : {repo}")
    print(f"GPU     : {torch.cuda.get_device_name(0)} -> precisione {nome_prec}")

    # ---- dati ------------------------------------------------------------
    train = C.load_split(a.split_dir, "train")
    dev = C.load_split(a.split_dir, "dev")
    if a.solo_golden:
        n0 = len(train)
        train = [r for r in train if r["fonte"] == "golden"]
        print(f"Solo golden: train {n0} -> {len(train)}")
    if a.max_train_samples:
        train = train[:a.max_train_samples]
    if a.max_dev_samples:
        dev = dev[:a.max_dev_samples]

    tok = AutoTokenizer.from_pretrained(repo, token=os.environ.get("HF_TOKEN"),
                                        use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # il dev resta a peso_prefisso=0: eval_loss deve misurare la stessa cosa
    # dei run precedenti, altrimenti la selezione del checkpoint cambia
    # significato da un esperimento all'altro
    ds_tr = C.DatasetT2(train, tok, a.max_seq_len, a.stile,
                        peso_prefisso=a.peso_prefisso)
    ds_dv = C.DatasetT2(dev, tok, a.max_seq_len, a.stile, peso_prefisso=0.0)
    print(f"peso della loss sul prefisso: {a.peso_prefisso}")
    if a.peso_prefisso > 0:
        print("! train loss ed eval loss NON sono la stessa quantita': il train "
              f"include il prefisso a peso {a.peso_prefisso}, il dev no (per "
              "restare confrontabile\n  fra esperimenti). Lo scarto fra le due "
              "curve non e' tutto overfitting. Per curve omogenee: "
              "--peso-prefisso 0")
        print("  Ordine di grandezza atteso: la train loss e' PIU' BASSA della "
              "eval loss di circa il fattore\n  (peso medio dei token), qui "
              f"~{a.peso_prefisso:.2f}-1.0. Se invece la vedi molte volte piu' "
              "ALTA (es. 40 contro 3),\n  non e' l'asimmetria: e' la "
              "normalizzazione della loss sull'accumulo — vedi TrainerPesato.")
    print("train:", ds_tr.diagnosi())
    print("dev  :", ds_dv.diagnosi())
    if ds_tr.prefix_mismatch:
        print(f"! {ds_tr.prefix_mismatch} item con ritokenizzazione al confine "
              "prompt/target: la maschera li segue comunque (si allinea sul "
              "prefisso comune reale), ma se il numero e' alto controlla il "
              "template prima di fidarti della loss")
    if ds_tr.troncati:
        print(f"! {ds_tr.troncati} item troncati: alza --max-seq-len")

    # ---- modello ---------------------------------------------------------
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=dtype)
    attn = C.attn_impl(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, quantization_config=bnb, device_map={"": 0},
        attn_implementation=attn, token=os.environ.get("HF_TOKEN"),
        **C.kw_dtype(dtype))
    model.config.use_cache = False        # incompatibile col gradient checkpointing
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})

    if a.init_adapter:
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
        print("Adapter iniziale CONTINUATO da:", a.init_adapter)
    else:
        lc = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha,
                        lora_dropout=a.lora_dropout, bias="none",
                        task_type="CAUSAL_LM", target_modules=moduli_lora(model))
        print("Moduli LoRA:", lc.target_modules)
        model = get_peft_model(model, lc)
    model.print_trainable_parameters()

    # ---- training --------------------------------------------------------
    eff = a.batch_size * a.grad_accum
    per_epoca = max(1, math.ceil(len(ds_tr) / eff))
    eval_steps = max(5, per_epoca // 2)     # DERIVATO: con 42 step/epoca un
    print(f"batch efficace {eff} | {per_epoca} step/epoca | "                 # eval_steps fisso a 100
          f"eval ogni {eval_steps} step")                                     # non partirebbe mai

    kw_targs = dict(
        output_dir=a.out_dir, overwrite_output_dir=True,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
        max_grad_norm=1.0, weight_decay=a.weight_decay,
        neftune_noise_alpha=(a.neftune or None),
        fp16=(nome_prec == "fp16"), bf16=(nome_prec == "bf16"),
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=max(1, eval_steps // 4),
        eval_steps=eval_steps, **{C.kw_strategia_eval(): "steps"},
        save_strategy="steps", save_steps=eval_steps, save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=a.metric,
        greater_is_better=(a.metric != "eval_loss"),
        report_to="none", seed=a.seed, dataloader_num_workers=0,
        remove_unused_columns=False, group_by_length=False,
    )
    kw_ok, scartati = C.kwargs_accettati(TrainingArguments, **kw_targs)
    if scartati:
        print("! parametri non supportati da questa versione di transformers, "
              "ignorati:", scartati)
    targs = TrainingArguments(**kw_ok)

    cb_ctx = CtxAccCallback(model, tok, dev, a.stile, a.max_seq_len, seed=a.seed)
    print(f"ctx_acc intra-training su {len(cb_ctx.records)} item di dev, "
          f"stratificati per conversazione: {cb_ctx.composizione}")
    trainer = TrainerPesato(
        model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_dv,
        data_collator=lambda b: C.collate(b, tok.pad_token_id),
        # ORDINE: cb_ctx PRIMA di EarlyStoppingCallback. Entrambi girano dentro
        # on_evaluate; cb_ctx inietta ctx_acc/acc_token nel dizionario metrics e
        # l'early stopping lo legge subito dopo. Invertendoli, con
        # --metric eval_ctx_acc l'early stopping non troverebbe la sua metrica.
        callbacks=[cb_ctx, StopSuNaN(),
                   EarlyStoppingCallback(early_stopping_patience=a.patience)],
    )

    t0 = time.time()
    trainer.train()
    secondi = time.time() - t0

    os.makedirs(a.out_dir, exist_ok=True)
    finale = os.path.join(a.out_dir, "adapter_final")
    trainer.model.save_pretrained(finale)
    tok.save_pretrained(finale)

    ev = trainer.evaluate()
    riepilogo = {
        "repo": repo, "task": "T2", "stile": a.stile,
        "gpu": torch.cuda.get_device_name(0), "precisione": nome_prec,
        "dati": {"train": ds_tr.diagnosi(), "dev": ds_dv.diagnosi(),
                 "solo_golden": a.solo_golden},
        "iperparametri": {k: getattr(a, k) for k in
                          ("epochs", "lr", "dtype", "batch_size", "grad_accum",
                           "max_seq_len", "lora_r", "lora_alpha",
                           "lora_dropout", "weight_decay", "neftune",
                           "peso_prefisso", "seed", "metric", "patience")},
        "step_per_epoca": per_epoca, "eval_steps": eval_steps,
        "loss_asimmetrica": a.peso_prefisso > 0,
        "ctx_acc_sottoinsieme": cb_ctx.composizione,
        "eval_finale": {k: v for k, v in ev.items() if isinstance(v, (int, float))},
        "storia_ctx": cb_ctx.storia,
        # La storia COMPLETA del Trainer: train loss a ogni logging_steps,
        # eval_loss e metriche a ogni valutazione, learning rate, norma del
        # gradiente. Senza questa riga l'unica traccia delle curve resta
        # l'output della cella, che se ne va con la sessione: i checkpoint
        # intermedi vengono cancellati da save_total_limit e /kaggle/working
        # non sopravvive alla disconnessione.
        "log_history": trainer.state.log_history,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "secondi": round(secondi, 1), "adapter": finale,
    }
    with open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(riepilogo, f, ensure_ascii=False, indent=2)

    # Le stesse curve anche in CSV: log_history e' una lista di dizionari con
    # chiavi diverse a seconda che la voce venga da un log di training o da una
    # valutazione, e leggerla a mano e' scomodo. Qui le voci vengono unite per
    # step, cosi' train loss ed eval_loss finiscono sulla stessa riga quando
    # cadono sullo stesso step.
    campi = ["step", "epoch", "loss", "eval_loss", "eval_ctx_acc",
             "eval_acc_token", "eval_ppl_target", "learning_rate", "grad_norm"]
    per_step = {}
    for voce in trainer.state.log_history:
        s_ = voce.get("step")
        if s_ is None:
            continue
        riga = per_step.setdefault(s_, {"step": s_})
        for k in campi:
            if k in voce and voce[k] is not None:
                riga[k] = voce[k]
    percorso_curve = os.path.join(a.out_dir, "curve.csv")
    with open(percorso_curve, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=campi, extrasaction="ignore")
        w.writeheader()
        for s_ in sorted(per_step):
            w.writerow(per_step[s_])
    print(f"curve in {percorso_curve} ({len(per_step)} punti)")
    print(json.dumps(riepilogo["eval_finale"], indent=2))
    print(f"\nAdapter in {finale} | {secondi/60:.1f} minuti")


if __name__ == "__main__":
    main()
