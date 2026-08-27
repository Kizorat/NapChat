#!/usr/bin/env python3
"""
t3_train.py — SFT di T3 (replica conversazionale in napoletano) su Kaggle T4.

Differenze rispetto alla versione precedente, tutte mirate alla cecita' al
contesto:

  1. LOSS CONTRASTIVA SUL CONTESTO (--lambda-ctx, il pezzo nuovo).
     La cross-entropy su ~750 target non puo' insegnare la pertinenza: per un
     contesto dato ci sono centinaia di repliche valide, quindi il gradiente
     dice "produci QUESTA stringa" quando la cosa da imparare e' "produci una
     replica che ci sta". Qui ogni esempio viene visto DUE volte, col contesto
     vero e con un contesto preso da un altro punto, e si aggiunge

         relu( margine - ( logP(target|ctx vero) - logP(target|ctx falso) ) )

     Non serve un modello di riferimento (non e' DPO) e non servono preferenze
     annotate: il negativo si costruisce dai dati che ci sono. E' l'unico
     termine che rende il contesto causalmente rilevante per la loss.

  2. SELEZIONE DEL CHECKPOINT SU ctx_delta, non su chrF.
     chrF contro l'unico turno realmente pronunciato premia chi indovina quelle
     parole. ctx_delta misura esattamente cio' che manca.

  3. Fine turno verificato prima di partire, non dopo.
  4. Nessuno stadio A/A2: si parte dal modello di base. La pertinenza sta nel
     modello di base ed e' quello che il continued pretraining erode; se serve
     dialettalita' in piu' si aggiunge dopo, con --init-adapter.

Uso:
    python t3_train.py --model minerva --split-dir /kaggle/working/split_t3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time

MODEL_REGISTRY = {
    "minerva":       "sapienzanlp/Minerva-7B-instruct-v1.0",
    "llama":         "meta-llama/Llama-2-7b-chat-hf",
    "gemma":         "google/gemma-3-4b-it",
    "minerva-small": "sapienzanlp/Minerva-3B-base-v1.0",
    "gemma-tiny":    "google/gemma-3-1b-it",
    # Stand-in per lo smoke test di Llama-2: stessa architettura e stesso
    # tokenizer SentencePiece a 32k, ma chat template diverso (usa i marcatori
    # <|user|>/<|assistant|>, non [INST]). Convalida quindi il cablaggio,
    # il targeting LoRA e la maschera della loss; NON convalida il formato del
    # prompt. Ha pero' lo stesso token di fine turno (</s>), quindi la riga
    # "fine turno" dello smoke test resta significativa. E non e' gated.
    "llama-tiny":    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}
VISION_MARKERS = ("vision_tower", "vision_model", "multi_modal_projector",
                  "visual", "image_encoder", "patch_embed")


# --------------------------------------------------------------------------- #
# Utility condivise con lo script di valutazione
# --------------------------------------------------------------------------- #

def resolve_model(nome: str) -> str:
    return MODEL_REGISTRY.get(nome, nome)


def slug(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def scegli_dtype(repo_id: str, scelta: str = "auto"):
    """(dtype, bf16_ok). Gemma in fp16 va in overflow: le attivazioni escono
    dal range del formato, i logit diventano NaN e generate() restituisce
    stringhe vuote (col campionamento: 'probability tensor contains inf/nan').
    La loss in teacher forcing puo' restare sana mentre la generazione e' gia'
    morta, quindi il sintomo non e' la loss. Su GPU senza bf16 (T4 = cc 7.5)
    per Gemma si usa fp32: piu' lento, ma finito."""
    import torch
    bf16_ok = torch.cuda.get_device_capability(0)[0] >= 8
    if scelta == "bf16":
        return torch.bfloat16, True
    if scelta == "fp16":
        return torch.float16, False
    if scelta == "fp32":
        return torch.float32, False
    if bf16_ok:
        return torch.bfloat16, True
    if "gemma" in repo_id.lower():
        return torch.float32, False
    return torch.float16, False


def controlla_finito(model, tokenizer, prompt):
    """Un forward solo, per fallire in 5 secondi invece che dopo un'ora. Se i
    logit non sono finiti nessun numero a valle e' interpretabile: chrF 0,
    uscite vuote e ctx_delta NaN sono conseguenze, non misure."""
    import torch
    inp = tokenizer(render(tokenizer, prompt), return_tensors="pt",
                    add_special_tokens=False).to(model.device)
    with torch.no_grad():
        logits = model(**inp).logits
    ok = bool(torch.isfinite(logits).all())
    print(f"logit finiti: {ok} (dtype {logits.dtype})")
    if not ok:
        sys.exit("LOGIT NON FINITI: overflow numerico con questa precisione. "
                 "Rilancia con --dtype fp32, oppure passa a una GPU con bf16 "
                 "(L4, A100) che e' piu' veloce e piu' stabile.")


def carica_token(esplicito=None):
    """Facoltativo: serve solo per i modelli gated (Llama-2, Gemma).
    Ordine: argomento -> variabile d'ambiente -> Colab userdata -> Kaggle Secrets."""
    if esplicito:
        return esplicito
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from google.colab import userdata
        t = userdata.get("HF_TOKEN")
        if t:
            return t
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def base_ambiente():
    """/kaggle/working su Kaggle, /content su Colab, la cwd altrove."""
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working"
    if os.path.isdir("/content"):
        return "/content"
    return os.getcwd()


def carica_split(split_dir, nome, max_samples=None):
    righe = json.load(open(os.path.join(split_dir, f"{nome}.json"), encoding="utf-8"))
    return righe[:max_samples] if max_samples else righe


def blocco_contesto(prompt: str):
    """(testa, contesto, istruzione). Il formato e' quello di t3_dati.py:
    riga di testa, righe di contesto, '---', istruzione."""
    testa_e_ctx, _, istr = prompt.partition("\n---\n")
    righe = testa_e_ctx.split("\n")
    return righe[0], righe[1:], istr


def sostituisci_contesto(prompt: str, altro: str) -> str:
    """Stesso item, contesto di un altro punto della conversazione.
    Testa e istruzione restano identiche: l'unica variabile e' il contenuto."""
    testa, _, istr = blocco_contesto(prompt)
    _, ctx_altro, _ = blocco_contesto(altro)
    return "\n".join([testa] + ctx_altro) + "\n---\n" + istr


def render(tokenizer, prompt, target=None):
    """Chat template del tokenizer. Nessun ruolo system: l'istruzione e' gia'
    dentro il prompt, byte-identica fra training e inferenza."""
    if not getattr(tokenizer, "chat_template", None):
        if target is None:
            return prompt + "\n"
        return prompt + "\n" + target + (tokenizer.eos_token or "")

    def applica(u, a):
        msg = [{"role": "user", "content": u}]
        if a is None:
            return tokenizer.apply_chat_template(msg, tokenize=False,
                                                 add_generation_prompt=True)
        return tokenizer.apply_chat_template(msg + [{"role": "assistant", "content": a}],
                                             tokenize=False, add_generation_prompt=False)
    try:
        return applica(prompt, target)
    except Exception:
        wrap = lambda s: [{"type": "text", "text": s}]
        return applica(wrap(prompt), None if target is None else wrap(target))


def target_lora(model):
    """Solo il language model: su Gemma-3 il vision tower ha moduli omonimi."""
    import torch.nn as nn
    nomi, saltati = set(), 0
    for nome, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) and mod.__class__.__name__ not in (
                "Linear4bit", "Linear8bitLt"):
            continue
        foglia = nome.split(".")[-1]
        if foglia in ("lm_head",):
            continue
        if any(m in nome for m in VISION_MARKERS):
            saltati += 1
            continue
        if foglia.endswith("_proj") or foglia in ("gate_proj", "up_proj", "down_proj"):
            nomi.add(foglia)
    if saltati:
        print(f"  esclusi {saltati} moduli del vision tower")
    return sorted(nomi)


def training_args_compatibili(TrainingArguments, **kw):
    """TrainingArguments cambia firma fra transformers 4.x e 5.x (`group_by_length`,
    per esempio, in 5.x non esiste piu'). Invece di inseguire i singoli argomenti,
    si tengono solo quelli che la classe installata accetta davvero e si dichiara
    a schermo cosa e' stato scartato: un argomento silenziosamente ignorato e'
    peggio di uno mancante."""
    import inspect
    ammessi = set(inspect.signature(TrainingArguments.__init__).parameters)
    try:                                     # dataclass: i campi sono la fonte vera
        import dataclasses
        ammessi |= {f.name for f in dataclasses.fields(TrainingArguments)}
    except Exception:
        pass
    tenuti = {k: v for k, v in kw.items() if k in ammessi}
    scartati = sorted(set(kw) - set(tenuti))
    if scartati:
        print(f"  TrainingArguments: argomenti non supportati da questa versione "
              f"di transformers, ignorati -> {', '.join(scartati)}")
    return TrainingArguments(**tenuti)


def id_fine_turno(tokenizer, esempio_prompt, esempio_target):
    """L'id che il modello deve emettere per chiudere il turno. Con i chat
    template moderni NON e' eos_token_id ma <|im_end|> / <end_of_turn>: se non
    entra in generation_config.eos_token_id, generate() non si ferma mai e le
    uscite escono 2-3 volte piu' lunghe dell'umano.

    Non basta prendere ids[-1]: il template di Gemma chiude con
    '<end_of_turn>\\n' e l'ultimo token e' il newline (id 107), che come
    criterio d'arresto tronca ogni uscita a zero token. Si risale indietro
    fino al primo token speciale. Su Minerva (<|eot_id|> in ultima
    posizione) il risultato non cambia."""
    ids = tokenizer(render(tokenizer, esempio_prompt, esempio_target),
                    add_special_tokens=False)["input_ids"]
    speciali = set(tokenizer.all_special_ids or [])
    for i in reversed(ids):
        t = tokenizer.convert_ids_to_tokens([i])[0]
        if i in speciali or (t.startswith("<") and t.endswith(">")):
            return i
    return ids[-1]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class DatasetT3:
    """prompt (mascherato a -100) + target. Con contrastivo=True ogni item
    porta anche la versione col contesto sbagliato."""

    def __init__(self, righe, tokenizer, max_len, contrastivo=False, seed=0):
        self.righe, self.tok, self.max_len = righe, tokenizer, max_len
        self.contrastivo = contrastivo
        self.n_troncati = self.n_prefix_mismatch = 0
        rng = random.Random(seed)
        self.neg = []
        for i, r in enumerate(righe):
            if contrastivo and len(righe) > 1:
                j = rng.randrange(len(righe) - 1)
                j = j + (j >= i)
                self.neg.append(sostituisci_contesto(r["prompt"], righe[j]["prompt"]))
            else:
                self.neg.append(None)
            if i < 200:
                p = tokenizer(render(tokenizer, r["prompt"]),
                              add_special_tokens=False)["input_ids"]
                f = tokenizer(render(tokenizer, r["prompt"], r["target"]),
                              add_special_tokens=False)["input_ids"]
                if len(f) > max_len:
                    self.n_troncati += 1
                if f[:len(p)] != p:
                    self.n_prefix_mismatch += 1

    def __len__(self):
        return len(self.righe)

    def _codifica(self, prompt, target):
        p = self.tok(render(self.tok, prompt), add_special_tokens=False)["input_ids"]
        f = self.tok(render(self.tok, prompt, target),
                     add_special_tokens=False)["input_ids"][:self.max_len]
        lab = list(f)
        for j in range(min(len(p), len(lab))):
            lab[j] = -100
        return f, lab

    def __getitem__(self, i):
        r = self.righe[i]
        ids, lab = self._codifica(r["prompt"], r["target"])
        out = {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": lab}
        if self.contrastivo:
            n_ids, n_lab = self._codifica(self.neg[i], r["target"])
            out |= {"neg_input_ids": n_ids, "neg_attention_mask": [1] * len(n_ids),
                    "neg_labels": n_lab}
        return out


def make_collate(pad_id, contrastivo):
    import torch

    def pad(seqs, valore):
        n = max(len(s) for s in seqs)
        return torch.tensor([s + [valore] * (n - len(s)) for s in seqs], dtype=torch.long)

    def collate(batch):
        out = {"input_ids": pad([b["input_ids"] for b in batch], pad_id),
               "attention_mask": pad([b["attention_mask"] for b in batch], 0),
               "labels": pad([b["labels"] for b in batch], -100)}
        if contrastivo and "neg_input_ids" in batch[0]:
            out |= {"neg_input_ids": pad([b["neg_input_ids"] for b in batch], pad_id),
                    "neg_attention_mask": pad([b["neg_attention_mask"] for b in batch], 0),
                    "neg_labels": pad([b["neg_labels"] for b in batch], -100)}
        return out
    return collate


# --------------------------------------------------------------------------- #
# Loss contrastiva
# --------------------------------------------------------------------------- #

def logp_per_sequenza(logits, labels):
    """(nll_somma_totale, n_token, logp_medio_per_sequenza).

    Si indicizzano PRIMA le posizioni etichettate: castare a fp32 tutti i logit
    di un batch 4x384x32000 sono ~200 MB per forward, e qui i forward sono due.
    Le posizioni che contano sono la decina di token del target.
    """
    import torch
    import torch.nn.functional as F
    lg = logits[:, :-1, :]
    lb = labels[:, 1:]
    maschera = lb != -100
    idx_b, idx_t = maschera.nonzero(as_tuple=True)
    if idx_b.numel() == 0:
        zero = logits.sum() * 0.0
        return zero, torch.tensor(1, device=logits.device), zero.expand(logits.size(0))
    nll = F.cross_entropy(lg[idx_b, idx_t].float(), lb[idx_b, idx_t], reduction="none")
    somma = torch.zeros(logits.size(0), device=logits.device, dtype=nll.dtype
                        ).index_add_(0, idx_b, nll)
    n_seq = torch.zeros_like(somma).index_add_(0, idx_b, torch.ones_like(nll)).clamp(min=1)
    return nll.sum(), maschera.sum(), -(somma / n_seq)


def costruisci_trainer_contrastivo(Trainer, lam, margine):
    class TrainerContrastivo(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None, **kw):
            neg = {k[4:]: inputs.pop(k) for k in list(inputs) if k.startswith("neg_")}
            out = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"])
            nll_somma, n_tok, logp_pos = logp_per_sequenza(out.logits, inputs["labels"])

            # Denominatore identico per i due termini: transformers non divide
            # la loss per gli step di accumulo quando il forward accetta **kwargs
            # (un PeftModel lo fa sempre), quindi la normalizzazione va fatta qui.
            denom = num_items_in_batch if num_items_in_batch else n_tok
            loss = nll_somma / denom

            if lam > 0 and neg:
                out_n = model(input_ids=neg["input_ids"],
                              attention_mask=neg["attention_mask"])
                _, _, logp_neg = logp_per_sequenza(out_n.logits, neg["labels"])
                margine_loss = (margine - (logp_pos - logp_neg)).clamp(min=0).mean()
                loss = loss + lam * margine_loss * (n_tok / denom)

            return (loss, out) if return_outputs else loss
    return TrainerContrastivo


# --------------------------------------------------------------------------- #
# Metriche di valutazione periodica
# --------------------------------------------------------------------------- #

def callback_contesto(TrainerCallback, torch, tokenizer, righe, n, seed, max_len):
    """ctx_delta = media di logP(target|ctx vero) - logP(target|ctx falso), in
    nat per token. Sopra 0 il contesto viene usato; a 0 il modello e' cieco.
    Nessuna generazione: costa due forward su n item."""
    rng = random.Random(seed)
    campione = list(righe)
    rng.shuffle(campione)
    campione = campione[:n]
    falsi = [sostituisci_contesto(r["prompt"], campione[(i + 1) % len(campione)]["prompt"])
             for i, r in enumerate(campione)]

    class CB(TrainerCallback):
        def on_evaluate(self, args, state, control, model=None, metrics=None, **kw):
            if model is None or metrics is None:
                return
            model.eval()
            deltas, vinti = [], 0
            with torch.no_grad():
                for r, falso in zip(campione, falsi):
                    lp = []
                    for prm in (r["prompt"], falso):
                        ids = tokenizer(render(tokenizer, prm, r["target"]),
                                        add_special_tokens=False,
                                        return_tensors="pt")["input_ids"][:, :max_len]
                        n_p = len(tokenizer(render(tokenizer, prm),
                                            add_special_tokens=False)["input_ids"])
                        lab = ids.clone()
                        lab[:, :n_p] = -100
                        ids, lab = ids.to(model.device), lab.to(model.device)
                        logits = model(input_ids=ids).logits
                        lp.append(logp_per_sequenza(logits, lab)[2].item())
                    deltas.append(lp[0] - lp[1])
                    vinti += lp[0] > lp[1]
            metrics["eval_ctx_delta"] = round(sum(deltas) / len(deltas), 4)
            metrics["eval_ctx_acc"] = round(vinti / len(deltas), 4)
            print(f"    ctx_delta {metrics['eval_ctx_delta']:+.4f} nat/token | "
                  f"ctx_acc {metrics['eval_ctx_acc']:.3f} (caso: 0.500)")
            model.train()
    return CB


def callback_generazione(TrainerCallback, torch, tokenizer, righe, n, eot_ids, max_new):
    """chrF++ su generazione REALE + rapporto di lunghezza. In teacher forcing
    l'ipotesi ha per costruzione la lunghezza del riferimento, quindi il
    fallimento sull'EOS resta invisibile: qui no."""
    import sacrebleu
    campione = righe[:n]

    class CB(TrainerCallback):
        def on_evaluate(self, args, state, control, model=None, metrics=None, **kw):
            if model is None or metrics is None:
                return
            model.eval()
            model.config.use_cache = True
            ipo, rif = [], []
            with torch.no_grad():
                for r in campione:
                    ids = tokenizer(render(tokenizer, r["prompt"]),
                                    add_special_tokens=False, return_tensors="pt"
                                    ).to(model.device)
                    o = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                       eos_token_id=eot_ids,
                                       pad_token_id=tokenizer.pad_token_id,
                                       repetition_penalty=1.15, no_repeat_ngram_size=3)
                    ipo.append(tokenizer.decode(o[0][ids["input_ids"].shape[1]:],
                                                skip_special_tokens=True).strip())
                    rif.append(r["target"])
            model.config.use_cache = False
            metrics["eval_gen_chrf"] = round(
                sacrebleu.corpus_chrf(ipo, [rif], word_order=2).score, 3)
            lr_ = (sum(len(h.split()) for h in ipo) /
                   max(1, sum(len(t.split()) for t in rif)))
            metrics["eval_rapporto_lunghezza"] = round(lr_, 3)
            print(f"    gen_chrf {metrics['eval_gen_chrf']:.2f} | "
                  f"lunghezza generato/umano {lr_:.2f}x (obiettivo ~1.0)")
            print(f"    esempio: {ipo[0][:90]!r}\n       umano: {rif[0][:90]!r}")
            model.train()
    return CB


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="minerva")
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--tag", default="T3")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=384)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lambda-ctx", type=float, default=0.5,
                    help="peso della loss contrastiva sul contesto (0 = spenta)")
    ap.add_argument("--margine", type=float, default=0.15,
                    help="margine in nat/token fra contesto vero e falso")
    ap.add_argument("--metric", default="ctx_delta",
                    choices=["ctx_delta", "gen_chrf", "loss"])
    ap.add_argument("--evals-per-epoch", type=int, default=2)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--ctx-n", type=int, default=64)
    ap.add_argument("--gen-n", type=int, default=24)
    ap.add_argument("--gen-max-new", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init-adapter", default="")
    ap.add_argument("--resume-dir", default="")
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--eval-subset", type=int, default=96)
    ap.add_argument("--no-gradient-checkpointing", action="store_true")
    ap.add_argument("--precision", default="qlora4bit", choices=["qlora4bit", "fp16"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                    help="auto: bf16 se la GPU lo supporta, fp32 per Gemma su T4, "
                         "fp16 altrove")
    a = ap.parse_args()
    a.out_dir = a.out_dir or os.path.join(base_ambiente(), "runs")

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              EarlyStoppingCallback, Trainer, TrainerCallback,
                              TrainingArguments)
    try:
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError:                      # transformers 5.x
        from transformers.trainer_callback import get_last_checkpoint
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    random.seed(a.seed)
    torch.manual_seed(a.seed)

    repo = resolve_model(a.model)
    dtype, bf16_ok = scegli_dtype(repo, a.dtype)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    import transformers
    print(f"{repo} | GPU {torch.cuda.get_device_name(0)} {vram:.1f} GB | "
          f"precisione {str(dtype).replace('torch.', '')} | "
          f"transformers {transformers.__version__}")

    token = carica_token()
    tokenizer = AutoTokenizer.from_pretrained(repo, token=token, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- dati ---------------------------------------------------------------
    train_rows = carica_split(a.split_dir, "train", a.max_train_samples or None)
    dev_rows = carica_split(a.split_dir, "dev")
    print(f"dati: train {len(train_rows)} | dev {len(dev_rows)}")

    # --- fine turno: si controlla PRIMA di addestrare ------------------------
    eot = id_fine_turno(tokenizer, train_rows[0]["prompt"], train_rows[0]["target"])
    eot_ids = sorted({eot, tokenizer.eos_token_id} - {None})
    print(f"fine turno: id={eot} repr={tokenizer.convert_ids_to_tokens([eot])[0]!r} | "
          f"eos_token_id del tokenizer={tokenizer.eos_token_id} | "
          f"generate() si fermera' su {eot_ids}")

    # --- modello ------------------------------------------------------------
    kw = dict(device_map={"": 0}, token=token, low_cpu_mem_usage=True,
              attn_implementation="eager")
    if a.precision == "qlora4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype, **kw)

    usa_ckpt = not a.no_gradient_checkpointing
    model.config.use_cache = False
    if a.precision == "qlora4bit":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=usa_ckpt,
            gradient_checkpointing_kwargs={"use_reentrant": False} if usa_ckpt else None)
    elif usa_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    if a.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
        print(f"adapter iniziale: {a.init_adapter}")
    else:
        model = get_peft_model(model, LoraConfig(
            r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
            target_modules=target_lora(model), bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    controlla_finito(model, tokenizer, train_rows[0]["prompt"])
    if usa_ckpt:
        model.enable_input_require_grads()

    # --- dataset ------------------------------------------------------------
    contrastivo = a.lambda_ctx > 0
    train_ds = DatasetT3(train_rows, tokenizer, a.max_seq_len, contrastivo, a.seed)
    dev_eval = dev_rows[:a.eval_subset] if a.eval_subset else dev_rows
    dev_ds = DatasetT3(dev_eval, tokenizer, a.max_seq_len, False, a.seed)
    if train_ds.n_prefix_mismatch:
        sys.exit(f"PREFIX MISMATCH su {train_ds.n_prefix_mismatch}/200: il prompt "
                 f"renderizzato non e' prefisso del testo completo, la maschera della "
                 f"loss e' disallineata. Non proseguire.")
    if train_ds.n_troncati:
        print(f"  ! {train_ds.n_troncati}/200 troncati a {a.max_seq_len}: alza --max-seq-len")

    # --- cadenza ------------------------------------------------------------
    eff = a.batch_size * a.grad_accum
    step_epoca = max(1, math.ceil(len(train_rows) / eff))
    max_steps = max(1, math.ceil(a.epochs * step_epoca))
    eval_steps = max(5, step_epoca // max(1, a.evals_per_epoch))
    n_eval = max_steps // eval_steps
    print(f"batch efficace {eff} | {step_epoca} step/epoca | {max_steps} totali | "
          f"eval ogni {eval_steps} -> ~{n_eval} valutazioni")
    if n_eval < a.patience + 1:
        print(f"  ! ~{n_eval} eval contro patience={a.patience}: l'early stopping "
              f"non potra' scattare.")

    run = f"{slug(repo)}__{a.tag}"
    out_dir = os.path.join(a.out_dir, run)
    os.makedirs(out_dir, exist_ok=True)
    if a.resume_dir:
        src = os.path.join(a.resume_dir, run)
        src = src if os.path.isdir(src) else a.resume_dir
        if os.path.isdir(src):
            for nome in os.listdir(src):
                if nome.startswith("checkpoint-") and not os.path.exists(
                        os.path.join(out_dir, nome)):
                    shutil.copytree(os.path.join(src, nome), os.path.join(out_dir, nome))
    riprendi = get_last_checkpoint(out_dir)
    if riprendi:
        print("riprendo da", riprendi)

    targs = training_args_compatibili(
        TrainingArguments,
        output_dir=out_dir, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.eval_batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, round(0.05 * max_steps)),
        weight_decay=0.01, max_grad_norm=0.3,
        eval_strategy="steps", eval_steps=eval_steps,
        save_strategy="steps", save_steps=eval_steps, save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=a.metric,
        greater_is_better=(a.metric != "loss"),
        logging_steps=max(1, eval_steps // 4), report_to="none",
        seed=a.seed, data_seed=a.seed,
        fp16=(dtype == torch.float16), bf16=bf16_ok,
        optim="paged_adamw_8bit", remove_unused_columns=False,
        group_by_length=False, dataloader_num_workers=2,
        label_names=["labels"],
        # senza questo il Trainer accumula i logit di tutto il dev per
        # compute_metrics (96 x 384 x |V|): OOM garantito su T4.
        prediction_loss_only=True,
    )

    # ORDINE: EarlyStoppingCallback legge metrics[metric_for_best_model], quindi
    # deve stare DOPO i callback che scrivono ctx_delta e gen_chrf.
    cbs = []
    if a.ctx_n:
        cbs.append(callback_contesto(TrainerCallback, torch, tokenizer, dev_rows,
                                     a.ctx_n, a.seed, a.max_seq_len)())
    if a.gen_n:
        cbs.append(callback_generazione(TrainerCallback, torch, tokenizer, dev_rows,
                                        a.gen_n, eot_ids, a.gen_max_new)())
    cbs.append(EarlyStoppingCallback(early_stopping_patience=a.patience))

    Base = costruisci_trainer_contrastivo(Trainer, a.lambda_ctx, a.margine)
    trainer = Base(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                   data_collator=make_collate(tokenizer.pad_token_id, contrastivo),
                   callbacks=cbs)

    print(f"\ncontrastiva: lambda={a.lambda_ctx} margine={a.margine} | "
          f"selezione su {a.metric}\n")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=riprendi)
    minuti = (time.time() - t0) / 60

    finale = os.path.join(out_dir, "adapter_final")
    trainer.model.save_pretrained(finale)
    tokenizer.save_pretrained(finale)

    json.dump({
        "repo_id": repo, "run": run, "minuti": round(minuti, 1),
        "split_dir": a.split_dir, "train": len(train_rows), "dev": len(dev_rows),
        "iperparametri": vars(a),
        "fine_turno": {"id": eot, "eos_usati": eot_ids},
        "best": {"step": trainer.state.best_model_checkpoint,
                 "metrica": a.metric, "valore": trainer.state.best_metric},
        "storia": [{k: v for k, v in h.items() if k.startswith("eval_") or k == "step"}
                   for h in trainer.state.log_history if "eval_loss" in h],
    }, open(os.path.join(out_dir, "summary.json"), "w"), indent=2, default=str)

    print(f"\nfatto in {minuti:.1f} min -> {finale}")
    print(f"best {a.metric} = {trainer.state.best_metric}")


if __name__ == "__main__":
    main()
