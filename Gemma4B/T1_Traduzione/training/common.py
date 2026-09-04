#!/usr/bin/env python3
"""
common.py — nucleo condiviso dei tre script di fine-tuning per task.

NON si esegue da solo. Viene importato da:
    finetune_t1_traduzione.py
    finetune_t2_completamento.py
    finetune_t3_replica.py

Ogni entrypoint definisce solo un TaskConfig (layout, lunghezze, epoche,
metrica di selezione, parametri di generazione qualitativa) e chiama run().
Tutto il resto - caricamento modello, QLoRA, masking della loss, metriche,
resume, summary - vive qui in una sola copia: se cambia, cambia per tutti e
tre i task contemporaneamente, e il confronto cross-task resta valido.

Differenze rispetto al finetune.py monolitico:
  * eval_steps DERIVATO dagli step per epoca (era fisso a 100: su T3, con 87
    step totali, non partiva nessuna eval e l'early stopping era inerte)
  * bf16/fp16 rilevati a runtime: le GPU Kaggle (T4, P100) sono pre-Ampere e
    NON supportano bf16. Il vecchio script forzava bnb_4bit_compute_dtype=
    bfloat16 e non impostava mai fp16/bf16 in TrainingArguments
  * attn_implementation="eager": obbligatorio per Gemma-2/3 (logit soft-capping,
    SDPA e FlashAttention lo implementano male o non lo implementano)
  * token HuggingFace dai Kaggle Secrets invece di .napoli/.api
  * percorsi split espliciti (/kaggle/input, read-only) separati dall'output
    (/kaggle/working, scrivibile)
  * resume da un dataset Kaggle montato in read-only (i checkpoint vengono
    copiati in working prima di riprendere)
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

LORA_TARGET_CANDIDATES = {"q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"}

# Alias -> repo_id. Sostituisce model.txt: su Kaggle i modelli arrivano dall'hub
# (o da un dataset montato), non da un file locale della macchina di sviluppo.
# I tre checkpoint sono quelli del tuo model.txt, non altri: cambiare versione
# a metà matrice renderebbe i 9 run non confrontabili.
# NOTA sulle taglie: 7B / 7B / 4B. Gemma-3-4b è la più piccola dei tre, quindi
# un suo punteggio più basso non è automaticamente un limite del modello: va
# dichiarato come asimmetria di taglia tra gli arm del confronto.
MODEL_REGISTRY = {
    "llama":   "meta-llama/Llama-2-7b-chat-hf",
    "minerva": "sapienzanlp/Minerva-7B-instruct-v1.0",
    "gemma":   "google/gemma-3-4b-it",
    # fallback per smoke test o se la taglia grande non regge la GPU
    "minerva-small": "sapienzanlp/Minerva-3B-base-v1.0",
    "gemma-tiny":    "google/gemma-3-1b-it",
}

# Gemma-3-4b-it è multimodale: contiene un vision tower con moduli che si
# chiamano q_proj/k_proj/v_proj/o_proj esattamente come quelli del language
# model. Selezionare i target LoRA per SUFFISSO ci attaccherebbe adapter anche
# sull'encoder visivo - parametri allenabili buttati e un confronto sporco con
# gli altri due modelli, che di vision tower non ne hanno.
VISION_MARKERS = ("vision_tower", "vision_model", "multi_modal_projector",
                  "visual", "image_encoder", "patch_embed")

LAYOUT_DIRS = {
    "T1": "layout1_traduzione_con_contesto",
    "T2": "layout2_completamento_turno",
    "T3": "layout3_replica_conversazionale",
    # Stadio A2: iniezione lessicale. Vive nella stessa cartella di split degli
    # altri layout e ha lo stesso formato ({prompt, target, layout, ...}), cosi'
    # load_split e ChatDataset non cambiano.
    "A2": "stadio_a2_lessico",
}


# --------------------------------------------------------------------------- #
# Configurazione per task
# --------------------------------------------------------------------------- #

@dataclass
class TaskConfig:
    """Tutto (e solo) ciò che distingue un task dagli altri due."""
    layout: str                      # "T1" | "T2" | "T3"
    nome: str                        # nome leggibile, finisce nel summary
    descrizione: str
    max_seq_len: int
    epochs: float
    lr: float = 1e-4
    metric: str = "chrf"             # metrica di selezione del checkpoint
    greater_is_better: bool = True
    gen_max_new_tokens: int = 64
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    patience: int = 4
    evals_per_epoch: int = 2         # da cui si deriva eval_steps
    # capacita' LoRA: su dataset da poche centinaia di esempi r=16 overfitta,
    # quindi ogni task puo' fissare la propria invece di ereditare un default
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    note: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #

def _library_versions():
    """Registrate nel summary: l'immagine Kaggle si aggiorna, e se il run 1 gira
    su una versione di transformers e il run 7 su un'altra il confronto
    cross-modello ha un confondente che dopo non puoi piu' rimuovere."""
    import importlib
    out = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate",
                 "sacrebleu", "trl"):
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception:
            out[name] = None
    return out


def slug(repo_id: str) -> str:
    return repo_id.split("/")[-1].lower()


def approx_params_b(repo_id: str):
    m = re.search(r"(\d+(?:\.\d+)?)b", slug(repo_id))
    return float(m.group(1)) if m else None


def resolve_model(name: str) -> str:
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    if "/" in name:                                   # repo_id esplicito
        return name
    sys.exit(f"ERRORE: modello {name!r} non riconosciuto. Alias disponibili: "
             f"{', '.join(MODEL_REGISTRY)} (oppure passa un repo_id org/nome).")


def load_hf_token(explicit=None):
    """Kaggle Secrets -> variabile d'ambiente -> None.
    Su Kaggle: Add-ons > Secrets, chiave HF_TOKEN, e spunta 'Attach to notebook'.
    Llama e Gemma sono repo gated: senza token il download fallisce (Minerva no)."""
    if explicit:
        return explicit
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def load_split(split_dir: str, layout: str, split_name: str, max_samples=None):
    path = os.path.join(split_dir, LAYOUT_DIRS[layout], f"{split_name}.json")
    if not os.path.exists(path):
        sys.exit(f"ERRORE: {path!r} non trovato.\n"
                 f"Su Kaggle lo split va caricato come Dataset e passato con --split-dir, "
                 f"es. --split-dir /kaggle/input/napoletano-split/split")
    rows = json.load(open(path, encoding="utf-8"))
    rows = [r for r in rows if r.get("layout") == layout]     # difesa: file misti
    return rows[:max_samples] if max_samples else rows


def find_lora_target_modules(model):
    """Ritorna i nomi COMPLETAMENTE QUALIFICATI dei moduli target, escludendo il
    vision tower.

    Selezione per nome e non per tipo: Linear4bit di bitsandbytes non è
    nn.Linear, ma la convenzione q/k/v/o/gate/up/down_proj è la stessa su Llama,
    Mistral (Minerva) e Gemma.

    Nomi completi e non suffissi perché su Gemma-3-4b-it il vision tower ha
    moduli omonimi: passando `["q_proj", ...]` PEFT li matcherebbe per endswith
    e attaccherebbe adapter anche all'encoder visivo. PEFT accetta nomi esatti
    (`key in target_modules`), quindi la lista completa è precisa.
    """
    names, skipped = [], 0
    for n, _ in model.named_modules():
        if n.rsplit(".", 1)[-1] not in LORA_TARGET_CANDIDATES:
            continue
        if any(m in n for m in VISION_MARKERS):
            skipped += 1
            continue
        names.append(n)
    if not names:
        sys.exit("ERRORE: nessun modulo target LoRA trovato (q/k/v/o/gate/up/down_proj). "
                 "Architettura non riconosciuta: adatta LORA_TARGET_CANDIDATES.")
    leaves = sorted({n.rsplit(".", 1)[-1] for n in names})
    print(f"Target LoRA: {len(names)} moduli, tipi {leaves}")
    if skipped:
        print(f"  esclusi {skipped} moduli omonimi nel vision tower "
              f"(modello multimodale: gli adapter vanno solo sul language model)")
    return names


def render_prompt(tokenizer, prompt, target=None):
    """Wrappa prompt (e opzionalmente target) nel chat template del tokenizer.

    Nessun ruolo 'system': l'istruzione è già dentro il testo del prompt scritto
    da split_dataset.py, identico byte per byte tra training e inferenza. Questo
    rende il formato compatibile anche con Gemma, che il ruolo system non lo
    ammette.

    Modelli BASE (senza chat template, es. Minerva-3B-base): fallback testuale
    semplice - stesso CONTENUTO, senza tag di ruolo mai visti in pretraining.

    Il template di Gemma-3 in alcune versioni itera su `content` come lista di
    blocchi tipizzati e va in errore su una stringa: da qui il secondo tentativo
    con il contenuto incapsulato.
    """
    if not getattr(tokenizer, "chat_template", None):
        if target is None:
            return prompt + "\n"
        return prompt + "\n" + target + (tokenizer.eos_token or "")

    def apply(user_content, target_content):
        msg = [{"role": "user", "content": user_content}]
        if target_content is None:
            return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        return tokenizer.apply_chat_template(
            msg + [{"role": "assistant", "content": target_content}],
            tokenize=False, add_generation_prompt=False)

    try:
        return apply(prompt, target)
    except Exception:
        wrap = lambda s: [{"type": "text", "text": s}]
        return apply(wrap(prompt), None if target is None else wrap(target))


# --------------------------------------------------------------------------- #
# Generazione reale durante il training
# --------------------------------------------------------------------------- #

# Token che chiudono il turno assistant nei chat template dei tre modelli. Sono
# gli stessi di evaluate_task.py: se qui mancassero, la generazione di controllo
# durante il training arriverebbe sempre a max_new_tokens e sembrerebbe che il
# modello non impari a fermarsi, mentre il problema sarebbe solo nel decoding.
CANDIDATI_EOT = ("<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|end|>",
                 "<|endoftext|>", "</s>", "<end_of_text>")


def _solo_spazi(tok, i):
    """True se l'id decodifica in soli spazi (o in niente).

    Serve a non usare come fine turno un token che non chiude nulla. Su Gemma-3
    il chat template scrive '<end_of_turn>' seguito da un a capo, quindi
    l'ULTIMO token del target renderizzato e' il 107, cioe' '\\n' (verificato
    in tokenizer.json: 106 = '<end_of_turn>', 107 = '\\n'). Passarlo a
    generate() come eos significa fermare la generazione al primo a capo: se il
    modello lo emette subito, l'uscita e' la stringa vuota.
    """
    try:
        return not tok.decode([int(i)], skip_special_tokens=False).strip()
    except Exception:
        return False


def _chiusura_target(tok, ids_testo):
    """Ultimo token NON di spaziatura della sequenza renderizzata.

    E' il token che chiude davvero il turno (su Gemma-3 il 106,
    '<end_of_turn>'), non l'a capo che il template gli mette dopo.
    """
    for i in reversed(list(ids_testo)):
        if not _solo_spazi(tok, i):
            return int(i)
    return None


def id_fine_turno(tok, rows=(), verbose=True):
    """Insieme degli id che chiudono il turno assistant, da passare al decoding.

    La fonte di verita' non e' la generation_config del modello ma l'ultimo
    token del testo renderizzato in training: e' esattamente cio' che il modello
    ha imparato a produrre per chiudere. Stessa logica di evaluate_task.py, cosi'
    la generazione di monitoraggio e quella di valutazione si fermano allo stesso
    punto e i due chrF sono confrontabili.
    """
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    for s in CANDIDATI_EOT:
        i = tok.convert_tokens_to_ids(s)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.add(int(i))
    coda = []
    for r in list(rows)[:8]:
        t = tok(render_prompt(tok, r["prompt"], r["target"]),
                add_special_tokens=False)["input_ids"]
        if t:
            ultimo = _chiusura_target(tok, t)
            if ultimo is not None:
                coda.append(ultimo)
    if coda:
        ids.add(Counter(coda).most_common(1)[0][0])
    # BUG CORRETTO. Nessun token di sola spaziatura fra gli eos: fermarsi su un
    # a capo prima di aver prodotto contenuto restituisce la stringa vuota, e
    # una stringa vuota vale 0.00 in tutte le metriche senza sembrare un errore.
    tenuti = set(i for i in ids if not _solo_spazi(tok, i))
    ids = tenuti or ids
    out = sorted(ids)
    if verbose:
        print(f"  fine turno per la generazione di monitoraggio: {out} "
              f"({[tok.convert_ids_to_tokens([i])[0] for i in out]})")
    return out


# Stato del motore di generazione: si stampa una volta sola per run, altrimenti
# ogni eval ripete lo stesso traceback.
_GEN_FALLBACK = {"annunciato": False, "forza_manuale": False}


def _decode_manuale(model, tokenizer, enc, max_new_tokens, eos_ids=(),
                    repetition_penalty=1.0, no_repeat_ngram_size=0,
                    usa_cache=True):
    """Loop di decoding greedy scritto a mano, batch 1, senza model.generate().

    PERCHE' ESISTE. Su google/gemma-3-4b-it (Gemma3ForConditionalGeneration)
    model.generate() e' rotto nella transformers dell'immagine Kaggle: _sample
    riceve next_tokens con una dimensione di troppo e crolla con
    "Tensors must have same number of dimensions: got 2 and 3". Il sintomo, nei
    log dello stadio A2, era

        [generazione reale saltata: RuntimeError: Tensors must have same ...]
        generato ( 3 parole): '[generazione fallita: RuntimeError]'

    cioe' l'UNICA metrica che vede il fallimento sull'EOS (eval_gen_chrf) non
    veniva mai calcolata, e la generazione di controllo stampava un segnaposto.
    evaluate_task.py aggirava gia' il bug con lo stesso loop; qui mancava, quindi
    il training girava cieco. Affettando esplicitamente logits[:, -1, :] la forma
    e' sempre [1, vocab] e il problema non si pone.

    Con cache KV (usa_cache=True) il costo e' un forward sul prompt piu' uno per
    token; senza cache si ricalcola tutto ad ogni passo, quindi e' solo il
    ripiego se la cache non e' supportata.
    """
    import torch
    ids = enc["input_ids"]
    attn = enc.get("attention_mask")
    if attn is None:
        attn = torch.ones_like(ids)
    eos_set = {int(e) for e in (eos_ids or ())}
    prompt_len = ids.shape[1]
    generati = []
    past, cur = None, ids
    for _ in range(max_new_tokens):
        kw = dict(input_ids=cur, attention_mask=attn, use_cache=usa_cache)
        if usa_cache and past is not None:
            kw["past_key_values"] = past
        out = model(**kw)
        if usa_cache:
            past = getattr(out, "past_key_values", None)
            if past is None:                       # il modello ignora la cache
                usa_cache = False
        logits = out.logits[:, -1, :].float()      # [1, vocab] sempre
        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "logit non finiti (NaN/inf) durante la generazione: il forward "
                "e' in overflow fp16, non e' un problema di decoding. "
                "Vedi stabilizza_fp16() in questo file.")
        seq = ids[0].tolist()
        if repetition_penalty and repetition_penalty != 1.0:
            for t in set(seq):
                v = logits[0, t]
                logits[0, t] = v / repetition_penalty if v > 0 else v * repetition_penalty
        if no_repeat_ngram_size and len(seq) >= no_repeat_ngram_size:
            n = no_repeat_ngram_size
            prefisso = tuple(seq[-(n - 1):]) if n > 1 else ()
            for i in range(len(seq) - n + 1):
                if tuple(seq[i:i + n - 1]) == prefisso:
                    logits[0, seq[i + n - 1]] = float("-inf")
        # Niente EOS finche' l'uscita sarebbe vuota. Fra gli id di fine turno
        # c'e' il 107, che e' un semplice '\n' e non un token speciale: se il
        # modello lo emette al primo passo, senza questa guardia il loop esce
        # subito e restituisce stringa vuota. In valutazione e' costato un
        # chrF++ 0.00 su tutte e 267 le uscite di test, senza nessun errore.
        if tokenizer is not None and not tokenizer.decode(
                generati, skip_special_tokens=True).strip():
            for e in eos_set:
                logits[0, e] = float("-inf")
        nxt = logits.argmax(dim=-1, keepdim=True)  # [1, 1]
        ids = torch.cat([ids, nxt], dim=-1)
        generati.append(int(nxt.item()))
        attn = torch.cat([attn, torch.ones_like(nxt)], dim=-1)
        cur = nxt if usa_cache else ids
        if int(nxt.item()) in eos_set:
            break
    return ids[0][prompt_len:]


def genera_una(model, tokenizer, testo, max_new_tokens, eos_ids=(),
               repetition_penalty=1.0, no_repeat_ngram_size=0):
    """Genera la continuazione di UN prompt gia' renderizzato. Ritorna testo.

    Un item alla volta, nessun padding: la generazione batched con left-padding
    su Gemma-3 crasha in transformers (_sample: shape [prompt_len] vs [batch]).
    Con i 12-32 item della metrica di generazione il costo e' accettabile.

    Su Gemma-3 si usa direttamente il loop manuale; sugli altri modelli si prova
    model.generate() e si ripiega sul loop se solleva un'eccezione, stampando il
    traceback UNA volta. Una generazione che fallisce non deve piu' sparire
    dietro un segnaposto: o si genera, o si vede perche' no.
    """
    import torch
    enc = tokenizer(testo, return_tensors="pt",
                    add_special_tokens=False).to(model.device)
    mt = (getattr(getattr(model, "config", None), "model_type", "") or "").lower()
    manuale = _GEN_FALLBACK["forza_manuale"] or "gemma3" in mt
    if not manuale:
        gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                      pad_token_id=tokenizer.pad_token_id)
        if eos_ids:
            gen_kw["eos_token_id"] = list(eos_ids)
        if repetition_penalty and repetition_penalty != 1.0:
            gen_kw["repetition_penalty"] = repetition_penalty
        if no_repeat_ngram_size:
            gen_kw["no_repeat_ngram_size"] = no_repeat_ngram_size
        try:
            with torch.no_grad():
                g = model.generate(**enc, **gen_kw)
            return tokenizer.decode(g[0][enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
        except Exception:
            import traceback
            _GEN_FALLBACK["forza_manuale"] = True
            if not _GEN_FALLBACK["annunciato"]:
                _GEN_FALLBACK["annunciato"] = True
                print("  [model.generate() ha fallito: passo al loop di decoding "
                      "manuale per tutto il run. Traceback una volta sola:]")
                traceback.print_exc()
    try:
        with torch.no_grad():
            out_ids = _decode_manuale(model, tokenizer, enc, max_new_tokens,
                                      eos_ids, repetition_penalty,
                                      no_repeat_ngram_size, usa_cache=True)
    except Exception:                              # cache KV non supportata qui
        with torch.no_grad():
            out_ids = _decode_manuale(model, tokenizer, enc, max_new_tokens,
                                      eos_ids, repetition_penalty,
                                      no_repeat_ngram_size, usa_cache=False)
    return tokenizer.decode(out_ids, skip_special_tokens=True)


def _forza_token_type_ids(model):
    """Gemma-3 (Gemma3ForConditionalGeneration) pretende token_type_ids in input
    durante il training: la maschera causale multimodale li usa per distinguere i
    token immagine da quelli testo. Qui addestriamo SOLO su testo, quindi sono
    tutti 0 (nessun token immagine). Li iniettiamo di default nel forward, cosi'
    nessuno degli script a valle (A, A2, SFT) deve saperlo. Patch applicata solo a
    Gemma-3: gli altri modelli (Llama, Minerva) non accettano token_type_ids.
    """
    mt = getattr(getattr(model, "config", None), "model_type", "") or ""
    if "gemma3" not in mt.lower() and "gemma3" not in type(model).__name__.lower():
        return model
    import torch
    _orig_forward = model.forward
    def forward(*args, **kw):
        ii = kw.get("input_ids")
        # solo in TRAINING Gemma-3 pretende token_type_ids; in eval/generate no,
        # e iniettarli li' rompe la generazione con la cache (shape mismatch).
        if model.training and kw.get("token_type_ids") is None and ii is not None:
            kw["token_type_ids"] = torch.zeros_like(ii)
        return _orig_forward(*args, **kw)
    model.forward = forward
    return model


def scegli_dtype(repo_id, scelta="auto"):
    """(dtype, bf16_ok) per QUESTA GPU e QUESTO modello. Regola del notebook T3.

    Non basta guardare la compute capability. Su una GPU senza bf16 nativo
    (T4, cc 7.5) il ripiego naturale e' fp16, ma Gemma in fp16 va in overflow:
    le attivazioni escono dal range del formato, i logit diventano NaN e la
    generazione restituisce stringhe vuote (in campionamento crolla con
    'probability tensor contains inf/nan'; in greedy no, perche' argmax su NaN
    restituisce l'id 0, che su Gemma e' '<pad>' - stesso guasto, silenzioso).

    La loss in teacher forcing puo' restare sana mentre la generazione e' gia'
    morta, quindi il sintomo NON e' la loss. In questo progetto il training
    sopravviveva in fp16 solo perche' prepare_model_for_kbit_training riporta a
    fp32 i parametri non quantizzati; gli script di inferenza, che caricano il
    modello nudo, no. Da qui le 267 uscite vuote su 267 in valutazione.

    Per Gemma senza bf16 si usa fp32: circa 2x piu' lento, ma finito. Il
    notebook T3, stesso modello e stessa T4, gira cosi' e riporta "vuote": 0 in
    tutti gli arm di valutazione.
    """
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
    if "gemma" in str(repo_id).lower():
        return torch.float32, False
    return torch.float16, False


def controlla_finito(model, tokenizer, testo, dove="", rendi=True):
    """Un forward solo, per fallire in cinque secondi invece che dopo un'ora.

    Se i logit non sono finiti nessun numero a valle e' interpretabile: loss che
    non scende, grad_norm=nan, chrF 0 e uscite vuote sono conseguenze, non
    misure. Va chiamato dopo aver costruito il PeftModel e prima del Trainer.
    """
    import torch
    t = render_prompt(tokenizer, testo) if rendi else testo
    enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=512,
                    add_special_tokens=False).to(model.device)
    era = model.training
    model.eval()
    with torch.no_grad():
        logits = model(**enc).logits
    model.train(era)
    ok = bool(torch.isfinite(logits).all())
    print(f"  logit finiti: {ok} (dtype {logits.dtype})" + (f" [{dove}]" if dove else ""))
    if not ok:
        sys.exit("LOGIT NON FINITI: overflow numerico con questa precisione. "
                 "Rilancia con --dtype fp32, oppure usa una GPU con bf16 nativo "
                 "(L4, A100), che e' piu' veloce e piu' stabile.")


def stabilizza_fp16(model):
    """Porta a fp32 i parametri NON quantizzati quando si lavora in float16.

    PERCHE' ESISTE (uscite di valutazione tutte vuote, chrF++ 0.00).
    Su GPU pre-Ampere (Kaggle T4, cc 7.5) il compute dtype e' fp16. Gemma-3 ha
    attivazioni molto grandi: nel flusso residuale in fp16 si supera 65504, l'inf
    entra nelle RMSNorm e i logit finali diventano NaN. `argmax` su un tensore di
    NaN restituisce l'indice 0, che su Gemma e' `<pad>`: il decoder emette 64 pad
    di fila, `decode(skip_special_tokens=True)` restituisce la stringa vuota e
    OGNI metrica vale 0.00 senza che nessuno sollevi un'eccezione.

    Le due prove che era questo e non il decoding:
      - `logprob_riferimento_norm: NaN` in tutti e quattro i .metrics.json, e la
        contrastiva non genera niente, fa solo forward: il forward era gia' rotto;
      - la generazione di monitoraggio DURANTE il training stampava napoletano
        corretto sugli stessi prompt e con lo stesso adapter, perche' li' il
        modello era passato per prepare_model_for_kbit_training, che fa
        esattamente questo upcast. evaluate_task.py carica il modello "nudo".

    Resta come seconda linea di difesa per i modelli che in fp16 ci girano
    davvero (Llama, Minerva): su Gemma la prima linea e' scegli_dtype(), che in
    assenza di bf16 sceglie direttamente fp32.

    I pesi 4-bit non vengono toccati (Params4bit e' uint8, non fp16). Con gli
    embedding in fp32 le hidden state sono fp32: bitsandbytes casta l'ingresso a
    compute_dtype per il matmul e riporta l'uscita al dtype d'ingresso, quindi il
    residuo si accumula in fp32. Costo ~1,3 GB su un 4B; nessun cambiamento nei
    pesi, solo nella precisione con cui vengono sommati.
    """
    import torch
    n = 0
    for _, p in model.named_parameters():
        if p.dtype == torch.float16:
            p.data = p.data.to(torch.float32)
            n += 1
    if n:
        print(f"  stabilizzazione numerica: {n} tensori non quantizzati da fp16 "
              f"a fp32 (residuo in fp32, matmul 4-bit in fp16)")
    return model


def load_backbone(repo_id, dtype, **kw):
    """AutoModelForCausalLM, con fallback per i checkpoint multimodali.

    google/gemma-3-4b-it è registrato come Gemma3ForConditionalGeneration:
    AutoModelForCausalLM può rifiutarlo. AutoModelForImageTextToText lo carica
    (vision tower incluso, che noi congeliamo escludendolo dai target LoRA).
    Su Gemma-3 il forward viene patchato per fornire token_type_ids=0 (testo).
    """
    import torch
    from transformers import AutoModelForCausalLM

    def _prepara(m):
        # solo sui caricamenti quantizzati: in fp16 pieno l'upcast raddoppierebbe
        # la memoria dei pesi e non ci starebbe in VRAM.
        if dtype == torch.float16 and kw.get("quantization_config") is not None:
            m = stabilizza_fp16(m)
        return _forza_token_type_ids(m)

    try:
        model = from_pretrained_compat(AutoModelForCausalLM, repo_id, dtype, **kw)
        return _prepara(model), "CausalLM"
    except Exception as e_causal:
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            raise e_causal
        print(f"  AutoModelForCausalLM non applicabile ({type(e_causal).__name__}): "
              f"ripiego su AutoModelForImageTextToText (checkpoint multimodale).")
        model = from_pretrained_compat(AutoModelForImageTextToText, repo_id, dtype, **kw)
        return _prepara(model), "ImageTextToText"


class ChatDataset:
    """Un esempio = turno user (prompt già pronto) + risposta assistant.
    Loss mascherata a -100 sui token del prompt: si allena solo sull'assistant."""

    def __init__(self, rows, tokenizer, max_len):
        self.rows, self.tok, self.max_len = rows, tokenizer, max_len
        self.n_troncati = 0
        self.n_prefix_mismatch = 0
        for k, r in enumerate(rows):                  # diagnostica, non silenziosa
            p = self.tok(render_prompt(self.tok, r["prompt"]),
                         add_special_tokens=False)["input_ids"]
            f = self.tok(render_prompt(self.tok, r["prompt"], r["target"]),
                         add_special_tokens=False)["input_ids"]
            if len(f) > max_len:
                self.n_troncati += 1
            # Il masking presuppone che il prompt renderizzato sia un PREFISSO
            # esatto del testo completo. Se il chat template o la tokenizzazione
            # al confine prompt/target rompono questa proprietà, la loss finisce
            # calcolata su posizioni sbagliate - in silenzio. Meglio saperlo.
            if k < 200 and f[:len(p)] != p:
                self.n_prefix_mismatch += 1

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt_ids = self.tok(render_prompt(self.tok, r["prompt"]),
                              add_special_tokens=False)["input_ids"]
        full_ids = self.tok(render_prompt(self.tok, r["prompt"], r["target"]),
                            add_special_tokens=False)["input_ids"][:self.max_len]
        labels = list(full_ids)
        for j in range(min(len(prompt_ids), len(labels))):
            labels[j] = -100
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def make_collate_fn(pad_id):
    import torch

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids, attn, labels = [], [], []
        for b in batch:                               # padding a DESTRA in training
            k = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_id] * k)
            attn.append(b["attention_mask"] + [0] * k)
            labels.append(b["labels"] + [-100] * k)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}
    return collate


def from_pretrained_compat(cls, repo_id, dtype, **kw):
    """transformers 5.x usa `dtype=`, le 4.x `torch_dtype=`. Prova il nuovo,
    ripiega sul vecchio: così lo stesso script gira sia sul venv locale sia
    sull'immagine Kaggle, che possono avere versioni diverse."""
    try:
        return cls.from_pretrained(repo_id, dtype=dtype, **kw)
    except TypeError:
        return cls.from_pretrained(repo_id, torch_dtype=dtype, **kw)


# --------------------------------------------------------------------------- #
# Metriche
# --------------------------------------------------------------------------- #

def build_metrics(tokenizer, task: TaskConfig):
    """compute_metrics + preprocess_logits_for_metrics.

    ATTENZIONE metodologica: tutto qui è calcolato in TEACHER FORCING (argmax
    dei logit posizione per posizione), non con generate() autoregressivo. È un
    proxy economico per il monitoraggio e per l'early stopping, NON il numero da
    riportare nel paper: un modello può sembrare buono qui e degenerare in
    ripetizioni a generazione libera. I numeri finali vanno da uno script di
    valutazione separato su test.json con generate() vero.
    """
    import sacrebleu
    try:
        from rouge_score import rouge_scorer
        rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    except ImportError:
        rouge = None
        print("  ! rouge_score non installato: rouge1/rougeL non calcolati "
              "(pip install rouge_score).")

    def preprocess_logits_for_metrics(logits, labels):
        # riduzione immediata ad argmax: tenere (batch x seq x vocab) per tutto il
        # dev satura la memoria (Gemma ha un vocabolario da 256k token)
        return logits.argmax(dim=-1)

    def word_prf1(hyp, ref):
        """Overlap bag-of-words (come l'F1 di SQuAD). Proxy LESSICALE su un task
        generativo, NON F1 di classificazione: etichettarlo come tale."""
        h, r = hyp.split(), ref.split()
        if not h or not r:
            return 0.0, 0.0, 0.0
        common = sum((Counter(h) & Counter(r)).values())
        if common == 0:
            return 0.0, 0.0, 0.0
        p, rc = common / len(h), common / len(r)
        return p, rc, 2 * p * rc / (p + rc)

    def score_set(hyps, refs):
        out = {"chrf": sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score,
               "bleu": sacrebleu.corpus_bleu(hyps, [refs]).score}
        prf1 = [word_prf1(h, r) for h, r in zip(hyps, refs)]
        out["word_precision"] = sum(p for p, _, _ in prf1) / len(prf1)
        out["word_recall"] = sum(r for _, r, _ in prf1) / len(prf1)
        out["word_f1"] = sum(f for _, _, f in prf1) / len(prf1)
        if rouge is not None:
            s = [rouge.score(r, h) for h, r in zip(hyps, refs)]   # (target, prediction)
            out["rouge1"] = sum(x["rouge1"].fmeasure for x in s) / len(s)
            out["rougeL"] = sum(x["rougeL"].fmeasure for x in s) / len(s)
        return out

    def compute_metrics(eval_pred):
        pred_ids, label_ids = eval_pred
        pred_ids = pred_ids[:, :-1]          # shift causal LM: pred[i] prevede label[i+1]
        label_ids = label_ids[:, 1:]
        hyps, refs, correct, total = [], [], 0, 0
        for p_row, l_row in zip(pred_ids, label_ids):
            mask = l_row != -100
            if mask.sum() == 0:
                continue
            hyps.append(tokenizer.decode(p_row[mask], skip_special_tokens=True))
            refs.append(tokenizer.decode(l_row[mask], skip_special_tokens=True))
            correct += int((p_row[mask] == l_row[mask]).sum())
            total += int(mask.sum())
        if not hyps:
            return {task.metric: 0.0}
        res = score_set(hyps, refs)
        res["token_accuracy"] = correct / total if total else 0.0
        return res

    return compute_metrics, preprocess_logits_for_metrics


def build_generation_metric_callback(TrainerCallback, torch, tokenizer, rows, task: TaskConfig,
                                     n=32, batch=8, eos_ids=()):
    """Aggiunge eval_gen_chrf e eval_gen_len_ratio, calcolate su GENERAZIONE REALE.

    Perche' serve: eval_chrf di compute_metrics e' in teacher forcing, cioe' si
    decodifica l'argmax su tante posizioni quante ne ha il target. L'ipotesi ha
    percio' SEMPRE la stessa lunghezza del riferimento, e la metrica e'
    strutturalmente cieca al fallimento sull'EOS. Misurato su Minerva T3:
    eval_chrf in teacher forcing 22.1, chrF su generazione reale 10.5, con
    output 2.7 volte piu' lunghi dell'umano. Nessuna metrica in teacher forcing
    (ne' chrF ne' loss) puo' selezionare un checkpoint su quella base.

    Decoding greedy anche su T3: qui serve un segnale STABILE fra una eval e
    l'altra per la selezione: il nucleus sampling introdurrebbe varianza fra
    valutazioni. Il nucleus va usato nella valutazione finale.

    Mutare il dict `metrics` dentro on_evaluate propaga alla selezione del
    checkpoint, quindi metric_for_best_model="gen_chrf" funziona.
    """
    import sacrebleu

    sub = rows[:n]

    class GenerationMetricCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
            if model is None or metrics is None or not sub:
                return
            was_training, was_cache = model.training, model.config.use_cache
            prev_side = tokenizer.padding_side
            model.eval()
            # Con gradient_checkpointing attivo transformers forza use_cache=False,
            # quindi generate() gira SENZA cache KV e ricalcola il forward su tutta
            # la sequenza per ogni token nuovo: con 32 item, 64 token e prompt da
            # ~200 sono ~64x il lavoro previsto, e il run sembra bloccato. Va
            # disattivato per la durata della generazione e riattivato dopo.
            ckpt_era_attivo = bool(getattr(model, "is_gradient_checkpointing", False))
            if ckpt_era_attivo:
                model.gradient_checkpointing_disable()
            model.config.use_cache = True
            tokenizer.padding_side = "left"          # obbligatorio in generazione
            hyps = []
            try:
                # Generazione UN ITEM ALLA VOLTA (batch=1, nessun padding), via
                # genera_una: su Gemma-3 usa il loop di decoding manuale, sugli
                # altri model.generate() con ripiego automatico. Prima qui c'era
                # una chiamata diretta a model.generate(), che su Gemma-3 solleva
                # RuntimeError e faceva saltare eval_gen_chrf ad ogni eval.
                for r in sub:
                    txt = genera_una(model, tokenizer,
                                     render_prompt(tokenizer, r["prompt"]),
                                     task.gen_max_new_tokens, eos_ids)
                    hyps.append(txt.strip().split("\n")[0].strip())
                refs = [r["target"] for r in sub]
                metrics["eval_gen_chrf"] = sacrebleu.corpus_chrf(
                    hyps, [refs], word_order=2).score
                lh = sum(len(x.split()) for x in hyps) / len(hyps)
                lr = sum(len(x.split()) for x in refs) / len(refs)
                metrics["eval_gen_len_ratio"] = lh / lr if lr else 0.0
                print(f"  [generazione reale su {len(sub)} item] chrF "
                      f"{metrics['eval_gen_chrf']:.2f} | lunghezza "
                      f"{metrics['eval_gen_len_ratio']:.2f}x l'umano")
            except Exception as e:
                import traceback
                print(f"  [generazione reale saltata: {type(e).__name__}: {e}]")
                print("   eval_gen_chrf non calcolato; la selezione del checkpoint "
                      "usa eval_chrf (teacher forcing) e non ne dipende.")
                print("   ATTENZIONE: senza eval_gen_chrf nessuna metrica vede il "
                      "fallimento sull'EOS. Traceback:")
                traceback.print_exc()
            finally:
                tokenizer.padding_side = prev_side
                model.config.use_cache = was_cache
                if ckpt_era_attivo:
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False})
                if was_training:
                    model.train()

    return GenerationMetricCallback()


def build_sample_callback(TrainerCallback, torch, tokenizer, samples, task: TaskConfig,
                          eos_ids=()):
    """Ad ogni eval genera per davvero (no teacher forcing) su 3 esempi FISSI di
    dev. È l'unico modo per accorgersi in tempo reale che il chrF proxy sale
    mentre il testo generato degenera in loop.

    Tre correzioni rispetto alla versione precedente:
      1. la generazione passa da genera_una (loop manuale su Gemma-3), quindi
         non stampa piu' '[generazione fallita: RuntimeError]';
      2. si passano gli id di fine turno: senza, il decoding arriva sempre a
         max_new_tokens e il messaggio "non sta imparando a emettere EOS" e'
         un artefatto del decoding, non una misura;
      3. il gradient checkpointing viene disattivato per la durata della
         generazione. Con il checkpointing attivo transformers forza
         use_cache=False, quindi ogni token nuovo ricalcolava il forward
         sull'intera sequenza: era il motivo per cui questa cella sembrava
         bloccata dopo ogni eval. Il callback della metrica lo faceva gia',
         questo no.
    """

    class SampleGenerationCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, model=None, **kwargs):
            if model is None or not samples:
                return
            was_training, was_cache = model.training, model.config.use_cache
            model.eval()
            ckpt_era_attivo = bool(getattr(model, "is_gradient_checkpointing", False))
            if ckpt_era_attivo:
                model.gradient_checkpointing_disable()
            model.config.use_cache = True
            print(f"\n--- Generazione reale ({task.layout}) allo step {state.global_step} ---")
            try:
                for prompt, target in samples:
                    try:
                        gen = genera_una(model, tokenizer,
                                         render_prompt(tokenizer, prompt),
                                         task.gen_max_new_tokens, eos_ids,
                                         task.repetition_penalty,
                                         task.no_repeat_ngram_size)
                    except Exception as e:
                        import traceback
                        gen = f"[generazione fallita: {type(e).__name__}: {e}]"
                        traceback.print_exc()
                    # prompt COMPLETO: troncarlo in stampa fa sembrare che il
                    # modello non veda il contesto, quando invece lo riceve tutto
                    print("  --- prompt (completo) ---")
                    for riga in prompt.split("\n"):
                        print(f"  | {riga}")
                    nt, ng = len(target.split()), len(gen.split())
                    print(f"  target   ({nt:2d} parole): {target!r}")
                    print(f"  generato ({ng:2d} parole): {gen!r}")
                    if ng > 3 * max(nt, 1):
                        print(f"  ! generato {ng/max(nt,1):.1f}x piu' lungo del target: "
                              f"il modello non sta imparando a emettere EOS")
                    print()
                print("---\n")
            finally:
                model.config.use_cache = was_cache
                if ckpt_era_attivo:
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False})
                if was_training:
                    model.train()

    return SampleGenerationCallback()


# --------------------------------------------------------------------------- #
# CLI condivisa
# --------------------------------------------------------------------------- #

def build_parser(task: TaskConfig) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=f"Fine-tuning QLoRA — {task.layout} ({task.nome}). {task.descrizione}",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help=f"alias ({'/'.join(list(MODEL_REGISTRY)[:3])}...) o repo_id org/nome")
    ap.add_argument("--split-dir", default="/kaggle/working/split",
                    help="cartella che contiene layout1_*/layout2_*/layout3_* (read-only su Kaggle)")
    ap.add_argument("--out-dir", default="/kaggle/working/runs")
    ap.add_argument("--resume-dir", default=None,
                    help="cartella (es. output di una sessione Kaggle precedente montata in "
                         "/kaggle/input) da cui copiare i checkpoint per riprendere il training")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--precision", choices=["qlora4bit", "bf16", "fp16"], default="qlora4bit")
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                    help="precisione di calcolo. auto = bf16 se la GPU ce l'ha, "
                         "fp32 per Gemma senza bf16 (in fp16 il GradScaler scarta "
                         "gli step: grad_norm=nan e learning_rate=0), fp16 per gli altri")
    ap.add_argument("--seed", type=int, default=42)
    # iperparametri: default dal TaskConfig, sovrascrivibili
    ap.add_argument("--max-seq-len", type=int, default=task.max_seq_len)
    ap.add_argument("--epochs", type=float, default=task.epochs)
    ap.add_argument("--lr", type=float, default=task.lr)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--eval-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="batch efficace = batch-size * grad-accum. Tenerlo COSTANTE (16) su "
                         "tutti e 9 i run: è una variabile di confronto, non un parametro libero")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--neftune-alpha", type=float, default=5.0)
    ap.add_argument("--lora-r", type=int, default=task.lora_r)
    ap.add_argument("--lora-alpha", type=int, default=task.lora_alpha)
    ap.add_argument("--lora-dropout", type=float, default=task.lora_dropout)
    ap.add_argument("--train-embeddings", action="store_true",
                    help="allena embed_tokens/lm_head per intero (non a basso rango). "
                         "Costoso in VRAM, soprattutto su Gemma (vocabolario 256k)")
    ap.add_argument("--evals-per-epoch", type=int, default=task.evals_per_epoch,
                    help="da cui si DERIVA eval_steps. Con dataset di poche centinaia di "
                         "esempi un eval_steps fisso rischia di non far partire nessuna eval")
    ap.add_argument("--eval-subset", type=int, default=200,
                    help="esempi di dev per l'eval periodica (0 = tutto il dev)")
    ap.add_argument("--patience", type=int, default=task.patience)
    ap.add_argument("--metric", default=task.metric,
                    help="metrica di selezione del checkpoint (chrf, bleu, loss, word_f1...)")
    ap.add_argument("--no-sample-generation", action="store_true")
    ap.add_argument("--gen-metric-n", type=int, default=12,
                    help="item di dev su cui calcolare chrF e rapporto di lunghezza con "
                         "GENERAZIONE REALE ad ogni eval (0 disattiva). Sono le uniche "
                         "metriche che vedono il fallimento sull'EOS: quelle in teacher "
                         "forcing hanno per costruzione la stessa lunghezza del riferimento")
    ap.add_argument("--no-gradient-checkpointing", action="store_true")
    ap.add_argument("--max-train-samples", type=int, default=None, help="smoke test")
    ap.add_argument("--max-dev-samples", type=int, default=None, help="smoke test")
    ap.add_argument("--init-adapter", default=None,
                    help="adapter di partenza, tipicamente l'output dello stadio A "
                         "(pretrain_dialect.py). Lo stesso adapter viene CONTINUATO, "
                         "non impilato: r/alpha/dropout li eredita da quello e i "
                         "corrispondenti --lora-* vengono ignorati")
    ap.add_argument("--ctx-metric-n", type=int, default=64,
                    help="item di dev su cui misurare la sensibilita' al contesto "
                         "(0 disattiva). Aggiunge eval_ctx_delta e eval_ctx_acc, "
                         "usabili come --metric ctx_acc. Costa 2 forward per item, "
                         "nessuna generazione")
    ap.add_argument("--lessico", default=None,
                    help="output di lessico.py. Se presente, la loss viene PESATA "
                         "sui token delle forme dialettali (vedi --peso-dial). "
                         "Senza questo flag il comportamento e' identico ai run "
                         "precedenti, quindi i due arm sono confrontabili")
    ap.add_argument("--peso-dial", type=float, default=3.0,
                    help="peso dei token dialettali contro 1.0 degli altri. La media "
                         "e' ponderata, non sommata: la scala della loss resta "
                         "confrontabile con i run non pesati e il lr non va ritoccato")
    ap.add_argument("--hf-token", default=None)
    return ap


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

def run(task: TaskConfig):
    args = build_parser(task).parse_args()

    import torch
    from transformers import (AutoTokenizer, BitsAndBytesConfig,
                              EarlyStoppingCallback, Trainer, TrainerCallback,
                              TrainingArguments, set_seed)
    from transformers.trainer_utils import get_last_checkpoint
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    set_seed(args.seed)
    repo_id = resolve_model(args.model)

    if not torch.cuda.is_available():
        sys.exit("ERRORE: nessuna GPU CUDA rilevata. Su Kaggle: Settings > Accelerator > "
                 "GPU P100 (preferibile) oppure T4 x2. Questo script non è pensato per la CPU.")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    # NON usare torch.cuda.is_bf16_supported(): nelle versioni recenti di PyTorch
    # ha including_emulation=True per default e risponde True anche sulla T4
    # (Turing, cc 7.5), che il bf16 in hardware non ce l'ha. Il risultato e' bf16
    # EMULATO, molto piu' lento dei tensor core fp16 nativi. Il bf16 vero parte da
    # Ampere, cc 8.0.
    _cc = torch.cuda.get_device_capability(0)
    bf16_ok = _cc[0] >= 8
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    compute_dtype, _ = scegli_dtype(repo_id, args.dtype)
    nome_dt = str(compute_dtype).replace("torch.", "")
    print(f"=== {task.layout} ({task.nome}) | {repo_id} ===")
    print(f"GPU: {gpu_name} (cc {_cc[0]}.{_cc[1]}), {vram_gb:.1f} GB VRAM | "
          f"bf16 nativo: {bf16_ok} -> precisione: {nome_dt}")
    if compute_dtype == torch.float32:
        print("  Gemma su GPU senza bf16: fp32, quindi niente autocast fp16 e niente "
              "GradScaler.\n"
              "  In fp16 il GradScaler scarta gli step con gradienti inf/nan e nei log "
              "compaiono\n"
              "  grad_norm=nan con learning_rate=0: sono step che NON hanno aggiornato "
              "l'adapter.\n"
              "  Costo: circa 2x di tempo per step.")
    elif not bf16_ok:
        print("  GPU pre-Ampere: fp16 (tensor core nativi). Se nei log compaiono "
              "grad_norm=nan e learning_rate=0 gli step vengono scartati: rilancia "
              "con --dtype fp32.")
    pb = approx_params_b(repo_id)
    if pb:
        peso = pb * (0.5 if args.precision == "qlora4bit" else 2)
        print(f"  ~{pb:.0f}B parametri -> ~{peso:.1f} GB di soli pesi in {args.precision} "
              f"(più adapter, ottimizzatore, attivazioni)")
        if peso > vram_gb * 0.6:
            print("  ATTENZIONE: frazione alta della VRAM. Rischio OOM.")

    token = load_hf_token(args.hf_token)
    if token is None:
        print("! Nessun HF_TOKEN: Llama e Gemma sono gated e il download fallirà. "
              "Su Kaggle: Add-ons > Secrets > HF_TOKEN.", file=sys.stderr)

    # --- tokenizer ---------------------------------------------------------
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, token=token, use_fast=True)
    except Exception as e:
        # 403 GatedRepoError: il repo esiste ma l'accesso non e' stato concesso a
        # QUESTO account. Il traceback di huggingface_hub e' lungo tre schermate
        # e nasconde la sola cosa da fare, quindi la si dice qui.
        if "gated" in str(e).lower() or "403" in str(e):
            sys.exit(
                f"ERRORE: {repo_id} e' un repo gated e questo account non ha "
                f"l'accesso.\n"
                f"  1. apri https://huggingface.co/{repo_id} e accetta la licenza "
                f"(per Minerva l'accesso va richiesto e approvato a mano);\n"
                f"  2. verifica che HF_TOKEN nei Kaggle Secrets sia dello STESSO "
                f"account che ha accettato;\n"
                f"  3. oppure usa un alias accessibile: "
                f"{', '.join(MODEL_REGISTRY)}.")
        raise
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if not getattr(tokenizer, "chat_template", None):
        print(f"  ! {repo_id} non ha chat template (modello BASE): fallback testuale.")

    # --- modello -----------------------------------------------------------
    print(f"Carico il modello ({args.precision})...")
    common_kw = dict(device_map={"": 0}, token=token, low_cpu_mem_usage=True,
                     attn_implementation="eager")   # obbligatorio per Gemma-2 (soft-capping)
    if args.precision == "qlora4bit":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=compute_dtype)
        try:
            model, arch_kind = load_backbone(repo_id, compute_dtype,
                                             quantization_config=quant, **common_kw)
        except Exception as e:
            sys.exit(f"ERRORE: caricamento 4-bit fallito: {e}\n"
                     f"Verifica che bitsandbytes sia compatibile con la CUDA dell'immagine "
                     f"Kaggle, oppure ripiega su --precision fp16 (per un 7B+ probabilmente "
                     f"non ci sta in {vram_gb:.0f} GB).")
    else:
        dt = torch.bfloat16 if args.precision == "bf16" else torch.float16
        model, arch_kind = load_backbone(repo_id, dt, **common_kw)

    use_ckpt = not args.no_gradient_checkpointing
    model.config.use_cache = False
    if args.precision == "qlora4bit":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=use_ckpt,
            gradient_checkpointing_kwargs={"use_reentrant": False} if use_ckpt else None)
    elif use_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    target_modules = find_lora_target_modules(model)
    if args.init_adapter:
        # Stadio B: si CONTINUA ad addestrare l'adapter dello stadio A (continued
        # pretraining sul napoletano), non se ne impila un secondo sopra.
        # is_trainable=True e' indispensabile: senza, l'adapter viene caricato
        # congelato e il training non aggiorna niente.
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        print(f"Adapter iniziale caricato da {args.init_adapter} (stadio A -> stadio B)")
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
            modules_to_save=["embed_tokens", "lm_head"] if args.train_embeddings else None))
    model.print_trainable_parameters()

    # --- dati (solo questo layout) -----------------------------------------
    train_rows = load_split(args.split_dir, task.layout, "train", args.max_train_samples)
    dev_full = load_split(args.split_dir, task.layout, "dev", args.max_dev_samples)
    if not train_rows:
        sys.exit(f"ERRORE: nessuna istanza {task.layout} nel train. Controlla --split-dir.")
    print(f"Dati {task.layout}: train {len(train_rows)} | dev {len(dev_full)}")
    controlla_finito(model, tokenizer, train_rows[0]["prompt"], task.layout)

    fixed_samples = []
    if not args.no_sample_generation:
        fixed_samples = [(r["prompt"], r["target"]) for r in dev_full[:3]]

    dev_rows = dev_full
    if args.eval_subset and len(dev_rows) > args.eval_subset:
        import random
        dev_rows = list(dev_rows)
        random.Random(args.seed).shuffle(dev_rows)
        dev_rows = dev_rows[:args.eval_subset]
        print(f"  dev sottocampionato a {len(dev_rows)} per l'eval periodica")

    # --- pesatura lessicale (opzionale) ------------------------------------
    # Senza --lessico si usano ChatDataset/Trainer come prima: e' l'arm di
    # confronto "senza pesatura", non un ripiego.
    lex = None
    if args.lessico:
        from pesi_lessicali import (ChatDatasetPesato, carica_lessico,
                                    make_collate_pesato)
        lex = carica_lessico(args.lessico)
        print(f"Pesatura lessicale attiva: {len(lex['dialettali'])} tipi dialettali, "
              f"peso {args.peso_dial}")

    def costruisci_ds(rows):
        if lex:
            return ChatDatasetPesato(rows, tokenizer, args.max_seq_len, lex,
                                     peso_dial=args.peso_dial)
        return ChatDataset(rows, tokenizer, args.max_seq_len)

    train_ds = costruisci_ds(train_rows)
    dev_ds = costruisci_ds(dev_rows)
    if lex:
        # Passata di controllo su un campione: se la quota di token pesati e'
        # vicina a zero il lessico non sta agganciando i target (tokenizer slow,
        # o forme normalizzate in modo diverso) e la pesatura e' inerte.
        for i in range(min(300, len(train_ds))):
            train_ds[i]
        print("  " + train_ds.riepilogo_pesi())
    if train_ds.n_prefix_mismatch or dev_ds.n_prefix_mismatch:
        print(f"  !! PREFIX MISMATCH: {train_ds.n_prefix_mismatch} train, "
              f"{dev_ds.n_prefix_mismatch} dev (sui primi 200 controllati). Il prompt "
              f"renderizzato non e' un prefisso esatto del testo completo: il masking "
              f"della loss e' disallineato. NON usare questo run, va risolto il "
              f"rendering del chat template per questo modello.")
    if train_ds.n_troncati or dev_ds.n_troncati:
        print(f"  ! troncati a --max-seq-len {args.max_seq_len}: "
              f"{train_ds.n_troncati} train, {dev_ds.n_troncati} dev. "
              f"Un target troncato è un target sbagliato: alza max-seq-len.")

    # --- cadenza di eval DERIVATA (il bug del monolite) --------------------
    eff_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, math.ceil(len(train_rows) / eff_batch))
    max_steps = max(1, math.ceil(args.epochs * steps_per_epoch))
    eval_steps = max(5, steps_per_epoch // max(1, args.evals_per_epoch))
    warmup_steps = max(1, round(args.warmup_ratio * max_steps))
    n_evals = max_steps // eval_steps
    print(f"Batch efficace {eff_batch} | {steps_per_epoch} step/epoca | {max_steps} step totali")
    print(f"eval ogni {eval_steps} step -> ~{n_evals} valutazioni | warmup {warmup_steps} step")
    if n_evals < args.patience + 1:
        print(f"  ! solo ~{n_evals} eval contro patience={args.patience}: l'early stopping non "
              f"potrà mai scattare. Alza --evals-per-epoch o --epochs.")

    metric_name = args.metric
    greater = task.greater_is_better if metric_name == task.metric else metric_name != "loss"
    compute_metrics, preprocess_logits = build_metrics(tokenizer, task)

    run_slug = f"{slug(repo_id)}__{task.layout}"
    out_dir = os.path.join(args.out_dir, run_slug)
    os.makedirs(out_dir, exist_ok=True)

    # --- resume ------------------------------------------------------------
    # /kaggle/working viene azzerato tra le sessioni: per riprendere davvero,
    # l'output della sessione precedente va montato come dataset di input e
    # passato con --resume-dir. Qui lo copiamo in working (Trainer deve scrivere).
    if args.resume_dir and not args.no_resume:
        src = os.path.join(args.resume_dir, run_slug)
        src = src if os.path.isdir(src) else args.resume_dir
        if get_last_checkpoint(src):
            for name in os.listdir(src):
                if name.startswith("checkpoint-"):
                    dst = os.path.join(out_dir, name)
                    if not os.path.exists(dst):
                        shutil.copytree(os.path.join(src, name), dst)
            print(f"Checkpoint copiati da {src} in {out_dir}")
    resume_from = None if args.no_resume else get_last_checkpoint(out_dir)
    if resume_from:
        print(f"Riprendo da {resume_from}")

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=0.3,
        neftune_noise_alpha=args.neftune_alpha if args.neftune_alpha > 0 else None,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=greater,
        logging_steps=max(1, eval_steps // 4),
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        # La precisione dell'ottimizzazione segue quella del modello (regola T3):
        # con compute_dtype fp32 non c'e' autocast e non c'e' GradScaler, quindi
        # nessuno step viene scartato per gradienti inf/nan.
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        group_by_length=True,
        dataloader_num_workers=2,
    )

    # ORDINE IMPORTANTE: i callback girano nell'ordine della lista, e
    # EarlyStoppingCallback.on_evaluate legge metrics[metric_for_best_model]. Se
    # sta prima di chi aggiunge eval_gen_chrf al dict, non lo trova e si
    # disattiva con "early stopping required metric_for_best_model, but did not
    # find eval_gen_chrf". La selezione del best checkpoint invece funziona in
    # ogni caso, perche' avviene dopo che tutti i callback hanno finito.
    # Id di fine turno, calcolati sui target renderizzati di train: servono a
    # entrambe le generazioni di monitoraggio. Senza, il decoding non si ferma
    # mai prima di max_new_tokens e ogni diagnosi sull'EOS e' falsata.
    EOT = id_fine_turno(tokenizer, train_rows)

    callbacks = []
    if args.gen_metric_n > 0:
        callbacks.append(build_generation_metric_callback(
            TrainerCallback, torch, tokenizer, dev_rows, task,
            n=args.gen_metric_n, eos_ids=EOT))
    # PRIMA di EarlyStoppingCallback, che legge metrics[metric_for_best_model]:
    # se il callback che scrive eval_ctx_acc girasse dopo, non lo troverebbe e
    # l'early stopping si disattiverebbe in silenzio.
    if args.ctx_metric_n > 0:
        from contesto_metrica import build_context_callback
        callbacks.append(build_context_callback(
            TrainerCallback, torch, tokenizer, dev_rows, task,
            n=args.ctx_metric_n, seed=args.seed))
    callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
    if fixed_samples:
        callbacks.append(build_sample_callback(TrainerCallback, torch, tokenizer,
                                               fixed_samples, task, eos_ids=EOT))

    ClasseTrainer = Trainer
    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    if lex:
        from pesi_lessicali import TrainerPesato
        ClasseTrainer = TrainerPesato
        collate_fn = make_collate_pesato(tokenizer.pad_token_id)

    trainer = ClasseTrainer(model=model, args=training_args,
                      train_dataset=train_ds, eval_dataset=dev_ds,
                      data_collator=collate_fn,
                      compute_metrics=compute_metrics,
                      preprocess_logits_for_metrics=preprocess_logits,
                      callbacks=callbacks)

    print(f"\nAvvio training. Output: {out_dir}\n")
    t0 = time.time()
    interrotto = False
    try:
        trainer.train(resume_from_checkpoint=resume_from)
    except KeyboardInterrupt:
        interrotto = True
        print("\nInterrotto (Ctrl+C / timeout sessione).")
    elapsed = time.time() - t0

    final_dir = os.path.join(out_dir, "adapter_final")
    if not interrotto:
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
    elif trainer.state.best_model_checkpoint:
        final_dir = trainer.state.best_model_checkpoint
        print(f"(interrotto: uso il best checkpoint {final_dir})")

    log = trainer.state.log_history
    for row in log:
        if "eval_loss" in row:
            row["eval_perplexity"] = math.exp(min(row["eval_loss"], 20))
    csv_path = os.path.join(out_dir, "metrics_history.csv")
    if log:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in log for k in r}),
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(log)

    summary = {
        "task": {"layout": task.layout, "nome": task.nome,
                 "descrizione": task.descrizione, "note": task.note},
        "repo_id": repo_id, "completed": not interrotto,
        "precision": args.precision, "seed": args.seed,
        "hardware": {"gpu": gpu_name, "vram_gb": round(vram_gb, 1), "bf16": bf16_ok,
                     "dtype": nome_dt},
        "architettura_caricata": arch_kind,
        "versioni": _library_versions(),
        "init_adapter": args.init_adapter,
        "due_stadi": bool(args.init_adapter),
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "target_modules": target_modules,
                 "embeddings_allenati": args.train_embeddings},
        "hyperparams": {"lr": args.lr, "epochs": args.epochs,
                        "effective_batch_size": eff_batch,
                        "steps_per_epoch": steps_per_epoch, "max_steps": max_steps,
                        "eval_steps": eval_steps, "n_evals_previste": n_evals,
                        "warmup_steps": warmup_steps, "max_seq_len": args.max_seq_len,
                        "weight_decay": args.weight_decay,
                        "neftune_alpha": args.neftune_alpha or None,
                        "patience": args.patience},
        "dataset": {"train": len(train_rows), "dev_full": len(dev_full),
                    "dev_in_eval": len(dev_rows),
                    "troncati_train": train_ds.n_troncati, "troncati_dev": dev_ds.n_troncati,
                    "prefix_mismatch_train": train_ds.n_prefix_mismatch,
                    "prefix_mismatch_dev": dev_ds.n_prefix_mismatch,
                    # Provenienza: quale strategia di generazione ha vinto l'arbitraggio in
                    # build_final_dataset.py per ciascun turno. NON usato per filtrare né per
                    # pesare: il training vede tutto il dataset. Registrato qui solo perché la
                    # sezione "costruzione del dataset" del paper deve poter dichiarare la
                    # composizione effettiva di ciò su cui si è addestrato.
                    "provenienza_train": dict(Counter(r.get("fonte", "n/d") for r in train_rows)),
                    "provenienza_dev": dict(Counter(r.get("fonte", "n/d") for r in dev_full))},
        "selezione_checkpoint": {"metrica": metric_name, "greater_is_better": greater,
                                 "best_value": trainer.state.best_metric,
                                 "best_step": trainer.state.best_global_step,
                                 "best_checkpoint": trainer.state.best_model_checkpoint},
        "training_seconds": round(elapsed, 1),
        "final_adapter_dir": final_dir,
        "metrics_history_csv": csv_path if log else None,
        "avvertenza_metriche": "Tutte le metriche di questo run sono in TEACHER FORCING "
                               "(argmax dei logit), proxy per il monitoraggio e l'early "
                               "stopping. I numeri da riportare vanno da uno script di "
                               "valutazione separato su test.json con generate() reale.",
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'Interrotto' if interrotto else 'Fatto'} in {elapsed/60:.1f} min")
    print(f"Adapter: {final_dir}")
    print(f"Miglior {metric_name} (dev): {trainer.state.best_metric}")
    print(f"Riepilogo: {os.path.join(out_dir, 'summary.json')}")
    if interrotto:
        print("Per riprendere: salva la versione del notebook, montane l'output come dataset "
              "e rilancia con --resume-dir /kaggle/input/<nome-dataset>")
