#!/usr/bin/env python3
"""
evaluate_task.py — valutazione di UN run (modello x layout) sul test set.

Importa da common.py: stesso registry di modelli, stesso rendering dei prompt.
Il prompt in valutazione e' byte-identico a quello visto in training.

Uso:
  # sistema fine-tuned
  python evaluate_task.py --model llama --task T3 --split-dir /kaggle/working/split \\
      --adapter /kaggle/working/runs/llama-2-7b-chat-hf__T3/adapter_final
  # baseline del modello base
  python evaluate_task.py --model llama --task T3 --split-dir ... --zero-shot
  python evaluate_task.py --model llama --task T3 --split-dir ... --few-shot 5

--------------------------------------------------------------------------
LE METRICHE, E PERCHE' QUESTE
--------------------------------------------------------------------------
T2 e T3 sono task UNO-A-MOLTI: date le stesse 3 battute di contesto, molte
repliche diverse sono corrette. Nessuna metrica calcolata contro l'unico turno
realmente pronunciato puo' misurare l'adeguatezza. Quindi:

A. DISCRIMINAZIONE AVVERSARIALE (reference-free, e' la metrica principale)
   Un classificatore prova a distinguere le repliche umane da quelle generate.
   Accuracy 5-fold CV: ~0.50 = indistinguibile, 1.00 = riconoscibile subito.
   Vengono riportati SEMPRE due controlli, senza i quali il numero non si legge:
     - umano-vs-umano: due meta' casuali del solo insieme umano. Deve dare
       ~0.50 (misurato su questo corpus: 0.520 +- 0.044). Se da' di piu', il
       classificatore sta leggendo artefatti e il risultato non vale.
     - solo-lunghezza: stessa classificazione usando come unica feature il
       numero di parole. Esclude che si stia misurando "la macchina e' piu'
       corta dell'umano".

B. CONTRASTIVA (trasforma uno-a-molti in risposta-unica)
   Il modello deve assegnare la log-likelihood piu' alta al target vero fra
   k+1 candidati. Distrattori = target reali di altri item, appaiati per
   lunghezza (negativi difficili). Accuracy, baseline casuale 1/(k+1).
   Nessuna generazione: deterministica e non contaminata dal decoding.

C. DISTRIBUZIONALI (corpus vs corpus, non item vs item)
   distinct-1/2/3, rep-2/3, distribuzione delle lunghezze con test di
   Kolmogorov-Smirnov contro i riferimenti umani. Sono bersagli BILATERALI:
   distinct troppo basso = collasso su formule frequenti, troppo alto =
   insalata di parole. I valori umani vengono ricalcolati sul test corrente,
   non hardcoded.

D. DIALETTALITA' (reference-free)
   Classificatore italiano-vs-napoletano addestrato sulle coppie parallele del
   TRAIN di T1 (mai sul test). Su questo corpus: accuracy 0.994, AUC 0.998,
   P(nap) 0.914 sui riferimenti umani contro 0.094 sulle frasi italiane.
   Restituisce P(napoletano) medio sugli output generati.

E. LIKELIHOOD DEL RIFERIMENTO, normalizzata per lunghezza
   Reference-based ma tollerante: non chiede di produrre quella stringa, solo
   di considerarla plausibile. Le altre continuazioni valide non competono.

F. chrF++ / BLEU
   Riportate, ma su T2 e T3 sono un LIMITE INFERIORE, non una misura di
   adeguatezza. Insieme vengono stampate le baseline non-banali misurate sul
   test corrente (copia dell'italiano per T1, ripeti-il-prefisso per T2,
   replica-da-un-altro-punto per T3): sono quelle il riferimento, non lo zero.

Nessuna di queste metriche, da sola, misura la qualita': ciascuna e' aggirabile
in isolamento (dialettalita' massima con dialetto senza senso, distinct massimo
con parole casuali). Si leggono come vettore.
"""

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from Minerva7B.T1_Traduzione.training.common import (LAYOUT_DIRS, MODEL_REGISTRY, load_backbone, load_hf_token,
                   load_split, render_prompt, resolve_model, slug)

MAX_NEW = {"T1": 64, "T2": 48, "T3": 32}   # T3 a 32: la mediana dei turni
# umani e' 4 parole (~8 token) e il 68% sta sotto le 6. A 64 token il modello
# continuava a generare fino a 40 parole contro un target di 10, e la deriva
# semantica si accumulava proprio nella coda. Con solo la lunghezza come
# feature un classificatore distingueva generato da umano al 57,9%: il
# troncamento va fatto prima di dare peso alla discriminazione avversariale.
ISTRUZIONE = {"T1": "Traduci in napoletano: ", "T2": "Continua il turno in napoletano: "}


# --------------------------------------------------------------------------- #
# utility testuali
# --------------------------------------------------------------------------- #

def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w' ]+", " ", (s or "").lower())).strip()


CANDIDATI_EOT = ("<|im_end|>", "<end_of_turn>", "<|eot_id|>", "<|end|>",
                 "<|endoftext|>", "</s>", "<end_of_text>")


def id_fine_turno(tok, rows, verbose=True):
    """Insieme degli id che chiudono il turno assistant, da passare a generate().

    IL PUNTO. Con un chat template tipo ChatML il turno assistant chiude con
    <|im_end|>, che NON e' tok.eos_token (</s>). In training la label c'e':
    apply_chat_template con add_generation_prompt=False lo include e il masking
    di ChatDataset lo lascia dentro, quindi il modello IMPARA a emetterlo. Ma se
    generate() non lo ha fra gli eos_token_id non si ferma mai e arriva sempre a
    max_new_tokens. Il sintomo misurato su minerva T3 nucleus: lunghezza media
    17.57 parole contro 6.99 umano, ks 0.589 con p=0, e soprattutto
    controllo_solo_lunghezza 0.776 su accuracy_umano_vs_macchina 0.80 - cioe' il
    97% del potere discriminante del classificatore avversariale veniva dalla
    lunghezza. Una frase tagliata al tetto si legge come "sconclusionata" anche
    quando la deriva semantica non c'e'.

    La fonte di verita' non e' la config del modello: e' l'ultimo token del
    testo renderizzato in training, che e' esattamente cio' che il modello ha
    imparato a produrre per chiudere.
    """
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    for s in CANDIDATI_EOT:
        i = tok.convert_tokens_to_ids(s)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.add(int(i))
    coda = []
    for r in rows[:8]:
        t = tok(render_prompt(tok, r["prompt"], r["target"]),
                add_special_tokens=False)["input_ids"]
        if t:
            coda.append(int(t[-1]))
    if coda:
        ultimo = Counter(coda).most_common(1)[0][0]
        ids.add(ultimo)
        if verbose:
            print(f"  ultimo token del target renderizzato: id={ultimo} "
                  f"repr={tok.convert_ids_to_tokens([ultimo])[0]!r}")
    out = sorted(ids)
    if verbose:
        print(f"  tok.eos_token = {tok.eos_token!r} (id {tok.eos_token_id})")
        print(f"  fine turno da usare = {out} "
              f"({[tok.convert_ids_to_tokens([i])[0] for i in out]})")
    return out


def frazione_al_tetto(tok, hyps, max_new):
    """Frazione di output che raggiunge max_new, cioe' che non ha mai emesso
    fine di turno. Sopra ~0.05 la qualita' apparente e' dominata dal
    troncamento e non dalla pragmatica: leggere le altre metriche prima di
    aver abbassato questa non ha senso."""
    if not hyps:
        return 0.0
    n = sum(1 for h in hyps
            if len(tok(h, add_special_tokens=False)["input_ids"]) >= max_new - 1)
    return n / len(hyps)


def fonte_italiana(prompt):
    """Estrae la frase italiana dal prompt di T1 (l'istruzione la contiene)."""
    marker = ISTRUZIONE["T1"]
    return prompt.split(marker)[-1].strip() if marker in prompt else None


def prefisso_nap(prompt):
    """Estrae il prefisso napoletano dal prompt di T2."""
    marker = ISTRUZIONE["T2"]
    return prompt.split(marker)[-1].strip() if marker in prompt else None


MARCATORI_RUOLO = ("<|im_start|>", "<|im_end|>", "<start_of_turn>",
                   "<end_of_turn>", "<|eot_id|>", "<|start_header_id|>",
                   "<|end_header_id|>")


def pulisci_generato(txt):
    """Primo turno utile della generazione.

    Come prima si taglia alla prima riga, ma PRIMA si togliono i marcatori di
    ruolo: con un template custom skip_special_tokens non sempre li copre, e
    tagliare alla prima riga quando la prima riga E' il marcatore restituisce
    stringa vuota, cioe' un item perso in silenzio.
    """
    t = txt
    for m in MARCATORI_RUOLO:
        t = t.replace(m, "\n")
    for riga in t.split("\n"):
        r = riga.strip()
        if r.lower() in ("assistant", "user", "model", "system"):
            continue
        if r:
            return r
    return ""


def distinct(texts, n):
    grams, tot = set(), 0
    for t in texts:
        w = t.split()
        for i in range(len(w) - n + 1):
            grams.add(tuple(w[i:i + n]))
            tot += 1
    return len(grams) / tot if tot else 0.0


def rep_n(texts, n):
    c = 0
    for t in texts:
        w = t.split()
        g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
        if g and len(set(g)) < len(g):
            c += 1
    return c / len(texts) if texts else 0.0


def loop_massimo(t):
    """Lunghezza della piu' lunga sequenza di token identici consecutivi."""
    w = t.split()
    best = cur = 1 if w else 0
    for i in range(1, len(w)):
        cur = cur + 1 if w[i].lower() == w[i - 1].lower() else 1
        best = max(best, cur)
    return best


def degenerazione(hyps, refs):
    """Due patologie che distinct-n e rep-n non separano: i loop del decoder e il
    collasso sull'apertura piu' frequente."""
    def incipit(ts):
        c = Counter(t.split()[0].lower() for t in ts if t.split())
        top, n = (c.most_common(1)[0] if c else ("", 0))
        return {"token": top, "frazione": round(n / max(len(ts), 1), 3)}
    lm = [loop_massimo(t) for t in hyps]
    lr = [loop_massimo(t) for t in refs]
    return {
        "frazione_con_loop_3plus": round(sum(1 for x in lm if x >= 3) / max(len(lm), 1), 3),
        "loop_massimo": max(lm) if lm else 0,
        "incipit_dominante": incipit(hyps),
        "_umano": {"frazione_con_loop_3plus": round(sum(1 for x in lr if x >= 3) / max(len(lr), 1), 3),
                  "loop_massimo": max(lr) if lr else 0,
                  "incipit_dominante": incipit(refs)},
        "_lettura": "loop = degenerazione del decoder (il greedy su task aperti la produce "
                    "per costruzione). incipit_dominante alto = collasso modale: se il 76% "
                    "delle repliche inizia con la stessa parola e negli umani il massimo e' "
                    "il 6%, il modello ha imparato una formula, non una distribuzione.",
    }


def distribuzionali(hyps, refs):
    from scipy import stats
    Lh = [len(x.split()) for x in hyps]
    Lr = [len(x.split()) for x in refs]
    ks = stats.ks_2samp(Lh, Lr)
    return {
        "distinct_1": round(distinct(hyps, 1), 4),
        "distinct_2": round(distinct(hyps, 2), 4),
        "distinct_3": round(distinct(hyps, 3), 4),
        "rep_2": round(rep_n(hyps, 2), 4),
        "rep_3": round(rep_n(hyps, 3), 4),
        "lunghezza_media": round(statistics.mean(Lh), 2) if Lh else 0,
        "lunghezza_mediana": statistics.median(Lh) if Lh else 0,
        "ks_lunghezza_stat": round(float(ks.statistic), 4),
        "ks_lunghezza_pvalue": round(float(ks.pvalue), 4),
        "_umano": {
            "distinct_1": round(distinct(refs, 1), 4),
            "distinct_2": round(distinct(refs, 2), 4),
            "distinct_3": round(distinct(refs, 3), 4),
            "rep_2": round(rep_n(refs, 2), 4),
            "rep_3": round(rep_n(refs, 3), 4),
            "lunghezza_media": round(statistics.mean(Lr), 2) if Lr else 0,
            "lunghezza_mediana": statistics.median(Lr) if Lr else 0,
        },
        "_lettura": "bersagli bilaterali: confronta con _umano, non massimizzare. "
                    "ks_pvalue alto = distribuzione delle lunghezze compatibile con l'umano.",
    }


# --------------------------------------------------------------------------- #
# A. discriminazione avversariale
# --------------------------------------------------------------------------- #

def _clf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=5))


def discriminazione_avversariale(hyps, refs, seed=0):
    """Accuracy 5-fold nel distinguere umano da macchina, piu' i due controlli."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X = [norm(x) for x in refs] + [norm(x) for x in hyps]
    y = [0] * len(refs) + [1] * len(hyps)
    if min(len(refs), len(hyps)) < 20:
        return {"errore": "troppi pochi esempi per la cross-validation"}
    sc = cross_val_score(_clf(), X, y, cv=5, scoring="accuracy")

    # controllo 1: umano-vs-umano. Deve stare a ~0.50.
    rng = random.Random(seed)
    hh = [norm(x) for x in refs]
    rng.shuffle(hh)
    yhh = [0] * (len(hh) // 2) + [1] * (len(hh) - len(hh) // 2)
    sc_hh = cross_val_score(_clf(), hh, yhh, cv=5, scoring="accuracy")

    # controllo 2: solo la lunghezza come feature
    L = np.array([[len(x.split())] for x in X])
    sc_len = cross_val_score(LogisticRegression(max_iter=1000), L, y, cv=5, scoring="accuracy")

    return {
        "accuracy_umano_vs_macchina": round(float(sc.mean()), 3),
        "std": round(float(sc.std()), 3),
        "controllo_umano_vs_umano": round(float(sc_hh.mean()), 3),
        "controllo_solo_lunghezza": round(float(sc_len.mean()), 3),
        "_lettura": "~0.50 = indistinguibile dall'umano (ottimo). 1.00 = riconoscibile "
                    "subito. Valido SOLO se controllo_umano_vs_umano resta vicino a 0.50: "
                    "altrimenti il classificatore legge artefatti. Se "
                    "controllo_solo_lunghezza e' alto, sta misurando la lunghezza.",
    }


# --------------------------------------------------------------------------- #
# D. dialettalita'
# --------------------------------------------------------------------------- #

def costruisci_discriminatore_dialetto(split_dir):
    """Addestrato sulle coppie parallele del TRAIN di T1: la frase italiana sta
    dentro il prompt, la resa napoletana e' il target. Mai addestrato sul test."""
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    rows = load_split(split_dir, "T1", "train")
    X, y = [], []
    for r in rows:
        ita, nap = fonte_italiana(r["prompt"]), r["target"]
        if not ita:
            continue
        a, b = norm(ita), norm(nap)
        if a == b or len(a.split()) < 3:      # coppie identiche: non separabili per definizione
            continue
        X += [a, b]
        y += [0, 1]
    if len(X) < 100:
        return None, {"errore": "coppie parallele insufficienti"}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, random_state=0, stratify=y)
    clf = _clf().fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    info = {
        "n_frasi_train": len(Xtr),
        "accuracy": round(float(accuracy_score(yte, p > 0.5)), 3),
        "auc": round(float(roc_auc_score(yte, p)), 3),
    }
    return clf, info


# --------------------------------------------------------------------------- #
# B/E. likelihood e contrastiva
# --------------------------------------------------------------------------- #

def logprob_target(model, tok, prompt, target, max_len, device):
    """log P(target | prompt) sommata e normalizzata per numero di token."""
    import torch
    p_ids = tok(render_prompt(tok, prompt), add_special_tokens=False)["input_ids"]
    f_ids = tok(render_prompt(tok, prompt, target), add_special_tokens=False)["input_ids"][:max_len]
    n_t = len(f_ids) - len(p_ids)
    if n_t <= 0:
        return None
    ids = torch.tensor([f_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0].float()
    lp = torch.log_softmax(logits[:-1], dim=-1)
    tgt = ids[0][1:]
    tot = sum(float(lp[i, tgt[i]]) for i in range(len(p_ids) - 1, len(tgt)))
    return tot / n_t


def distrattori_per_lunghezza(rows, i, k, rng):
    """Negativi difficili: target reali di altri item, con lunghezza simile."""
    n_target = len(rows[i]["target"].split())
    cand = [j for j in range(len(rows))
            if j != i and abs(len(rows[j]["target"].split()) - n_target) <= 2]
    if len(cand) < k:
        cand = [j for j in range(len(rows)) if j != i]
    return [rows[j]["target"] for j in rng.sample(cand, min(k, len(cand)))]


def contrastiva(model, tok, rows, k, max_len, device, seed=0, limite=None):
    rng = random.Random(seed)
    sub = rows if limite is None else rows[:limite]
    ok, n, lp_ref = 0, 0, []
    for i, r in enumerate(sub):
        cand = [r["target"]] + distrattori_per_lunghezza(rows, i, k, rng)
        scores = [logprob_target(model, tok, r["prompt"], c, max_len, device) for c in cand]
        if scores[0] is None or any(s is None for s in scores):
            continue
        lp_ref.append(scores[0])
        ok += int(max(range(len(scores)), key=lambda j: scores[j]) == 0)
        n += 1
        if n % 25 == 0:
            print(f"  contrastiva {n}/{len(sub)}", end="\r")
    if not n:
        return {}
    return {
        "accuracy": round(ok / n, 3),
        "baseline_casuale": round(1 / (k + 1), 3),
        "k_distrattori": k,
        "n": n,
        "logprob_riferimento_norm": round(statistics.mean(lp_ref), 4),
        "_lettura": "distrattori = target reali di altri item appaiati per lunghezza. "
                    "logprob normalizzata per token: piu' alta (meno negativa) = il "
                    "modello considera plausibile una continuazione valida.",
    }


# --------------------------------------------------------------------------- #
# F. metriche di riferimento e baseline
# --------------------------------------------------------------------------- #

def riferimento(hyps, refs):
    import sacrebleu
    return {
        "chrf++": round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2),
        "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
    }


def bootstrap_chrf(hyps, refs, n=1000, seed=0):
    import sacrebleu
    rng = random.Random(seed)
    N = len(refs)
    vals = []
    for _ in range(n):
        s = [rng.randrange(N) for _ in range(N)]
        vals.append(sacrebleu.corpus_chrf([hyps[i] for i in s], [[refs[i] for i in s]],
                                          word_order=2).score)
    vals.sort()
    return {"ci95_basso": round(vals[int(0.025 * n)], 2),
            "ci95_alto": round(vals[int(0.975 * n)], 2)}


def baseline_non_banali(task, rows, refs):
    """Il riferimento non e' lo zero: e' quello che ottiene una strategia stupida."""
    import sacrebleu
    out = {}
    def chrf(h, r):
        return round(sacrebleu.corpus_chrf(h, [r], word_order=2).score, 2)
    if task == "T1":
        ita = [fonte_italiana(r["prompt"]) or "" for r in rows]
        out["copia_italiano"] = chrf(ita, refs)
    if task == "T2":
        out["ripeti_prefisso"] = chrf([prefisso_nap(r["prompt"]) or "" for r in rows], refs)
    sh = refs[:]
    random.Random(0).shuffle(sh)
    out["riferimento_di_un_altro_item"] = chrf(sh, refs)
    out["_lettura"] = "un sistema che non batte queste in modo netto non ha imparato nulla."
    return out


def stratifica_per_lunghezza(hyps, refs, soglia=2):
    """Riscontri (<= soglia parole) contro turni contenutistici.

    Nel corpus il 33.7% dei turni napoletani sta a <= 2 parole e i piu'
    frequenti sono segnali di ascolto (mh 95, si' 71, mhmh 58, no 32). Sono un
    sotto-task diverso: chrF++ contro un riferimento di una parola non misura
    niente, e aggregare i due annega il segnale sui turni contenutistici, che
    sono quelli di cui si parla quando si dice "generazione libera".
    """
    out = {}
    for nome, tieni in (("riscontri", lambda n: n <= soglia),
                        ("contenutistici", lambda n: n > soglia)):
        idx = [i for i, r in enumerate(refs) if tieni(len(r.split()))]
        if len(idx) < 20:
            out[nome] = {"n": len(idx), "nota": "troppi pochi item"}
            continue
        h = [hyps[i] for i in idx]
        f = [refs[i] for i in idx]
        out[nome] = {
            "n": len(idx),
            "chrf++": riferimento(h, f)["chrf++"],
            "lunghezza_media_generata": round(
                statistics.mean(len(x.split()) for x in h), 2),
            "lunghezza_media_riferimento": round(
                statistics.mean(len(x.split()) for x in f), 2),
        }
    out["_lettura"] = ("il numero aggregato e' una media fra due distribuzioni "
                       "diverse: guarda 'contenutistici' per la generazione libera.")
    return out


# --------------------------------------------------------------------------- #
# generazione
# --------------------------------------------------------------------------- #

def prompt_few_shot(tok, r, shots):
    msgs = []
    for s in shots:
        msgs.append({"role": "user", "content": s["prompt"]})
        msgs.append({"role": "assistant", "content": s["target"]})
    msgs.append({"role": "user", "content": r["prompt"]})
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return "".join(m["content"] + "\n" for m in msgs)


def genera(model, tok, rows, shots, max_new, device, batch=8, decoding="greedy",
           top_p=0.9, temperature=0.8, seed=42, eos_token_id=None,
           repetition_penalty=1.0, no_repeat_ngram_size=0, typical_p=None):
    """decoding='greedy' per T1/T2, 'nucleus' per T3.

    Il greedy e' mode-seeking: su un task APERTO cerca sempre la continuazione
    piu' probabile e il risultato e' collasso sull'apertura piu' frequente piu'
    loop di ripetizione (Holtzman et al. 2020). Misurato su Minerva T3: il 76%
    delle repliche iniziava con 'ma' contro il 5.7% del token piu' frequente
    negli umani, e il 14.6% conteneva un token ripetuto >=3 volte di fila, con
    un massimo di 39 ripetizioni consecutive.

    Le repliche umane sono campioni da una distribuzione, non modi. Per T3 serve
    nucleus sampling; il seed fisso preserva la riproducibilita'.

    BUG CORRETTO. `repetition_penalty` e `no_repeat_ngram_size` erano nel
    TaskConfig di T3 (1.2 e 3) e agivano sulla generazione di monitoraggio
    durante il training, ma qui non arrivavano: il ramo nucleus impostava solo
    do_sample/top_p/temperature/top_k. La valutazione di test girava quindi
    senza NESSUN controllo di ripetizione, da cui rep_2 0.151 contro 0.0365
    umano e loop_massimo 63. Ora sono parametri espliciti e finiscono nel
    metrics.json, cosi' un run non e' confrontabile con un altro per sbaglio.

    `eos_token_id` va passato: vedi id_fine_turno(). Senza, generate() usa
    generation_config, che su piu' di un modello del registry non contiene il
    token di fine turno del chat template.

    `typical_p`: locally typical sampling (Meister et al. 2023). Su target con
    mediana 4 parole il nucleus a top_p 0.9 e T 0.8 e' troppo caldo - un
    campionamento sbagliato sui primi due token non ha spazio per essere
    recuperato. Alternativa a top_p, non aggiuntiva.
    """
    import torch
    if decoding == "nucleus":
        torch.manual_seed(seed)
    hyps = []
    for i in range(0, len(rows), batch):
        b = rows[i:i + batch]
        testi = [prompt_few_shot(tok, r, shots) if shots else render_prompt(tok, r["prompt"])
                 for r in b]
        enc = tok(testi, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen_kw = dict(max_new_tokens=max_new, min_new_tokens=1,
                      pad_token_id=tok.pad_token_id)
        if eos_token_id:
            gen_kw["eos_token_id"] = eos_token_id
        if repetition_penalty and repetition_penalty != 1.0:
            gen_kw["repetition_penalty"] = repetition_penalty
        if no_repeat_ngram_size:
            gen_kw["no_repeat_ngram_size"] = no_repeat_ngram_size
        if decoding == "nucleus":
            gen_kw.update(do_sample=True, temperature=temperature, top_k=0)
            if typical_p:
                gen_kw["typical_p"] = typical_p
            else:
                gen_kw["top_p"] = top_p
        else:
            gen_kw.update(do_sample=False, num_beams=1)
        with torch.no_grad():
            g = model.generate(**enc, **gen_kw)
        for j in range(len(b)):
            txt = tok.decode(g[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            hyps.append(pulisci_generato(txt))
        print(f"  generazione {min(i + batch, len(rows))}/{len(rows)}", end="\r")
    print()
    return hyps


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=f"alias ({'/'.join(MODEL_REGISTRY)}) o repo_id")
    ap.add_argument("--task", required=True, choices=["T1", "T2", "T3"])
    ap.add_argument("--split-dir", default="/kaggle/working/split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--few-shot", type=int, default=0)
    ap.add_argument("--out", default="/kaggle/working/eval")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--decoding", choices=["greedy", "nucleus", "auto"], default="auto",
                    help="auto = greedy su T1/T2 (hanno un modo), nucleus su T3 (aperto). "
                         "Riporta ENTRAMBI su T3: la differenza e' essa stessa un risultato")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--typical-p", type=float, default=None,
                    help="locally typical sampling invece di top_p (alternativa, "
                         "non aggiuntiva). Su T3 provare 0.9 con --temperature 0.7")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--repetition-penalty", type=float, default=None,
                    help="default: 1.15 su T3 nucleus, 1.0 altrove. Prima non "
                         "arrivava affatto a generate()")
    ap.add_argument("--no-repeat-ngram", type=int, default=None,
                    help="default: 3 su T3 nucleus, 0 altrove")
    ap.add_argument("--gen-seed", type=int, default=42)
    ap.add_argument("--contrastive-k", type=int, default=4)
    ap.add_argument("--contrastive-limit", type=int, default=100,
                    help="la contrastiva costa (k+1) forward per item: limitala")
    ap.add_argument("--skip-contrastive", action="store_true")
    ap.add_argument("--hf-token", default=None)
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    repo_id = resolve_model(a.model)
    max_new = a.max_new or MAX_NEW[a.task]
    token = load_hf_token(a.hf_token)
    device = 0

    tok = AutoTokenizer.from_pretrained(repo_id, token=token, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                          # obbligatorio in generazione

    cc = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
    print(f"GPU cc {cc[0]}.{cc[1]} -> {dtype}")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model, arch = load_backbone(repo_id, dtype, quantization_config=quant,
                               device_map={"": device}, token=token,
                               low_cpu_mem_usage=True, attn_implementation="eager")
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    model.config.use_cache = True

    rows = load_split(a.split_dir, a.task, a.split)
    refs = [r["target"] for r in rows]
    shots = []
    if a.few_shot:
        tr = load_split(a.split_dir, a.task, "train")
        shots = tr[:a.few_shot]
    tag = "ft" if a.adapter else (f"fs{a.few_shot}" if a.few_shot else "zs")
    print(f"=== {slug(repo_id)} {a.task} {tag} | {len(rows)} esempi di {a.split} ===")

    dec = a.decoding if a.decoding != "auto" else ("nucleus" if a.task == "T3" else "greedy")

    # Fine turno: senza questo generate() non si ferma e arriva sempre al tetto.
    print("Fine turno:")
    eot = id_fine_turno(tok, rows)
    gc_eos = getattr(model.generation_config, "eos_token_id", None)
    gc_set = set(gc_eos if isinstance(gc_eos, (list, tuple))
                 else ([gc_eos] if gc_eos is not None else []))
    mancanti = [i for i in eot if i not in gc_set]
    if mancanti:
        print(f"  ! {mancanti} NON erano in generation_config: senza il fix "
              f"generate() arrivava sempre a max_new={max_new}")

    rep = a.repetition_penalty if a.repetition_penalty is not None else (
        1.15 if dec == "nucleus" else 1.0)
    nrn = a.no_repeat_ngram if a.no_repeat_ngram is not None else (
        3 if dec == "nucleus" else 0)
    print(f"Decoding: {dec}" + (
        f" ({'typical_p=%s' % a.typical_p if a.typical_p else 'top_p=%s' % a.top_p}, "
        f"T={a.temperature}, rep={rep}, no_repeat={nrn}, seed={a.gen_seed})"
        if dec == "nucleus" else ""))
    hyps = genera(model, tok, rows, shots, max_new, device, a.batch,
                  decoding=dec, top_p=a.top_p, temperature=a.temperature,
                  seed=a.gen_seed, eos_token_id=eot, repetition_penalty=rep,
                  no_repeat_ngram_size=nrn, typical_p=a.typical_p)
    tetto = frazione_al_tetto(tok, hyps, max_new)
    print(f"Frazione al tetto: {tetto:.3f}" +
          ("   <- ALTO: stai misurando un troncamento, non la pragmatica"
           if tetto > 0.05 else "   ok"))

    etichetta_dec = "" if dec == "greedy" else (
        "__typical" if a.typical_p else f"__{dec}")
    run = f"{slug(repo_id)}__{a.task}__{tag}" + etichetta_dec
    res = {"run": run, "decoding": dec, "repo_id": repo_id, "task": a.task, "tag": tag,
           "architettura": arch, "n_test": len(rows),
           "F_riferimento": riferimento(hyps, refs),
           "F_bootstrap_chrf": bootstrap_chrf(hyps, refs),
           "F_baseline": baseline_non_banali(a.task, rows, refs),
           "C_distribuzionali": distribuzionali(hyps, refs),
           "C_degenerazione": degenerazione(hyps, refs),
           "A_avversariale": discriminazione_avversariale(hyps, refs),
           "G_troncamento": {
               "frazione_al_tetto": round(tetto, 3),
               "max_new_tokens": max_new,
               "_lettura": "frazione di output che raggiunge max_new senza "
                           "emettere fine di turno. Sopra ~0.05 le altre "
                           "metriche misurano il troncamento.",
           },
           "H_stratificato": stratifica_per_lunghezza(hyps, refs),
           "gen_kw": {"decoding": dec, "temperature": a.temperature,
                      "top_p": None if a.typical_p else a.top_p,
                      "typical_p": a.typical_p, "repetition_penalty": rep,
                      "no_repeat_ngram_size": nrn, "eos_token_id": eot,
                      "max_new_tokens": max_new, "seed": a.gen_seed}}

    clf_dial, info = costruisci_discriminatore_dialetto(a.split_dir)
    if clf_dial is not None:
        pm = clf_dial.predict_proba([norm(x) for x in hyps])[:, 1]
        pr = clf_dial.predict_proba([norm(x) for x in refs])[:, 1]
        res["D_dialettalita"] = {
            "P_nap_generato": round(float(pm.mean()), 3),
            "P_nap_riferimenti_umani": round(float(pr.mean()), 3),
            "discriminatore": info,
            "_lettura": "confronta P_nap_generato con P_nap_riferimenti_umani: "
                        "il bersaglio e' quello, non 1.0.",
        }
    else:
        res["D_dialettalita"] = info

    if not a.skip_contrastive:
        tok.padding_side = "right"                     # scoring, non generazione
        print("Contrastiva (nessuna generazione, solo forward)...")
        res["B_contrastiva"] = contrastiva(model, tok, rows, a.contrastive_k, a.max_len,
                                           device, limite=a.contrastive_limit)

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / f"{run}.preds.jsonl").open("w", encoding="utf-8") as f:
        for r, h in zip(rows, hyps):
            f.write(json.dumps({"id": r.get("id"), "prompt": r["prompt"],
                                "target": r["target"], "hyp": h}, ensure_ascii=False) + "\n")
    (outdir / f"{run}.metrics.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\nSalvato in {outdir}/{run}.metrics.json")


if __name__ == "__main__":
    main()
