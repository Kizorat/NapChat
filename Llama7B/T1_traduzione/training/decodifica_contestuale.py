#!/usr/bin/env python3
"""
decodifica_contestuale.py — costringe il modello a usare il contesto in
INFERENZA, senza riaddestrare niente.

L'idea
------
A ogni passo si calcolano i logit due volte, con il prompt completo e con il
blocco di contesto rimosso, e si combinano:

    logit = (1 + gamma) * logit(y | contesto)  -  gamma * logit(y | senza contesto)

E' il classifier-free guidance applicato al testo (context-aware decoding).
La differenza fra i due termini e' esattamente il contributo del contesto alla
predizione: moltiplicarla per gamma amplifica cio' che il contesto suggerisce e
sopprime cio' che il modello direbbe comunque. Sul fallimento tipico di T2/T3 —
napoletano fluente e morfologicamente corretto ma semanticamente slegato da cio'
che precede — e' il rimedio piu' diretto, e costa solo 2x il forward in
generazione.

gamma
-----
gamma=0 e' la generazione normale. Valori utili stanno fra 0,3 e 1,5. Troppo
alto degrada la fluenza: il modello inizia a preferire token rari solo perche'
il ramo senza contesto li sconsiglia. Non esiste un valore giusto a priori, si
sceglie sul dev con --sweep guardando DUE numeri insieme (chrF e ripetizioni):
un gamma che alza la pertinenza distruggendo la lingua non e' un miglioramento.

Cosa NON e'
-----------
Non insegna niente al modello: e' un intervento sul decoding. Se il modello non
ha proprio codificato il contesto nei suoi stati, non c'e' segnale da
amplificare e il gamma non produrra' effetto. In quel caso il problema e' a
monte (dati, curriculum) e questo script te lo dice: se il chrF resta piatto per
ogni gamma, il contesto non e' nella rappresentazione.

Uso
---
    # scegli gamma sul dev
    python decodifica_contestuale.py --model minerva --task T3 \\
        --adapter /kaggle/working/runs/<slug>__T3/adapter_final \\
        --split-dir /kaggle/working/split --split dev --sweep 0 0.5 1.0 1.5 --n 60

    # genera con il gamma scelto
    python decodifica_contestuale.py --model minerva --task T3 --adapter ... \\
        --split test --gamma 1.0 --out predizioni_cfg.json

Come libreria (per evaluate_task.py o prova_esempi.py):
    from decodifica_contestuale import genera_cfg, rimuovi_contesto
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from contesto_metrica import _e_separatore, estrai_blocco_contesto


def rimuovi_contesto(prompt):
    """Toglie intestazione, turni di contesto e la riga vuota che li chiude.

    Ritorna None se il prompt non ha contesto: in quel caso i due rami sarebbero
    identici, la differenza sarebbe zero e il CFG un puro raddoppio di costo.
    Quegli item vanno generati normalmente.
    """
    trovato = estrai_blocco_contesto(prompt)
    if trovato is None:
        return None
    i, j, _ = trovato
    righe = prompt.split("\n")
    k = j
    while k < len(righe) and _e_separatore(righe[k]):
        k += 1                     # via anche il separatore (riga vuota o ---)
    return "\n".join(righe[:i] + righe[k:])


def genera_cfg(model, tokenizer, torch, testo_con, testo_senza, gamma=1.0,
               max_new_tokens=64, do_sample=False, top_p=0.9, temperature=0.8,
               eos_id=None):
    """Generazione con guidance sul contesto. Ritorna la stringa generata.

    Si tengono due cache KV separate e si avanzano in parallelo. Non si usa
    model.generate perche' servirebbe un LogitsProcessor con accesso a un
    secondo stato, che con la cache diventa piu' fragile di un ciclo esplicito.
    """
    eos_id = eos_id if eos_id is not None else tokenizer.eos_token_id
    dev = model.device
    a = tokenizer(testo_con, return_tensors="pt", add_special_tokens=False).to(dev)
    b = tokenizer(testo_senza, return_tensors="pt", add_special_tokens=False).to(dev)

    with torch.no_grad():
        oa = model(**a, use_cache=True)
        ob = model(**b, use_cache=True)
    ca, cb = oa.past_key_values, ob.past_key_values
    la, lb = oa.logits[:, -1, :].float(), ob.logits[:, -1, :].float()

    prodotti = []
    for _ in range(max_new_tokens):
        # la differenza fra i due rami e' il contributo del contesto
        logit = (1.0 + gamma) * la - gamma * lb if gamma else la

        if do_sample:
            logit = logit / max(temperature, 1e-5)
            ordinati, indici = torch.sort(logit, descending=True, dim=-1)
            cum = torch.softmax(ordinati, dim=-1).cumsum(dim=-1)
            taglia = cum - torch.softmax(ordinati, dim=-1) > top_p
            ordinati = ordinati.masked_fill(taglia, float("-inf"))
            scelto = indici.gather(-1, torch.multinomial(
                torch.softmax(ordinati, dim=-1), 1))
        else:
            scelto = logit.argmax(dim=-1, keepdim=True)

        tid = int(scelto[0, 0])
        if tid == eos_id:
            break
        prodotti.append(tid)

        with torch.no_grad():
            oa = model(input_ids=scelto, past_key_values=ca, use_cache=True)
            ob = model(input_ids=scelto, past_key_values=cb, use_cache=True)
        ca, cb = oa.past_key_values, ob.past_key_values
        la, lb = oa.logits[:, -1, :].float(), ob.logits[:, -1, :].float()

    return tokenizer.decode(prodotti, skip_special_tokens=True)


def rerank_contestuale(model, tokenizer, torch, testo_con, testo_senza,
                       candidati, lam=1.0, render=None):
    """Alternativa piu' semplice al CFG: si generano k candidati e si ordinano
    per logP(y|contesto) - lam * logP(y|senza contesto).

    Piu' debole del CFG (sceglie fra cio' che il campionamento ha prodotto,
    invece di guidare la produzione) ma non richiede un ciclo di decoding
    custom: si puo' innestare in evaluate_task.py cambiando poche righe.
    """
    from contesto_metrica import _logp_per_token
    if render is None:
        from common import render_prompt
        render = render_prompt
    n = len(candidati)
    lc = _logp_per_token(model, tokenizer, torch, [testo_con] * n, candidati,
                         render, 512)
    ls = _logp_per_token(model, tokenizer, torch, [testo_senza] * n, candidati,
                         render, 512)
    punteggi = [c - lam * s for c, s in zip(lc, ls)]
    ordine = sorted(range(n), key=lambda k: -punteggi[k])
    return candidati[ordine[0]], [punteggi[k] for k in ordine]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["T1", "T2", "T3"])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--split-dir", default="/kaggle/working/split")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=60, help="item da valutare")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--sweep", type=float, nargs="*", default=None,
                    help="prova piu' gamma e confronta, es. --sweep 0 0.5 1.0 1.5")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hf-token", default=None)
    a = ap.parse_args()

    import sacrebleu
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, BitsAndBytesConfig

    from common import (load_backbone, load_hf_token, load_split, render_prompt,
                        resolve_model)

    righe = load_split(a.split_dir, a.task, a.split)[:a.n]
    if not righe:
        sys.exit(f"Nessuna riga {a.task}/{a.split} in {a.split_dir}")
    coppie = []
    senza_ctx = 0
    for r in righe:
        nudo = rimuovi_contesto(r["prompt"])
        if nudo is None:
            senza_ctx += 1
            continue
        coppie.append((r["prompt"], nudo, r["target"]))
    print(f"{a.task}/{a.split}: {len(coppie)} item con contesto "
          f"({senza_ctx} senza, esclusi: per loro il CFG non ha effetto)")
    if not coppie:
        sys.exit("Nessun item con contesto: il CFG non e' applicabile a questo layout.")

    repo_id = resolve_model(a.model)
    token = load_hf_token(a.hf_token)
    tok = AutoTokenizer.from_pretrained(repo_id, token=token, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cc = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dtype)
    model, _ = load_backbone(repo_id, dtype, quantization_config=quant,
                             device_map={"": 0}, token=token,
                             low_cpu_mem_usage=True, attn_implementation="eager")
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    model.config.use_cache = True

    def ripetizioni(testi):
        """quota di 3-grammi ripetuti: sale quando il gamma degrada la lingua"""
        tot = rip = 0
        for t in testi:
            w = t.split()
            g = [tuple(w[i:i + 3]) for i in range(max(0, len(w) - 2))]
            tot += len(g)
            rip += len(g) - len(set(g))
        return round(rip / tot, 4) if tot else 0.0

    gammi = a.sweep if a.sweep is not None else [a.gamma]
    rifer = [c[2] for c in coppie]
    risultati = {}
    for g in gammi:
        torch.manual_seed(a.seed)
        ipo = []
        for con, nudo, _ in coppie:
            t = genera_cfg(model, tok, torch, render_prompt(tok, con),
                           render_prompt(tok, nudo), gamma=g,
                           max_new_tokens=a.max_new, do_sample=a.sample)
            ipo.append(t.strip().split("\n")[0].strip())
        chrf = round(sacrebleu.corpus_chrf(ipo, [rifer], word_order=2).score, 2)
        lung = sum(len(h.split()) for h in ipo) / max(1, sum(len(r.split()) for r in rifer))
        risultati[g] = {"chrf": chrf, "rapporto_lunghezza": round(lung, 3),
                        "rep3": ripetizioni(ipo), "ipotesi": ipo}
        print(f"  gamma={g:<4} chrF {chrf:6.2f} | lunghezza {lung:.2f}x | "
              f"rep-3 {risultati[g]['rep3']:.4f}")

    if len(gammi) > 1:
        best = max(risultati, key=lambda g: risultati[g]["chrf"])
        print(f"\nMiglior chrF a gamma={best}. Ma controlla rep-3 e lunghezza: "
              f"un gamma che alza il chrF\nfacendo esplodere le ripetizioni non e' "
              f"un miglioramento, e' un artefatto del decoding.")
        piatto = max(r["chrf"] for r in risultati.values()) - \
            min(r["chrf"] for r in risultati.values())
        if piatto < 1.0:
            print("! chrF piatto su tutti i gamma: non c'e' segnale di contesto da "
                  "amplificare.\n  Il problema e' a monte (dati o curriculum), non "
                  "nel decoding.")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {str(g): v for g, v in risultati.items()}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"\nScritto {a.out}")


if __name__ == "__main__":
    main()
