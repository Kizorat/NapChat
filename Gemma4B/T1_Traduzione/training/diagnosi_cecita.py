#!/usr/bin/env python3
"""
diagnosi_cecita.py — il modello e' cieco al contesto? Tre controlli, due gratis.

"Sembra cieco" e' un'impressione, e questa e' l'unica cosa di tutto il progetto
che non va trattata come tale: se il contesto non arriva al modello per un
motivo meccanico, ogni ora spesa su prompt, DPO o reranking e' buttata.

I tre controlli, in ordine di costo:

[1] CONTATORI DI TRAINING (zero GPU, legge i summary.json)
    troncati_train        prompt tagliati a max_seq_len. Se e' alto, parte del
                          contesto NON ARRIVA MAI al modello: cieco alla
                          lettera, e nessun intervento a valle serve.
    prefix_mismatch_train confine prompt/target sfasato: la loss e'
                          mascherata sul punto sbagliato. Deve essere 0.

[2] ABLAZIONE SUL CONTESTO (~20 min di GPU)
    Genera lo stesso item due volte: una col contesto reale, una col contesto
    preso da un ALTRO punto della conversazione (sostituisci_contesto di
    contesto_metrica.py), poi misura il chrF++ FRA LE DUE USCITE.

        chrF alto (>60)  le due uscite sono quasi identiche: il modello ignora
                         il contesto. E' la misura diretta della cecita', ed e'
                         il numero da riportare.
        chrF basso       il contesto entra nella generazione. Il problema non e'
                         che sia cieco, e' che ci vede male.

    Perche' e' meglio della contrastiva: la contrastiva misura il PUNTEGGIO che
    il modello assegna a stringhe date, questa misura la GENERAZIONE - cioe'
    esattamente il comportamento che stai osservando quando dici che sembra
    cieco. Un modello puo' avere contrastiva 0.33 (qualche preferenza) e
    generare comunque la stessa cosa a prescindere dal contesto.

[3] ISPEZIONE DEI PROMPT (zero GPU)
    Stampa prompt reali per layout e controlla quattro cose concrete:
      - i turni di contesto hanno l'etichetta del parlante?
      - l'istruzione dice A QUALE parlante rispondere, con lo stesso
        identificativo usato nel contesto?
      - quante parole di contesto ci sono davvero? 3 turni da ~7 parole sono
        ~21 parole di parlato frammentario: bastano a stabilire l'argomento?
      - il separatore distingue visibilmente contesto e istruzione?
    Piu' un controllo automatico: l'istruzione e' COSTANTE su tutti gli esempi?
    Se lo e', ha informazione mutua zero col target, il modello non puo' usarla
    per abbassare la loss e impara a ignorarla. Riformularla meglio non serve:
    va resa variabile (vedi arricchisci_split.py).

Uso
---
    # solo i controlli gratis
    python diagnosi_cecita.py --split-dir /kaggle/working/split --solo-statico

    # tutto, incluso il [2]
    python diagnosi_cecita.py --model minerva --split-dir /kaggle/working/split \\
        --adapter /kaggle/working/runs/<slug>__T3/adapter_final --task T3
"""

import argparse
import glob
import json
import os
import random
import statistics
from collections import Counter
from pathlib import Path

from common import load_split, render_prompt, resolve_model
from contesto_metrica import estrai_blocco_contesto, sostituisci_contesto
from evaluate_task import MAX_NEW, id_fine_turno, norm


# --------------------------------------------------------------------------- #
# [1] contatori di training
# --------------------------------------------------------------------------- #

def contatori(runs_dir):
    righe = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "summary.json"))):
        s = json.load(open(p, encoding="utf-8"))
        d = s.get("dataset", {})
        n_tr = d.get("train") or 1
        righe.append({
            "run": os.path.basename(os.path.dirname(p)),
            "layout": s.get("task", {}).get("layout"),
            "max_seq_len": s.get("hyperparams", {}).get("max_seq_len"),
            "train": d.get("train"),
            "troncati_train": d.get("troncati_train"),
            "frazione_troncati": round((d.get("troncati_train") or 0) / n_tr, 3),
            "prefix_mismatch_train": d.get("prefix_mismatch_train"),
        })
    return righe


def leggi_contatori(righe):
    problemi = []
    for r in righe:
        if (r["frazione_troncati"] or 0) > 0.02:
            problemi.append(
                f"{r['run']}: {r['frazione_troncati']:.1%} dei prompt di train "
                f"troncati a max_seq_len={r['max_seq_len']}. Parte del contesto "
                f"non arriva al modello. Alza --max-seq-len o accorcia la "
                f"finestra: nessun intervento a valle recupera informazione "
                f"che non e' mai entrata.")
        if r["prefix_mismatch_train"]:
            problemi.append(
                f"{r['run']}: prefix_mismatch_train={r['prefix_mismatch_train']} "
                f"(deve essere 0). Il confine prompt/target e' sfasato: la loss "
                f"e' mascherata sul punto sbagliato e stai addestrando su un "
                f"target che non e' quello che credi.")
    return problemi


# --------------------------------------------------------------------------- #
# [3] ispezione dei prompt (statico)
# --------------------------------------------------------------------------- #

def analizza_prompt(rows, layout):
    """Cosa c'e' e cosa manca nei prompt, in numeri."""
    con_ctx, turni_n, parole_ctx = 0, [], []
    etichettati, non_etichettati = 0, 0
    istruzioni = Counter()

    for r in rows:
        trovato = estrai_blocco_contesto(r["prompt"])
        if trovato is None:
            continue
        con_ctx += 1
        i0, i1, turni = trovato
        turni_n.append(len(turni))
        parole_ctx.append(sum(len(t.split()) for t in turni))
        # etichetta del parlante: "X:" all'inizio della riga
        for t in turni:
            testa = t.strip().split(":", 1)
            if len(testa) == 2 and 0 < len(testa[0]) <= 12:
                etichettati += 1
            else:
                non_etichettati += 1
        # l'istruzione: tutto cio' che segue il blocco, separatori esclusi
        righe = r["prompt"].split("\n")
        istr = "\n".join(x for x in righe[i1:] if x.strip()
                         and not set(x.strip()) <= set("-=*_"))
        istruzioni[istr] += 1

    n = len(rows)
    tot_t = etichettati + non_etichettati
    piu_comune, freq = (istruzioni.most_common(1)[0] if istruzioni else ("", 0))
    return {
        "layout": layout,
        "n": n,
        "con_contesto": con_ctx,
        "senza_contesto": n - con_ctx,
        "turni_medi": round(statistics.mean(turni_n), 2) if turni_n else 0,
        "parole_contesto_medie": round(statistics.mean(parole_ctx), 1) if parole_ctx else 0,
        "parole_contesto_mediane": statistics.median(parole_ctx) if parole_ctx else 0,
        "turni_etichettati": round(etichettati / tot_t, 3) if tot_t else 0,
        "istruzioni_distinte": len(istruzioni),
        "frazione_istruzione_piu_comune": round(freq / con_ctx, 3) if con_ctx else 0,
        "istruzione_piu_comune": piu_comune,
    }


def leggi_prompt(a_):
    """Le quattro domande, con la risposta che i numeri danno."""
    p = []
    if a_["istruzioni_distinte"] <= 1 or a_["frazione_istruzione_piu_comune"] > 0.98:
        p.append(
            f"{a_['layout']}: l'istruzione e' COSTANTE su tutti gli esempi "
            f"({a_['istruzioni_distinte']} varianti distinte). Una feature "
            f"costante ha informazione mutua zero col target: il modello non "
            f"puo' usarla per abbassare la loss, quindi impara a ignorarla e "
            f"la legge come un delimitatore, non come un'istruzione. "
            f"Riformularla meglio NON serve: va resa variabile.")
    if a_["turni_etichettati"] < 0.9:
        p.append(
            f"{a_['layout']}: solo il {a_['turni_etichettati']:.0%} dei turni di "
            f"contesto ha l'etichetta del parlante. Senza, il modello non puo' "
            f"sapere che i turni si alternano ne' a chi tocca rispondere. "
            f"Il CSV ha la colonna `speaker`: l'informazione c'e' e non e' usata.")
    if a_["parole_contesto_medie"] < 30:
        p.append(
            f"{a_['layout']}: {a_['parole_contesto_medie']} parole di contesto in "
            f"media ({a_['turni_medi']} turni). E' parlato frammentario: puo' non "
            f"bastare a stabilire l'argomento. Prova ad allargare la finestra "
            f"(igiene_split.py --finestre) e vedi se la contrastiva si muove: se "
            f"non si muove, il contesto non e' il collo di bottiglia e lo sai.")
    if a_["senza_contesto"]:
        p.append(
            f"{a_['layout']}: {a_['senza_contesto']}/{a_['n']} item senza blocco "
            f"di contesto estraibile. Su quelli il delta di contesto e il CFG "
            f"non sono definiti e vengono esclusi dalle misure.")
    return p


# --------------------------------------------------------------------------- #
# [2] ablazione: contesto vero contro contesto falso
# --------------------------------------------------------------------------- #

def coppie_contesto(rows, seed=0):
    """(prompt_vero, prompt_falso, target). Il contesto falso viene da un ALTRO
    item, non da turni sintetici: cosi' i due prompt sono entrambi plausibili e
    ben formati, e la differenza fra le uscite misura la pertinenza e non la
    stranezza del testo."""
    rng = random.Random(seed)
    con = [(i, estrai_blocco_contesto(r["prompt"])) for i, r in enumerate(rows)]
    con = [(i, t) for i, t in con if t is not None]
    fuori = []
    for i, _ in con:
        j = i
        for _ in range(20):
            j = rng.choice(con)[0]
            if j != i:
                break
        alt = estrai_blocco_contesto(rows[j]["prompt"])
        falso = sostituisci_contesto(rows[i]["prompt"], alt[2])
        if falso is None or falso == rows[i]["prompt"]:
            continue
        fuori.append((rows[i]["prompt"], falso, rows[i]["target"]))
    return fuori


def ablazione(model, tok, torch, coppie, max_new, eot, batch=8, limite=64,
              temperature=0.0):
    """chrF++ fra uscita-con-contesto-vero e uscita-con-contesto-falso.

    temperature=0 (greedy) di proposito: col campionamento due uscite
    differiscono anche a parita' di prompt, e il numero misurerebbe la
    stocasticita' del decoder invece della sensibilita' al contesto.
    """
    import sacrebleu
    from evaluate_task import pulisci_generato

    sub = coppie[:limite]
    prev = tok.padding_side
    tok.padding_side = "left"

    # BUG CORRETTO. Qui c'era una generate() batched con left-padding: su
    # Gemma-3 (Gemma3ForConditionalGeneration) crasha dentro transformers
    # ("Tensors must have same number of dimensions"), quindi l'ablazione non
    # produceva nessun numero. Si passa da common.genera_una, che su Gemma-3 usa
    # il loop di decoding manuale (batch 1, greedy) e sugli altri modelli
    # continua a usare generate(). L'ablazione e' greedy per costruzione, quindi
    # il loop manuale calcola esattamente la stessa cosa.
    from common import genera_una

    def genera(testi):
        out = []
        for i, x in enumerate(testi, 1):
            txt = genera_una(model, tok, render_prompt(tok, x), max_new, eot)
            out.append(pulisci_generato(txt))
            print(f"    {i}/{len(testi)}", end="\r")
        return out

    try:
        veri = genera([c[0] for c in sub])
        print()
        falsi = genera([c[1] for c in sub])
        print()
    finally:
        tok.padding_side = prev

    def chrf(h, r):
        return sacrebleu.sentence_chrf(h, [r], word_order=2).score

    per_item = [chrf(v, f) if (v.strip() and f.strip()) else None
                for v, f in zip(veri, falsi)]
    validi = [x for x in per_item if x is not None]
    identici = sum(1 for v, f in zip(veri, falsi) if norm(v) == norm(f))
    refs = [c[2] for c in sub]
    chrf_vero = round(sacrebleu.corpus_chrf(veri, [refs], word_order=2).score, 2)
    chrf_falso = round(sacrebleu.corpus_chrf(falsi, [refs], word_order=2).score, 2)

    return {
        "n": len(validi),
        "chrf_tra_le_due_uscite": round(statistics.mean(validi), 2) if validi else None,
        "frazione_uscite_identiche": round(identici / len(sub), 3) if sub else None,
        "chrf_contro_riferimento_contesto_vero": chrf_vero,
        "chrf_contro_riferimento_contesto_falso": chrf_falso,
        "delta_chrf_vero_meno_falso": round(chrf_vero - chrf_falso, 2),
        "_lettura": "chrf_tra_le_due_uscite ALTO (>60) o frazione_identiche alta "
                    "= il modello ignora il contesto: cecita' misurata. "
                    "delta_chrf_vero_meno_falso <= 0 = il contesto reale non "
                    "aiuta nemmeno contro il riferimento, che e' la forma piu' "
                    "netta dello stesso risultato.",
        "esempi": [{"contesto_vero": v, "contesto_falso": f, "riferimento": r}
                   for v, f, r in list(zip(veri, falsi, refs))[:10]],
    }


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Il modello e' cieco al contesto? Tre controlli.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", default="/kaggle/working/split")
    ap.add_argument("--runs-dir", default="/kaggle/working/runs")
    ap.add_argument("--out", default="/kaggle/working/eval/diagnosi_cecita.json")
    ap.add_argument("--task", default="T3", nargs="+", choices=["T1", "T2", "T3"])
    ap.add_argument("--split", default="dev",
                    help="dev per default: la diagnosi non deve consumare il test")
    ap.add_argument("--solo-statico", action="store_true",
                    help="solo [1] e [3]: nessuna GPU")
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limite", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf-token", default=None)
    a = ap.parse_args()

    tasks = a.task if isinstance(a.task, list) else [a.task]
    rapporto = {"split_dir": a.split_dir, "split": a.split}

    print("=" * 74)
    print("[1] CONTATORI DI TRAINING  (troncamento e masking)")
    print("=" * 74)
    righe = contatori(a.runs_dir)
    if not righe:
        print(f"  nessun summary.json in {a.runs_dir}: salto")
    else:
        for r in righe:
            print(f"  {r['run']:<38} {r['layout']}  troncati "
                  f"{r['troncati_train']}/{r['train']} "
                  f"({r['frazione_troncati']:.1%})  mismatch "
                  f"{r['prefix_mismatch_train']}")
    problemi = leggi_contatori(righe)
    rapporto["contatori"] = {"righe": righe, "problemi": problemi,
                             "verificato": bool(righe)}
    if not righe:
        print("\n  ? NON VERIFICATO: senza summary.json non si puo' escludere il "
              "troncamento.\n     Questo controllo resta aperto: rilancialo dopo "
              "il primo run di training.")
    elif problemi:
        print("\n  ! PROBLEMI MECCANICI (da risolvere PRIMA di tutto il resto):")
        for p in problemi:
            print(f"    - {p}")
    else:
        print("\n  ok: nessun troncamento significativo, masking allineato.")
        print("     Il contesto ARRIVA al modello: se sembra cieco, non e' per "
              "questo.")

    print("\n" + "=" * 74)
    print("[3] ISPEZIONE DEI PROMPT")
    print("=" * 74)
    rapporto["prompt"] = {}
    for t in tasks:
        rows = load_split(a.split_dir, t, "train")
        an = analizza_prompt(rows, t)
        rapporto["prompt"][t] = an
        print(f"\n  {t}: {an['con_contesto']}/{an['n']} con contesto | "
              f"{an['turni_medi']} turni | {an['parole_contesto_medie']} parole | "
              f"turni etichettati {an['turni_etichettati']:.0%} | "
              f"istruzioni distinte {an['istruzioni_distinte']}")
        print(f"  --- un prompt reale di {t} " + "-" * 40)
        for riga in rows[0]["prompt"].split("\n"):
            print(f"  | {riga}")
        print(f"  | TARGET: {rows[0]['target']}")
        oss = leggi_prompt(an)
        rapporto["prompt"][t]["osservazioni"] = oss
        for o in oss:
            print(f"    - {o}")

    if a.solo_statico:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rapporto, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"\nSalvato {a.out}  (ablazione [2] non eseguita: --solo-statico)")
        return

    if not (a.model and a.adapter):
        print("\n[2] salto l'ablazione: servono --model e --adapter")
        return

    print("\n" + "=" * 74)
    print("[2] ABLAZIONE SUL CONTESTO  (contesto vero contro contesto falso)")
    print("=" * 74)
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from common import load_backbone, load_hf_token, scegli_dtype

    repo = resolve_model(a.model)
    token = load_hf_token(a.hf_token)
    tok = AutoTokenizer.from_pretrained(repo, token=token, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt, _ = scegli_dtype(repo)            # fp32 su Gemma senza bf16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dt)
    model, arch = load_backbone(repo, dt, quantization_config=quant,
                                device_map={"": 0}, token=token,
                                low_cpu_mem_usage=True,
                                attn_implementation="eager")
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    model.config.use_cache = True

    rapporto["ablazione"] = {}
    for t in tasks:
        rows = load_split(a.split_dir, t, a.split)
        coppie = coppie_contesto(rows, a.seed)
        if not coppie:
            print(f"  {t}: nessuna coppia costruibile (contesto non estraibile)")
            continue
        eot = id_fine_turno(tok, rows, verbose=False)
        print(f"\n  {t}: {len(coppie)} coppie, ne uso {min(a.limite, len(coppie))}")
        res = ablazione(model, tok, torch, coppie, MAX_NEW[t], eot,
                        batch=a.batch, limite=a.limite)
        rapporto["ablazione"][t] = res
        print(f"    chrF fra le due uscite      {res['chrf_tra_le_due_uscite']}")
        print(f"    uscite identiche            "
              f"{res['frazione_uscite_identiche']:.1%}")
        print(f"    chrF vs riferimento  vero   "
              f"{res['chrf_contro_riferimento_contesto_vero']}")
        print(f"                         falso  "
              f"{res['chrf_contro_riferimento_contesto_falso']}")
        print(f"    delta (vero - falso)        "
              f"{res['delta_chrf_vero_meno_falso']}")
        c = res["chrf_tra_le_due_uscite"] or 0
        d = res["delta_chrf_vero_meno_falso"]
        if c > 60 or (res["frazione_uscite_identiche"] or 0) > 0.3:
            print(f"    -> CIECO: cambiare il contesto non cambia l'uscita.")
        elif d <= 0:
            print(f"    -> CIECO in forma netta: il contesto reale non aiuta "
                  f"nemmeno contro il riferimento.")
        else:
            print(f"    -> il contesto ENTRA nella generazione. Il problema non "
                  f"e' la cecita': e' che ci vede male. Prompt e DPO sono la "
                  f"strada, non il troncamento.")
        print("\n    tre esempi (stesso item, contesto diverso):")
        for e in res["esempi"][:3]:
            print(f"      RIF   {e['riferimento']}")
            print(f"      vero  {e['contesto_vero']}")
            print(f"      falso {e['contesto_falso']}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rapporto, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\nSalvato {a.out}")


if __name__ == "__main__":
    main()
