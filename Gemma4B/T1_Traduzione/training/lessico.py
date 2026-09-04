#!/usr/bin/env python3
"""
lessico.py — estrae un lessico allineato italiano -> napoletano dai SOLI turni
di train, piu' l'inventario dei tipi lessicali dialettali.

Perche' serve
-------------
"Insegnare il dialetto termine per termine" richiede di sapere QUALI sono i
termini. Non esiste un glossario allegato al dataset, ma il dataset e' un corpus
parallelo allineato a livello di turno: da 2.568 coppie si ricava un lessico per
allineamento statistico (IBM Model 1, EM su P(nap|ita) con token NULL) senza
scrivere a mano nemmeno una voce.

L'output serve a tre cose diverse, tutte a valle:
  1. costruire il dataset dello stadio A2 (dati_lessicali.py);
  2. pesare la loss sui token dialettali (pesi_lessicali.py);
  3. misurare se il dialetto e' stato imparato (recall lessicale, italianismi).

Anti-leakage
------------
Il lessico e' un artefatto ADDESTRATO sui dati: se lo estrai da tutto il CSV,
dentro ci finiscono le corrispondenze osservate in dev e test, e ogni metrica
che lo usa e' contaminata. Con --split-dir le coppie vengono filtrate sulle
chiavi (conversazione, turn_index) presenti nei train.json dei tre layout.
La modalita' --solo-csv esiste per l'ispezione esplorativa e lo dichiara a
schermo: non usarla per produrre il lessico dei run finali.

Uso
---
    python lessico.py --csv dataset_finale.csv --split-dir /kaggle/working/split \\
        --out lessico_train.json
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LAYOUT_DIRS = ("layout1_traduzione_con_contesto", "layout2_completamento_turno",
               "layout3_replica_conversazionale")

# Le apostrofate sono la spina dorsale del napoletano scritto ('o, 'e, pe',
# 'nfatti): un \w+ le spezzerebbe e il lessico verrebbe fuori a pezzi.
TOK = re.compile(r"[a-zàèéìòóùâêîôû'\u2019\-]+")


def tokenizza(s):
    return TOK.findall(str(s).lower().replace("\u2019", "'"))


def chiavi_train(split_dir):
    """(conversazione, turn_index) dei turni che compaiono nel train di almeno
    un layout. Si leggono i JSON dello split invece di ri-splittare il CSV:
    l'unica definizione di 'train' resta quella di split_dataset.py."""
    chiavi = set()
    for d in LAYOUT_DIRS:
        p = Path(split_dir) / d / "train.json"
        if not p.exists():
            print(f"  ! manca {p}", file=sys.stderr)
            continue
        for r in json.loads(p.read_text(encoding="utf-8")):
            chiavi.add((r["conversazione"], int(r["turn_index"])))
    return chiavi


def ibm1(coppie, iterazioni=12):
    """EM su P(nap | ita). Ritorna il dizionario di traduzione lessicale.

    Model 1 ignora l'ordine delle parole: e' un limite che qui non morde, perche'
    italiano e napoletano hanno sintassi vicinissima e le frasi sono corte
    (mediana 4 parole). Model 2 o HMM aggiungerebbero un modello di distorsione
    per un guadagno trascurabile su 2.500 coppie.
    """
    vocab_nap = {w for _, n in coppie for w in n}
    t = defaultdict(lambda: 1.0 / len(vocab_nap))
    for _ in range(iterazioni):
        conta, totale = defaultdict(float), defaultdict(float)
        for it, nap in coppie:
            sorgente = ["<NULL>"] + it
            for nw in nap:
                z = sum(t[(nw, sw)] for sw in sorgente)
                if z == 0:
                    continue
                for sw in sorgente:
                    d = t[(nw, sw)] / z
                    conta[(nw, sw)] += d
                    totale[sw] += d
        t = defaultdict(float, {k: v / totale[k[1]] for k, v in conta.items()
                                if totale[k[1]] > 0})
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split-dir", default=None,
                    help="cartella con layout1_*/layout2_*/layout3_*: filtra sul train")
    ap.add_argument("--solo-csv", action="store_true",
                    help="usa TUTTO il csv (dev e test compresi): solo esplorazione")
    ap.add_argument("--out", default="lessico_train.json")
    ap.add_argument("--p-min", type=float, default=0.15,
                    help="probabilita' minima di traduzione")
    ap.add_argument("--cooc-min", type=int, default=3,
                    help="co-occorrenze minime: sotto le 3 il lessico si riempie di rumore")
    ap.add_argument("--escludi-fonte", nargs="*", default=[],
                    help="fonti da escludere, es. --escludi-fonte gemma4")
    a = ap.parse_args()

    import pandas as pd

    if not a.split_dir and not a.solo_csv:
        sys.exit("Serve --split-dir (oppure --solo-csv se stai solo esplorando).")

    df = pd.read_csv(a.csv)
    n0 = len(df)
    if a.escludi_fonte:
        df = df[~df["fonte"].isin(a.escludi_fonte)]
        print(f"Fonti escluse {a.escludi_fonte}: {n0} -> {len(df)} righe")

    if a.split_dir:
        chiavi = chiavi_train(a.split_dir)
        prima = len(df)
        df = df[[(c, int(i)) in chiavi
                 for c, i in zip(df["conversazione"], df["turn_index"])]]
        print(f"Filtro sul train dello split: {prima} -> {len(df)} righe "
              f"({len(chiavi)} chiavi di train)")
        if len(df) < 0.2 * prima:
            print("  ! pochissime righe sopravvissute: le chiavi del CSV e dello "
                  "split non combaciano, controlla conversazione/turn_index")
    else:
        print("!!! --solo-csv: il lessico include dev e test. Artefatto NON "
              "utilizzabile per training o valutazione.")

    coppie = [(tokenizza(i), tokenizza(n))
              for i, n in zip(df["italiano"], df["napoletano"])]
    coppie = [(i, n) for i, n in coppie if i and n]
    print(f"Coppie allineate: {len(coppie)}")

    freq_it, freq_nap, cooc = Counter(), Counter(), Counter()
    for it, nap in coppie:
        for sw in set(it):
            freq_it[sw] += 1
            for nw in set(nap):
                cooc[(sw, nw)] += 1
        for nw in set(nap):
            freq_nap[nw] += 1

    # Tipi dialettali: forme napoletane che non esistono sul lato italiano del
    # corpus. E' un'approssimazione (una forma napoletana omografa dell'italiano
    # non viene contata) ma e' quella che si puo' calcolare senza un dizionario
    # esterno, e cattura il segnale che ci interessa: 'o, nun, ca, pecche',
    # accussi', cchiu', quanno.
    tipi_dialettali = sorted({w for w in freq_nap if w not in freq_it},
                            key=lambda w: -freq_nap[w])
    massa = sum(freq_nap[w] for w in tipi_dialettali)
    print(f"Tipi dialettali: {len(tipi_dialettali)} "
          f"({massa} occorrenze, {massa/max(1,sum(freq_nap.values())):.1%} dei token napoletani)")

    t = ibm1(coppie)
    voci = []
    for (nw, sw), p in t.items():
        if sw == "<NULL>" or nw == sw or p < a.p_min or cooc[(sw, nw)] < a.cooc_min:
            continue
        voci.append({"italiano": sw, "napoletano": nw, "p": round(p, 4),
                     "cooc": cooc[(sw, nw)], "freq_it": freq_it[sw],
                     "freq_nap": freq_nap[nw],
                     "dialettale": nw in set(tipi_dialettali)})
    voci.sort(key=lambda v: (-v["cooc"], -v["p"]))
    print(f"Voci di lessico: {len(voci)} "
          f"({sum(v['dialettale'] for v in voci)} con forma dialettale)")

    # Una entrata italiana puo' avere piu' rese (di -> 'e / 'o): si tengono
    # tutte, e chi consuma il lessico decide se prendere l'argmax o l'insieme.
    fuori = {
        "_meta": {"fonte_csv": a.csv, "split_dir": a.split_dir,
                  "solo_csv": a.solo_csv, "fonti_escluse": a.escludi_fonte,
                  "coppie": len(coppie), "p_min": a.p_min, "cooc_min": a.cooc_min},
        "tipi_dialettali": tipi_dialettali,
        "freq_nap": {w: freq_nap[w] for w in tipi_dialettali},
        # Vocabolario napoletano COMPLETO del train (non solo i tipi dialettali):
        # serve a misurare le forme inedite, cioe' le parole che il modello
        # produce e che non compaiono da nessuna parte nei dati di training.
        # Senza questo, forme inventate per analogia morfologica come
        # "puparuolo" o "cusinajo" passano per napoletano valido in ogni metrica.
        "vocabolario_nap": dict(freq_nap),
        "lessico": voci,
    }
    Path(a.out).write_text(json.dumps(fuori, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print(f"\nScritto {a.out}")
    print("Prime 25 voci:")
    for v in voci[:25]:
        print(f"  {v['italiano']:>10s} -> {v['napoletano']:<12s} "
              f"p={v['p']:.2f} cooc={v['cooc']}")


if __name__ == "__main__":
    main()
