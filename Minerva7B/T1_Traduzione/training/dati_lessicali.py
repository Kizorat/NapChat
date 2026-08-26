#!/usr/bin/env python3
"""
dati_lessicali.py — costruisce il dataset dello STADIO A2 (iniezione lessicale),
il pezzo che manca fra il continued pretraining e l'SFT sui tre task.

Il problema che risolve
-----------------------
Lo stadio A (pretrain_dialect.py) sposta la distribuzione con language modeling
puro, ma non insegna nessuna CORRISPONDENZA: il modello vede napoletano, non
vede che "pecche'" sta al posto di "perche'". L'SFT sui task insegna la
corrispondenza ma a livello di frase intera, dove il segnale lessicale e'
diluito su decine di token e la strada piu' facile per abbassare la loss e'
copiare l'italiano. Nel mezzo manca la supervisione a livello di TERMINE.

Tre forme di item, dalla piu' isolata alla piu' contestuale:

  glossa       "In napoletano «perche'» si dice ___"  -> pecche'
               Ancora la corrispondenza. Da sola produrrebbe un glossario
               parlante, per questo e' la quota minoritaria.

  cloze        contesto napoletano + frase italiana + resa napoletana con un
               buco -> la parola dialettale mancante.
               E' l'item centrale: il termine va prodotto DENTRO la frase e
               dentro la conversazione, quindi il modello impara anche quando
               quella forma e' appropriata, che e' esattamente la richiesta del
               prompt con contesto pragmatico.

  turno_breve  turni corti (<= --max-parole) con almeno una parola dialettale,
               tradotti per intero. Fa da ponte verso T1/T2/T3: stessa forma di
               compito, ma su una finestra dove il termine dialettale pesa molto
               nella loss.

Gli item escono nello stesso formato dei tre layout ({prompt, target, layout,
conversazione, turn_index}), quindi ChatDataset e il resto di common.py li
digeriscono senza modifiche. Serve solo aggiungere la voce in LAYOUT_DIRS:

    LAYOUT_DIRS["A2"] = "stadio_a2_lessico"

Anti-leakage
------------
Solo turni di train (chiavi lette dai train.json dello split), lessico
estratto dal train, e contesto conversazionale costruito solo con turni di
train: se il contesto pescasse dai turni vicini senza filtro, i target di
dev/test entrerebbero nel training come testo di contesto.
Il dev di A2 e' un campione tenuto fuori dagli item di train (--quota-dev),
serve solo per l'early stopping dello stadio A2.

Uso
---
    python lessico.py --csv dataset_finale.csv --split-dir split --out lessico_train.json
    python dati_lessicali.py --csv dataset_finale.csv --split-dir split \\
        --lessico lessico_train.json --out-dir split/stadio_a2_lessico
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

LAYOUT_DIRS = ("layout1_traduzione_con_contesto", "layout2_completamento_turno",
               "layout3_replica_conversazionale")
TOK = re.compile(r"[a-zàèéìòóùâêîôû'\u2019\-]+")

ISTR_GLOSSA = ("Il napoletano usa una forma diversa dall'italiano per questa "
               "parola. Scrivi soltanto la forma napoletana.")
ISTR_CLOZE = ("Completa la frase napoletana inserendo la parola che manca al "
              "posto di ___. Scrivi soltanto quella parola.")
ISTR_TURNO = "Traduci in napoletano."


def norm(s):
    return str(s).replace("\u2019", "'")


def tokenizza(s):
    return TOK.findall(norm(s).lower())


def chiavi_train(split_dir):
    chiavi = set()
    for d in LAYOUT_DIRS:
        p = Path(split_dir) / d / "train.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text(encoding="utf-8")):
            chiavi.add((r["conversazione"], int(r["turn_index"])))
    if not chiavi:
        sys.exit(f"Nessun train.json trovato sotto {split_dir}")
    return chiavi


def costruisci_contesto(righe_conv, i, chiavi_ok, n=3):
    """Fino a n turni napoletani precedenti, ma solo quelli di train.

    Se un turno precedente non e' di train si TRONCA la finestra invece di
    saltarlo: un contesto con un buco in mezzo e' una conversazione che non e'
    mai esistita, e il modello impara a raccordare turni non adiacenti.
    """
    fuori = []
    for j in range(i - 1, max(-1, i - 1 - n), -1):
        r = righe_conv[j]
        if (r["conversazione"], int(r["turn_index"])) not in chiavi_ok:
            break
        fuori.append(f"{r['speaker']}: {norm(r['napoletano'])}")
    return list(reversed(fuori))


def blocco_contesto(ctx):
    if not ctx:
        return ""
    return "Conversazione finora (in napoletano):\n" + "\n".join(ctx) + "\n\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--lessico", required=True, help="output di lessico.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-parole", type=int, default=8,
                    help="soglia per gli item turno_breve")
    ap.add_argument("--cloze-per-turno", type=int, default=2,
                    help="quanti buchi diversi generare dallo stesso turno")
    ap.add_argument("--min-freq-glossa", type=int, default=3,
                    help="frequenza minima nel train perche' un termine diventi glossa")
    ap.add_argument("--richiedi-contesto", action="store_true",
                    help="scarta gli item di cloze e turno_breve senza blocco di "
                         "contesto. Con un train sparso la finestra si tronca spesso "
                         "e la maggioranza degli item finisce senza contesto: quegli "
                         "item insegnano a produrre napoletano senza guardare indietro, "
                         "che e' il fallimento osservato su T2 e T3")
    ap.add_argument("--quota-glossa", type=float, default=None,
                    help="con --richiedi-contesto, tetto alla quota di item glossa "
                         "(che per natura non hanno contesto), es. 0.10")
    ap.add_argument("--quota-dev", type=float, default=0.08)
    ap.add_argument("--escludi-fonte", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import pandas as pd

    rnd = random.Random(a.seed)
    lex = json.loads(Path(a.lessico).read_text(encoding="utf-8"))
    if lex["_meta"].get("solo_csv"):
        sys.exit("Il lessico e' stato estratto con --solo-csv (include dev/test). "
                 "Rigeneralo con --split-dir prima di costruire i dati.")
    dialettali = set(lex["tipi_dialettali"])
    freq_nap = lex["freq_nap"]

    df = pd.read_csv(a.csv)
    if a.escludi_fonte:
        df = df[~df["fonte"].isin(a.escludi_fonte)]
    df = df.sort_values(["conversazione", "turn_index"]).reset_index(drop=True)
    chiavi = chiavi_train(a.split_dir)

    per_conv = {c: g.to_dict("records") for c, g in df.groupby("conversazione")}
    items, conta = [], Counter()

    # ---- glossa ------------------------------------------------------------
    # Una voce per forma napoletana: si prende l'italiano con la co-occorrenza
    # piu' alta, cosi' "'o" non genera tre item quasi identici (il/lo/nel).
    migliore = {}
    for v in lex["lessico"]:
        if not v["dialettale"] or freq_nap.get(v["napoletano"], 0) < a.min_freq_glossa:
            continue
        pre = migliore.get(v["napoletano"])
        if pre is None or v["cooc"] > pre["cooc"]:
            migliore[v["napoletano"]] = v
    for nw, v in migliore.items():
        items.append({"layout": "A2", "tipo": "glossa",
                      "conversazione": "_lessico", "turn_index": -1,
                      "prompt": f"{ISTR_GLOSSA}\n\nItaliano: {v['italiano']}\nNapoletano:",
                      "target": nw})
        conta["glossa"] += 1

    # ---- cloze e turno_breve ----------------------------------------------
    for conv, righe in per_conv.items():
        for i, r in enumerate(righe):
            chiave = (r["conversazione"], int(r["turn_index"]))
            if chiave not in chiavi:
                continue
            ita, nap = norm(r["italiano"]).strip(), norm(r["napoletano"]).strip()
            if not ita or not nap:
                continue
            parole = nap.split()
            idx_dial = [k for k, p in enumerate(parole)
                        if TOK.findall(p.lower()) and TOK.findall(p.lower())[0] in dialettali]
            if not idx_dial:
                continue                      # nessun segnale dialettale: si salta
            ctx = costruisci_contesto(righe, i, chiavi)

            for k in rnd.sample(idx_dial, min(a.cloze_per_turno, len(idx_dial))):
                bucato = list(parole)
                bucato[k] = "___"
                items.append({
                    "layout": "A2", "tipo": "cloze",
                    "conversazione": conv, "turn_index": int(r["turn_index"]),
                    "prompt": (f"{ISTR_CLOZE}\n\n{blocco_contesto(ctx)}"
                               f"Italiano: {ita}\nNapoletano: {' '.join(bucato)}\n"
                               f"Parola mancante:"),
                    "target": parole[k]})
                conta["cloze"] += 1

            if len(parole) <= a.max_parole:
                items.append({
                    "layout": "A2", "tipo": "turno_breve",
                    "conversazione": conv, "turn_index": int(r["turn_index"]),
                    "prompt": (f"{ISTR_TURNO}\n\n{blocco_contesto(ctx)}"
                               f"Italiano: {ita}\nNapoletano:"),
                    "target": nap})
                conta["turno_breve"] += 1

    # --- filtro sul contesto -----------------------------------------------
    if a.richiedi_contesto:
        prima = len(items)
        items = [i for i in items
                 if i["tipo"] == "glossa" or "Conversazione finora" in i["prompt"]]
        conta_f = Counter(i["tipo"] for i in items)
        print(f"\n--richiedi-contesto: {prima} -> {len(items)} item "
              f"({dict(conta_f)})")
        if a.quota_glossa is not None:
            gl = [i for i in items if i["tipo"] == "glossa"]
            altri = [i for i in items if i["tipo"] != "glossa"]
            tetto = int(a.quota_glossa * len(altri) / max(1e-9, 1 - a.quota_glossa))
            if len(gl) > tetto:
                gl = rnd.sample(gl, tetto)
                print(f"  glossa ridotta a {len(gl)} item "
                      f"(tetto {a.quota_glossa:.0%} del totale)")
            items = gl + altri
        senza = sum(1 for i in items if "Conversazione finora" not in i["prompt"])
        print(f"  item senza contesto residui: {senza}/{len(items)} "
              f"({senza/max(1,len(items)):.1%}) — sono le glossa")

    rnd.shuffle(items)
    n_dev = int(len(items) * a.quota_dev)
    dev, train = items[:n_dev], items[n_dev:]

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for nome, dati in (("train", train), ("dev", dev)):
        (out / f"{nome}.json").write_text(
            json.dumps(dati, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"Item per tipo: {dict(conta)}")
    print(f"Totale {len(items)} -> train {len(train)}, dev {len(dev)}")
    print(f"Scritto in {out}")
    print("\nEsempio cloze:\n" + "-" * 60)
    ex = next(i for i in items if i["tipo"] == "cloze")
    print(ex["prompt"] + f"\n--- TARGET: {ex['target']}")


if __name__ == "__main__":
    main()
