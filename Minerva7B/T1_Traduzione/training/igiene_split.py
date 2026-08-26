#!/usr/bin/env python3
"""
igiene_split.py — riscrive lo split di T3 (e T2) senza aggiungere dati.

Tre patologie del corpus che l'addestramento attuale trasforma in
comportamento appreso, e che si correggono ricombinando i dati che hai:

1. I RISCONTRI COME TARGET.
   nel corpus:  <=1 parola 20.9%   |  mh    95 volte
                <=2 parole 33.7%   |  si'   71
                <=3 parole 43.8%   |  mhmh  58
                                   |  no    32
   Addestrare su questi insegna che una replica senza contenuto e' corretta, e
   il modello media fra due distribuzioni molto diverse: "decidi se serve un
   riscontro" e "produci un contenuto". Due rimedi possibili:
     --tipo-turno   li tiene, ma sotto un marcatore nel prompt, cosi' il modello
                    impara QUANDO un riscontro e' giusto invece di mediare
     --filtra-riscontri  li toglie dal train (non da dev/test: la valutazione
                    resta sulla distribuzione reale)

2. I DUPLICATI. 495 target su 2568 sono duplicati esatti. "mh" pesa 95 volte
   quanto un turno contenutistico. `--sottocampiona-duplicati` tiene al massimo
   ceil(sqrt(freq)) copie per target: e' il peso 1/sqrt(freq) ottenuto per
   sottocampionamento, senza toccare il Trainer.

3. UNA SOLA FINESTRA DI CONTESTO. Ogni item ha 3 turni di contesto, sempre.
   `--finestre 1,2,3,4` genera un item per ogni ampiezza dagli stessi turni:
   ~3x le istanze, zero dati nuovi, e insegna robustezza all'ampiezza del
   contesto - utile anche in inferenza, dove non e' garantito che ce ne siano 3.

Sul marcatore di tipo turno
---------------------------
Bucket derivato dalla lunghezza del target:

    riscontro  <= 2 parole      breve  3-6      medio  7-12      lungo  13+

In TRAIN il bucket viene dal target reale. In DEV e TEST il target non si puo'
guardare, quindi il bucket viene CAMPIONATO dalla distribuzione empirica del
train con seme fisso. Questo e' il punto: la valutazione resta onesta (il
modello non vede la lunghezza del riferimento) e la distribuzione delle
lunghezze generate diventa compatibile con l'umano per costruzione, che e'
quello che fa crollare `controllo_solo_lunghezza` - 0.776 su 0.80 di accuracy
avversariale nel run misurato, cioe' il 97% del potere discriminante.

E' controllable generation standard (Kikuchi et al. 2016; Fan et al. 2018).

Sicurezza
---------
Scrive in una cartella NUOVA: lo split originale non viene toccato, e i due
arm sono confrontabili perche' esistono entrambi. Prima di scrivere verifica che
`estrai_blocco_contesto` e `rimuovi_contesto` funzionino ancora sui prompt
modificati: se il marcatore rompe il parsing, contesto_metrica.py e
decodifica_contestuale.py darebbero numeri plausibili e sbagliati, quindi lo
script si ferma invece di procedere.

Uso
---
    python igiene_split.py --split-dir /kaggle/working/split \\
        --out-dir /kaggle/working/split_igiene \\
        --tipo-turno --sottocampiona-duplicati --finestre 1,2,3,4

    # variante minimale: solo il marcatore
    python igiene_split.py --split-dir ... --out-dir ... --tipo-turno
"""

import argparse
import json
import math
import random
import shutil
import statistics
from collections import Counter
from pathlib import Path

from Minerva7B.T1_Traduzione.training.common import LAYOUT_DIRS, load_split
from Minerva7B.T1_Traduzione.training.contesto_metrica import (_e_separatore, estrai_blocco_contesto,
                              sostituisci_contesto)
from Minerva7B.T1_Traduzione.training.decodifica_contestuale import rimuovi_contesto

MARCATORE = "Tipo di turno: {b}."


def bucket(n_parole):
    if n_parole <= 2:
        return "riscontro"
    if n_parole <= 6:
        return "breve"
    if n_parole <= 12:
        return "medio"
    return "lungo"


def inserisci_marcatore(prompt, b):
    """Il marcatore va DOPO il separatore che chiude il contesto.

    Non prima: estrai_blocco_contesto delimita il blocco fra l'intestazione e il
    primo separatore (riga vuota o trattini), quindi una riga inserita a indice
    i1 finisce DENTRO i turni di contesto - verrebbe letta come un turno, e
    sostituisci_contesto la cancellerebbe. Il delta di contesto e il CFG
    misurerebbero la cosa sbagliata senza dare errore.

    Non all'inizio: sposterebbe l'indice dell'intestazione. Non alla fine: deve
    stare adiacente all'istruzione, dove il modello lo legge prima di generare.
    """
    trovato = estrai_blocco_contesto(prompt)
    if trovato is None:
        return None
    _, i1, _ = trovato
    righe = prompt.split("\n")
    k = i1
    while k < len(righe) and _e_separatore(righe[k]):
        k += 1                       # oltre il separatore, prima dell'istruzione
    righe.insert(k, MARCATORE.format(b=b))
    return "\n".join(righe)


def finestra(prompt, n):
    """Lo stesso item con solo gli ultimi n turni di contesto."""
    trovato = estrai_blocco_contesto(prompt)
    if trovato is None:
        return None
    _, _, turni = trovato
    if n >= len(turni):
        return prompt if n == len(turni) else None
    return sostituisci_contesto(prompt, turni[-n:])


def sottocampiona(rows, seed):
    """Tiene al massimo ceil(sqrt(freq)) item per ogni target distinto."""
    rng = random.Random(seed)
    per_target = {}
    for r in rows:
        per_target.setdefault(r["target"].strip().lower(), []).append(r)
    fuori, tolti = [], 0
    for _, gruppo in per_target.items():
        tetto = math.ceil(math.sqrt(len(gruppo)))
        if len(gruppo) <= tetto:
            fuori.extend(gruppo)
        else:
            fuori.extend(rng.sample(gruppo, tetto))
            tolti += len(gruppo) - tetto
    rng.shuffle(fuori)
    return fuori, tolti


def statistiche(rows, etichetta):
    L = [len(r["target"].split()) for r in rows]
    c = Counter(bucket(x) for x in L)
    return {"nome": etichetta, "n": len(rows),
            "lunghezza_media": round(statistics.mean(L), 2) if L else 0,
            "lunghezza_mediana": statistics.median(L) if L else 0,
            "bucket": {k: round(v / len(rows), 3) for k, v in c.items()} if rows else {}}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", default="/kaggle/working/split")
    ap.add_argument("--out-dir", default="/kaggle/working/split_igiene")
    ap.add_argument("--task", default="T3", choices=["T2", "T3"], nargs="+")
    ap.add_argument("--tipo-turno", action="store_true")
    ap.add_argument("--filtra-riscontri", action="store_true",
                    help="toglie i target <=2 parole dal TRAIN (dev/test intatti). "
                         "Alternativa a --tipo-turno, non complementare")
    ap.add_argument("--sottocampiona-duplicati", action="store_true")
    ap.add_argument("--finestre", default=None,
                    help="es. 1,2,3,4 — un item per ogni ampiezza di contesto")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    tasks = a.task if isinstance(a.task, list) else [a.task]
    if a.tipo_turno and a.filtra_riscontri:
        print("! --tipo-turno e --filtra-riscontri fanno la stessa cosa in due "
              "modi opposti. Con entrambi attivi il marcatore 'riscontro' non "
              "compare mai in train ma verra' campionato in test: incoerente.")
        return

    out = Path(a.out_dir)
    src = Path(a.split_dir)
    rapporto = {"origine": str(src), "opzioni": vars(a), "task": {}}
    finestre = ([int(x) for x in a.finestre.split(",")] if a.finestre else None)
    rng_bucket = random.Random(a.seed)

    for t in tasks:
        d = out / LAYOUT_DIRS[t]
        d.mkdir(parents=True, exist_ok=True)
        rows_train = load_split(str(src), t, "train")
        dist_bucket = Counter(bucket(len(r["target"].split()))
                              for r in rows_train)
        chiavi = sorted(dist_bucket)
        pesi = [dist_bucket[k] for k in chiavi]
        info_t = {"distribuzione_bucket_train":
                  {k: round(dist_bucket[k] / len(rows_train), 3) for k in chiavi}}

        for nome in ("train", "dev", "test"):
            rows = load_split(str(src), t, nome)
            prima = statistiche(rows, f"{t}/{nome} prima")
            n_ctx_persi = 0

            # 1. finestre variabili: solo sul train
            if finestre and nome == "train":
                espanse = []
                for r in rows:
                    for n in finestre:
                        p = finestra(r["prompt"], n)
                        if p is None:
                            continue
                        q = dict(r)
                        q["prompt"] = p
                        q["finestra_contesto"] = n
                        espanse.append(q)
                if espanse:
                    rows = espanse
                else:
                    n_ctx_persi = 1

            # 2. filtro riscontri: solo sul train
            if a.filtra_riscontri and nome == "train":
                rows = [r for r in rows if len(r["target"].split()) > 2]

            # 3. sottocampionamento dei duplicati: solo sul train
            tolti = 0
            if a.sottocampiona_duplicati and nome == "train":
                rows, tolti = sottocampiona(rows, a.seed)

            # 4. marcatore di tipo turno
            senza_marcatore = 0
            if a.tipo_turno:
                nuove = []
                for r in rows:
                    if nome == "train":
                        b = bucket(len(r["target"].split()))
                    else:
                        # dev/test: il target non si guarda. Bucket campionato
                        # dalla distribuzione del TRAIN, seme fisso.
                        b = rng_bucket.choices(chiavi, weights=pesi, k=1)[0]
                    p = inserisci_marcatore(r["prompt"], b)
                    if p is None:
                        senza_marcatore += 1
                        nuove.append(r)
                        continue
                    q = dict(r)
                    q["prompt"] = p
                    q["tipo_turno"] = b
                    nuove.append(q)
                rows = nuove

            # 5. verifica che i parser a valle funzionino ancora
            campione = rows[:min(32, len(rows))]
            rotti_ctx = sum(1 for r in campione
                            if estrai_blocco_contesto(r["prompt"]) is None)
            rotti_rm = sum(1 for r in campione
                           if rimuovi_contesto(r["prompt"]) is None)
            if rotti_ctx or rotti_rm:
                print(f"ERRORE su {t}/{nome}: il marcatore rompe il parsing "
                      f"({rotti_ctx}/{len(campione)} estrai_blocco_contesto, "
                      f"{rotti_rm}/{len(campione)} rimuovi_contesto).")
                print("  contesto_metrica.py e decodifica_contestuale.py "
                      "darebbero numeri sbagliati. Niente scritto.")
                return

            json.dump(rows, open(d / f"{nome}.json", "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            info_t[nome] = {"prima": prima,
                            "dopo": statistiche(rows, f"{t}/{nome} dopo"),
                            "duplicati_tolti": tolti,
                            "prompt_senza_marcatore": senza_marcatore,
                            "contesto_non_estraibile": n_ctx_persi}
            print(f"  {t}/{nome}: {prima['n']} -> {len(rows)}"
                  + (f"  (-{tolti} duplicati)" if tolti else "")
                  + (f"  ! {senza_marcatore} senza marcatore"
                     if senza_marcatore else ""))
        rapporto["task"][t] = info_t

    # A2 e gli altri layout vanno copiati intatti, altrimenti lo split nuovo e'
    # incompleto e i run che li usano fallirebbero con un errore oscuro.
    for k, sub in LAYOUT_DIRS.items():
        if k in tasks:
            continue
        s = src / sub
        if s.exists():
            shutil.copytree(s, out / sub, dirs_exist_ok=True)
            print(f"  {k}: copiato intatto")

    (out / "igiene.json").write_text(
        json.dumps(rapporto, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nScritto {out}")
    print(f"Rapporto: {out}/igiene.json")
    print(f"Usalo con --split-dir {out} negli script di finetune E in "
          f"evaluate_task.py: i prompt devono essere gli stessi da entrambe "
          f"le parti.")


if __name__ == "__main__":
    main()
