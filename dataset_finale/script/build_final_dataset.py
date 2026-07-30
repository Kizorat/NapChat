#!/usr/bin/env python3
"""
Fase 1.2 (chiusura) — Dataset finale IT-NAP dalla fusione bozza + golden.

Legge il report di coerenza prodotto da golden_consistency.py e costruisce il
dataset definitivo applicando, turno per turno, la decisione gia' presa li':

    Sostituire con golden = True   ->  si prende il napoletano del GOLDEN
    Sostituire con golden = False  ->  si tiene il napoletano di GEMMA4

Ogni turno porta con se' il campo `fonte` ("golden" | "gemma4"), cosi' nella
Fase 1.4 si sa quali coppie vengono dalla validazione umana e quali dalla bozza
LLM (distinzione gold / silver) senza doverlo ricalcolare.

NOTA sul risultato: i turni con False sono per costruzione quelli con
similarita' sopra soglia, cioe' dove le due rese sono quasi identiche. Il
dataset che ne esce e' quindi il golden con un numero limitato di sostituzioni
minime. Con --solo-golden si ottiene il golden puro, per confronto.

Input (default):
    validation_gemma_golden/results/coerenza_golden_vs_gemma.csv   decisioni
    golden_translate/golden_validator.json                          struttura+metadati
Output (default):
    dataset_finale/dataset_finale.json   (stesso formato degli altri dataset)
    dataset_finale/dataset_finale.csv

Uso tipico (dalla radice del progetto):
    py -3.12 dataset_finale/script/build_final_dataset.py
    # rifai la scelta con un'altra soglia, ignorando la colonna del report:
    py -3.12 dataset_finale/script/build_final_dataset.py --soglia 70
    # mostra piu' esempi dei turni tenuti da gemma4:
    py -3.12 dataset_finale/script/build_final_dataset.py --anteprima 25
    # schema identico agli altri dataset, senza il campo fonte:
    py -3.12 dataset_finale/script/build_final_dataset.py --no-fonte
"""

import argparse, csv, json, os, sys

# Questo script sta in dataset_finale/script/, ma il serializzatore condiviso
# (stesso formato .json/.csv degli altri dataset) sta in scripts/ alla radice:
# senza questo aggancio l'import fallisce con ModuleNotFoundError.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from build_conversation_dataset import write_json, CSV_FIELDS

DEFAULT_COERENZA = os.path.join("validation_gemma_golden", "results",
                                "coerenza_golden_vs_gemma.csv")
DEFAULT_STRUTTURA = os.path.join("golden_translate", "golden_validator.json")
OUT_DIR = "dataset_finale"
DEFAULT_OUT = os.path.join(OUT_DIR, "dataset_finale.json")

OUT_CSV_FIELDS = CSV_FIELDS + ["fonte", "similarita"]


def load_samples(path):
    """Carica il dataset in qualsiasi forma: oggetto JSON, array JSON o JSONL."""
    raw = open(path, encoding="utf-8").read().strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        return [json.loads(l) for l in raw.splitlines() if l.strip()]


def load_decisioni(path, soglia):
    """Dal CSV di coerenza a {(conversazione, turn_index): riga}.

    Se `soglia` e' data, il booleano viene ricalcolato dalla colonna
    `similarita` invece di fidarsi della colonna `sostituire` del report.
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        righe = list(csv.DictReader(f))
    if not righe:
        sys.exit(f"ERRORE: nessuna riga in {path!r}")

    mancanti = [c for c in ("conversazione", "turn_index", "gemma", "golden",
                            "similarita", "sostituire") if c not in righe[0]]
    if mancanti:
        sys.exit(f"ERRORE: in {path!r} mancano le colonne: {', '.join(mancanti)}\n"
                 f"Rigeneralo con:  py -3.12 validation_gemma_golden/"
                 f"golden_consistency.py --csv")

    dec = {}
    for r in righe:
        sim = float(r["similarita"])
        r["_sim"] = sim
        r["_sostituire"] = (sim < soglia) if soglia is not None \
            else (r["sostituire"].strip().lower() == "true")
        dec[(r["conversazione"], int(r["turn_index"]))] = r
    return dec


def fondi(samples, decisioni, con_fonte):
    """Applica la decisione a ogni turno; ritorna (campioni, righe piatte, conteggi)."""
    n = {"golden": 0, "gemma4": 0}
    senza_decisione, italiano_diverso = [], []
    out_samples, flat = [], []

    for s in samples:
        turni = []
        for t in s["turni"]:
            key = (s["id"], int(t["turn_index"]))
            d = decisioni.get(key)
            if d is None:
                senza_decisione.append(f"{key[0]} turno {key[1]}")
                continue
            if d["italiano"] != t["italiano"]:
                italiano_diverso.append(f"{key[0]} turno {key[1]}")

            fonte = "golden" if d["_sostituire"] else "gemma4"
            nap = d["golden"] if d["_sostituire"] else d["gemma"]
            n[fonte] += 1

            turno = {"turn_index": int(t["turn_index"]), "tu_id": t.get("tu_id", ""),
                     "speaker": t["speaker"], "italiano": t["italiano"],
                     "napoletano": nap}
            if con_fonte:
                turno["fonte"] = fonte
            turni.append(turno)

            flat.append({
                "conversazione": s["id"], "regione": s.get("regione", ""),
                "macro_regione": s.get("macro_regione", ""),
                "languages": s.get("languages", ""),
                "turn_index": turno["turn_index"], "tu_id": turno["tu_id"],
                "speaker": turno["speaker"], "italiano": turno["italiano"],
                "napoletano": nap, "note": "",
                "fonte": fonte, "similarita": round(d["_sim"], 2),
            })

        out_samples.append({
            "id": s["id"], "regione": s.get("regione", ""),
            "macro_regione": s.get("macro_regione", ""),
            "languages": s.get("languages", ""),
            "n_turni_italiani": len(turni), "turni": turni,
        })

    return out_samples, flat, n, senza_decisione, italiano_diverso


def anteprima_gemma(flat, decisioni, quanti):
    """I turni tenuti da gemma4 che piu' si discostano dal golden: i piu' a rischio.

    Sono quelli appena sopra soglia: la metrica li ha giudicati equivalenti, ma
    sono anche gli unici punti in cui il dataset finale si allontana dal testo
    validato a mano. Vale la pena guardarli.
    """
    cand = []
    for r in flat:
        if r["fonte"] != "gemma4":
            continue
        d = decisioni[(r["conversazione"], r["turn_index"])]
        if d["gemma"] != d["golden"]:
            cand.append((r["similarita"], r["conversazione"], r["turn_index"],
                         d["gemma"], d["golden"]))
    cand.sort()
    return cand[:quanti], len(cand)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coerenza", default=DEFAULT_COERENZA,
                    help=f"CSV del report di coerenza (default: {DEFAULT_COERENZA})")
    ap.add_argument("--struttura", default=DEFAULT_STRUTTURA,
                    help=f"JSON da cui prendere metadati e ordine dei turni "
                         f"(default: {DEFAULT_STRUTTURA})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"JSON di output, il .csv gemello gli sta accanto "
                         f"(default: {DEFAULT_OUT})")
    ap.add_argument("--soglia", type=float, default=None,
                    help="ricalcola la scelta da `similarita` con questa soglia "
                         "invece di usare la colonna `sostituire` del report")
    ap.add_argument("--solo-golden", action="store_true",
                    help="ignora le decisioni e prende sempre il golden (per confronto)")
    ap.add_argument("--no-fonte", action="store_true",
                    help="non scrivere il campo `fonte` nel JSON (schema identico "
                         "agli altri dataset)")
    ap.add_argument("--anteprima", type=int, default=10, metavar="N",
                    help="quanti turni tenuti da gemma4 mostrare (default: 10)")
    args = ap.parse_args()

    for p in (args.coerenza, args.struttura):
        if not os.path.exists(p):
            sys.exit(f"ERRORE: file non trovato: {p!r}")

    samples = load_samples(args.struttura)
    decisioni = load_decisioni(args.coerenza, args.soglia)
    if args.solo_golden:
        for d in decisioni.values():
            d["_sostituire"] = True

    out_samples, flat, n, senza, it_div = fondi(samples, decisioni, not args.no_fonte)

    if senza:
        print(f"! {len(senza)} turni senza decisione nel report, esclusi: "
              f"{', '.join(senza[:5])}" + (" ..." if len(senza) > 5 else ""),
              file=sys.stderr)
    if it_div:
        print(f"! {len(it_div)} turni con italiano diverso fra report e struttura: "
              f"{', '.join(it_div[:5])}" + (" ..." if len(it_div) > 5 else ""),
              file=sys.stderr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_json(out_samples, args.out)
    out_csv = os.path.splitext(args.out)[0] + ".csv"
    campi = OUT_CSV_FIELDS if not args.no_fonte else CSV_FIELDS
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campi, extrasaction="ignore")
        w.writeheader()
        w.writerows(flat)

    tot = n["golden"] + n["gemma4"]
    print(f"Scritto: {args.out}")
    print(f"Scritto: {out_csv}\n")
    print(f"Turni totali : {tot}  ("
          + ", ".join(f"{s['id']} {s['n_turni_italiani']}" for s in out_samples) + ")")
    soglia_txt = f"soglia {args.soglia}" if args.soglia is not None \
        else "colonna `sostituire` del report"
    print(f"Criterio     : {'sempre golden' if args.solo_golden else soglia_txt}")
    print(f"  da golden  : {n['golden']:5d}  ({n['golden'] / tot * 100:.1f}%)")
    print(f"  da gemma4  : {n['gemma4']:5d}  ({n['gemma4'] / tot * 100:.1f}%)")

    esempi, quanti_div = anteprima_gemma(flat, decisioni, args.anteprima)
    if quanti_div:
        print(f"\nDei {n['gemma4']} turni tenuti da gemma4, {quanti_div} differiscono "
              f"dal golden.\nSono gli unici punti in cui il dataset si discosta dal "
              f"testo validato a mano;\nqui i {len(esempi)} piu' distanti:\n")
        for sim, conv, ti, gem, gol in esempi:
            print(f"  [{conv} t{ti}] {sim:.1f}%")
            print(f"     tenuto (gemma4): {gem}")
            print(f"     scartato (gold): {gol}")


if __name__ == "__main__":
    main()
