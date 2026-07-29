#!/usr/bin/env python3
"""
Costruisce il dataset per il Progetto 2 a livello di CONVERSAZIONE.
Ogni CAMPIONE = una conversazione intera del corpus KIPasti, presa per intero e
in ordine di turno. Dentro ciascuna conversazione si tengono SOLO i turni in
italiano (si escludono i turni con dialetto: variation=some).

Piu' conversazioni possono essere unite in un UNICO dataset: il .json contiene
un array di conversazioni (stessa struttura di KPN001.json) e il .csv contiene
un turno per riga, raggruppato per conversazione.

Uso:
    # dataset unico con KPN001 + KPN003 (default)
    python build_conversation_dataset.py --kipasti ./dataset_Kipasti

    # scelta esplicita delle conversazioni
    python build_conversation_dataset.py --kipasti ./dataset_Kipasti \
        --conversations KPN001 KPN003 --out dataset_structured

    # tutte le conversazioni disponibili
    python build_conversation_dataset.py --kipasti "D:/Progetti/KIPasti" --all

Output (dentro la cartella dataset_filter/, creata se non esiste):
    <out>.json    array di conversazioni (o singolo oggetto se ne elabori una sola),
                  con la lista ordinata dei turni italiani
    <out>.csv     una riga per turno (per annotare comodamente il napoletano)
"""

import argparse, csv, glob, json, os
from collections import defaultdict

# indici colonne del formato .vert.tsv
C_SPEAKER, C_TU, C_FORM, C_TYPE, C_VAR, C_FEATS = 1, 2, 4, 5, 6, 7
BAD_TYPES = {"error", "unknown", "anonymized"}

# cartella di destinazione dei file prodotti (creata se non esiste)
OUT_DIR = "dataset_filter"

# conversazioni incluse se non si passa ne' --conversations ne' --all
DEFAULT_CONVERSATIONS = ["KPN001", "KPN003"]

CSV_FIELDS = ["conversazione", "regione", "macro_regione", "languages",
              "turn_index", "tu_id", "speaker", "italiano", "napoletano", "note"]


def find_meta_file(kipasti):
    """conversations.tsv sta in <kipasti>/metadata/ (corpus originale) o in <kipasti>/."""
    for cand in (os.path.join(kipasti, "metadata", "conversations.tsv"),
                 os.path.join(kipasti, "conversations.tsv")):
        if os.path.exists(cand):
            return cand
    raise SystemExit(f"ERRORE: conversations.tsv non trovato sotto {kipasti!r}")


def find_tsv_files(kipasti):
    """I .vert.tsv stanno in <kipasti>/tsv/ (corpus originale) o in <kipasti>/."""
    for sub in ("tsv", ""):
        paths = sorted(glob.glob(os.path.join(kipasti, sub, "*.vert.tsv")))
        if paths:
            return paths
    raise SystemExit(f"ERRORE: nessun file *.vert.tsv trovato sotto {kipasti!r}")


def load_conv_meta(kipasti):
    meta = {}
    with open(find_meta_file(kipasti), encoding="utf-8") as f:
        header = f.readline().lstrip("\ufeff").rstrip("\n").split("\t")
        idx = {n: i for i, n in enumerate(header)}
        for line in f:
            c = line.rstrip("\n").split("\t")
            meta[c[idx["code"]]] = {
                "regione": c[idx["collection-region"]],
                "macro_regione": c[idx["macro-region"]],
                "languages": c[idx["languages"]],
            }
    return meta


def reconstruct(tokens):
    """Testo dai soli token linguistici, rispettando SpaceAfter=No."""
    parts = [(t[C_FORM], "SpaceAfter=No" in t[C_FEATS])
             for t in tokens if t[C_TYPE] == "linguistic"]
    if not parts:
        return "", 0
    text = ""
    for i, (form, no_space) in enumerate(parts):
        text += form
        if i < len(parts) - 1 and not no_space:
            text += " "
    return text.strip(), len(parts)


def load_conversation(path):
    """Ritorna la lista ordinata dei turni italiani: [(tu_id, speaker, testo, n_parole)]."""
    tus = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) <= C_FEATS:
                continue
            tus[c[C_TU]].append(c)
    turns = []
    for tu in sorted(tus, key=lambda x: int(x) if x.isdigit() else 0):
        toks = tus[tu]
        if any(t[C_VAR] == "some" for t in toks):      # turno con dialetto -> escluso
            continue
        if any(t[C_TYPE] in BAD_TYPES for t in toks):  # errori/anon/inintelligibile -> escluso
            continue
        text, n = reconstruct(toks)
        if n == 0:                                     # turno senza contenuto linguistico
            continue
        turns.append((tu, toks[0][C_SPEAKER], text, n))
    return turns


def dump_sample(sample, indent):
    """Serializza una conversazione: metadati indentati, un turno per riga (come KPN001.json)."""
    pad = " " * indent
    inner = " " * (indent + 2)
    lines = [pad + "{"]
    keys = [k for k in sample if k != "turni"]
    for k in keys:
        lines.append(f"{inner}{json.dumps(k)}: {json.dumps(sample[k], ensure_ascii=False)},")
    lines.append(f"{inner}\"turni\": [")
    turni = sample["turni"]
    for i, t in enumerate(turni):
        comma = "," if i < len(turni) - 1 else ""
        lines.append(" " * (indent + 4) + json.dumps(t, ensure_ascii=False) + comma)
    lines.append(f"{inner}]")
    lines.append(pad + "}")
    return "\n".join(lines)


def write_json(samples, path):
    """Un solo campione -> oggetto (formato KPN001.json); piu' campioni -> array."""
    with open(path, "w", encoding="utf-8") as f:
        if len(samples) == 1:
            f.write(dump_sample(samples[0], 0) + "\n")
            return
        f.write("[\n")
        for i, s in enumerate(samples):
            f.write(dump_sample(s, 2) + ("," if i < len(samples) - 1 else "") + "\n")
        f.write("]\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kipasti", default="dataset_Kipasti",
                    help="Cartella del corpus (con tsv/ e metadata/, oppure file .vert.tsv "
                         "e conversations.tsv direttamente dentro)")
    ap.add_argument("--conversations", nargs="+", default=None, metavar="CODE",
                    help=f"Codici conversazione da unire nel dataset "
                         f"(default: {' '.join(DEFAULT_CONVERSATIONS)})")
    ap.add_argument("--conversation", default=None,
                    help="Alias per una singola conversazione (es. KPN001)")
    ap.add_argument("--all", action="store_true",
                    help="Elabora tutte le conversazioni trovate")
    ap.add_argument("--out", default="dataset_structured")
    args = ap.parse_args()

    # quali conversazioni includere
    if args.all:
        wanted = None
    elif args.conversations:
        wanted = list(args.conversations)
    elif args.conversation:
        wanted = [args.conversation]
    else:
        wanted = list(DEFAULT_CONVERSATIONS)

    # se --out e' un nome semplice, i file finiscono in OUT_DIR
    out_base = args.out if os.path.dirname(args.out) else os.path.join(OUT_DIR, args.out)
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)

    meta = load_conv_meta(args.kipasti)
    by_conv = {os.path.basename(p).split(".")[0]: p for p in find_tsv_files(args.kipasti)}

    if wanted is None:
        wanted = sorted(by_conv)
    missing = [c for c in wanted if c not in by_conv]
    if missing:
        raise SystemExit(f"ERRORE: file .vert.tsv mancanti per: {', '.join(missing)}\n"
                         f"Disponibili: {', '.join(sorted(by_conv))}")

    samples, flat_rows, tot_turns = [], [], 0
    for conv in wanted:
        m = meta.get(conv, {"regione": "", "macro_regione": "", "languages": ""})
        turns = load_conversation(by_conv[conv])
        if not turns:
            print(f"  ! {conv}: nessun turno italiano, saltata")
            continue
        tot_turns += len(turns)

        turni_json = []
        for k, (tu, speaker, text, n) in enumerate(turns, 1):
            turni_json.append({
                "turn_index": k, "tu_id": tu, "speaker": speaker,
                "italiano": text, "napoletano": "",
            })
            flat_rows.append({
                "conversazione": conv, "regione": m["regione"],
                "macro_regione": m["macro_regione"], "languages": m["languages"],
                "turn_index": k, "tu_id": tu, "speaker": speaker,
                "italiano": text, "napoletano": "", "note": "",
            })
        samples.append({
            "id": conv, "regione": m["regione"], "macro_regione": m["macro_regione"],
            "languages": m["languages"], "n_turni_italiani": len(turns), "turni": turni_json,
        })

    if not samples:
        print("Nessuna conversazione trovata (controlla --conversations e --kipasti).")
        return

    json_name = out_base + ".json"
    write_json(samples, json_name)

    csv_name = out_base + ".csv"
    with open(csv_name, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(flat_rows)

    print(f"Conversazioni (campioni)        : {len(samples)}  ({', '.join(s['id'] for s in samples)})")
    for s in samples:
        print(f"  - {s['id']}: {s['n_turni_italiani']} turni italiani  "
              f"[{s['regione']} / {s['macro_regione']} / {s['languages']}]")
    print(f"Turni italiani totali           : {tot_turns}")
    print(f"Media turni italiani/conversaz. : {tot_turns/len(samples):.0f}")
    it_only = [s for s in samples if s["languages"] == "italian"]
    print(f"Di cui conversazioni ITALIAN-only: {len(it_only)}")
    print(f"\nFile scritti: {json_name}  |  {csv_name}")


if __name__ == "__main__":
    main()
