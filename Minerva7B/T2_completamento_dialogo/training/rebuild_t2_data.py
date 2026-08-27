#!/usr/bin/env python3
"""
rebuild_t2_data.py — ricostruisce gli split T2 dal CSV parallelo.

Risolve tre problemi dei file attuali:

  1. UNA sola istanza per turno, con taglio fisso a meta' delle parole.
     Il modello impara "fermati dopo k parole", non "completa il turno".
     Qui il taglio e' MOBILE: ogni posizione che lascia almeno --min-lato
     parole per parte diventa un'istanza. Da 980 istanze si passa a ~9.000.

  2. Turni sotto le 6 parole scartati del tutto (62% del corpus).
     Qui la soglia scende a --min-turno (default 4).

  3. Split cronologico a tre blocchi: dev/test cadono nell'ultimo terzo di
     ogni conversazione, cioe' su argomenti che nel train non compaiono mai
     (Jaccard delle parole di contenuto train/test = 0.10). Qui lo split e'
     a BLOCCHI INTERLACCIATI con una zona cuscinetto: la copertura tematica
     e' uniforme e nessun turno di dev/test finisce nel contesto di un turno
     di train.

     --split cronologico  riproduce lo schema attuale (per il confronto).

Uso:
    python rebuild_t2_data.py --csv dataset_finale.csv --out-dir split_v2
    python rebuild_t2_data.py --csv dataset_finale.csv --out-dir split_v1b \
        --split cronologico --tagli 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict

MARCA = "Continua il turno in napoletano:"
CONTESTO = 3          # turni precedenti nel prompt; e' anche l'ampiezza del cuscinetto


def parole(s: str) -> list[str]:
    return (s or "").split()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


# --------------------------------------------------------------------------- #
# 1. Lettura e ordinamento
# --------------------------------------------------------------------------- #

def leggi(csv_path: str) -> dict[str, list[dict]]:
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        righe = list(csv.DictReader(f))
    conv = defaultdict(list)
    for r in righe:
        r["turn_index"] = int(r["turn_index"])
        r["napoletano"] = norm(r["napoletano"])
        conv[r["conversazione"]].append(r)
    for k in conv:
        conv[k].sort(key=lambda r: r["turn_index"])
    return dict(conv)


# --------------------------------------------------------------------------- #
# 2. Assegnazione degli split
# --------------------------------------------------------------------------- #

PATTERN = ["train"] * 4 + ["dev"] + ["train"] * 4 + ["test"]   # 80/10/10 di blocchi


def assegna_blocchi(n: int, blocco: int) -> list[str]:
    """Etichetta per posizione. I blocchi si alternano, cosi' train/dev/test
    attraversano tutta la conversazione invece di occuparne tre segmenti
    consecutivi. Il cuscinetto applicato dopo riduce dev/test di ~CONTESTO
    turni per blocco, quindi le quote finali sono un po' sotto il 10%."""
    return [PATTERN[(i // blocco) % len(PATTERN)] for i in range(n)]


def assegna_cronologico(n: int, quote=(0.70, 0.15, 0.15)) -> list[str]:
    a = int(n * quote[0])
    b = a + int(n * quote[1])
    return ["train"] * a + ["dev"] * (b - a) + ["test"] * (n - b)


def applica_cuscinetto(etichette: list[str], k: int = CONTESTO) -> list[str | None]:
    """None = turno da NON usare come bersaglio. Un turno resta utilizzabile
    solo se i k turni che lo precedono (il suo contesto) appartengono al suo
    stesso split: altrimenti il contesto di un item di train conterrebbe un
    turno di test, e viceversa. Il turno scartato resta comunque disponibile
    COME contesto: e' il bersaglio che sparisce, non il testo."""
    fuori = []
    for i, e in enumerate(etichette):
        finestra = etichette[max(0, i - k):i + 1]
        fuori.append(e if all(x == e for x in finestra) else None)
    return fuori


# --------------------------------------------------------------------------- #
# 3. Costruzione degli item
# --------------------------------------------------------------------------- #

def contesto_testo(turni: list[dict], i: int, mappa: dict[str, str],
                   k: int = CONTESTO) -> str:
    righe = []
    for j in range(max(0, i - k), i):
        t = turni[j]
        if not t["napoletano"]:
            continue
        righe.append(f"{mappa[t['tu_id']]}: {t['napoletano']}")
    return "\n".join(righe)


def mappa_parlanti(turni: list[dict]) -> dict[str, str]:
    """speaker_id -> lettera, in ordine di prima comparsa. Stabile su tutta la
    conversazione: se cambiasse fra un item e l'altro, 'A' nel contesto e 'A'
    nell'istruzione indicherebbero persone diverse."""
    lettere, out = "ABCDEFGH", {}
    for t in turni:
        sid = t["speaker"]
        if sid not in out:
            out[sid] = lettere[min(len(out), len(lettere) - 1)]
        t["tu_id"] = sid
    return out


def tagli(n_parole: int, min_lato: int, massimo: int, rng) -> list[int]:
    """Posizioni di taglio ammesse (numero di parole nel prefisso)."""
    possibili = list(range(min_lato, n_parole - min_lato + 1))
    if not possibili:
        return []
    if massimo <= 0 or len(possibili) <= massimo:
        return possibili
    if massimo == 1:                       # meta' esatta: lo schema originale
        return [len(possibili) // 2 + min_lato]
    return sorted(rng.sample(possibili, massimo))


def costruisci(conv: dict[str, list[dict]], modo: str, blocco: int,
               min_turno: int, min_lato: int, tagli_train: int,
               tagli_eval: int, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    out = {"train": [], "dev": [], "test": []}
    stat = Counter()
    for nome, turni in conv.items():
        mappa = mappa_parlanti(turni)
        base = (assegna_blocchi(len(turni), blocco) if modo == "blocchi"
                else assegna_cronologico(len(turni)))
        etich = applica_cuscinetto(base)
        for i, t in enumerate(turni):
            split = etich[i]
            if split is None:
                stat["scartati_cuscinetto"] += 1
                continue
            w = parole(t["napoletano"])
            if len(w) < min_turno:
                stat["scartati_corti"] += 1
                continue
            ctx = contesto_testo(turni, i, mappa)
            if not ctx:
                stat["scartati_senza_contesto"] += 1
                continue
            k_max = tagli_train if split == "train" else tagli_eval
            for c in tagli(len(w), min_lato, k_max, rng):
                pref, targ = " ".join(w[:c]), " ".join(w[c:])
                # il parlante si ripete: il target e' gia' nel contesto,
                # l'item e' gratis e gonfia le metriche
                if targ and targ in ctx:
                    stat["scartati_target_nel_contesto"] += 1
                    continue
                out[split].append({
                    "id": f"{nome}_T2_{t['turn_index']}_c{c}",
                    "layout": "T2",
                    "conversazione": nome,
                    "turn_index": t["turn_index"],
                    "speaker": mappa[t["speaker"]],
                    "speaker_id": t["speaker"],
                    "prompt": f"Conversazione finora:\n{ctx}\n---\n{MARCA} {pref}",
                    "target": targ,
                    "n_parole_prefisso": c,
                    "n_parole_target": len(w) - c,
                    "fonte": t.get("fonte", ""),
                    "split": split,
                })
    return out, stat


# --------------------------------------------------------------------------- #
# 4. Controlli
# --------------------------------------------------------------------------- #

def controlla(out: dict[str, list[dict]]) -> None:
    def chiavi(ds):
        return {(r["conversazione"], r["turn_index"]) for r in ds}
    tr, dv, te = (chiavi(out[k]) for k in ("train", "dev", "test"))
    print("\n=== controlli ===")
    print("  turni condivisi train/dev :", len(tr & dv))
    print("  turni condivisi train/test:", len(tr & te))
    print("  turni condivisi dev/test  :", len(dv & te))
    for nome in ("train", "dev", "test"):
        ds = out[nome]
        lt = [r["n_parole_target"] for r in ds]
        conv = Counter(r["conversazione"] for r in ds)
        print(f"  {nome:5s} n={len(ds):5d} | turni={len({(r['conversazione'], r['turn_index']) for r in ds}):4d} "
              f"| target medio {sum(lt)/max(1,len(lt)):.2f} parole | {dict(conv)}")
    # deriva lessicale: quota di token di TARGET mai visti nel train. E' la
    # misura che conta (il Jaccard fra vocabolari di taglia molto diversa dice
    # soprattutto che uno e' piu' grande dell'altro).
    voc = set()
    for r in out["train"]:
        voc |= set(re.findall(r"[a-zàèéìòùâêîôû']+",
                              (r["prompt"] + " " + r["target"]).lower()))
    for nome in ("dev", "test"):
        tok = [w for r in out[nome]
               for w in re.findall(r"[a-zàèéìòùâêîôû']+", r["target"].lower())]
        oov = sum(1 for w in tok if w not in voc)
        print(f"  token di target {nome} mai visti nel train: "
              f"{oov}/{len(tok)} ({100*oov/max(1,len(tok)):.1f}%)")


def anti_eco(out: dict[str, list[dict]]) -> Counter:
    """Toglie da dev/test gli item il cui target compare ALLA LETTERA in un
    prompt di train.

    Non e' lo stesso turno (il cuscinetto lo garantisce gia'): e' la stessa
    frase detta altrove. Su due sole conversazioni le formule brevi si
    ripetono di continuo, e un item il cui target il modello ha gia' letto in
    addestramento e' un item regalato: gonfia le metriche senza dire niente
    sulla capacita' di completare un turno."""
    testi = "\n".join(r["prompt"] + " " + r["target"] for r in out["train"])
    stat = Counter()
    for s in ("dev", "test"):
        tenuti = [r for r in out[s] if r["target"] not in testi]
        stat[f"eco_rimossi_{s}"] = len(out[s]) - len(tenuti)
        out[s] = tenuti
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="blocchi", choices=["blocchi", "cronologico"])
    ap.add_argument("--blocco", type=int, default=25,
                    help="turni per blocco nello split interlacciato")
    ap.add_argument("--min-turno", type=int, default=4,
                    help="parole minime del turno perche' diventi un item")
    ap.add_argument("--min-lato", type=int, default=2,
                    help="parole minime da ogni lato del taglio")
    ap.add_argument("--tagli", type=int, default=4,
                    help="tagli per turno nel TRAIN (0 = tutti quelli ammessi)")
    ap.add_argument("--tagli-eval", type=int, default=1,
                    help="tagli per turno in dev/test. 1 = taglio a meta', "
                         "come nei file originali: la valutazione resta confrontabile")
    ap.add_argument("--anti-eco", dest="anti_eco", action="store_true",
                    default=True,
                    help="toglie da dev/test i target che compaiono alla "
                         "lettera nel train (default: attivo)")
    ap.add_argument("--no-anti-eco", dest="anti_eco", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    conv = leggi(a.csv)
    print(f"conversazioni: { {k: len(v) for k, v in conv.items()} }")
    out, stat = costruisci(conv, a.split, a.blocco, a.min_turno, a.min_lato,
                           a.tagli, a.tagli_eval, a.seed)
    if a.anti_eco:
        stat.update(anti_eco(out))
    print("scarti:", dict(stat))
    controlla(out)

    os.makedirs(a.out_dir, exist_ok=True)
    for nome, ds in out.items():
        p = os.path.join(a.out_dir, f"{nome}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(ds, f, ensure_ascii=False, indent=2)
        print(f"scritto {p} ({len(ds)} item)")


if __name__ == "__main__":
    main()
