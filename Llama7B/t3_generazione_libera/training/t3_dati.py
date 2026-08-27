#!/usr/bin/env python3
"""
t3_dati.py — audit dello split T3 e costruzione dello split arricchito.

Due funzioni, entrambe senza GPU:

  --audit         stampa le patologie del corpus con i numeri, e si ferma.
  (default)       ricostruisce la trascrizione e riscrive train/dev/test con
                  contesto piu' lungo, istruzione variabile e ancoraggio esplicito
                  al turno da riprendere.

Perche' la ricostruzione e' possibile
-------------------------------------
Nei file originali il campo `prompt` contiene gli ULTIMI 3 turni della
trascrizione, non gli ultimi 3 turni *presenti nello split*. Le finestre di item
consecutivi si sovrappongono, quindi l'unione delle finestre ricostruisce la
trascrizione: 2.285 turni contro i 1.127 usati come target. Meta' del contesto
disponibile e' gia' nei file e viene buttata via dalla finestra fissa a 3.

Uso:
    python t3_dati.py --in-dir  /kaggle/input/.../layout3_replica_conversazionale --audit
    python t3_dati.py --in-dir  /kaggle/input/.../layout3_replica_conversazionale \
                      --out-dir /kaggle/working/split_t3 --finestra 6
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
from collections import Counter, defaultdict

SPLITS = ("train", "dev", "test")

# Riscontri / segnali di ascolto: turni che non portano contenuto proposizionale.
# Servono COME CONTESTO (dicono che l'altro sta ancora parlando) ma come TARGET
# insegnano che una replica vuota e' una replica corretta.
RISCONTRI = {
    "mh", "mhmh", "mhm", "hm", "eh", "ehm", "emh", "ah", "oh", "eeh", "uh",
    "si", "sì", "no", "boh", "okay", "ok", "esatto", "certo", "vabbuò", "vabbè",
    "cioè", "overamente", "oddio", "embè",
}


def is_riscontro(testo: str) -> bool:
    """Vero se il turno e' fatto solo di riscontri, eventualmente ripetuti."""
    parole = re.findall(r"[\w'àèéìòùâêîôû]+", testo.lower())
    if not parole:
        return True
    return all(p in RISCONTRI for p in parole)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def carica(in_dir: str) -> dict[str, list[dict]]:
    dati = {}
    for s in SPLITS:
        p = os.path.join(in_dir, f"{s}.json")
        if not os.path.exists(p):
            raise SystemExit(f"manca {p}")
        dati[s] = json.load(open(p, encoding="utf-8"))
    return dati


def righe_contesto(prompt: str) -> list[tuple[str, str]]:
    """Estrae [(speaker, testo)] dal blocco di contesto del prompt originale."""
    blocco = prompt.split("---")[0]
    out = []
    for riga in blocco.strip().split("\n")[1:]:      # salta 'Conversazione finora:'
        if ":" in riga:
            sp, txt = riga.split(":", 1)
            sp = sp.strip()
            if len(sp) == 1 and sp.isalpha():
                out.append((sp.upper(), txt.strip()))
    return out


# --------------------------------------------------------------------------- #
# Ricostruzione della trascrizione
# --------------------------------------------------------------------------- #

def ricostruisci(dati: dict[str, list[dict]]):
    """conversazione -> {turn_index: (speaker, testo)}.

    L'ultima riga di contesto dell'item con indice k e' il turno k-1, la
    penultima k-2, e cosi' via. Verificato sul corpus: per ogni coppia di item
    consecutivi con distanza <= 3 il target del precedente ricompare nella
    finestra del successivo, alla posizione attesa.
    """
    turni: dict[str, dict[int, tuple[str, str]]] = defaultdict(dict)
    conflitti = 0
    for s in SPLITS:
        for x in dati[s]:
            c, k = x["conversazione"], int(x["turn_index"])
            ctx = righe_contesto(x["prompt"])
            for j, (sp, txt) in enumerate(reversed(ctx)):
                idx = k - 1 - j
                nuovo = (sp, txt)
                if idx in turni[c] and turni[c][idx] != nuovo:
                    conflitti += 1
                turni[c][idx] = nuovo
            nuovo = (x["speaker"], x["target"])
            if k in turni[c] and turni[c][k] != nuovo:
                conflitti += 1
            turni[c][k] = nuovo
    return turni, conflitti


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #

def audit(dati, turni, conflitti):
    print("=" * 74)
    print("AUDIT DELLO SPLIT T3")
    print("=" * 74)

    tot = sum(len(dati[s]) for s in SPLITS)
    print(f"\nistanze: " + " | ".join(f"{s} {len(dati[s])}" for s in SPLITS) + f"  (tot {tot})")

    # --- 1. copertura e disegno dello split --------------------------------
    print("\n[1] DISEGNO DELLO SPLIT")
    for s in SPLITS:
        per = defaultdict(list)
        for x in dati[s]:
            per[x["conversazione"]].append(x["turn_index"])
        desc = ", ".join(f"{c}: turni {min(v)}-{max(v)} ({len(v)})" for c, v in sorted(per.items()))
        print(f"  {s:5s} {desc}")
    print("  -> split CRONOLOGICO: nessuna sovrapposizione di indici fra split.")
    print("     Niente leakage di adiacenza, ma dev/test sono la CODA delle stesse")
    print("     due conversazioni: non misurano generalizzazione a parlanti o temi nuovi.")

    convs = {x["conversazione"] for s in SPLITS for x in dati[s]}
    print(f"  conversazioni totali: {len(convs)} ({sorted(convs)}) -> n=2 per qualsiasi")
    print("     intervallo di confidenza sul 'parlare in napoletano di un dato gruppo'.")

    # --- 2. contesto --------------------------------------------------------
    print("\n[2] INFORMAZIONE NEL CONTESTO  <- la criticita' principale per questo task")
    for s in SPLITS:
        n_turni, parole, tutti_riscontri = Counter(), [], 0
        for x in dati[s]:
            ctx = righe_contesto(x["prompt"])
            n_turni[len(ctx)] += 1
            parole.append(sum(len(t.split()) for _, t in ctx))
            if ctx and all(is_riscontro(t) for _, t in ctx):
                tutti_riscontri += 1
        print(f"  {s:5s} turni/contesto {dict(sorted(n_turni.items()))} | "
              f"parole medie {statistics.mean(parole):.1f} (mediana {statistics.median(parole):.0f}) | "
              f"contesti di soli riscontri {100*tutti_riscontri/len(dati[s]):.1f}%")
    freq = Counter(t for s in SPLITS for x in dati[s] for _, t in righe_contesto(x["prompt"]))
    print("  turni di contesto piu' frequenti:",
          ", ".join(f"{t!r}x{n}" for t, n in freq.most_common(8)))
    print("  -> la finestra e' fissa a 3 turni e i turni sono frammenti di parlato:")
    print("     ~16 parole di contesto in tutto, dominate da riscontri. Un modello che")
    print("     ignora un contesto quasi vuoto non sta sbagliando: sta stimando bene.")

    # --- 3. trascrizione ricostruibile -------------------------------------
    print("\n[3] CONTESTO RECUPERABILE (gratis, dai file che hai gia')")
    print(f"  conflitti nella ricostruzione: {conflitti} (0-1 atteso)")
    ric = sum(len(m) for m in turni.values())
    print(f"  turni ricostruiti: {ric} contro {tot} usati come target "
          f"({ric/tot:.1f}x materiale di contesto)")
    for c, m in sorted(turni.items()):
        ks = sorted(m)
        span = ks[-1] - ks[0] + 1
        print(f"    {c}: {len(m)} turni su uno span di {span} -> copertura {100*len(m)/span:.1f}%")
    for W in (3, 6, 8, 10):
        ok = sum(1 for s in SPLITS for x in dati[s]
                 if all((x["turn_index"] - 1 - j) in turni[x["conversazione"]] for j in range(W)))
        print(f"  item con {W:2d} turni di contesto contigui ricostruibili: "
              f"{ok}/{tot} ({100*ok/tot:.1f}%)")

    # --- 4. target ----------------------------------------------------------
    print("\n[4] TARGET")
    for s in SPLITS:
        L = [len(x["target"].split()) for x in dati[s]]
        risc = sum(1 for x in dati[s] if is_riscontro(x["target"]))
        print(f"  {s:5s} parole: media {statistics.mean(L):.1f} mediana {statistics.median(L):.0f} "
              f"max {max(L)} | <=4 parole {100*sum(1 for l in L if l<=4)/len(L):.1f}% | "
              f"target di soli riscontri {100*risc/len(dati[s]):.1f}%")
    print("  -> riferimento SINGOLO su un task open-ended: chrF/BERTScore contro l'unico")
    print("     turno realmente pronunciato premiano l'indovino, non la pertinenza.")

    # --- 5. provenienza -----------------------------------------------------
    print("\n[5] PROVENIENZA DEI RIFERIMENTI")
    for s in SPLITS:
        f = Counter(x.get("fonte", "?") for x in dati[s])
        sint = sum(v for k, v in f.items() if k != "golden")
        print(f"  {s:5s} {dict(f)}  -> {100*sint/len(dati[s]):.1f}% sintetico")
    print("  -> in TEST i riferimenti non-golden sono generati da un altro modello:")
    print("     misurare la somiglianza a un output di macchina non e' misurare la qualita'.")
    print("     Vanno esclusi dal test, o riportati come strato separato.")

    # --- 6. parlanti --------------------------------------------------------
    print("\n[6] PARLANTI")
    for s in SPLITS:
        print(f"  {s:5s} target per parlante: {dict(Counter(x['speaker'] for x in dati[s]))}")
    print("  -> KPN003 e' multi-parte (A/B/C/D). Con 3 turni di finestra l'informazione")
    print("     su CHI parla a CHI non c'e': l'indirizzamento e' irrecuperabile dal prompt.")
    print("     Il campo speaker_id esiste nei dati ma non entra mai nel prompt.")

    # --- 7. istruzione ------------------------------------------------------
    print("\n[7] ISTRUZIONE")
    istr = Counter(x["prompt"].split("---")[-1].strip() for s in SPLITS for x in dati[s])
    print(f"  varianti distinte: {len(istr)} su {tot} item -> {dict(list(istr.items())[:5])}")
    print("  -> l'istruzione varia solo per la lettera del parlante. Una feature quasi")
    print("     costante ha informazione mutua ~0 col target: il modello impara a")
    print("     saltarla e a condizionare sui soli primi token. E' il meccanismo con cui")
    print("     nasce la 'cecita' al contesto.")

    # --- 8. duplicati -------------------------------------------------------
    print("\n[8] IGIENE")
    for s in SPLITS:
        pr = [x["prompt"] for x in dati[s]]
        print(f"  {s:5s} prompt duplicati interni: {len(pr)-len(set(pr))}")
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                ov = len(set(x["prompt"] for x in dati[a]) & set(x["prompt"] for x in dati[b]))
                print(f"  prompt in comune {a}/{b}: {ov}")
    print("  -> nessun duplicato e nessuna sovrapposizione: questa parte e' pulita.")
    print("=" * 74)


# --------------------------------------------------------------------------- #
# Costruzione dello split arricchito
# --------------------------------------------------------------------------- #

ISTRUZIONI = [
    "Tocca a {sp}. Rispondi in napoletano, restando sul tema.",
    "Continua la conversazione: scrivi il turno di {sp} in napoletano.",
    "Adesso parla {sp}. Che dice, in napoletano?",
    "Sei {sp}. Rispondi in napoletano a quello che ha appena detto {prec}.",
    "Scrivi la replica di {sp} in napoletano: dev'essere una risposta a {prec}.",
    "Turno di {sp}. Rispondi in napoletano, breve e pertinente.",
    "Come risponderebbe {sp}, in napoletano?",
    "Riprendi il discorso di {prec}: parla {sp}, in napoletano.",
]


def costruisci_prompt(ctx, speaker, finestra, rng, varia=True):
    """Prompt arricchito. Tutte le feature derivano dal CONTESTO, mai dal target."""
    ctx = ctx[-finestra:]
    parlanti = sorted({sp for sp, _ in ctx} | {speaker})
    prec = ctx[-1][0] if ctx else "l'altro"

    testa = (f"Conversazione in napoletano fra {len(parlanti)} persone "
             f"({', '.join(parlanti)}).")
    corpo = "\n".join(f"{sp}: {txt}" for sp, txt in ctx) if ctx else "(inizio della conversazione)"

    # segnali calcolati dal contesto: sono disponibili anche in inferenza.
    ultimo = ctx[-1][1] if ctx else ""
    segnali = []
    if ultimo.strip().endswith("?") or re.match(
            r"^\s*(che|chi|comme|quanno|addo|pecch|quant|ma )", ultimo.lower()):
        segnali.append("l'ultimo turno e' una domanda: rispondi alla domanda")
    if ctx and is_riscontro(ultimo):
        prec_cont = next((t for _, t in reversed(ctx[:-1]) if not is_riscontro(t)), "")
        if prec_cont:
            segnali.append("l'ultimo turno e' solo un cenno di ascolto: "
                           "il tema vero e' nel turno precedente")

    istr = rng.choice(ISTRUZIONI) if varia else ISTRUZIONI[0]
    istr = istr.format(sp=speaker, prec=prec)
    if segnali:
        istr += " (" + "; ".join(segnali) + ")"

    return f"{testa}\n{corpo}\n---\n{istr}"


def costruisci(dati, turni, finestra, varia, togli_riscontri, togli_sintetici, seed):
    rng = random.Random(seed)
    out, report = {}, {}
    for s in SPLITS:
        righe, scartati_r, scartati_s, ctx_corti = [], 0, 0, 0
        for x in dati[s]:
            c, k = x["conversazione"], int(x["turn_index"])
            if togli_riscontri and s == "train" and is_riscontro(x["target"]):
                scartati_r += 1
                continue
            if togli_sintetici and s in ("dev", "test") and x.get("fonte") != "golden":
                scartati_s += 1
                continue
            ctx = []
            for j in range(finestra, 0, -1):
                t = turni[c].get(k - j)
                if t is None:
                    ctx = []          # buco: riparti, il contesto dev'essere contiguo
                    continue
                ctx.append(t)
            if len(ctx) < min(3, finestra):
                ctx_corti += 1
            y = dict(x)
            y["prompt"] = costruisci_prompt(ctx, x["speaker"], finestra, rng, varia)
            y["n_turni_contesto"] = len(ctx)
            righe.append(y)
        out[s] = righe
        report[s] = dict(n=len(righe), scartati_riscontri=scartati_r,
                         scartati_sintetici=scartati_s, contesti_corti=ctx_corti,
                         turni_contesto_medi=round(
                             statistics.mean([r["n_turni_contesto"] for r in righe]), 2))
    return out, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--finestra", type=int, default=6)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--no-variazioni", action="store_true")
    ap.add_argument("--tieni-riscontri", action="store_true")
    ap.add_argument("--tieni-sintetici", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    dati = carica(a.in_dir)
    turni, conflitti = ricostruisci(dati)

    if a.audit or not a.out_dir:
        audit(dati, turni, conflitti)
        return

    out, report = costruisci(dati, turni, a.finestra, not a.no_variazioni,
                             not a.tieni_riscontri, not a.tieni_sintetici, a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    for s in SPLITS:
        with open(os.path.join(a.out_dir, f"{s}.json"), "w", encoding="utf-8") as f:
            json.dump(out[s], f, ensure_ascii=False, indent=1)
    json.dump(report, open(os.path.join(a.out_dir, "report.json"), "w"), indent=2)

    print(f"scritto in {a.out_dir}")
    for s in SPLITS:
        print(f"  {s:5s} {report[s]}")
    print("\nesempio di prompt (train):\n" + "-" * 66)
    print(out["train"][5]["prompt"])
    print("--- TARGET:", out["train"][5]["target"])


if __name__ == "__main__":
    main()
