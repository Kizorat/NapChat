#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grafici delle metriche di T1 (traduzione con contesto) e della loss di T3,
per ognuno dei modelli fine-tunati (Minerva7B, Gemma4B, Llama7B).

Per T1 (da runs/, cpt/, metriche_finali.json, eval/):
  - la curva della loss (train + eval) dei tre stadi di addestramento
  - le curve di Precision, Recall, F1 e chrF++ in validazione
  - le barre di Precision, Recall, F1 finali su test
  - le barre di chrF++ finale contro le baseline

Per T3 (dagli output salvati nel notebook fine_tuning_T3.ipynb):
  - la curva della loss (train + eval)

I task assenti per un modello vengono semplicemente saltati: solo Minerva7B ha
anche T1. Senza --out ogni modello scrive nella propria cartella dei grafici
(Minerva7B/grafici_minerva7B/, Gemma4B/grafici_gemma4B/, Llama7B/grafici_Llama7B/),
in una sottocartella per task.

Uso:
    python scripts/grafici_metriche.py                     # tutto, tutti i modelli
    python scripts/grafici_metriche.py --modello llama     # solo Llama7B
    python scripts/grafici_metriche.py --task T3
    python scripts/grafici_metriche.py --out mia_cartella --dpi 300
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RADICE = Path(__file__).resolve().parent.parent


def prima_esistente(base, *nomi):
    """Prima sottocartella esistente fra quelle indicate.

    Serve perche' i nomi delle cartelle non sono uniformi fra i modelli:
    Minerva usa T3_gemerazione_libera (con un refuso), Gemma t3_generazione_libera."""
    for nome in nomi:
        if (base / nome).is_dir():
            return base / nome
    return base / nomi[0]


class Modello:
    """Un modello fine-tunato: dove stanno i suoi task e dove finiscono i grafici."""

    NOMI_T1 = ("T1_Traduzione", "t1_traduzione")
    NOMI_T3 = ("T3_generazione_libera", "t3_generazione_libera",
               "T3_gemerazione_libera")

    def __init__(self, chiave, etichetta, cartella, cartella_grafici):
        self.chiave = chiave
        self.etichetta = etichetta
        self.radice = RADICE / cartella
        self.grafici = self.radice / cartella_grafici
        self.dir_t1 = prima_esistente(self.radice, *self.NOMI_T1)
        self.dir_t3 = prima_esistente(self.radice, *self.NOMI_T3)
        self.notebook_t3 = self.dir_t3 / "notebook" / "fine_tuning_T3.ipynb"


MODELLI = {
    "minerva": Modello("minerva", "Minerva7B", "Minerva7B", "grafici_minerva7B"),
    "gemma": Modello("gemma", "Gemma4B", "Gemma4B", "grafici_gemma4B"),
    "llama": Modello("llama", "Llama7B", "Llama7B", "grafici_Llama7B"),
}

# palette coerente fra tutti i grafici
C_TRAIN, C_EVAL = "#1f77b4", "#d62728"
C_P, C_R, C_F1 = "#4C72B0", "#DD8452", "#55A868"
C_CHRF, C_BASE = "#8172B3", "#C0C0C0"


# ---------------------------------------------------------------- utilita' --

def leggi_json(percorso):
    with open(percorso, encoding="utf-8") as f:
        return json.load(f)


def num(valore):
    """Converte in float una cella CSV, restituendo None se vuota o non numerica."""
    if valore is None or valore == "":
        return None
    try:
        return float(valore)
    except ValueError:
        return None


def leggi_storico(percorso):
    """metrics_history.csv -> lista di dict con i valori gia' convertiti."""
    with open(percorso, encoding="utf-8-sig", newline="") as f:
        return [{k: num(v) for k, v in r.items()} for r in csv.DictReader(f)]


def serie(righe, x, y):
    """Estrae le coppie (x, y) dove entrambi i campi sono presenti."""
    coppie = [(r[x], r[y]) for r in righe
              if r.get(x) is not None and r.get(y) is not None]
    coppie.sort(key=lambda c: c[0])
    return [c[0] for c in coppie], [c[1] for c in coppie]


def salva(fig, cartella, nome, dpi):
    cartella.mkdir(parents=True, exist_ok=True)
    percorso = cartella / nome
    fig.savefig(percorso, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  ok  " + str(percorso))
    return percorso


def etichette_barre(ax, barre, formato="{:.3f}", scarto=3):
    for b in barre:
        ax.annotate(formato.format(b.get_height()),
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, scarto), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)


def griglia(ax):
    ax.grid(alpha=.3, linestyle="--", linewidth=.7)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------- T1 -----

def run_t1(modello, suffisso):
    """Cartella del run che termina con il suffisso dato (es. __T1), se esiste.

    Il nome completo dipende dal repo_id del modello, quindi non lo fissiamo."""
    candidati = sorted(d for d in (modello.dir_t1 / "runs").glob("*" + suffisso)
                       if d.is_dir())
    return candidati[-1] if candidati else None


def curve_loss_t1(modello, cartella, dpi):
    """Loss di training e di validazione dei tre stadi: CPT -> A2 lessico -> T1."""
    stadi = []

    stato = sorted((modello.dir_t1 / "cpt").glob("*/checkpoint-*/trainer_state.json"))
    if stato:
        log = leggi_json(stato[-1])["log_history"]
        tr = [(d["epoch"], d["loss"]) for d in log if "loss" in d]
        ev = [(d["epoch"], d["eval_loss"]) for d in log if "eval_loss" in d]
        stadi.append(("Stadio CPT (pretraining dialettale)", tr, ev))

    for suffisso, titolo in (("__A2", "Stadio A2 (lessico)"),
                             ("__T1", "Stadio T1 (traduzione)")):
        run = run_t1(modello, suffisso)
        csv_run = run / "metrics_history.csv" if run else None
        if csv_run is None or not csv_run.exists():
            continue
        righe = leggi_storico(csv_run)
        tr = list(zip(*serie(righe, "epoch", "loss")))
        ev = list(zip(*serie(righe, "epoch", "eval_loss")))
        stadi.append((titolo, tr, ev))

    if not stadi:
        print("  ! nessuno storico di loss trovato per T1")
        return

    fig, assi = plt.subplots(1, len(stadi), figsize=(5.2 * len(stadi), 4.2))
    assi = assi if len(stadi) > 1 else [assi]
    for ax, (titolo, tr, ev) in zip(assi, stadi):
        if tr:
            x, y = zip(*tr)
            ax.plot(x, y, color=C_TRAIN, lw=1.4, alpha=.85,
                    marker="o", ms=3, label="train loss")
        if ev:
            x, y = zip(*ev)
            ax.plot(x, y, color=C_EVAL, lw=1.8, marker="s", ms=5,
                    label="eval loss")
            migliore = min(ev, key=lambda c: c[1])
            ax.axvline(migliore[0], color=C_EVAL, ls=":", lw=1, alpha=.6)
            ax.annotate("min {:.3f}".format(migliore[1]), xy=migliore,
                        xytext=(4, 8), textcoords="offset points",
                        fontsize=8, color=C_EVAL)
        ax.set_title(titolo, fontsize=11)
        ax.set_xlabel("epoca")
        ax.set_ylabel("loss")
        ax.legend(fontsize=9)
        griglia(ax)
    fig.suptitle("{} - T1 - Loss function per stadio di addestramento"
                 .format(modello.etichetta), fontsize=13)
    fig.tight_layout()
    salva(fig, cartella, "T1_loss.png", dpi)


def curve_prf1_t1(modello, cartella, dpi):
    """Precision / Recall / F1 lessicali (e chrF++) durante l'addestramento T1."""
    run = run_t1(modello, "__T1")
    csv_run = run / "metrics_history.csv" if run else None
    if csv_run is None or not csv_run.exists():
        print("  ! metrics_history.csv di T1 assente")
        return
    righe = leggi_storico(csv_run)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for campo, colore, etichetta in (("eval_word_precision", C_P, "Precision"),
                                     ("eval_word_recall", C_R, "Recall"),
                                     ("eval_word_f1", C_F1, "F1-Score")):
        x, y = serie(righe, "epoch", campo)
        if x:
            ax1.plot(x, y, color=colore, lw=1.8, marker="o", ms=5, label=etichetta)
    ax1.set_title("Precision / Recall / F1 (overlap bag-of-words) su dev", fontsize=11)
    ax1.set_xlabel("epoca")
    ax1.set_ylabel("valore")
    ax1.legend(fontsize=9)
    griglia(ax1)

    x, y = serie(righe, "epoch", "eval_chrf")
    if x:
        ax2.plot(x, y, color=C_CHRF, lw=1.8, marker="o", ms=5, label="chrF++")
    x, y = serie(righe, "epoch", "eval_bleu")
    if x:
        ax2.plot(x, y, color="#937860", lw=1.4, ls="--", marker="^", ms=5,
                 alpha=.8, label="BLEU")
    ax2.set_title("chrF++ e BLEU su dev", fontsize=11)
    ax2.set_xlabel("epoca")
    ax2.set_ylabel("punteggio (0-100)")
    ax2.legend(fontsize=9)
    griglia(ax2)

    fig.suptitle("{} - T1 - Andamento delle metriche in validazione "
                 "(teacher forcing, proxy di monitoraggio)"
                 .format(modello.etichetta), fontsize=12)
    fig.tight_layout()
    salva(fig, cartella, "T1_prf1_curve.png", dpi)


def barre_prf1_t1(modello, cartella, dpi):
    """P/R/F1 finali su test: BERTScore (sistema vs pavimenti) + lessicali."""
    percorso = modello.dir_t1 / "metriche_finali.json"
    if not percorso.exists():
        print("  ! metriche_finali.json assente")
        return
    dati = leggi_json(percorso)
    chiave = next(iter(dati))
    m = dati[chiave]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    gruppi = [("sistema (fine-tuned)", m.get("bertscore_sistema"), C_F1),
              ("copia dell'italiano", m.get("bertscore_copia_italiano"), "#8C8C8C"),
              ("pavimento (nap. non correlato)", m.get("bertscore_pavimento"), "#D3D3D3")]
    gruppi = [g for g in gruppi if g[1]]
    etichette = ["Precision", "Recall", "F1-Score"]
    larghezza = 0.8 / max(len(gruppi), 1)
    posizioni = range(len(etichette))
    for i, (nome, valori, colore) in enumerate(gruppi):
        altezze = [valori["P"], valori["R"], valori["F1"]]
        x = [p - 0.4 + larghezza * (i + .5) for p in posizioni]
        barre = ax1.bar(x, altezze, larghezza * .92, label=nome, color=colore,
                        edgecolor="white", linewidth=.6)
        etichette_barre(ax1, barre)
    ax1.set_xticks(list(posizioni))
    ax1.set_xticklabels(etichette)
    ax1.set_ylim(0, 1.25)
    ax1.set_ylabel("BERTScore")
    ax1.set_title("BERTScore su test (n={})\n".format(m.get("n", "?")) +
                  "il pavimento e' il vero zero della scala", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right", framealpha=.95)
    griglia(ax1)

    les = m.get("lessicali", {})
    valori = [les.get("precisione_dialettale"), les.get("recall_dialettale"),
              les.get("f1_dialettale")]
    if all(v is not None for v in valori):
        barre = ax2.bar(etichette, valori, 0.55, color=[C_P, C_R, C_F1],
                        edgecolor="white", linewidth=.6)
        etichette_barre(ax2, barre)
    extra = {"tasso italianismi": les.get("tasso_italianismi"),
             "tasso di copia": les.get("tasso_copia")}
    extra = dict((k, v) for k, v in extra.items() if v is not None)
    if extra:
        barre = ax2.bar(list(extra), list(extra.values()), 0.55, color="#C44E52",
                        alpha=.75, edgecolor="white", linewidth=.6)
        etichette_barre(ax2, barre)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("valore")
    ax2.set_title("Metriche lessicali dialettali su test\n"
                  "(a destra: piu' basso e' meglio)", fontsize=10)
    ax2.tick_params(axis="x", labelrotation=12)
    griglia(ax2)

    fig.suptitle("{} - T1 - Precision, Recall e F1-Score finali"
                 .format(modello.etichetta), fontsize=13)
    fig.tight_layout()
    salva(fig, cartella, "T1_prf1_finale.png", dpi)


def barre_chrf_t1(modello, cartella, dpi):
    """chrF++ finale del sistema contro le baseline obbligatorie."""
    finali = modello.dir_t1 / "metriche_finali.json"
    valutazione = next(iter(sorted((modello.dir_t1 / "eval").glob("*.metrics.json"))),
                       None)
    if not finali.exists() and valutazione is None:
        print("  ! nessuna metrica finale di chrF per T1")
        return

    sistema = ci = None
    baseline = {}
    if finali.exists():
        m = next(iter(leggi_json(finali).values()))
        sistema = m.get("chrf++")
    if valutazione is not None:
        e = leggi_json(valutazione)
        sistema = e.get("F_riferimento", {}).get("chrf++", sistema)
        boot = e.get("F_bootstrap_chrf", {})
        if "ci95_basso" in boot and "ci95_alto" in boot and sistema is not None:
            ci = ([sistema - boot["ci95_basso"]], [boot["ci95_alto"] - sistema])
        b = e.get("F_baseline", {})
        if "copia_italiano" in b:
            baseline["copia\ndell'italiano"] = b["copia_italiano"]
        if "riferimento_di_un_altro_item" in b:
            baseline["riferimento di\nun altro item"] = b["riferimento_di_un_altro_item"]

    if sistema is None:
        print("  ! chrF++ del sistema non trovato")
        return

    nomi = ["sistema\n(fine-tuned)"] + list(baseline)
    valori = [sistema] + list(baseline.values())
    colori = [C_CHRF] + [C_BASE] * len(baseline)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    barre = ax.bar(nomi, valori, 0.55, color=colori, edgecolor="white", linewidth=.6)
    if ci:
        ax.errorbar([0], [sistema], yerr=ci, fmt="none", ecolor="#333",
                    capsize=6, lw=1.4, label="IC 95% (bootstrap)")
        ax.legend(fontsize=9, loc="upper right")
        # l'etichetta del sistema va sopra il baffo, non dentro
        ax.annotate("{:.2f}".format(sistema), xy=(0, sistema + ci[1][0]),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
        etichette_barre(ax, barre[1:], "{:.2f}")
    else:
        etichette_barre(ax, barre, "{:.2f}")
    ax.set_ylabel("chrF++ (0-100)")
    ax.set_ylim(0, max(valori) * 1.25)
    ax.set_title("{} - T1 - chrF++ su test contro le baseline"
                 .format(modello.etichetta), fontsize=12)
    griglia(ax)
    fig.tight_layout()
    salva(fig, cartella, "T1_chrf_finale.png", dpi)


# ------------------------------------------------------------------- T3 -----
#
# Il run di T3 non produce un metrics_history.csv e summary.json conserva la sola
# eval_loss: la train loss esiste soltanto negli output della cella di training
# salvati dentro fine_tuning_T3.ipynb, ed e' da li' che viene letta.

RE_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RE_RECORD = re.compile(r"\{[^{}]*'(?:eval_)?loss'[^{}]*\}")
RE_COPPIA = re.compile(r"'(\w+)':\s*'?(-?[\d.]+(?:e[-+]?\d+)?)'?")
RE_CTX_DELTA = re.compile(r"ctx_delta\s+([-+]?[\d.]+)\s*nat/token")


def testo_output(cella):
    """Concatena l'output testuale di una cella di notebook (stream + text/plain)."""
    pezzi = []
    for o in cella.get("outputs", []):
        testo = o.get("text") or o.get("data", {}).get("text/plain", "")
        pezzi.append("".join(testo) if isinstance(testo, list) else str(testo))
    return RE_ANSI.sub("", "".join(pezzi))


def log_training_t3(percorso):
    """Estrae (train, eval, ctx_delta) dalla cella di training di fine_tuning_T3.

    Le celle candidate sono quelle che lanciano t3_train.py; lo smoke test viene
    escluso per tag. Fra le rimanenti si tiene quella con piu' record, cosi' un
    eventuale rilancio parziale non sostituisce il run completo."""
    nb = leggi_json(percorso)
    migliore = ([], [], [])
    for cella in nb.get("cells", []):
        if cella.get("cell_type") != "code":
            continue
        sorgente = "".join(cella.get("source", []))
        if "t3_train.py" not in sorgente or "SMOKE" in sorgente:
            continue
        testo = testo_output(cella)
        train, valutazione = [], []
        for record in RE_RECORD.findall(testo):
            campi = dict(RE_COPPIA.findall(record))
            if "epoch" not in campi:
                continue
            epoca = float(campi["epoch"])
            if "eval_loss" in campi:
                valutazione.append((epoca, float(campi["eval_loss"])))
            elif "loss" in campi:          # train_loss finale non ha 'loss'
                train.append((epoca, float(campi["loss"])))
        ctx = [float(v) for v in RE_CTX_DELTA.findall(testo)]
        if len(train) + len(valutazione) > len(migliore[0]) + len(migliore[1]):
            migliore = (sorted(train), sorted(valutazione), ctx)
    return migliore


def sottotitolo_t3(valutazione, ctx):
    """Commento sotto il titolo, ricavato dai dati.

    Minerva e Gemma divergono in punti diversi e scelgono checkpoint diversi:
    la riga non puo' essere fissata a mano."""
    pezzi = []
    if valutazione:
        migliore = min(valutazione, key=lambda c: c[1])
        pezzi.append("minimo della eval loss all'epoca {:.2f}".format(migliore[0]))
    if ctx and len(ctx) == len(valutazione):
        i = max(range(len(ctx)), key=lambda k: ctx[k])
        pezzi.append("checkpoint scelto sul massimo di ctx_delta "
                     "(epoca {:.2f}), non sulla loss".format(valutazione[i][0]))
    return "; ".join(pezzi)


def curve_loss_t3(modello, cartella, dpi):
    """Loss di training e di validazione di T3, lette dal notebook."""
    if not modello.notebook_t3.exists():
        print("  ! notebook non trovato: " + str(modello.notebook_t3))
        return
    train, valutazione, ctx = log_training_t3(modello.notebook_t3)
    if not train and not valutazione:
        print("  ! nessun record di loss negli output del notebook "
              "(la cella di training e' stata ripulita?)")
        return

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    if train:
        x, y = zip(*train)
        ax.plot(x, y, color=C_TRAIN, lw=1.4, alpha=.85, marker="o", ms=3,
                label="train loss")
    if valutazione:
        x, y = zip(*valutazione)
        ax.plot(x, y, color=C_EVAL, lw=1.8, marker="s", ms=6, label="eval loss")
        migliore = min(valutazione, key=lambda c: c[1])
        ax.axvline(migliore[0], color=C_EVAL, ls=":", lw=1, alpha=.6)
        ax.annotate("min eval {:.3f}".format(migliore[1]), xy=migliore,
                    xytext=(8, -12), textcoords="offset points",
                    fontsize=8, color=C_EVAL, va="top")
        # il checkpoint non viene scelto sulla loss ma sul massimo di ctx_delta
        if len(ctx) == len(valutazione):
            i = max(range(len(ctx)), key=lambda k: ctx[k])
            ax.axvline(valutazione[i][0], color="#2ca02c", ls="--", lw=1.2,
                       label="checkpoint scelto (ctx_delta={:+.4f})".format(ctx[i]))

    ax.set_xlabel("epoca")
    ax.set_ylabel("loss")
    ax.set_title("{} - T3 - Loss function\n{}"
                 .format(modello.etichetta,
                         sottotitolo_t3(valutazione, ctx)),
                 fontsize=10)
    ax.legend(fontsize=9)
    griglia(ax)
    fig.tight_layout()
    salva(fig, cartella, "T3_loss.png", dpi)


# ----------------------------------------------------------------- main -----

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["T1", "T3", "tutti"], default="tutti")
    ap.add_argument("--modello", choices=list(MODELLI) + ["tutti"], default="tutti",
                    help="modello da graficare (default: tutti)")
    ap.add_argument("--out", default=None,
                    help="cartella di destinazione: i grafici finiscono in "
                         "<out>/<modello>/<task>/. Senza questa opzione ogni "
                         "modello scrive nella propria cartella dei grafici")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    scelti = (list(MODELLI.values()) if args.modello == "tutti"
              else [MODELLI[args.modello]])

    for modello in scelti:
        if not modello.radice.is_dir():
            print("== {} ==  cartella assente ({}), salto"
                  .format(modello.etichetta, modello.radice))
            continue
        base = (modello.grafici if args.out is None
                else Path(args.out) / modello.etichetta)
        print("== " + modello.etichetta + " ==")

        if args.task in ("T1", "tutti"):
            if modello.dir_t1.is_dir():
                print("T1  <- " + str(modello.dir_t1.relative_to(RADICE)))
                cartella = base / "T1"
                curve_loss_t1(modello, cartella, args.dpi)
                curve_prf1_t1(modello, cartella, args.dpi)
                barre_prf1_t1(modello, cartella, args.dpi)
                barre_chrf_t1(modello, cartella, args.dpi)
            else:
                print("T1  - non presente per " + modello.etichetta + ", salto")

        if args.task in ("T3", "tutti"):
            if modello.dir_t3.is_dir():
                print("T3  <- " + str(modello.dir_t3.relative_to(RADICE)))
                curve_loss_t3(modello, base / "T3", args.dpi)
            else:
                print("T3  - non presente per " + modello.etichetta + ", salto")


if __name__ == "__main__":
    main()
