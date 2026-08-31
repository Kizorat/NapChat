#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grafici delle metriche di T1 (traduzione con contesto), di T2 (completamento di
turno) e della loss di T3, per ognuno dei modelli fine-tunati (Minerva7B,
Gemma4B, Llama7B).

Per T1 (da runs/, cpt/, metriche_finali.json, eval/):
  - la curva della loss (train + eval) dei tre stadi di addestramento
  - le curve di Precision, Recall, F1 e chrF++ in validazione
  - le barre di Precision, Recall, F1 finali su test
  - le barre di chrF++ finale contro le baseline

Per T2 (da runs/<run>/summary.json, da eval_v2/ e dal notebook; Minerva tiene la
loss negli output del notebook, Gemma nel log_history del summary):
  - la curva della loss (train + eval)
  - l'andamento della perplessita' sul target (ppl_target) in addestramento
  - l'andamento della ctx_accuracy (quanto il modello usa davvero il contesto)
  - il confronto fra zero-shot, few-shot-4 e fine-tuned

Per T3 (dagli output salvati nel notebook fine_tuning_T3.ipynb):
  - la curva della loss (train + eval)

I task assenti per un modello vengono semplicemente saltati: T1 esiste solo per
Minerva7B, T2 per Minerva7B e Gemma4B. Senza --out ogni modello scrive nella
propria cartella dei grafici (Minerva7B/grafici_minerva7B/,
Gemma4B/grafici_Gemma4B/, Llama7B/grafici_Llama7B/), in una sottocartella per task.

Uso:
    python scripts/grafici_metriche.py                     # tutto, tutti i modelli
    python scripts/grafici_metriche.py --modello llama     # solo Llama7B
    python scripts/grafici_metriche.py --task T2     # solo i grafici di T2
    python scripts/grafici_metriche.py --out mia_cartella --dpi 300
"""

import argparse
import csv
import json
import re
import textwrap
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


def primo_file(cartella, motivo):
    """Primo file che corrisponde al motivo, None se non ce n'e' nessuno.

    Serve per i notebook, il cui nome non e' uniforme fra i task:
    fine-tuning_T2.ipynb con il trattino, fine_tuning_T3.ipynb con l'underscore."""
    trovati = sorted(cartella.glob(motivo)) if cartella.is_dir() else []
    return trovati[0] if trovati else None


class Modello:
    """Un modello fine-tunato: dove stanno i suoi task e dove finiscono i grafici."""

    NOMI_T1 = ("T1_Traduzione", "t1_traduzione")
    NOMI_T2 = ("T2_completamento_dialogo", "t2_completamento_dialogo")
    NOMI_T3 = ("T3_generazione_libera", "t3_generazione_libera",
               "T3_gemerazione_libera")

    def __init__(self, chiave, etichetta, cartella, cartella_grafici):
        self.chiave = chiave
        self.etichetta = etichetta
        self.radice = RADICE / cartella
        self.grafici = self.radice / cartella_grafici
        self.dir_t1 = prima_esistente(self.radice, *self.NOMI_T1)
        self.dir_t2 = prima_esistente(self.radice, *self.NOMI_T2)
        self.dir_t3 = prima_esistente(self.radice, *self.NOMI_T3)
        self.notebook_t2 = primo_file(self.dir_t2 / "notebook", "*T2*.ipynb")
        self.notebook_t3 = self.dir_t3 / "notebook" / "fine_tuning_T3.ipynb"


MODELLI = {
    "minerva": Modello("minerva", "Minerva7B", "Minerva7B", "grafici_minerva7B"),
    "gemma": Modello("gemma", "Gemma4B", "Gemma4B", "grafici_Gemma4B"),
    "llama": Modello("llama", "Llama7B", "Llama7B", "grafici_Llama7B"),
}

# palette coerente fra tutti i grafici
C_TRAIN, C_EVAL = "#1f77b4", "#d62728"
C_P, C_R, C_F1 = "#4C72B0", "#DD8452", "#55A868"
C_CHRF, C_BASE = "#8172B3", "#C0C0C0"
# i tre sistemi confrontati in T2: stesso colore in tutti i grafici del task
COLORI_T2 = {"zero-shot": "#8C8C8C", "few-shot-4": "#DD8452",
             "fine-tuned": "#55A868"}


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


def a_capo(testo, larghezza=96):
    """Manda a capo un titolo lungo: senza questo bbox_inches="tight" allarga
    la figura fino a farci stare il titolo su una riga sola."""
    return "\n".join(textwrap.wrap(testo, larghezza)) if testo else ""


def griglia(ax):
    ax.grid(alpha=.3, linestyle="--", linewidth=.7)
    ax.set_axisbelow(True)


# ------------------------------------------------- lettura dei notebook -----
#
# Ne' T2 ne' T3 salvano su disco la loss di training per intero: il
# trainer_state del checkpoint si ferma alla propria epoca e summary.json
# conserva la sola eval_loss finale. La serie completa esiste solo negli output
# della cella di training rimasti dentro il notebook, ed e' da li' che si legge.

RE_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RE_RECORD = re.compile(r"\{[^{}]*'(?:eval_)?loss'[^{}]*\}")
RE_COPPIA = re.compile(r"'(\w+)':\s*'?(-?[\d.]+(?:e[-+]?\d+)?)'?")


def testo_output(cella):
    """Concatena l'output testuale di una cella di notebook (stream + text/plain)."""
    pezzi = []
    for o in cella.get("outputs", []):
        testo = o.get("text") or o.get("data", {}).get("text/plain", "")
        pezzi.append("".join(testo) if isinstance(testo, list) else str(testo))
    return RE_ANSI.sub("", "".join(pezzi))


def testo_cella_training(percorso, script):
    """Output della cella che lancia `script`, "" se non ce n'e' nessuna.

    Lo smoke test viene escluso per tag. Fra le celle rimaste si tiene quella
    con piu' record di loss: cosi' un rilancio parziale non sostituisce il run
    completo, e la cella che si limita a scrivere lo script su disco (stesso
    nome nel sorgente, nessun output) non viene mai scelta."""
    nb = leggi_json(percorso)
    migliore, punteggio = "", 0
    for cella in nb.get("cells", []):
        if cella.get("cell_type") != "code":
            continue
        sorgente = "".join(cella.get("source", []))
        if script not in sorgente or "SMOKE" in sorgente:
            continue
        testo = testo_output(cella)
        n = len(RE_RECORD.findall(testo))
        if n > punteggio:
            migliore, punteggio = testo, n
    return migliore


def loss_da_testo(testo):
    """I record di log del Trainer -> (train, eval), liste di (epoca, loss)."""
    train, valutazione = [], []
    for record in RE_RECORD.findall(testo):
        campi = dict(RE_COPPIA.findall(record))
        if "epoch" not in campi:
            continue
        epoca = float(campi["epoch"])
        if "eval_loss" in campi:
            valutazione.append((epoca, float(campi["eval_loss"])))
        elif "loss" in campi:              # train_loss finale non ha 'loss'
            train.append((epoca, float(campi["loss"])))
    return sorted(train), sorted(valutazione)


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


# ------------------------------------------------------------------- T2 -----
#
# T2 (completamento del turno in napoletano). Le fonti:
#   * runs/<run>/summary.json -> storia_ctx (ppl_target, acc_token e ctx_acc a
#     ogni valutazione intermedia), log_history (la loss), iperparametri e
#     checkpoint scelto
#   * eval_v2/*.metrics.json  -> le metriche finali dei sistemi a confronto
#     (zero-shot, few-shot-4, fine-tuned) su dev e su test
#   * il notebook             -> la loss di training, quando il run non salva
#     log_history (e' il caso di Minerva)

# ordine di preferenza fra le decodifiche, usato solo per sciogliere le parita':
# quale sia davvero quella comune ai sistemi lo decide decodifica_comune()
DECOD_PREFERITA = ("beam", "greedy", "contrastiva", "campionamento")
SISTEMI_T2 = ("zero-shot", "few-shot-4", "fine-tuned")

RE_CTX_STEP = re.compile(r"\[ctx\] step (\d+): ctx_acc=([\d.]+)\s+"
                         r"acc_token=([\d.]+)\s+ppl_target=([\d.]+)")
RE_CHECKPOINT = re.compile(r"checkpoint-(\d+)")


# --- lettura -----------------------------------------------------------------

def sommario_t2(modello):
    """runs/<run>/summary.json, {} se il run non e' in locale.

    La cartella runs/ e' esclusa dal versionamento (pesa quanto gli adapter):
    su un clone senza run i grafici ricadono sul notebook."""
    trovati = sorted((modello.dir_t2 / "runs").glob("*/summary.json"))
    return leggi_json(trovati[-1]) if trovati else {}


def storia_ctx_t2(modello, sommario):
    """Le valutazioni intermedie: [{step, ppl_target, acc_token, ctx_acc}, ...].

    L'ultima voce di storia_ctx ripete la valutazione finale, che e' quella del
    checkpoint migliore ricaricato e non un nuovo step: gli step gia' visti si
    scartano. Senza summary.json si ripiega sulle righe "[ctx] step ..." che il
    training stampa nel notebook."""
    voci = sommario.get("storia_ctx") or []
    if not voci and modello.notebook_t2:
        testo = testo_cella_training(modello.notebook_t2, "train_t2.py")
        voci = [{"step": int(s), "ctx_acc": float(c), "acc_token": float(a),
                 "ppl_target": float(p)}
                for s, c, a, p in RE_CTX_STEP.findall(testo)]
    fuori, visti = [], set()
    for v in voci:
        if v.get("step") in visti:
            continue
        visti.add(v.get("step"))
        fuori.append(v)
    return sorted(fuori, key=lambda v: v["step"])


def loss_da_log_history(sommario):
    """(train, eval) dal log_history del Trainer conservato in summary.json.

    E' la sorgente per i run il cui notebook e' stato ripulito degli output
    (Gemma). Come per storia_ctx, l'ultima valutazione ripete quella del
    checkpoint migliore ricaricato a fine addestramento invece di misurare
    l'ultimo step: degli step duplicati si tiene la prima occorrenza, l'unica
    che dice davvero come stava andando la eval loss li'."""
    train, valutazione, visti = [], [], set()
    for r in sommario.get("log_history") or []:
        if "epoch" not in r:
            continue
        if "eval_loss" in r:
            if r.get("step") in visti:
                continue
            visti.add(r.get("step"))
            valutazione.append((r["epoch"], r["eval_loss"]))
        elif "loss" in r:              # il record finale ha 'train_loss', non 'loss'
            train.append((r["epoch"], r["loss"]))
    return sorted(train), sorted(valutazione)


def step_scelto_t2(sommario):
    """Lo step del checkpoint promosso ad adapter finale, None se non risulta."""
    trovato = RE_CHECKPOINT.search(sommario.get("best_checkpoint") or "")
    return int(trovato.group(1)) if trovato else None


def indice_decod(nome):
    """Posizione nell'ordine di preferenza; le decodifiche ignote vanno in coda."""
    return (DECOD_PREFERITA.index(nome) if nome in DECOD_PREFERITA
            else len(DECOD_PREFERITA))


def ordine_decod(m, preferita=None):
    """Quanto e' preferibile la valutazione m: piu' basso, meglio."""
    nome = m.get("decodifica")
    return -1 if preferita is not None and nome == preferita else indice_decod(nome)


def decodifica_comune(valutazioni, split):
    """La decodifica con cui e' stato valutato il maggior numero di sistemi.

    Non e' la stessa per tutti i modelli: su dev Minerva ha zero-shot e
    few-shot in beam, Gemma in contrastiva. Fissarne una a priori farebbe
    confrontare il fine-tuned in beam con le baseline in contrastiva, cioe'
    misurare la decodifica invece del sistema: la si ricava dai dati. A parita'
    di copertura vince l'ordine di DECOD_PREFERITA."""
    copertura = {}
    for m in valutazioni:
        if m.get("split") != split or not m.get("decodifica"):
            continue
        copertura.setdefault(m["decodifica"], set()).add(m.get("sistema"))
    if not copertura:
        return None
    return min(copertura,
               key=lambda d: (-len(copertura[d]), indice_decod(d)))


def valutazioni_t2(modello):
    """Le metriche finali salvate in eval_v2/*.metrics.json."""
    return [leggi_json(p) for p in
            sorted((modello.dir_t2 / "eval_v2").glob("*.metrics.json"))]


def sistemi_su(valutazioni, split):
    """[(sistema, metriche)] su uno split, un sistema solo per decodifica.

    Del fine-tuned su dev esistono tre valutazioni (beam, greedy, contrastiva):
    si tiene quella nella decodifica comune agli altri sistemi, altrimenti il
    confronto misura la decodifica invece del sistema. Le sezioni assenti dal
    file scelto (la 'scelta' manca dai run in beam e greedy) si ripescano dalle
    altre valutazioni dello stesso sistema, che condividono modello, adapter e
    split: teacher forcing e accuratezza di scelta non dipendono dalla
    decodifica."""
    preferita = decodifica_comune(valutazioni, split)
    scelte = {}
    for m in valutazioni:
        if m.get("split") != split:
            continue
        attuale = scelte.get(m.get("sistema"))
        if attuale is None or ordine_decod(m, preferita) < ordine_decod(attuale,
                                                                       preferita):
            scelte[m["sistema"]] = m
    for sistema, m in scelte.items():
        for altra in valutazioni:
            if altra.get("split") != split or altra.get("sistema") != sistema:
                continue
            for sezione in ("teacher_forcing", "scelta"):
                if sezione not in m and sezione in altra:
                    m[sezione] = altra[sezione]
    return [(s, scelte[s]) for s in SISTEMI_T2 if s in scelte]


# --- grafici -----------------------------------------------------------------

def asse_epoche(ax, step_per_epoca):
    """Secondo asse x in epoche sopra quello in step, se il passo e' noto."""
    if not step_per_epoca:
        return
    secondo = ax.secondary_xaxis(
        "top", functions=(lambda s: s / step_per_epoca,
                          lambda e: e * step_per_epoca))
    secondo.set_xlabel("epoca", fontsize=9)
    secondo.tick_params(labelsize=8)


def sottotitolo_loss_t2(sommario, train, valutazione):
    """Commento sotto il titolo, ricavato dai dati e non fissato a mano."""
    pezzi = []
    if valutazione:
        migliore = min(valutazione, key=lambda c: c[1])
        pezzi.append("minimo della eval loss all'epoca {:.2f}".format(migliore[0]))
    iper = sommario.get("iperparametri", {})
    previste = iper.get("epochs")
    ultima = max([c[0] for c in train + valutazione] or [0])
    if previste and ultima < previste - .5:
        pezzi.append("addestramento fermato a {:.2f} epoche su {:g} "
                     "(early stopping, patience={})"
                     .format(ultima, previste, iper.get("patience", "?")))
    if train and len(valutazione) > 1:
        i = min(range(len(valutazione)), key=lambda k: valutazione[k][1])
        if (i < len(valutazione) - 1 and valutazione[-1][1] > valutazione[i][1]
                and train[-1][1] < train[0][1]):
            pezzi.append("dopo il minimo la train continua a scendere e la eval "
                         "risale: da li' in poi il modello memorizza")
    return "; ".join(pezzi)


def curve_loss_t2(modello, cartella, dpi):
    """Loss di training e di validazione di T2, lette dal notebook."""
    sommario = sommario_t2(modello)
    train, valutazione = [], []
    if modello.notebook_t2:
        train, valutazione = loss_da_testo(
            testo_cella_training(modello.notebook_t2, "train_t2.py"))
    if not train and not valutazione:
        train, valutazione = loss_da_log_history(sommario)
    parziale = False
    if not train and not valutazione:
        # ripiego: il trainer_state del checkpoint salvato. Si ferma alla
        # propria epoca, quindi mostra la discesa ma non la divergenza dopo
        stati = sorted((modello.dir_t2 / "runs")
                       .glob("*/checkpoint-*/trainer_state.json"))
        if stati:
            log = leggi_json(stati[-1])["log_history"]
            train = sorted((d["epoch"], d["loss"]) for d in log if "loss" in d)
            valutazione = sorted((d["epoch"], d["eval_loss"]) for d in log
                                 if "eval_loss" in d)
            parziale = True
    if not train and not valutazione:
        print("  ! nessuno storico di loss trovato per T2")
        return

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    if train:
        x, y = zip(*train)
        ax.plot(x, y, color=C_TRAIN, lw=1.4, alpha=.85, marker="o", ms=3,
                label="train loss")
    if valutazione:
        x, y = zip(*valutazione)
        ax.plot(x, y, color=C_EVAL, lw=1.8, marker="s", ms=6, label="eval loss")
        migliore = min(valutazione, key=lambda c: c[1])
        ax.axvline(migliore[0], color="#2ca02c", ls="--", lw=1.2,
                   label="checkpoint scelto (eval loss {:.3f})".format(migliore[1]))
        ax.annotate("min eval {:.3f}".format(migliore[1]), xy=migliore,
                    xytext=(8, -12), textcoords="offset points",
                    fontsize=8, color=C_EVAL, va="top")
    ax.set_xlabel("epoca")
    ax.set_ylabel("loss")
    sotto = sottotitolo_loss_t2(sommario, train, valutazione)
    if parziale:
        sotto = ("serie ricavata dal solo checkpoint salvato, si ferma li'"
                 + ("; " + sotto if sotto else ""))
    ax.set_title("{} - T2 - Loss function\n{}"
                 .format(modello.etichetta, a_capo(sotto, 78)), fontsize=10)
    ax.legend(fontsize=9)
    griglia(ax)
    fig.tight_layout()
    salva(fig, cartella, "T2_loss.png", dpi)


def curva_ppl_t2(modello, cartella, dpi):
    """Andamento della perplessita' sul target durante l'addestramento."""
    sommario = sommario_t2(modello)
    storia = storia_ctx_t2(modello, sommario)
    if not storia:
        print("  ! nessuna storia di ppl_target per T2")
        return
    passi = [v["step"] for v in storia]
    ppl = [v["ppl_target"] for v in storia]

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    ax.plot(passi, ppl, color=C_CHRF, lw=1.8, marker="o", ms=6,
            label="ppl_target (dev, teacher forcing)")
    i = min(range(len(ppl)), key=lambda k: ppl[k])
    ax.annotate("min {:.2f}".format(ppl[i]), xy=(passi[i], ppl[i]),
                xytext=(6, 8), textcoords="offset points", fontsize=9,
                color=C_CHRF)
    scelto = step_scelto_t2(sommario)
    if scelto in passi:
        ax.axvline(scelto, color="#2ca02c", ls="--", lw=1.2,
                   label="checkpoint scelto (step {})".format(scelto))

    # la ppl di partenza e' quella del modello non addestrato sullo stesso dev.
    # Tenerla dentro l'asse costa poco su Minerva (99.6 contro una curva che
    # arriva a 43.8) ma non su Gemma, dove lo zero-shot sta a 3095 e la curva
    # 37-102 finirebbe appiattita sull'asse: sopra la soglia si rinuncia alla
    # riga e il valore resta nella legenda, dichiarato fuori scala
    QUOTA_MINIMA = .10       # frazione dell'asse che la curva deve occupare
    base = dict(sistemi_su(valutazioni_t2(modello), "dev")).get("zero-shot", {})
    partenza = base.get("teacher_forcing", {}).get("ppl_target")
    if partenza:
        alto = max(ppl + [partenza]) * 1.08
        if (max(ppl) - min(ppl)) / alto >= QUOTA_MINIMA:
            ax.axhline(partenza, color=C_BASE, ls=":", lw=1.4,
                       label="zero-shot su dev ({:.1f})".format(partenza))
            ax.set_ylim(top=alto)
        else:
            # riga fantasma: sta fuori dai limiti, serve solo per la legenda
            ax.plot([], [], color=C_BASE, ls=":", lw=1.4,
                    label="zero-shot su dev ({:.1f}, fuori scala)".format(partenza))
            ax.set_ylim(top=max(ppl) * 1.15)

    ax.set_xlabel("step di addestramento")
    ax.set_ylabel("perplessita' sul target (piu' bassa e' meglio)")
    ax.set_xticks(passi)
    asse_epoche(ax, sommario.get("step_per_epoca"))
    ax.set_title("{} - T2 - Andamento della ppl_target\n"
                 "misurata sul solo target, con contesto e prefisso veri in input"
                 .format(modello.etichetta), fontsize=10)
    ax.legend(fontsize=9)
    griglia(ax)
    fig.tight_layout()
    salva(fig, cartella, "T2_ppl_target.png", dpi)


def curva_ctx_acc_t2(modello, cartella, dpi):
    """Andamento della ctx_accuracy: quanto il modello usa davvero il contesto."""
    sommario = sommario_t2(modello)
    storia = storia_ctx_t2(modello, sommario)
    if not storia:
        print("  ! nessuna storia di ctx_acc per T2")
        return
    passi = [v["step"] for v in storia]
    ctx = [v["ctx_acc"] for v in storia]

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    ax.plot(passi, ctx, color=C_F1, lw=1.8, marker="o", ms=6,
            label="ctx_accuracy (dev)")
    i = max(range(len(ctx)), key=lambda k: ctx[k])
    ax.annotate("max {:.3f}".format(ctx[i]), xy=(passi[i], ctx[i]),
                xytext=(6, 8), textcoords="offset points", fontsize=9, color=C_F1)
    scelto = step_scelto_t2(sommario)
    if scelto in passi:
        ax.axvline(scelto, color="#2ca02c", ls="--", lw=1.2,
                   label="checkpoint scelto (step {})".format(scelto))
    # il confronto e' fra contesto vero e contesto di un altro turno: due
    # alternative, quindi il caso vale 0.5 e non 0
    ax.axhline(.5, color="#C44E52", ls=":", lw=1.4, label="caso (0.5)")
    base = dict(sistemi_su(valutazioni_t2(modello), "dev")).get("zero-shot", {})
    partenza = base.get("teacher_forcing", {}).get("ctx_acc")
    if partenza:
        ax.axhline(partenza, color=C_BASE, ls=":", lw=1.4,
                   label="zero-shot su dev ({:.3f})".format(partenza))

    n = storia[0].get("n_item_ctx")
    ax.set_xlabel("step di addestramento")
    ax.set_ylabel("ctx_accuracy")
    ax.set_ylim(.4, 1.0)
    ax.set_xticks(passi)
    asse_epoche(ax, sommario.get("step_per_epoca"))
    ax.set_title("{} - T2 - Andamento della ctx_accuracy\n"
                 "quota di item in cui il contesto vero batte un contesto "
                 "estraneo{}".format(modello.etichetta,
                                     " (n={})".format(n) if n else ""),
                 fontsize=10)
    ax.legend(fontsize=9, loc="lower left")
    griglia(ax)
    fig.tight_layout()
    salva(fig, cartella, "T2_ctx_accuracy.png", dpi)


def barre_sistemi(ax, coppie, valore, colori=None, formato="{:.3f}"):
    """Una barra per sistema con l'etichetta sopra; salta i valori assenti."""
    nomi = [s for s, m in coppie if valore(m) is not None]
    altezze = [valore(m) for s, m in coppie if valore(m) is not None]
    if not nomi:
        return []
    barre = ax.bar(nomi, altezze, .55,
                   color=colori or [COLORI_T2.get(s, C_BASE) for s in nomi],
                   edgecolor="white", linewidth=.6)
    etichette_barre(ax, barre, formato)
    ax.tick_params(axis="x", labelrotation=8)
    return altezze


def confronto_sistemi_t2(modello, cartella, dpi, split="dev"):
    """Zero-shot, few-shot-4 e fine-tuned a confronto sullo stesso split."""
    coppie = sistemi_su(valutazioni_t2(modello), split)
    if len(coppie) < 2:
        print("  ! meno di due sistemi valutati su " + split + ", salto")
        return
    tf = dict((s, m.get("teacher_forcing", {})) for s, m in coppie)
    gen = dict((s, m.get("generazione", {})) for s, m in coppie)
    baseline = next((m.get("baseline", {}) for s, m in coppie if m.get("baseline")), {})
    n = next((m.get("n") for s, m in coppie if m.get("n")), "?")

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(12.6, 8.8))

    # (a) chrF++ della generazione, contro le baseline obbligatorie
    altezze = barre_sistemi(a1, coppie,
                            lambda m: m.get("generazione", {}).get("chrf++"),
                            formato="{:.2f}")
    for nome, chiave, colore in (
            ("ripeti il prefisso", "ripeti_il_prefisso", "#C44E52"),
            ("ultimo turno di contesto", "ultimo_turno_di_contesto", "#937860"),
            ("pavimento (target mescolati)", "pavimento_target_mescolati", C_BASE)):
        if baseline.get(chiave) is not None:
            a1.axhline(baseline[chiave], ls=":", lw=1.4, color=colore,
                       label="{} ({:.2f})".format(nome, baseline[chiave]))
    a1.set_ylim(0, max(altezze + list(baseline.values()) or [1]) * 1.55)
    a1.set_ylabel("chrF++ (0-100)")
    # quanto vale davvero un punto di chrF++ su questo task lo dice la distanza
    # fra il pavimento e la baseline banale, non la scala 0-100
    forbice = ""
    if baseline.get("pavimento_target_mescolati") and baseline.get("ripeti_il_prefisso"):
        forbice = ("\nfra pavimento e baseline banale ci sono {:.1f} punti: "
                   "la scala utile e' tutta li'"
                   .format(baseline["ripeti_il_prefisso"]
                           - baseline["pavimento_target_mescolati"]))
    a1.set_title("Somiglianza al riferimento: chrF++" + forbice, fontsize=10)
    a1.legend(fontsize=7.5, loc="upper center", framealpha=.95)
    griglia(a1)

    # (b) perplessita' sul target. Scala lineare da zero: su una scala
    # logaritmica le barre partirebbero dal fondo dell'asse invece che da zero
    # e il divario letto a occhio sarebbe falso
    altezze = barre_sistemi(a2, coppie,
                            lambda m: m.get("teacher_forcing", {}).get("ppl_target"),
                            formato="{:.1f}")
    a2.set_ylim(0, max(altezze or [1]) * 1.25)
    a2.set_ylabel("ppl_target (piu' bassa e' meglio)")
    a2.set_title("Perplessita' sul target in teacher forcing", fontsize=10)
    griglia(a2)

    # (c) le accuratezze, tutte sulla stessa scala 0-1
    misure = [("acc_token", lambda m: m.get("teacher_forcing", {}).get("acc_token"),
               "Accuracy\n(token)"),
              ("ctx_acc", lambda m: m.get("teacher_forcing", {}).get("ctx_acc"),
               "ctx_accuracy"),
              ("scelta", lambda m: m.get("scelta", {}).get("accuratezza@N"),
               "Accuratezza\ndi scelta@N")]
    gruppi(a3, coppie, misure)
    a3.axhline(.5, color="#C44E52", ls=":", lw=1.2, label="caso per ctx_accuracy")
    fondo = next((m.get("scelta", {}).get("fondo_scala") for s, m in coppie
                  if m.get("scelta")), None)
    if fondo:
        a3.axhline(fondo, color="#937860", ls=":", lw=1.2,
                   label="caso per la scelta@N ({:g})".format(fondo))
    a3.set_ylim(0, 1.05)
    a3.set_ylabel("valore")
    a3.set_title("Accuratezze (piu' alte sono meglio)", fontsize=10)
    a3.legend(fontsize=7.5, ncol=2, loc="upper center", framealpha=.95)
    griglia(a3)

    # (d) come e' fatto il testo generato, non quanto somiglia al riferimento
    misure = [("dial", lambda m: m.get("generazione", {}).get("densita_dial_gen"),
               "Densita'\ndialettale"),
              ("copia", lambda m: m.get("generazione", {}).get("tasso_copia"),
               "Tasso\ndi copia"),
              ("len", lambda m: m.get("generazione", {}).get("rapporto_lunghezza"),
               "Rapporto di\nlunghezza")]
    gruppi(a4, coppie, misure)
    rif = next((g.get("densita_dial_rif") for g in gen.values()
                if g.get("densita_dial_rif") is not None), None)
    if rif is not None:
        a4.axhline(rif, color="#C44E52", ls=":", lw=1.4,
                   label="densita' dei riferimenti ({:.3f})".format(rif))
    a4.axhline(1.0, color="#937860", ls=":", lw=1.2,
               label="lunghezza pari al riferimento")
    a4.set_ylim(0, 1.6)
    a4.set_ylabel("valore")
    a4.set_title("Forma del testo generato\n"
                 "il bersaglio non e' il massimo ma la riga tratteggiata",
                 fontsize=10)
    a4.legend(fontsize=7.5, loc="upper left", framealpha=.95)
    griglia(a4)

    # la decodifica cambia da modello a modello: dirla evita di confrontare a
    # occhio grafici prodotti con impostazioni di generazione diverse
    decodifiche = sorted(set(m.get("decodifica") for _, m in coppie
                             if m.get("decodifica")))
    titolo = "{} - T2 - {} a confronto su {} (n={}{})".format(
        modello.etichetta, " / ".join(s for s, _ in coppie), split, n,
        ", decodifica " + "/".join(decodifiche) if decodifiche else "")
    if (tf.get("few-shot-4") and tf.get("zero-shot")
            and tf["few-shot-4"] == tf["zero-shot"]):
        # non e' un errore dei dati: il teacher forcing non vede il prompt
        titolo += "\n" + a_capo("gli esempi few-shot entrano solo nel prompt "
                                "di generazione: in teacher forcing few-shot e "
                                "zero-shot sono lo stesso modello", 88)
    fig.suptitle(titolo, fontsize=12)
    fig.tight_layout()
    salva(fig, cartella, "T2_sistemi_" + split + ".png", dpi)


def gruppi(ax, coppie, misure):
    """Barre raggruppate: un gruppo per misura, una barra per sistema."""
    larghezza = .8 / max(len(coppie), 1)
    posizioni = range(len(misure))
    for i, (sistema, m) in enumerate(coppie):
        altezze = [funzione(m) for _, funzione, _ in misure]
        x = [p - .4 + larghezza * (i + .5) for p in posizioni]
        presenti = [(xx, h) for xx, h in zip(x, altezze) if h is not None]
        if not presenti:
            continue
        barre = ax.bar([c[0] for c in presenti], [c[1] for c in presenti],
                       larghezza * .9, label=sistema,
                       color=COLORI_T2.get(sistema, C_BASE),
                       edgecolor="white", linewidth=.6)
        etichette_barre(ax, barre)
    ax.set_xticks(list(posizioni))
    ax.set_xticklabels([e for _, _, e in misure], fontsize=9)


# ------------------------------------------------------------------- T3 -----

RE_CTX_DELTA = re.compile(r"ctx_delta\s+([-+]?[\d.]+)\s*nat/token")


def log_training_t3(percorso):
    """(train, eval, ctx_delta) dalla cella di training di fine_tuning_T3."""
    testo = testo_cella_training(percorso, "t3_train.py")
    train, valutazione = loss_da_testo(testo)
    return train, valutazione, [float(v) for v in RE_CTX_DELTA.findall(testo)]


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
    ap.add_argument("--task", choices=["T1", "T2", "T3", "tutti"], default="tutti")
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

        if args.task in ("T2", "tutti"):
            if modello.dir_t2.is_dir():
                print("T2  <- " + str(modello.dir_t2.relative_to(RADICE)))
                cartella = base / "T2"
                curve_loss_t2(modello, cartella, args.dpi)
                curva_ppl_t2(modello, cartella, args.dpi)
                curva_ctx_acc_t2(modello, cartella, args.dpi)
                for split in ("dev", "test"):
                    confronto_sistemi_t2(modello, cartella, args.dpi, split)
            else:
                print("T2  - non presente per " + modello.etichetta + ", salto")

        if args.task in ("T3", "tutti"):
            if modello.dir_t3.is_dir():
                print("T3  <- " + str(modello.dir_t3.relative_to(RADICE)))
                curve_loss_t3(modello, base / "T3", args.dpi)
            else:
                print("T3  - non presente per " + modello.etichetta + ", salto")


if __name__ == "__main__":
    main()
