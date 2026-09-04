#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Valutazione umana con scala Likert (1-5) delle frasi generate dal chatbot
napoletano, sulle tre dimensioni previste dal progetto:

    naturalezza dialettale, accuratezza grammaticale, coerenza lessicale

Si valutano SEI sistemi: un modello per ciascuno dei tre fine-tuning (Minerva7B,
Llama7B, Gemma4B) su ciascuno dei due task generativi (T2 completamento di
dialogo, T3 generazione libera). Per ogni modello e per ogni task si prende la
configurazione che risponde meglio al contesto, non quella con il chrF piu' alto:
su questi due task il chrF confronta la frase con un solo riferimento fra i molti
completamenti validi, e premia percio' chi ricopia la forma del bersaglio, non
chi capisce la conversazione. Il criterio di scelta e' quindi il ctx_delta (T3) e
la ctx_accuracy (T2), che misurano quanto il modello cambi risposta al cambiare
del contesto, cioe' esattamente il "non risponde a caso" richiesto dal task.

Il disegno sperimentale ha due caratteristiche:

  - e' APPAIATO fra modelli: i tre modelli vengono valutati sulle stesse
    conversazioni di partenza, cosi' una differenza di punteggio e' attribuibile
    al modello e non alle frasi capitate in sorte all'uno o all'altro;
  - e' MESCOLATO: le frasi dei tre modelli sono alternate fra loro e con un
    campione di turni umani presi dal corpus, in ordine casuale e diverso per
    ciascun annotatore. I turni umani sono un termine di paragone: fissano il
    tetto realistico della scala, dato che nemmeno un parlante vero trascritto
    da parlato spontaneo prende punteggi pieni.

    Di norma la scheda dichiara da quale sistema viene ogni frase, cosi' che i
    voti restino rileggibili accanto al modello che li ha meritati; con
    'prepara --in-cieco' la colonna sparisce e la valutazione diventa cieca, il
    che rende il confronto fra modelli piu' difendibile ma le schede meno
    utili da rileggere.

Tre sottocomandi:

    elenca    mostra le metriche automatiche di tutti i sistemi disponibili e
              quale sarebbe scelto per ciascun modello e task.
    prepara   campiona gli item, li mescola e produce una scheda per ciascun
              annotatore (CSV da compilare + modulo HTML da aprire nel browser),
              piu' la chiave con la corrispondenza item -> sistema.
    analizza  rilegge le schede compilate, calcola medie, distribuzioni,
              classifica fra i modelli con confronti appaiati, differenza
              rispetto al riferimento umano e accordo fra annotatori, e produce
              grafici, tabelle LaTeX e report testuale.

Uso:
    python likert/likert_human_eval.py elenca
    python likert/likert_human_eval.py prepara
    # ... compilare likert/schede/scheda_<nnn>.csv (o il .html) ...
    python likert/likert_human_eval.py analizza
"""

import argparse
import csv
import html
import itertools
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# lo script vive dentro la cartella likert/, che e' anche la sua cartella di
# lavoro: schede, chiave e risultati stanno accanto a lui
CARTELLA = Path(__file__).resolve().parent
RADICE = CARTELLA.parent
GRAFICI_TESI = RADICE / "Progetto NLP" / "grafici_Likert"

# le tre dimensioni della scala, nell'ordine in cui compaiono ovunque
DIMENSIONI = [
    ("naturalezza", "Naturalezza dialettale"),
    ("grammatica", "Accuratezza grammaticale"),
    ("lessico", "Coerenza lessicale"),
]
CHIAVI_DIM = [c for c, _ in DIMENSIONI]

# chiave usata per il gruppo di controllo umano, accanto ai codici dei modelli
UMANO = "riferimento"

# Un sistema per modello e per task. Su T2 lo split di test espone un solo
# sistema fine-tunato per modello, quindi la scelta e' obbligata; su T3, dove
# convivono piu' strategie di decodifica, si prende quella con il ctx_delta piu'
# alto fra quelle valutate sull'intero test (le varianti g0.0-g1.2 sono
# un'ablazione su 64 item e non sono confrontabili con le altre).
SISTEMI = {
    "T2": {
        "etichetta": "Completamento di dialogo (T2)",
        "modelli": {
            "minerva7b": {
                "etichetta": "Minerva7B",
                "preds": "Minerva7B/T2_completamento_dialogo/eval_v2/"
                         "minerva__T2__finetuned__test.preds.jsonl",
                "motivo": "unico sistema fine-tunato su test; ctx_accuracy 0,660",
            },
            "llama7b": {
                "etichetta": "Llama7B",
                "preds": "Llama7B/T2_completamento_dialogo/eval_v2/"
                         "llama__T2__finetuned__test.preds.jsonl",
                "motivo": "unico sistema fine-tunato su test; ctx_accuracy 0,689",
            },
            "gemma4b": {
                "etichetta": "Gemma4B",
                "preds": "Gemma4B/T2_completamento_dialogo/eval_v2/"
                         "gemma__T2__finetuned__test.preds.jsonl",
                "motivo": "unico sistema fine-tunato su test; ctx_accuracy 0,689, "
                          "migliore scelta@N (0,396)",
            },
        },
    },
    "T3": {
        "etichetta": "Generazione libera (T3)",
        "modelli": {
            "minerva7b": {
                "etichetta": "Minerva7B",
                "preds": "Minerva7B/T3_gemerazione_libera/eval/"
                         "Minerva-7B-instruct-v1.0__T3__ft__cad.preds.jsonl",
                "motivo": "decodifica contestuale: ctx_delta 2,385 contro 1,048 "
                          "in greedy e 0,992 in nucleus",
            },
            "llama7b": {
                "etichetta": "Llama7B",
                "preds": "Llama7B/t3_generazione_libera/eval/"
                         "Llama-2-7b-chat-hf__T3__ft__cad.preds.jsonl",
                "motivo": "decodifica contestuale: ctx_delta 1,953 contro 0,026 "
                          "in greedy, dove il modello e' cieco al contesto",
            },
            "gemma4b": {
                "etichetta": "Gemma4B",
                "preds": "Gemma4B/t3_generazione_libera/eval/"
                         "gemma-3-4b-it__T3__ft__cad.preds.jsonl",
                "motivo": "decodifica contestuale: ctx_delta 1,796 contro 0,278 "
                          "in greedy; anche il rapporto di lunghezza migliora",
            },
        },
    },
}

ETICHETTE_GRUPPO = {"minerva7b": "Minerva7B", "llama7b": "Llama7B",
                    "gemma4b": "Gemma4B", UMANO: "Riferimento umano"}
COLORI = {"minerva7b": "#c1121f", "llama7b": "#457b9d", "gemma4b": "#7d8c5c",
          UMANO: "#8a8279"}
COLORI_SCALA = ["#b02418", "#dd7a4e", "#c9a227", "#6f9e5a", "#33683c"]

INTESTAZIONE_SCHEDA = ["item", "task", "contesto", "istruzione", "inizio_turno",
                       "testo_valutato"] + CHIAVI_DIM + ["note"]


def gruppi(task):
    """Codici dei gruppi da confrontare su un task: i tre modelli piu' l'umano."""
    return list(SISTEMI[task]["modelli"]) + [UMANO]


# ----------------------------------------------------------------------------
# lettura delle predizioni
# ----------------------------------------------------------------------------

def leggi_jsonl(percorso):
    righe = []
    with open(percorso, encoding="utf-8") as f:
        for riga in f:
            riga = riga.strip()
            if riga:
                righe.append(json.loads(riga))
    return righe


def normalizza(testo):
    """Spazi ridotti a uno solo: serve solo per confronti e controlli di vuoto."""
    return re.sub(r"\s+", " ", (testo or "")).strip()


def spezza_prompt_t3(prompt):
    """Dal prompt di T3 estrae (contesto conversazionale, istruzione finale).

    Il prompt ha forma:
        Conversazione in napoletano fra N persone (A, B).
        A: ...
        B: ...
        ---
        <istruzione>
    La prima riga di intestazione non serve all'annotatore e viene scartata."""
    if "\n---\n" in prompt:
        testa, istruzione = prompt.split("\n---\n", 1)
    else:
        testa, istruzione = prompt, ""
    righe = [r for r in testa.split("\n") if r.strip()]
    if righe and not re.match(r"^[A-Z]:", righe[0]):
        righe = righe[1:]
    return "\n".join(righe), istruzione.strip()


def campo_comune(task, riga):
    """Parte dell'item indipendente dal modello: contesto, istruzione, prefisso."""
    if task == "T2":
        return {"contesto": riga.get("contesto", ""),
                "istruzione": "Completa il turno gia' iniziato, in napoletano.",
                "inizio_turno": normalizza(riga.get("prefisso", ""))}
    contesto, istruzione = spezza_prompt_t3(riga.get("prompt", ""))
    return {"contesto": contesto, "istruzione": istruzione, "inizio_turno": ""}


def testo_generato(task, riga):
    """In T2 si preferisce il campo gia' ripulito dalla coda oltre il turno."""
    if task == "T2":
        return normalizza(riga.get("generato_tagliato") or riga.get("generato", ""))
    return normalizza(riga.get("generato", ""))


# ----------------------------------------------------------------------------
# sottocomando: elenca
# ----------------------------------------------------------------------------

def metriche_sintetiche(d):
    """Le poche metriche che contano, appiattite: T2 le annida, T3 no."""
    tf = d.get("teacher_forcing", {})
    gen = d.get("generazione", {})
    fuori = {"n": d.get("n"), "split": d.get("split"),
             "decodifica": d.get("decodifica")}
    for chiave, valore in (("ctx_acc", tf.get("ctx_acc")),
                           ("ppl_target", tf.get("ppl_target")),
                           ("ctx_delta", d.get("ctx_delta")),
                           ("chrf++", gen.get("chrf++")),
                           ("chrf", d.get("chrf")),
                           ("rapp_lung", gen.get("rapporto_lunghezza")
                            or d.get("rapporto_lunghezza"))):
        if valore is not None:
            fuori[chiave] = valore
    return {k: v for k, v in fuori.items() if v is not None}


def comando_elenca(_args):
    for task in ("T2", "T3"):
        for codice, mod in SISTEMI[task]["modelli"].items():
            cartella = (RADICE / mod["preds"]).parent
            print(f"\n=== {task} - {mod['etichetta']} ===")
            scelto = Path(mod["preds"]).name.replace(".preds.jsonl", "")
            for f in sorted(cartella.glob("*.metrics.json")):
                nome = f.name[:-13]
                d = json.load(open(f, encoding="utf-8"))
                marchio = " <-- SCELTO" if nome == scelto else ""
                print(f"  {nome:<48} {metriche_sintetiche(d)}{marchio}")
            print(f"  motivo: {mod['motivo']}")


# ----------------------------------------------------------------------------
# sottocomando: prepara
# ----------------------------------------------------------------------------

def campiona(task, n_sistema, n_umano, rng, min_parole):
    """Item del task: n_sistema conversazioni valutate su tutti e tre i modelli,
    piu' n_umano conversazioni distinte da cui si prende il turno umano vero.

    Le righe usate per il controllo umano sono disgiunte da quelle dei modelli:
    se lo stesso contesto comparisse sia con la frase generata sia con quella
    umana, l'annotatore finirebbe per giudicare l'una in funzione dell'altra
    invece che per quello che e'. Fra i modelli, invece, le righe sono le stesse
    di proposito, perche' e' cio' che rende il confronto appaiato."""
    modelli = SISTEMI[task]["modelli"]
    righe = {}
    for codice, mod in modelli.items():
        percorso = RADICE / mod["preds"]
        if not percorso.exists():
            raise SystemExit(f"ERRORE: predizioni non trovate: {percorso}")
        righe[codice] = leggi_jsonl(percorso)

    lunghezze = {c: len(v) for c, v in righe.items()}
    if len(set(lunghezze.values())) != 1:
        raise SystemExit(f"ERRORE: {task}, i modelli hanno un numero diverso di "
                         f"predizioni: {lunghezze}")
    riferimento = list(modelli)[0]
    for codice in modelli:
        if [r["id"] for r in righe[codice]] != [r["id"] for r in righe[riferimento]]:
            raise SystemExit(f"ERRORE: {task}, gli id di {codice} non coincidono "
                             f"con quelli di {riferimento}: il disegno appaiato "
                             f"richiede le stesse righe di partenza.")

    # una riga e' utilizzabile solo se tutti e tre i modelli e il riferimento
    # hanno prodotto qualcosa di abbastanza lungo da poter essere giudicato:
    # altrimenti i modelli verrebbero confrontati su insiemi diversi di frasi
    utilizzabili = []
    for i in range(len(righe[riferimento])):
        testi = {c: testo_generato(task, righe[c][i]) for c in modelli}
        rif = normalizza(righe[riferimento][i].get("riferimento", ""))
        if all(len(t.split()) >= min_parole for t in testi.values()) \
                and len(rif.split()) >= min_parole:
            utilizzabili.append((i, testi, rif))

    if len(utilizzabili) < n_sistema + n_umano:
        raise SystemExit(
            f"ERRORE: {task} ha solo {len(utilizzabili)} righe utilizzabili "
            f"(ne servono {n_sistema + n_umano}). Riduci --n-sistema / --n-umano.")

    ordine = list(range(len(utilizzabili)))
    rng.shuffle(ordine)
    item = []
    for k in ordine[:n_sistema]:
        i, testi, _ = utilizzabili[k]
        comune = campo_comune(task, righe[riferimento][i])
        for codice in modelli:
            item.append(dict(comune, task=task, gruppo=codice, condizione="sistema",
                             origine=righe[riferimento][i]["id"], testo=testi[codice]))
    for k in ordine[n_sistema:n_sistema + n_umano]:
        i, _, rif = utilizzabili[k]
        comune = campo_comune(task, righe[riferimento][i])
        item.append(dict(comune, task=task, gruppo=UMANO, condizione="umano",
                         origine=righe[riferimento][i]["id"], testo=rif))

    return item, {"righe_totali": len(righe[riferimento]),
                  "righe_utilizzabili": len(utilizzabili),
                  "conversazioni_per_modello": n_sistema,
                  "conversazioni_umane": n_umano,
                  "item_prodotti": len(item)}


def scrivi_scheda_csv(percorso, item, in_chiaro=False):
    """Scheda da compilare. Con in_chiaro=True aggiunge la colonna del modello:
    comodo per rileggere, ma toglie il cieco e quindi invalida i confronti."""
    intestazione = list(INTESTAZIONE_SCHEDA)
    if in_chiaro:
        intestazione.insert(2, "modello")
    with open(percorso, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=intestazione)
        w.writeheader()
        for it in item:
            riga = {
                "item": it["item"], "task": it["task"],
                "contesto": it["contesto"], "istruzione": it["istruzione"],
                "inizio_turno": it["inizio_turno"], "testo_valutato": it["testo"],
                "naturalezza": "", "grammatica": "", "lessico": "", "note": "",
            }
            if in_chiaro:
                riga["modello"] = ETICHETTE_GRUPPO[it["gruppo"]]
            w.writerow(riga)


def scrivi_chiave_csv(percorso, item):
    """Chiave leggibile: per ogni item, da quale task e da quale modello viene la
    frase, e da quale conversazione del corpus.

    La chiave e' cio' che rende tracciabile il disegno appaiato: gli item con la
    stessa `origine` sono le risposte dei tre modelli alla medesima
    conversazione. Va consultata solo a valutazione conclusa."""
    with open(percorso, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "task", "modello", "condizione", "origine",
                    "contesto", "istruzione", "inizio_turno", "testo"])
        for it in sorted(item, key=lambda x: (x["task"], x["origine"], x["gruppo"])):
            w.writerow([it["item"], it["task"], ETICHETTE_GRUPPO[it["gruppo"]],
                        it["condizione"], it["origine"], it["contesto"],
                        it["istruzione"], it["inizio_turno"], it["testo"]])


MODULO_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
       margin: 0 auto; max-width: 60rem; padding: 1.5rem; line-height: 1.5;
       background: #fbfaf8; color: #1c1a17; }
h1 { font-size: 1.4rem; margin: 0 0 .3rem; }
.sottotitolo { color: #6b645c; margin: 0 0 1.2rem; font-size: .9rem; }
details { margin-bottom: 1rem; font-size: .9rem; }
.barra { position: sticky; top: 0; background: #fbfaf8; padding: .7rem 0;
         border-bottom: 1px solid #ddd8d0; margin-bottom: 1rem; z-index: 5;
         display: flex; gap: .6rem; align-items: center; flex-wrap: wrap; }
button { font: inherit; padding: .45rem .9rem; border-radius: .4rem;
         border: 1px solid #b9b2a8; background: #fff; cursor: pointer; }
button.primario { background: #1c1a17; color: #fff; border-color: #1c1a17; }
.avanzamento { font-size: .85rem; color: #6b645c; }
.scheda { border: 1px solid #ddd8d0; border-radius: .6rem; background: #fff;
          padding: 1rem 1.2rem; margin-bottom: 1rem; }
.testata { display: flex; justify-content: space-between; font-size: .8rem;
           color: #6b645c; margin-bottom: .6rem; }
.contesto { white-space: pre-wrap; background: #f4f1ec; border-radius: .4rem;
            padding: .6rem .8rem; font-size: .9rem; margin-bottom: .5rem; }
.istruzione { font-size: .85rem; font-style: italic; color: #6b645c;
              margin-bottom: .6rem; }
.frase { font-size: 1.05rem; padding: .6rem .8rem; border-left: 3px solid #c1121f;
         background: #fdf6f5; border-radius: .3rem; margin-bottom: .9rem; }
.frase .prefisso { color: #8a8279; }
.modello { font-weight: 700; color: #c1121f; }
.dim { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap;
       margin-bottom: .35rem; font-size: .9rem; }
.dim .nome { width: 13rem; }
.dim label { border: 1px solid #ccc6bd; border-radius: .3rem; padding: .15rem .55rem;
             cursor: pointer; }
.dim input { display: none; }
.dim label:has(input:checked) { background: #1c1a17; color: #fff; border-color: #1c1a17; }
textarea { width: 100%; font: inherit; font-size: .85rem; border-radius: .3rem;
           border: 1px solid #ccc6bd; padding: .35rem .5rem; }
.fatta { border-color: #6f9e5a; }
@media (prefers-color-scheme: dark) {
  body { background: #171614; color: #ece8e1; }
  .barra { background: #171614; border-color: #35322d; }
  .scheda { background: #1f1e1b; border-color: #35322d; }
  .contesto { background: #26241f; }
  .frase { background: #2a1f1e; }
  button { background: #26241f; color: #ece8e1; border-color: #4a453e; }
  button.primario { background: #ece8e1; color: #171614; }
  textarea { background: #26241f; color: #ece8e1; border-color: #4a453e; }
}
"""

MODULO_JS = r"""
const DIM = ["naturalezza", "grammatica", "lessico"];
const CHIAVE = "likert_" + ANNOTATORE;

function carica() {
  try { return JSON.parse(localStorage.getItem(CHIAVE) || "{}"); }
  catch (e) { return {}; }
}
function salva(stato) {
  try { localStorage.setItem(CHIAVE, JSON.stringify(stato)); } catch (e) {}
}
let stato = carica();

function completo(id) {
  const v = stato[id] || {};
  return DIM.every(d => v[d]);
}
function aggiorna() {
  let fatti = 0;
  for (const it of ITEM) {
    const card = document.getElementById("card_" + it.item);
    if (!card) continue;
    if (completo(it.item)) { fatti++; card.classList.add("fatta"); }
    else { card.classList.remove("fatta"); }
  }
  document.getElementById("avanzamento").textContent =
    fatti + " / " + ITEM.length + " item valutati";
}
function ripristina() {
  for (const it of ITEM) {
    const v = stato[it.item] || {};
    for (const d of DIM) {
      if (v[d]) {
        const el = document.querySelector(
          'input[name="' + it.item + '_' + d + '"][value="' + v[d] + '"]');
        if (el) el.checked = true;
      }
    }
    const nota = document.getElementById("note_" + it.item);
    if (nota && v.note) nota.value = v.note;
  }
  aggiorna();
}
function registra(id, campo, valore) {
  stato[id] = stato[id] || {};
  stato[id][campo] = valore;
  salva(stato);
  aggiorna();
}
function virgolette(s) {
  s = (s === undefined || s === null) ? "" : String(s);
  return '"' + s.replace(/"/g, '""') + '"';
}
function costruisciCsv() {
  // la colonna del modello compare solo se il modulo e' stato generato in
  // chiaro: cosi' il CSV scaricato ha le stesse colonne della scheda
  const testata = ["item", "task"].concat(IN_CHIARO ? ["modello"] : [])
    .concat(["contesto", "istruzione", "inizio_turno", "testo_valutato",
             "naturalezza", "grammatica", "lessico", "note"]);
  const righe = [testata.join(",")];
  for (const it of ITEM) {
    const v = stato[it.item] || {};
    righe.push([it.item, it.task].concat(IN_CHIARO ? [it.modello] : [])
      .concat([it.contesto, it.istruzione, it.inizio_turno, it.testo,
               v.naturalezza || "", v.grammatica || "", v.lessico || "",
               v.note || ""]).map(virgolette).join(","));
  }
  return "﻿" + righe.join("\r\n");
}
function scarica() {
  const b = new Blob([costruisciCsv()], {type: "text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b);
  a.download = "scheda_" + ANNOTATORE + ".csv";
  document.body.appendChild(a); a.click(); a.remove();
}
async function copia() {
  try { await navigator.clipboard.writeText(costruisciCsv());
        alert("CSV copiato negli appunti."); }
  catch (e) { alert("Copia non riuscita: usa il pulsante di download."); }
}
function azzera() {
  if (!confirm("Cancellare tutte le valutazioni inserite?")) return;
  stato = {}; salva(stato);
  document.querySelectorAll('input[type=radio]').forEach(e => e.checked = false);
  document.querySelectorAll('textarea').forEach(e => e.value = "");
  aggiorna();
}
document.addEventListener("DOMContentLoaded", ripristina);
"""


def scrivi_modulo_html(percorso, annotatore, item, guida_html, in_chiaro=False):
    """Modulo di annotazione da aprire in locale nel browser.

    E' un file .html sul disco, non un artifact: il salvataggio del CSV con
    <a download> funziona percio' senza restrizioni."""
    schede = []
    for i, it in enumerate(item, 1):
        prefisso = (f'<span class="prefisso">{html.escape(it["inizio_turno"])} </span>'
                    if it["inizio_turno"] else "")
        targhetta = (f' &middot; <span class="modello">'
                     f'{html.escape(ETICHETTE_GRUPPO[it["gruppo"]])}</span>'
                     if in_chiaro else "")
        gruppi_html = []
        for chiave, etichetta in DIMENSIONI:
            bottoni = "".join(
                f'<label><input type="radio" name="{it["item"]}_{chiave}" value="{v}" '
                f'onchange="registra(\'{it["item"]}\',\'{chiave}\',\'{v}\')">'
                f'<span>{v}</span></label>' for v in range(1, 6))
            gruppi_html.append(f'<div class="dim"><span class="nome">{etichetta}</span>'
                               f'{bottoni}</div>')
        schede.append(f"""
<div class="scheda" id="card_{it['item']}">
  <div class="testata"><span>{i} / {len(item)} &middot; {it['item']}</span>
    <span>{it['task']}{targhetta}</span></div>
  <div class="contesto">{html.escape(it['contesto']) or '(nessun contesto precedente)'}</div>
  <div class="istruzione">{html.escape(it['istruzione'])}</div>
  <div class="frase">{prefisso}<strong>{html.escape(it['testo'])}</strong></div>
  {''.join(gruppi_html)}
  <textarea id="note_{it['item']}" rows="1" placeholder="note (facoltative)"
    oninput="registra('{it['item']}','note',this.value)"></textarea>
</div>""")

    dati = json.dumps([dict({k: it[k] for k in
                             ("item", "task", "contesto", "istruzione",
                              "inizio_turno", "testo")},
                            modello=ETICHETTE_GRUPPO[it["gruppo"]])
                       for it in item], ensure_ascii=False)
    documento = f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scheda Likert - {html.escape(annotatore)}</title>
<style>{MODULO_CSS}</style></head><body>
<h1>Valutazione Likert - annotatore: {html.escape(annotatore)}</h1>
<p class="sottotitolo">Le risposte restano salvate in questo browser mentre lavori.
Al termine premi <em>Scarica CSV</em> e metti il file in
<code>likert/schede/</code>.</p>
{guida_html}
<div class="barra">
  <button class="primario" onclick="scarica()">Scarica CSV</button>
  <button onclick="copia()">Copia CSV</button>
  <button onclick="azzera()">Azzera</button>
  <span class="avanzamento" id="avanzamento"></span>
</div>
{''.join(schede)}
<script>const ANNOTATORE = {json.dumps(annotatore)};
const IN_CHIARO = {json.dumps(bool(in_chiaro))};
const ITEM = {dati};
{MODULO_JS}</script></body></html>"""
    percorso.write_text(documento, encoding="utf-8")


LINEE_GUIDA = """# Linee guida per la valutazione Likert

Valuti dei turni di conversazione in napoletano. Alcuni sono stati prodotti da uno
dei tre modelli in gara, altri sono turni reali presi dal corpus; la colonna
`modello` della scheda (e la targhetta in alto a destra nel modulo HTML) dice
quale sia il caso.

Proprio perche' la provenienza e' visibile, si sforzi di **giudicare la frase e
non il sistema**: e' facile, sapendo che un turno viene da un modello, essere
piu' severi di quanto si sarebbe stati con lo stesso testo attribuito a un
parlante vero, e viceversa. Se si accorge di stare valutando l'etichetta invece
del testo, copra la colonna e rilegga.

Lo stesso contesto conversazionale puo' comparire piu' volte nella scheda, con
frasi diverse: e' il modo in cui i modelli vengono confrontati sulle stesse
conversazioni. Valuti ogni occorrenza per conto proprio, senza tornare indietro
a correggere quelle gia' fatte.

Per ogni item legga prima il **contesto** (i turni che precedono), poi
l'**istruzione** (che cosa doveva fare chi parla) e infine la **frase da
valutare**. In T2 la frase e' un completamento: la parte in grigio e' l'inizio
del turno gia' dato, quella in nero e' cio' che va valutato, ma il giudizio
grammaticale va dato sul turno intero.

Assegni tre punteggi da 1 a 5, indipendenti fra loro.

## 1. Naturalezza dialettale
Quanto suona come napoletano parlato davvero, non come italiano travestito ne'
come napoletano "da cartolina".

- **1** non e' napoletano: e' italiano, o una lingua mista senza tratti dialettali.
- **2** qualche parola dialettale isolata dentro un impianto italiano.
- **3** riconoscibilmente napoletano ma legnoso, scritto piu' che parlato.
- **4** napoletano credibile, con al piu' qualche punto artificiale.
- **5** indistinguibile da un parlante nativo in conversazione spontanea.

## 2. Accuratezza grammaticale
Correttezza morfologica e sintattica del napoletano: articoli e loro forme
contratte, raddoppiamento fonosintattico, accordi, coniugazioni, clitici,
preposizioni articolate.

- **1** frase agrammaticale o incomprensibile.
- **2** piu' errori strutturali (verbi, accordi, clitici sbagliati).
- **3** un errore evidente, oppure diversi errori minori di forma.
- **4** sostanzialmente corretta, al piu' un'incertezza ortografica.
- **5** nessun errore.

Non penalizzi le varianti ortografiche: il napoletano non ha una grafia
standard, e forme come *chiu'* / *cchiu'* o l'apostrofo messo diversamente sono
entrambe legittime.

## 3. Coerenza lessicale
Se le parole scelte stanno insieme e stanno con il contesto: registro uniforme,
nessun italianismo stonato in mezzo al dialetto, termini che hanno senso rispetto
a cio' di cui si sta parlando, e frase che nel suo insieme sta in piedi come
risposta a quella conversazione e non a una qualunque.

- **1** lessico incoerente o fuori tema rispetto alla conversazione: una risposta
  che potrebbe stare dopo qualsiasi altro discorso, o che non c'entra niente.
- **2** parole che stonano nettamente con il registro o con il contesto.
- **3** accettabile ma con qualche scelta lessicale discutibile.
- **4** lessico coerente, con al piu' un termine opinabile.
- **5** lessico del tutto appropriato al contesto e al registro.

## Casi particolari
- Frase vuota o troncata a meta': 1 su tutte e tre le dimensioni.
- Frase brevissima ma corretta e pertinente (*"eh gia'"*): la valuti normalmente,
  senza penalizzarne la lunghezza.
- Frase che ripete piu' volte la stessa cosa: penalizzi la coerenza lessicale.
- Se il contesto e' oscuro, giudichi comunque la frase in se' e lo annoti nelle note.

## Come si lavora
Ciascun annotatore compila la propria scheda **da solo**, senza confrontarsi con
l'altro: l'accordo fra le due schede e' esso stesso un risultato da misurare, e
se le valutazioni vengono concordate durante il lavoro quel numero non significa
piu' niente. Gli item compaiono in ordine diverso nelle due schede, ed e' voluto.
"""


def comando_prepara(args):
    rng = random.Random(args.seed)
    CARTELLA.mkdir(exist_ok=True)
    (CARTELLA / "schede").mkdir(exist_ok=True)

    tutti, riepilogo = [], {}
    for task in ("T2", "T3"):
        item, info = campiona(task, args.n_sistema, args.n_umano, rng, args.min_parole)
        tutti.extend(item)
        riepilogo[task] = dict(
            info, modelli={c: {"etichetta": m["etichetta"], "preds": m["preds"],
                               "motivo": m["motivo"]}
                           for c, m in SISTEMI[task]["modelli"].items()})

    # i codici vengono assegnati DOPO il mescolamento, altrimenti l'ordine di
    # costruzione (modello per modello) trapelerebbe dalla numerazione
    rng.shuffle(tutti)
    for i, it in enumerate(tutti, 1):
        it["item"] = f"V{i:03d}"

    chiave = {"seed": args.seed, "dimensioni": CHIAVI_DIM, "disegno": "appaiato",
              "sistemi": riepilogo, "annotatori": args.annotatori, "item": tutti}
    (CARTELLA / "chiave.json").write_text(
        json.dumps(chiave, ensure_ascii=False, indent=2), encoding="utf-8")
    scrivi_chiave_csv(CARTELLA / "chiave.csv", tutti)
    (CARTELLA / "linee_guida.md").write_text(LINEE_GUIDA, encoding="utf-8")

    guida_html = ('<details><summary><strong>Linee guida (apri per rileggerle)'
                  '</strong></summary><div class="contesto">'
                  + html.escape(LINEE_GUIDA) + "</div></details>")

    for i, nome in enumerate(args.annotatori):
        # ordine diverso per ciascun annotatore: neutralizza gli effetti di
        # stanchezza e di trascinamento fra item consecutivi
        ordine = list(tutti)
        random.Random(args.seed + 1000 + i).shuffle(ordine)
        scrivi_scheda_csv(CARTELLA / "schede" / f"scheda_{nome}.csv", ordine,
                          args.in_chiaro)
        scrivi_modulo_html(CARTELLA / "schede" / f"scheda_{nome}.html",
                           nome, ordine, guida_html, args.in_chiaro)

    print(f"Item totali: {len(tutti)}")
    for task, r in riepilogo.items():
        print(f"  {task}: {r['conversazioni_per_modello']} conversazioni x "
              f"{len(r['modelli'])} modelli + {r['conversazioni_umane']} turni umani "
              f"= {r['item_prodotti']} item "
              f"(su {r['righe_utilizzabili']}/{r['righe_totali']} righe utilizzabili)")
    print("\nChiave        : likert/chiave.csv (leggibile) e likert/chiave.json")
    print("                item -> task, modello, conversazione di origine")
    print("Linee guida   : likert/linee_guida.md")
    for nome in args.annotatori:
        print(f"Scheda {nome:<8}: likert/schede/scheda_{nome}.csv (+ .html)")
    if args.in_chiaro:
        print("\nLe schede riportano la colonna 'modello': l'annotatore vede da quale "
              "sistema viene ogni frase. Per nasconderla: prepara --in-cieco.")
    print("\nCompilare le schede, poi: python likert/likert_human_eval.py analizza")


# ----------------------------------------------------------------------------
# statistica
# ----------------------------------------------------------------------------

def media(v):
    return sum(v) / len(v) if v else float("nan")


def devstd(v):
    if len(v) < 2:
        return 0.0
    m = media(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def bootstrap_media(v, rng, n=5000, livello=0.95):
    """Intervallo percentile sulla media: niente assunzioni di normalita', che su
    una scala ordinale a cinque punti non reggerebbero."""
    if len(v) < 2:
        return (float("nan"), float("nan"))
    medie = []
    k = len(v)
    for _ in range(n):
        medie.append(media([v[rng.randrange(k)] for _ in range(k)]))
    medie.sort()
    return (medie[int((1 - livello) / 2 * n)],
            medie[min(n - 1, int((1 + livello) / 2 * n))])


def bootstrap_appaiato(differenze, rng, n=5000, livello=0.95):
    """Intervallo sulla media delle differenze osservate item per item.

    Appaiato perche' i due modelli hanno risposto alle stesse conversazioni: la
    variabilita' dovuta alla difficolta' del singolo contesto si cancella, e
    l'intervallo risulta molto piu' stretto di quello non appaiato."""
    if len(differenze) < 2:
        return (float("nan"), float("nan"))
    campioni = []
    k = len(differenze)
    for _ in range(n):
        campioni.append(media([differenze[rng.randrange(k)] for _ in range(k)]))
    campioni.sort()
    return (campioni[int((1 - livello) / 2 * n)],
            campioni[min(n - 1, int((1 + livello) / 2 * n))])


def bootstrap_differenza(a, b, rng, n=5000, livello=0.95):
    """Intervallo sulla differenza fra due campioni indipendenti.

    Serve per il confronto con il riferimento umano, che per costruzione viene da
    conversazioni diverse e non e' quindi appaiabile con i modelli."""
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    diff = []
    for _ in range(n):
        ca = media([a[rng.randrange(len(a))] for _ in range(len(a))])
        cb = media([b[rng.randrange(len(b))] for _ in range(len(b))])
        diff.append(ca - cb)
    diff.sort()
    return (diff[int((1 - livello) / 2 * n)],
            diff[min(n - 1, int((1 + livello) / 2 * n))])


def kappa_pesata(a, b, k=5):
    """Cohen quadratica: penalizza i disaccordi in proporzione al loro quadrato,
    cosi' 4 contro 5 pesa molto meno di 1 contro 5."""
    if not a:
        return float("nan")
    n = len(a)
    osservata = defaultdict(int)
    for x, y in zip(a, b):
        osservata[(x, y)] += 1
    ma, mb = Counter(a), Counter(b)

    def peso(i, j):
        return 1 - ((i - j) ** 2) / ((k - 1) ** 2)

    num = sum(peso(i, j) * c / n for (i, j), c in osservata.items())
    den = sum(peso(i, j) * (ma.get(i, 0) / n) * (mb.get(j, 0) / n)
              for i in range(1, k + 1) for j in range(1, k + 1))
    return (num - den) / (1 - den) if abs(1 - den) > 1e-12 else float("nan")


def alpha_krippendorff_ordinale(unita):
    """Alpha su scala ordinale. `unita` e' una lista di liste di voti per item.

    Rispetto alla kappa non richiede esattamente due annotatori e tollera i buchi;
    la distanza fra due categorie tiene conto di quante osservazioni cadono fra
    di esse, che e' cio' che distingue il caso ordinale da quello nominale."""
    unita = [u for u in unita if len(u) >= 2]
    if not unita:
        return float("nan")
    categorie = sorted({v for u in unita for v in u})
    coincidenza = defaultdict(float)
    for u in unita:
        m = len(u)
        for i, ci in enumerate(u):
            for j, cj in enumerate(u):
                if i != j:
                    coincidenza[(ci, cj)] += 1.0 / (m - 1)
    marginali = {c: sum(coincidenza.get((c, k), 0.0) for k in categorie)
                 for c in categorie}
    n = sum(marginali.values())
    if n <= 1:
        return float("nan")

    def delta2(c, k):
        if c == k:
            return 0.0
        lo, hi = (c, k) if c < k else (k, c)
        fra = sum(marginali[g] for g in categorie if lo <= g <= hi)
        return (fra - (marginali[lo] + marginali[hi]) / 2.0) ** 2

    d_oss = sum(coincidenza.get((c, k), 0.0) * delta2(c, k)
                for c in categorie for k in categorie)
    d_att = sum(marginali[c] * (marginali[k] - (1 if c == k else 0)) / (n - 1)
                * delta2(c, k) for c in categorie for k in categorie)
    return 1 - d_oss / d_att if d_att > 0 else float("nan")


def ranghi(v):
    """Ranghi medi in caso di pari merito, come richiede lo Spearman su scale corte."""
    ordinati = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(ordinati):
        j = i
        while j + 1 < len(ordinati) and v[ordinati[j + 1]] == v[ordinati[i]]:
            j += 1
        medio = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[ordinati[k]] = medio
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = ranghi(a), ranghi(b)
    ma, mb = media(ra), media(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den > 0 else float("nan")


# ----------------------------------------------------------------------------
# sottocomando: analizza
# ----------------------------------------------------------------------------

def leggi_schede(cartella, meta=None):
    """Schede compilate: {annotatore: {item: {dimensione: voto}}}.

    Se la scheda porta la colonna 'modello' la si confronta con la chiave: e' un
    controllo a costo nullo contro schede rimescolate o incollate male, in cui i
    voti finirebbero attribuiti al sistema sbagliato."""
    schede = {}
    for percorso in sorted(Path(cartella).glob("scheda_*.csv")):
        nome = percorso.stem[len("scheda_"):]
        voti = {}
        with open(percorso, encoding="utf-8-sig", newline="") as f:
            for n_riga, riga in enumerate(csv.DictReader(f), 2):
                item = (riga.get("item") or "").strip()
                if not item:
                    continue
                dichiarato = (riga.get("modello") or "").strip()
                if dichiarato and meta and item in meta:
                    atteso = ETICHETTE_GRUPPO[meta[item]["gruppo"]]
                    if dichiarato != atteso:
                        raise SystemExit(
                            f"ERRORE: {percorso.name} riga {n_riga}, item {item}: "
                            f"la scheda dice {dichiarato!r} ma la chiave dice "
                            f"{atteso!r}. La scheda non corrisponde alla chiave.")
                valori = {}
                for dim in CHIAVI_DIM:
                    grezzo = (riga.get(dim) or "").strip()
                    if not grezzo:
                        continue
                    try:
                        v = int(float(grezzo.replace(",", ".")))
                    except ValueError:
                        raise SystemExit(
                            f"ERRORE: {percorso.name} riga {n_riga}, colonna {dim}: "
                            f"valore non numerico {grezzo!r}")
                    if not 1 <= v <= 5:
                        raise SystemExit(
                            f"ERRORE: {percorso.name} riga {n_riga}, colonna {dim}: "
                            f"{v} fuori dalla scala 1-5")
                    valori[dim] = v
                if valori:
                    if len(valori) < len(CHIAVI_DIM):
                        mancanti = [d for d in CHIAVI_DIM if d not in valori]
                        print(f"  ATTENZIONE: {percorso.name} item {item}: "
                              f"dimensioni non compilate: {', '.join(mancanti)}")
                    voti[item] = valori
        if voti:
            schede[nome] = voti
            print(f"  {percorso.name}: {len(voti)} item valutati")
        else:
            print(f"  {percorso.name}: nessuna valutazione, scheda ignorata")
    return schede


def distribuzione(task, gruppo, dim, meta, schede, annotatori):
    """Quante volte e' stato assegnato ciascun punteggio 1..5 (voti grezzi).

    Qui non si usa la media fra annotatori: la forma della distribuzione andrebbe
    persa arrotondando due voti a un valore intermedio."""
    conteggio = Counter()
    for item, m in meta.items():
        if m["task"] != task or m["gruppo"] != gruppo:
            continue
        for a in annotatori:
            v = schede.get(a, {}).get(item, {}).get(dim)
            if v:
                conteggio[v] += 1
    tot = sum(conteggio.values())
    return {str(v): (conteggio.get(v, 0) / tot if tot else 0.0) for v in range(1, 6)}


def comando_analizza(args):
    percorso_chiave = CARTELLA / "chiave.json"
    if not percorso_chiave.exists():
        raise SystemExit("ERRORE: chiave.json assente: esegui prima "
                         "'python likert/likert_human_eval.py prepara'")
    chiave = json.loads(percorso_chiave.read_text(encoding="utf-8"))
    meta = {it["item"]: it for it in chiave["item"]}

    print("Schede trovate:")
    schede = leggi_schede(CARTELLA / "schede", meta)
    if not schede:
        raise SystemExit(
            "\nERRORE: nessuna scheda compilata in likert/schede/.\n"
            "Le valutazioni Likert devono essere date da annotatori umani: "
            "compila\nalmeno una scheda (CSV o modulo HTML) e rilancia l'analisi.")

    rng = random.Random(args.seed)
    annotatori = sorted(schede)

    # voto medio fra annotatori per ciascun item, usato per le medie di gruppo
    aggregati = defaultdict(dict)
    for item in meta:
        for dim in CHIAVI_DIM:
            voti = [schede[a][item][dim] for a in annotatori
                    if item in schede[a] and dim in schede[a][item]]
            if voti:
                aggregati[item][dim] = media(voti)

    def raccogli(task, gruppo, dim):
        return [aggregati[i][dim] for i, m in meta.items()
                if m["task"] == task and m["gruppo"] == gruppo
                and dim in aggregati.get(i, {})]

    def complessive_per_origine(task, gruppo):
        """Media delle tre dimensioni, indicizzata per conversazione di origine.

        Serve al confronto appaiato: e' l'origine, non il codice dell'item, a
        legare fra loro le risposte dei tre modelli alla stessa conversazione."""
        fuori = {}
        for i, m in meta.items():
            if m["task"] != task or m["gruppo"] != gruppo:
                continue
            if all(d in aggregati.get(i, {}) for d in CHIAVI_DIM):
                fuori[m["origine"]] = media([aggregati[i][d] for d in CHIAVI_DIM])
        return fuori

    risultati = {"annotatori": annotatori, "disegno": chiave.get("disegno", "appaiato"),
                 "sistemi": chiave["sistemi"], "per_task": {}, "accordo": {}}

    for task in ("T2", "T3"):
        blocco = {"gruppi": {}}
        for gruppo in gruppi(task):
            voci = {}
            for dim in CHIAVI_DIM:
                v = raccogli(task, gruppo, dim)
                lo, hi = bootstrap_media(v, rng, args.bootstrap)
                voci[dim] = {"n": len(v), "media": media(v), "ds": devstd(v),
                             "ic95": [lo, hi],
                             "distribuzione": distribuzione(task, gruppo, dim,
                                                            meta, schede, annotatori)}
            complessiva = list(complessive_per_origine(task, gruppo).values())
            lo, hi = bootstrap_media(complessiva, rng, args.bootstrap)
            voci["complessiva"] = {"n": len(complessiva), "media": media(complessiva),
                                   "ds": devstd(complessiva), "ic95": [lo, hi]}
            blocco["gruppi"][gruppo] = voci

        # confronti appaiati fra modelli, sulle conversazioni valutate per entrambi
        codici = list(SISTEMI[task]["modelli"])
        confronti = {}
        for a, b in itertools.combinations(codici, 2):
            ca, cb = complessive_per_origine(task, a), complessive_per_origine(task, b)
            comuni = sorted(set(ca) & set(cb))
            differenze = [ca[o] - cb[o] for o in comuni]
            lo, hi = bootstrap_appaiato(differenze, rng, args.bootstrap)
            confronti[f"{a}_vs_{b}"] = {
                "n_conversazioni": len(comuni), "delta": media(differenze),
                "ic95": [lo, hi], "significativa": not (lo <= 0 <= hi)}
        blocco["confronti_appaiati"] = confronti

        # confronto con il riferimento umano: non appaiabile, perche' i turni
        # umani vengono per costruzione da conversazioni diverse
        rispetto_umano = {}
        for gruppo in codici:
            per_dim = {}
            for dim in CHIAVI_DIM + ["complessiva"]:
                if dim == "complessiva":
                    a = list(complessive_per_origine(task, gruppo).values())
                    b = list(complessive_per_origine(task, UMANO).values())
                else:
                    a, b = raccogli(task, gruppo, dim), raccogli(task, UMANO, dim)
                lo, hi = bootstrap_differenza(a, b, rng, args.bootstrap)
                per_dim[dim] = {"delta": media(a) - media(b), "ic95": [lo, hi],
                                "significativa": not (lo <= 0 <= hi)}
            rispetto_umano[gruppo] = per_dim
        blocco["rispetto_umano"] = rispetto_umano

        blocco["classifica"] = sorted(
            codici, key=lambda g: blocco["gruppi"][g]["complessiva"]["media"],
            reverse=True)
        risultati["per_task"][task] = blocco

    # accordo fra annotatori: solo sugli item valutati da almeno due persone
    for dim in CHIAVI_DIM:
        unita = []
        for item in meta:
            voti = [schede[a][item][dim] for a in annotatori
                    if item in schede[a] and dim in schede[a][item]]
            if len(voti) >= 2:
                unita.append(voti)
        voce = {"n_item": len(unita),
                "alpha_krippendorff": alpha_krippendorff_ordinale(unita)}
        if len(annotatori) == 2:
            a1, a2 = annotatori
            comuni = [i for i in meta
                      if i in schede[a1] and i in schede[a2]
                      and dim in schede[a1][i] and dim in schede[a2][i]]
            x = [schede[a1][i][dim] for i in comuni]
            y = [schede[a2][i][dim] for i in comuni]
            voce.update({
                "kappa_pesata": kappa_pesata(x, y),
                "spearman": spearman(x, y),
                "accordo_esatto": media([1.0 if p == q else 0.0
                                         for p, q in zip(x, y)]) if x else float("nan"),
                "accordo_entro_1": media([1.0 if abs(p - q) <= 1 else 0.0
                                          for p, q in zip(x, y)]) if x else float("nan"),
            })
        risultati["accordo"][dim] = voce

    scrivi_risultati(risultati, meta, aggregati, args)
    stampa_report(risultati)
    return risultati


def scrivi_risultati(risultati, meta, aggregati, args):
    uscita = CARTELLA / "risultati"
    uscita.mkdir(exist_ok=True)
    (uscita / "likert_risultati.json").write_text(
        json.dumps(risultati, ensure_ascii=False, indent=2), encoding="utf-8")

    # dettaglio per item, utile per rileggere i casi peggiori
    with open(uscita / "likert_per_item.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "task", "modello", "origine", "testo"]
                   + CHIAVI_DIM + ["media"])
        for item, m in sorted(meta.items()):
            voti = aggregati.get(item, {})
            if not voti:
                continue
            valori = [voti.get(d) for d in CHIAVI_DIM]
            completi = [v for v in valori if v is not None]
            w.writerow([item, m["task"], ETICHETTE_GRUPPO[m["gruppo"]], m["origine"],
                        m["testo"]]
                       + [f"{v:.2f}" if v is not None else "" for v in valori]
                       + [f"{media(completi):.2f}" if completi else ""])

    scrivi_tabelle_latex(risultati, uscita / "likert_tabella.tex")
    disegna_grafici(risultati, args)
    print(f"\nRisultati   : {uscita}")


def scrivi_tabelle_latex(risultati, percorso):
    nomi = dict(DIMENSIONI)
    nomi["complessiva"] = "\\textbf{Media complessiva}"
    pezzi = []

    # tabella 1: punteggi per modello e per dimensione, un blocco per task
    righe = []
    for task in ("T2", "T3"):
        b = risultati["per_task"][task]
        righe.append("        \\midrule")
        righe.append(f"        \\multicolumn{{5}}{{l}}{{\\emph{{"
                     f"{SISTEMI[task]['etichetta']}}}}} \\\\")
        righe.append("        \\midrule")
        ordine = gruppi(task)
        for dim in CHIAVI_DIM + ["complessiva"]:
            celle = []
            migliore = max((g for g in ordine if g != UMANO),
                           key=lambda g: b["gruppi"][g][dim]["media"])
            for g in ordine:
                v = b["gruppi"][g][dim]
                testo = f"{v['media']:.2f} ({v['ds']:.2f})"
                celle.append(f"\\textbf{{{testo}}}" if g == migliore else testo)
            righe.append(f"        {nomi[dim]} & " + " & ".join(celle) + " \\\\")

    intestazione = " & ".join(f"\\textbf{{{ETICHETTE_GRUPPO[g]}}}" for g in gruppi("T2"))
    pezzi.append(f"""% generata da likert/likert_human_eval.py
\\begin{{table}}[H]
    \\centering
    \\small
    \\begin{{tabular}}{{lcccc}}
        \\toprule
        \\textbf{{Dimensione}} & {intestazione} \\\\
{chr(10).join(righe)}
        \\bottomrule
    \\end{{tabular}}
    \\caption{{Valutazione umana su scala Likert 1--5: media (deviazione standard)
    per ciascun modello e per i turni umani di riferimento. In grassetto il
    migliore fra i tre modelli su ogni riga.}}\\label{{tab:likert}}
\\end{{table}}""")

    # tabella 2: confronti appaiati fra modelli sulla media complessiva
    righe = []
    for task in ("T2", "T3"):
        b = risultati["per_task"][task]
        righe.append("        \\midrule")
        righe.append(f"        \\multicolumn{{4}}{{l}}{{\\emph{{"
                     f"{SISTEMI[task]['etichetta']}}}}} \\\\")
        righe.append("        \\midrule")
        for coppia, v in b["confronti_appaiati"].items():
            a, c = coppia.split("_vs_")
            stella = "$^{*}$" if v["significativa"] else ""
            righe.append(
                f"        {ETICHETTE_GRUPPO[a]} vs {ETICHETTE_GRUPPO[c]} & "
                f"{v['n_conversazioni']} & {v['delta']:+.2f}{stella} & "
                f"[{v['ic95'][0]:+.2f}; {v['ic95'][1]:+.2f}] \\\\")
    pezzi.append(f"""
% confronti appaiati fra modelli
\\begin{{table}}[H]
    \\centering
    \\small
    \\begin{{tabular}}{{lccc}}
        \\toprule
        \\textbf{{Confronto}} & \\textbf{{$n$}} & \\textbf{{$\\Delta$ media}} &
        \\textbf{{IC 95\\%}} \\\\
{chr(10).join(righe)}
        \\bottomrule
    \\end{{tabular}}
    \\caption{{Confronti appaiati fra modelli sulla media complessiva delle tre
    dimensioni: i modelli hanno risposto alle stesse conversazioni, quindi la
    differenza e' calcolata conversazione per conversazione. $^{{*}}$ indica un
    intervallo bootstrap al 95\\% che non comprende lo
    zero.}}\\label{{tab:likert_confronti}}
\\end{{table}}""")

    # tabella 3: accordo fra annotatori
    righe = []
    for dim, etichetta in DIMENSIONI:
        a = risultati["accordo"][dim]

        def formatta(chiave):
            v = a.get(chiave)
            return f"{v:.3f}" if isinstance(v, float) and not math.isnan(v) else "---"

        entro = a.get("accordo_entro_1")
        entro_s = (f"{entro * 100:.1f}\\%" if isinstance(entro, float)
                   and not math.isnan(entro) else "---")
        righe.append(f"        {etichetta} & {formatta('alpha_krippendorff')} & "
                     f"{formatta('kappa_pesata')} & {formatta('spearman')} & "
                     f"{entro_s} \\\\")
    pezzi.append(f"""
% accordo fra annotatori
\\begin{{table}}[H]
    \\centering
    \\small
    \\begin{{tabular}}{{lcccc}}
        \\toprule
        \\textbf{{Dimensione}} & \\textbf{{$\\alpha$ Krippendorff}} &
        \\textbf{{$\\kappa_w$}} & \\textbf{{$\\rho$ Spearman}} &
        \\textbf{{Accordo $\\pm 1$}} \\\\
        \\midrule
{chr(10).join(righe)}
        \\bottomrule
    \\end{{tabular}}
    \\caption{{Accordo fra annotatori sulle tre dimensioni della scala
    Likert.}}\\label{{tab:likert_accordo}}
\\end{{table}}""")

    percorso.write_text("\n".join(pezzi) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------
# grafici
# ----------------------------------------------------------------------------

def disegna_grafici(risultati, args):
    cartelle = [CARTELLA / "risultati"]
    if args.grafici_tesi:
        GRAFICI_TESI.mkdir(parents=True, exist_ok=True)
        cartelle.append(GRAFICI_TESI)

    figure = {
        "likert_medie.png": grafico_medie(risultati),
        "likert_distribuzione.png": grafico_distribuzione(risultati),
        "likert_confronti.png": grafico_confronti(risultati),
        "likert_accordo.png": grafico_accordo(risultati),
    }
    for nome, fig in figure.items():
        if fig is None:
            continue
        for c in cartelle:
            fig.savefig(c / nome, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)


def grafico_medie(risultati):
    etichette = [e.replace(" ", "\n") for _, e in DIMENSIONI] + ["Media\ncomplessiva"]
    chiavi = CHIAVI_DIM + ["complessiva"]
    fig, assi = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, task in zip(assi, ("T2", "T3")):
        b = risultati["per_task"][task]["gruppi"]
        ordine = gruppi(task)
        x = range(len(chiavi))
        larghezza = 0.8 / len(ordine)
        for k, gruppo in enumerate(ordine):
            medie = [b[gruppo][d]["media"] for d in chiavi]
            barre = [[b[gruppo][d]["media"] - b[gruppo][d]["ic95"][0] for d in chiavi],
                     [b[gruppo][d]["ic95"][1] - b[gruppo][d]["media"] for d in chiavi]]
            pos = [i + (k - (len(ordine) - 1) / 2) * larghezza for i in x]
            ax.bar(pos, medie, larghezza, yerr=barre, capsize=2,
                   color=COLORI[gruppo], label=ETICHETTE_GRUPPO[gruppo],
                   edgecolor="white",
                   hatch="//" if gruppo == UMANO else None)
            for p, m, alto in zip(pos, medie, barre[1]):
                ax.text(p, m + alto + 0.07, f"{m:.2f}", ha="center", fontsize=6.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels(etichette, fontsize=8)
        ax.set_ylim(1, 5.4)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_title(SISTEMI[task]["etichetta"], fontsize=10)
    assi[0].set_ylabel("punteggio Likert (1-5)")
    assi[0].legend(fontsize=7.5, loc="upper left", ncol=2)
    fig.suptitle("Valutazione umana su scala Likert: i tre modelli a confronto",
                 fontsize=12)
    fig.tight_layout()
    fig.text(0.5, -0.02, "barre di errore: intervallo bootstrap al 95% sulla media; "
             "il riferimento umano (tratteggiato) non e' un sistema in gara ma il "
             "tetto della scala", ha="center", fontsize=7.5, color="#555")
    return fig


def grafico_distribuzione(risultati):
    fig, assi = plt.subplots(1, 2, figsize=(12.5, 6.4), sharex=True)
    for ax, task in zip(assi, ("T2", "T3")):
        b = risultati["per_task"][task]["gruppi"]
        ordine = gruppi(task)
        etichette, posizioni, y = [], [], 0
        for dim, nome in reversed(DIMENSIONI):
            for gruppo in reversed(ordine):
                dist = b[gruppo][dim]["distribuzione"]
                sinistra = 0.0
                for v in range(1, 6):
                    quota = dist[str(v)]
                    ax.barh(y, quota, left=sinistra, color=COLORI_SCALA[v - 1],
                            edgecolor="white", height=0.75)
                    if quota > 0.07:
                        ax.text(sinistra + quota / 2, y, f"{quota * 100:.0f}",
                                ha="center", va="center", fontsize=6.5, color="white")
                    sinistra += quota
                posizioni.append(y)
                etichette.append(ETICHETTE_GRUPPO[gruppo])
                y += 1
            # etichetta della dimensione, a sinistra dei nomi dei gruppi: va
            # tenuta fuori dall'area delle etichette per non sovrapporvisi
            ax.text(-0.27, y - len(ordine) / 2 - 0.5, nome, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", rotation=90,
                    transform=ax.get_yaxis_transform())
            y += 0.8
        ax.set_yticks(posizioni)
        ax.set_yticklabels(etichette, fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
        ax.set_title(SISTEMI[task]["etichetta"], fontsize=10)
    maniglie = [plt.Rectangle((0, 0), 1, 1, color=COLORI_SCALA[i]) for i in range(5)]
    fig.suptitle("Distribuzione dei punteggi assegnati", fontsize=12)
    fig.tight_layout()
    fig.legend(maniglie, [str(i + 1) for i in range(5)], ncol=5, fontsize=8,
               loc="lower center", title="punteggio Likert", title_fontsize=8,
               bbox_to_anchor=(0.5, -0.06))
    return fig


def grafico_confronti(risultati):
    """Differenze appaiate fra modelli: barra orizzontale con l'intervallo.

    E' il grafico che risponde alla domanda 'quale modello e' migliore': se
    l'intervallo attraversa lo zero, i due modelli non sono distinguibili."""
    voci = []
    for task in ("T2", "T3"):
        for coppia, v in risultati["per_task"][task]["confronti_appaiati"].items():
            a, b = coppia.split("_vs_")
            voci.append((f"{task}  {ETICHETTE_GRUPPO[a]} - {ETICHETTE_GRUPPO[b]}",
                         v["delta"], v["ic95"], v["significativa"]))
    if not voci:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(voci) + 1.6))
    y = list(range(len(voci)))[::-1]
    for pos, (_, delta, ic, signif) in zip(y, voci):
        colore = "#c1121f" if signif else "#9a938a"
        ax.plot([ic[0], ic[1]], [pos, pos], color=colore, lw=2.2, solid_capstyle="round")
        ax.plot([delta], [pos], "o", color=colore, ms=6)
    ax.axvline(0, color="#444", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([v[0] for v in voci], fontsize=8)
    ax.set_xlabel("differenza di punteggio Likert medio (appaiata per conversazione)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_title("Confronti appaiati fra modelli", fontsize=11)
    fig.text(0.5, -0.02, "in rosso le differenze il cui intervallo al 95% non "
             "comprende lo zero", ha="center", fontsize=7.5, color="#555")
    fig.tight_layout()
    return fig


def grafico_accordo(risultati):
    misure = [("alpha_krippendorff", "$\\alpha$ Krippendorff"),
              ("kappa_pesata", "$\\kappa$ pesata"),
              ("spearman", "$\\rho$ Spearman")]
    disponibili = [(k, e) for k, e in misure
                   if any(not math.isnan(risultati["accordo"][d].get(k, float("nan")))
                          for d in CHIAVI_DIM)]
    if not disponibili:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = range(len(CHIAVI_DIM))
    larghezza = 0.8 / len(disponibili)
    colori = ["#c1121f", "#457b9d", "#7d8c5c"]
    for k, (chiave, etichetta) in enumerate(disponibili):
        valori = [risultati["accordo"][d].get(chiave, float("nan")) for d in CHIAVI_DIM]
        pos = [i + (k - (len(disponibili) - 1) / 2) * larghezza for i in x]
        ax.bar(pos, valori, larghezza, color=colori[k % 3], label=etichetta,
               edgecolor="white")
        for p, v in zip(pos, valori):
            if not math.isnan(v):
                ax.text(p, max(v, 0) + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    for soglia, testo in ((0.667, "soglia minima (0,667)"),
                          (0.8, "accordo solido (0,80)")):
        ax.axhline(soglia, ls="--", lw=1, color="#888")
        ax.text(len(CHIAVI_DIM) - 0.5, soglia + 0.012, testo, fontsize=7,
                ha="right", color="#666")
    ax.set_xticks(list(x))
    ax.set_xticklabels([e for _, e in DIMENSIONI], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accordo")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)
    ax.set_title("Accordo fra annotatori per dimensione", fontsize=11)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# report testuale
# ----------------------------------------------------------------------------

def stampa_report(risultati):
    nomi = dict(DIMENSIONI)
    nomi["complessiva"] = "MEDIA COMPLESSIVA"
    print("\n" + "=" * 86)
    print("VALUTAZIONE UMANA SU SCALA LIKERT (1-5)")
    print("=" * 86)
    print(f"Annotatori: {', '.join(risultati['annotatori'])}")

    for task in ("T2", "T3"):
        b = risultati["per_task"][task]
        ordine = gruppi(task)
        print(f"\n--- {SISTEMI[task]['etichetta']} ---")
        print("    " + "dimensione".ljust(26)
              + "".join(ETICHETTE_GRUPPO[g].rjust(18) for g in ordine))
        for dim in CHIAVI_DIM + ["complessiva"]:
            celle = []
            for g in ordine:
                v = b["gruppi"][g][dim]
                celle.append(f"{v['media']:.2f} ({v['ds']:.2f})".rjust(18))
            print(f"    {nomi[dim]:<26}" + "".join(celle))
        print("    n = " + ", ".join(
            f"{ETICHETTE_GRUPPO[g]} {b['gruppi'][g]['complessiva']['n']}"
            for g in ordine))

        print("\n    confronti appaiati (media complessiva):")
        for coppia, v in b["confronti_appaiati"].items():
            a, c = coppia.split("_vs_")
            marchio = "*" if v["significativa"] else " "
            print(f"      {ETICHETTE_GRUPPO[a]:>10} - {ETICHETTE_GRUPPO[c]:<10} "
                  f"{v['delta']:+6.2f} {marchio} "
                  f"[{v['ic95'][0]:+.2f}, {v['ic95'][1]:+.2f}]  "
                  f"(n={v['n_conversazioni']})")

        print("    distanza dal riferimento umano (media complessiva):")
        for g in SISTEMI[task]["modelli"]:
            v = b["rispetto_umano"][g]["complessiva"]
            marchio = "*" if v["significativa"] else " "
            print(f"      {ETICHETTE_GRUPPO[g]:<12} {v['delta']:+6.2f} {marchio} "
                  f"[{v['ic95'][0]:+.2f}, {v['ic95'][1]:+.2f}]")

        classifica = " > ".join(ETICHETTE_GRUPPO[g] for g in b["classifica"])
        print(f"    classifica: {classifica}")

    print("\n--- accordo fra annotatori ---")
    for dim, etichetta in DIMENSIONI:
        a = risultati["accordo"][dim]
        pezzi = [f"alpha={a['alpha_krippendorff']:.3f}"]
        if "kappa_pesata" in a:
            pezzi += [f"kappa_w={a['kappa_pesata']:.3f}",
                      f"rho={a['spearman']:.3f}",
                      f"esatto={a['accordo_esatto'] * 100:.1f}%",
                      f"entro1={a['accordo_entro_1'] * 100:.1f}%"]
        print(f"    {etichetta:<26} " + "  ".join(pezzi) + f"  (n={a['n_item']})")
    print("\n* differenza il cui intervallo bootstrap al 95% non comprende lo zero")
    print("=" * 86)


# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="comando", required=True)

    e = sub.add_parser("elenca", help="metriche automatiche dei sistemi disponibili")
    e.set_defaults(func=comando_elenca)

    pr = sub.add_parser("prepara", help="costruisce le schede di annotazione")
    pr.add_argument("--n-sistema", type=int, default=20,
                    help="conversazioni per task, valutate su tutti e tre i modelli "
                         "(default 20, cioe' 60 item per task)")
    pr.add_argument("--n-umano", type=int, default=15,
                    help="turni umani di controllo per task (default 15)")
    pr.add_argument("--min-parole", type=int, default=2,
                    help="lunghezza minima in parole di una frase valutabile")
    pr.add_argument("--annotatori", nargs="+", default=["001", "002"],
                    help="etichette delle schede da produrre; numeriche e non "
                         "nominative, cosi' i punteggi non restano associati a "
                         "una persona (default: 001 002)")
    pr.add_argument("--in-chiaro", action="store_true", default=True,
                    help="schede con la colonna 'modello' visibile (predefinito)")
    pr.add_argument("--in-cieco", dest="in_chiaro", action="store_false",
                    help="nasconde il modello nelle schede: l'annotatore non sa "
                         "quale sistema sta giudicando, il che rende il confronto "
                         "fra modelli piu' difendibile")
    pr.add_argument("--seed", type=int, default=20252026)
    pr.set_defaults(func=comando_prepara)

    an = sub.add_parser("analizza", help="analizza le schede compilate")
    an.add_argument("--bootstrap", type=int, default=5000,
                    help="ricampionamenti bootstrap (default 5000)")
    an.add_argument("--seed", type=int, default=20252026)
    an.add_argument("--dpi", type=int, default=200)
    an.add_argument("--grafici-tesi", action="store_true", default=True,
                    help="copia i grafici anche in 'Progetto NLP/grafici_Likert/'")
    an.add_argument("--no-grafici-tesi", dest="grafici_tesi", action="store_false")
    an.set_defaults(func=comando_analizza)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
