#!/usr/bin/env python3
"""
Fase 1.2 (controllo) — Coerenza fra la bozza LLM e il golden validato a mano.

Confronta turno per turno il napoletano prodotto da gemma4:cloud
(`results/napoletano.json`) con quello corretto dal validatore nativo
(`golden_translate/golden_validator.json`) e produce un report .txt con, per
ogni turno, le quattro metriche della Fase 4.3, un grado di similarita' unico
in percentuale e il booleano "questa bozza va sostituita col golden?".

Direzione del confronto: il golden e' il RIFERIMENTO, la bozza gemma4 e'
l'IPOTESI (stessa direzione della valutazione di Fase 4/5).

Metriche:
- chrF       (sacrebleu)  -> primaria: lavora sui caratteri, quindi non punisce
                             le varianti ortografiche del napoletano non standard;
- BERTScore  (F1, mBERT)  -> coerenza semantica; punteggio grezzo, non ribasato
                             (per il napoletano non esistono baseline), quindi
                             va letto in relativo, non in assoluto;
- BLEU       (sacrebleu, sentence-level con smoothing) -> rumoroso sui turni corti;
- ROUGE-L    (rouge-score con tokenizer Unicode: quello di default cancella le
                             lettere accentate, cioe' mezzo napoletano).

Grado di similarita' (pesi = importanza dichiarata in Fase 4.3):

    sim% = 0.40*chrF + 0.30*BERTScore_F1 + 0.15*BLEU + 0.15*ROUGE-L

Booleano: `sostituire = sim% < soglia` (default 85). Sopra la soglia le due
frasi sono di fatto equivalenti e riscrivere la bozza col golden non cambia
nulla; sotto, il golden porta informazione e va tenuto lui.

Prerequisiti — ATTENZIONE ALL'INTERPRETE:
    su questa macchina `python` e' Python 3.8, che NON ha le librerie giuste.
    Serve Python 3.12 (l'unico con transformers e un torch recente):

        py -3.12 -m pip install sacrebleu rouge-score bert-score
        py -3.12 validation_gemma_golden/golden_consistency.py

    (bert-score scarica il modello mBERT al primo avvio: serve rete)

Output: tutto quello che lo script produce (report .txt e, con --csv, la tabella
per le analisi successive) finisce in `validation_gemma_golden/results/`.

Uso tipico (lanciare dalla radice del progetto, come gli altri script):
    py -3.12 validation_gemma_golden/golden_consistency.py
    # solo una conversazione, primi 50 turni, per una prova rapida:
    py -3.12 validation_gemma_golden/golden_consistency.py --only KPN001 --limit 50
    # niente BERTScore (nessun download, molto piu' veloce):
    py -3.12 validation_gemma_golden/golden_consistency.py --no-bertscore
    # anche il CSV per le analisi successive:
    py -3.12 validation_gemma_golden/golden_consistency.py --csv
"""

import argparse, csv, json, os, re, sys

DEFAULT_LLM = "results/napoletano.json"
DEFAULT_GOLDEN = os.path.join("golden_translate", "golden_validator.json")

# cartella unica per tutto cio' che questo script produce
OUT_DIR = os.path.join("validation_gemma_golden", "results")
DEFAULT_OUT = os.path.join(OUT_DIR, "coerenza_golden_vs_gemma.txt")
DEFAULT_CSV = os.path.join(OUT_DIR, "coerenza_golden_vs_gemma.csv")

DEFAULT_BERT_MODEL = "bert-base-multilingual-cased"
DEFAULT_SOGLIA = 85.0

# pesi del grado di similarita' (devono sommare a 1)
PESI = {"chrf": 0.40, "bertscore": 0.30, "bleu": 0.15, "rouge": 0.15}

# Tokenizer Unicode-aware: tiene lettere accentate e apostrofi interni, cosi'
# "l'atu", "vuo'", "cca'" restano token sensati invece di essere sbriciolati.
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


class TokenizerNapoletano:
    """Tokenizer per rouge-score che non butta via le lettere non ASCII."""

    def tokenize(self, text):
        return _TOKEN_RE.findall(text.lower())


def verifica_dipendenze(usa_bertscore):
    """Fallisce subito e con istruzioni, invece di un ModuleNotFoundError a meta' run.

    Le librerie di metrica si importano dentro le funzioni (BERTScore e' pesante
    e con --no-bertscore non serve), quindi senza questo controllo l'errore
    arriverebbe dopo aver gia' caricato e allineato i 2568 turni.
    """
    from importlib.util import find_spec

    richiesti = [("sacrebleu", "sacrebleu"), ("rouge_score", "rouge-score")]
    if usa_bertscore:
        richiesti += [("bert_score", "bert-score"), ("transformers", "transformers"),
                      ("torch", "torch")]
    mancanti = [pip for mod, pip in richiesti if find_spec(mod) is None]
    if not mancanti:
        return

    v = f"{sys.version_info.major}.{sys.version_info.minor}"
    sys.exit(
        f"ERRORE: mancano {', '.join(mancanti)} in Python {v} ({sys.executable}).\n\n"
        f"Su questa macchina `python` e' Python 3.8, che non ha le librerie giuste:\n"
        f"servono con Python 3.12, l'unico con transformers e un torch recente.\n\n"
        f"  py -3.12 -m pip install {' '.join(mancanti)}\n"
        f"  py -3.12 {os.path.relpath(os.path.abspath(__file__))} ...\n\n"
        f"In alternativa, senza BERTScore bastano sacrebleu e rouge-score:\n"
        f"  py -3.12 {os.path.relpath(os.path.abspath(__file__))} --no-bertscore")


def load_samples(path):
    """Carica il dataset in qualsiasi forma: oggetto JSON, array JSON o JSONL."""
    raw = open(path, encoding="utf-8").read().strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        return [json.loads(l) for l in raw.splitlines() if l.strip()]


def index_turni(samples):
    """Mappa (conversazione, turn_index) -> turno, per allineare i due file."""
    idx = {}
    for s in samples:
        for t in s["turni"]:
            idx[(s["id"], int(t["turn_index"]))] = t
    return idx


def build_pairs(llm_samples, golden_samples, only, limit):
    """Allinea i due dataset e restituisce (coppie, avvisi di disallineamento)."""
    llm_idx, gold_idx = index_turni(llm_samples), index_turni(golden_samples)
    avvisi = []

    solo_llm = sorted(llm_idx.keys() - gold_idx.keys())
    solo_gold = sorted(gold_idx.keys() - llm_idx.keys())
    for k in solo_llm[:20]:
        avvisi.append(f"{k[0]} turno {k[1]}: presente in gemma4 ma non nel golden")
    for k in solo_gold[:20]:
        avvisi.append(f"{k[0]} turno {k[1]}: presente nel golden ma non in gemma4")
    if len(solo_llm) + len(solo_gold) > 40:
        avvisi.append(f"... e altri {len(solo_llm) + len(solo_gold) - 40} disallineamenti")

    pairs = []
    for s in llm_samples:                              # ordine del file gemma4
        if only and s["id"] not in only:
            continue
        for t in s["turni"]:
            key = (s["id"], int(t["turn_index"]))
            g = gold_idx.get(key)
            if g is None:
                continue
            if t.get("italiano", "") != g.get("italiano", ""):
                avvisi.append(f"{key[0]} turno {key[1]}: l'italiano differisce fra i due file")
            pairs.append({
                "conversazione": s["id"],
                "turn_index": key[1],
                "tu_id": t.get("tu_id", ""),
                "speaker": t.get("speaker", ""),
                "italiano": t.get("italiano", ""),
                "gemma": (t.get("napoletano") or "").strip(),
                "golden": (g.get("napoletano") or "").strip(),
                "note": g.get("note", ""),
            })
            if limit is not None and len(pairs) >= limit:
                return pairs, avvisi
    return pairs, avvisi


# ---------------------------------------------------------------- metriche ----

def calcola_chrf_bleu(pairs, word_order):
    """chrF e BLEU frase per frase (sacrebleu), + i valori a livello di corpus."""
    from sacrebleu.metrics import BLEU, CHRF

    chrf = CHRF(word_order=word_order)
    bleu = BLEU(effective_order=True)                  # sentence-level su turni corti

    for p in pairs:
        if not p["gemma"] or not p["golden"]:
            p["chrf"] = p["bleu"] = 0.0
            continue
        p["chrf"] = chrf.sentence_score(p["gemma"], [p["golden"]]).score
        p["bleu"] = bleu.sentence_score(p["gemma"], [p["golden"]]).score

    hyps = [p["gemma"] for p in pairs]
    refs = [[p["golden"] for p in pairs]]
    return (CHRF(word_order=word_order).corpus_score(hyps, refs).score,
            BLEU().corpus_score(hyps, refs).score)


def calcola_rouge(pairs):
    """ROUGE-L F1 con tokenizer Unicode (quello di default perde gli accenti)."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False,
                                      tokenizer=TokenizerNapoletano())
    for p in pairs:
        if not p["gemma"] or not p["golden"]:
            p["rouge"] = 0.0
            continue
        p["rouge"] = scorer.score(p["golden"], p["gemma"])["rougeL"].fmeasure * 100


def calcola_bertscore(pairs, model_type, batch_size):
    """BERTScore F1 su tutte le coppie in un colpo solo (batch)."""
    import torch
    from bert_score import score as bert_score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"BERTScore: modello {model_type} su {device} "
          f"({len(pairs)} coppie, batch {batch_size})...", file=sys.stderr)

    # le coppie con un lato vuoto non si possono valutare: restano a 0
    valide = [p for p in pairs if p["gemma"] and p["golden"]]
    for p in pairs:
        p["bertscore"] = 0.0
    if not valide:
        return

    _, _, f1 = bert_score([p["gemma"] for p in valide],
                          [p["golden"] for p in valide],
                          model_type=model_type, batch_size=batch_size,
                          device=device, verbose=True)
    for p, v in zip(valide, f1.tolist()):
        p["bertscore"] = v * 100


def similarita(p, usa_bertscore):
    """Composito pesato in [0, 100]. Senza BERTScore i pesi si rinormalizzano."""
    pesi = dict(PESI)
    if not usa_bertscore:
        pesi.pop("bertscore")
        tot = sum(pesi.values())
        pesi = {k: v / tot for k, v in pesi.items()}
    val = sum(w * p[k] for k, w in pesi.items())
    return max(0.0, min(100.0, val))


# ------------------------------------------------------------------ report ----

def fmt_bert(p, usa_bertscore):
    return f"{p['bertscore'] / 100:.4f}" if usa_bertscore else "n/d (--no-bertscore)"


def scrivi_report(path, pairs, args, corpus_chrf, corpus_bleu, avvisi, usa_bertscore):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = len(pairs)
    da_sostituire = sum(1 for p in pairs if p["sostituire"])
    media_sim = sum(p["similarita"] for p in pairs) / n if n else 0.0

    def media(k):
        return sum(p[k] for p in pairs) / n if n else 0.0

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write("COERENZA TRADUZIONI NAPOLETANE — gemma4:cloud vs golden validato a mano\n")
        f.write("=" * 78 + "\n")
        f.write(f"Ipotesi (candidato) : {args.llm}\n")
        f.write(f"Riferimento (gold)  : {args.golden}\n")
        f.write(f"Turni confrontati   : {n}\n")
        f.write(f"BERTScore           : {args.bert_model if usa_bertscore else 'disattivato'}\n")
        f.write(f"chrF                : word_order={args.chrf_word_order}"
                f"{' (chrF++)' if args.chrf_word_order else ''}\n")
        pesi_txt = ", ".join(f"{k} {v:.0%}" for k, v in PESI.items()) if usa_bertscore \
            else "chrf 57%, bleu 21%, rouge 21% (rinormalizzati senza BERTScore)"
        f.write(f"Pesi similarita'    : {pesi_txt}\n")
        f.write(f"Soglia booleano     : sostituire = similarita' < {args.soglia:.1f}%\n\n")

        f.write("MEDIE SULL'INTERO INSIEME\n")
        f.write(f"  BERTScore F1 medio : {media('bertscore') / 100:.4f}\n"
                if usa_bertscore else "  BERTScore F1 medio : n/d\n")
        f.write(f"  BLEU medio (frase) : {media('bleu'):.2f}\n")
        f.write(f"  ROUGE-L F1 medio   : {media('rouge') / 100:.4f}\n")
        f.write(f"  chrF medio (frase) : {media('chrf'):.2f}\n")
        f.write(f"  BLEU di corpus     : {corpus_bleu:.2f}\n")
        f.write(f"  chrF di corpus     : {corpus_chrf:.2f}\n")
        f.write(f"  Similarita' media  : {media_sim:.2f} %\n")
        f.write(f"  Da sostituire      : {da_sostituire}/{n} "
                f"({da_sostituire / n * 100:.1f}%) — identici (100%): "
                f"{sum(1 for p in pairs if p['similarita'] >= 99.995)}\n\n")

        if avvisi:
            f.write(f"AVVISI ({len(avvisi)})\n")
            for a in avvisi[:30]:
                f.write(f"  - {a}\n")
            if len(avvisi) > 30:
                f.write(f"  ... e altri {len(avvisi) - 30}\n")
            f.write("\n")

        f.write("=" * 78 + "\nDETTAGLIO PER TURNO\n" + "=" * 78 + "\n\n")
        for p in pairs:
            f.write(f"[{p['conversazione']} | turno {p['turn_index']} | "
                    f"tu_id {p['tu_id']} | speaker {p['speaker']}]\n")
            f.write(f"Italiano                    : {p['italiano']}\n")
            f.write(f"Frase dialetto gemma4:cloud : {p['gemma']}\n")
            f.write(f"Frase dialetto golden       : {p['golden']}\n")
            f.write(f"Bert-Score (F1)             : {fmt_bert(p, usa_bertscore)}\n")
            f.write(f"BLEU                        : {p['bleu']:.2f}\n")
            f.write(f"Rouge (ROUGE-L F1)          : {p['rouge'] / 100:.4f}\n")
            f.write(f"ChrF                        : {p['chrf']:.2f}\n")
            f.write(f"Grado di similarita'        : {p['similarita']:.2f} %\n")
            f.write(f"Sostituire con golden       : {p['sostituire']}\n")
            if p["note"]:
                f.write(f"Note del validatore         : {p['note']}\n")
            f.write("\n")

    return n, da_sostituire, media_sim


CSV_FIELDS = ["conversazione", "turn_index", "tu_id", "speaker", "italiano",
              "gemma", "golden", "bertscore", "bleu", "rouge", "chrf",
              "similarita", "sostituire", "note"]


def scrivi_csv(path, pairs):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for p in pairs:
            r = dict(p)
            for k in ("bertscore", "bleu", "rouge", "chrf", "similarita"):
                r[k] = round(p[k], 4)
            w.writerow(r)


# -------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default=DEFAULT_LLM,
                    help=f"JSON con la bozza LLM (default: {DEFAULT_LLM})")
    ap.add_argument("--golden", default=DEFAULT_GOLDEN,
                    help=f"JSON golden validato a mano (default: {DEFAULT_GOLDEN})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"Report di testo (default: {DEFAULT_OUT})")
    ap.add_argument("--csv", nargs="?", const=DEFAULT_CSV, default=None, metavar="PATH",
                    help=f"Scrive anche un CSV con gli stessi dati, per le analisi "
                         f"(senza argomento: {DEFAULT_CSV})")
    ap.add_argument("--only", nargs="+", default=None, metavar="CODE",
                    help="Confronta solo queste conversazioni (es. KPN001)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Confronta solo i primi N turni (prova rapida)")
    ap.add_argument("--soglia", type=float, default=DEFAULT_SOGLIA,
                    help=f"Sotto questa similarita' il booleano e' True "
                         f"(default: {DEFAULT_SOGLIA})")
    ap.add_argument("--bert-model", default=DEFAULT_BERT_MODEL,
                    help=f"Modello per BERTScore (default: {DEFAULT_BERT_MODEL})")
    ap.add_argument("--bert-batch", type=int, default=64, help="Batch size di BERTScore")
    ap.add_argument("--no-bertscore", action="store_true",
                    help="Salta BERTScore (nessun download, molto piu' veloce)")
    ap.add_argument("--chrf-word-order", type=int, default=0,
                    help="0 = chrF (default), 2 = chrF++")
    args = ap.parse_args()

    verifica_dipendenze(not args.no_bertscore)

    # un nome di file senza cartella finisce comunque in validation_gemma_golden/results
    args.out = args.out if os.path.dirname(args.out) else os.path.join(OUT_DIR, args.out)
    if args.csv and not os.path.dirname(args.csv):
        args.csv = os.path.join(OUT_DIR, args.csv)

    for p in (args.llm, args.golden):
        if not os.path.exists(p):
            sys.exit(f"ERRORE: file non trovato: {p!r}")

    llm_samples = load_samples(args.llm)
    golden_samples = load_samples(args.golden)

    only = set(args.only) if args.only else None
    if only:
        ignoti = only - {s["id"] for s in llm_samples}
        if ignoti:
            sys.exit(f"ERRORE: conversazioni non presenti in {args.llm}: {', '.join(sorted(ignoti))}")

    pairs, avvisi = build_pairs(llm_samples, golden_samples, only, args.limit)
    if not pairs:
        sys.exit("ERRORE: nessun turno allineato fra i due file (controlla id e turn_index).")
    print(f"Turni allineati: {len(pairs)}"
          + (f"  |  {len(avvisi)} avvisi" if avvisi else ""))

    usa_bertscore = not args.no_bertscore
    corpus_chrf, corpus_bleu = calcola_chrf_bleu(pairs, args.chrf_word_order)
    calcola_rouge(pairs)
    if usa_bertscore:
        calcola_bertscore(pairs, args.bert_model, args.bert_batch)
    else:
        for p in pairs:
            p["bertscore"] = 0.0

    for p in pairs:
        p["similarita"] = similarita(p, usa_bertscore)
        p["sostituire"] = p["similarita"] < args.soglia

    n, da_sostituire, media_sim = scrivi_report(
        args.out, pairs, args, corpus_chrf, corpus_bleu, avvisi, usa_bertscore)
    if args.csv:
        scrivi_csv(args.csv, pairs)

    print(f"\nScritto: {args.out}" + (f"\nScritto: {args.csv}" if args.csv else ""))
    print(f"Turni confrontati : {n}")
    print(f"Similarita' media : {media_sim:.2f} %")
    print(f"Da sostituire     : {da_sostituire}/{n} ({da_sostituire / n * 100:.1f}%) "
          f"con soglia {args.soglia:.1f}%")


if __name__ == "__main__":
    main()
