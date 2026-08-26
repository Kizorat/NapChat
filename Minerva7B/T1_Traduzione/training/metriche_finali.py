#!/usr/bin/env python3
"""
metriche_finali.py — dai .preds.jsonl di evaluate_task.py produce
  1. UN CSV con tutte le generazioni di test (T1, T2, T3, tutti i modelli)
  2. BERTScore + chrF++ + BLEU + metriche lessicali, con le baseline di taratura

Non rigenera niente: `evaluate_task.py` scrive gia' {run}.preds.jsonl con
{id, prompt, target, hyp}. La generazione e' la parte cara e non va rifatta se
una metrica fallisce.

BERTScore sul napoletano: leggere con attenzione
------------------------------------------------
Nessun encoder pubblico conosce il napoletano. mBERT, XLM-R e i modelli italiani
lo trattano come italiano rumoroso, quindi BERTScore qui misura la similarita'
semantica *vista attraverso lo spazio dell'italiano*. Due conseguenze concrete:

  * e' quasi cieco alla correttezza dialettale: un output in italiano e un
    riferimento in napoletano finiscono vicini;
  * **premia la copia dell'italiano**, che e' esattamente il fallimento che stai
    cercando di misurare.

Per questo non va mai riportato da solo. Lo script calcola sempre due tarature:

  pavimento    riferimento di UN ALTRO item (napoletano non correlato). E' il
               valore che BERTScore da' a due frasi della stessa varieta' che
               non c'entrano niente: sotto quel numero non si scende, e la
               distanza fra pavimento e sistema e' l'intervallo utile reale.
  copia-it     solo T1: la frase italiana di partenza come ipotesi. Se il tuo
               modello non stacca nettamente questo valore, BERTScore non sta
               premiando la traduzione.

Il tetto (riferimento contro se stesso) e' 1.0 per costruzione e non si stampa.

Uso
---
    pip install bert-score --quiet
    python metriche_finali.py --preds-dir /kaggle/working/eval \\
        --lessico /kaggle/working/lessico_train.json \\
        --csv /kaggle/working/predizioni_test.csv \\
        --out /kaggle/working/metriche_finali.json

    # senza BERTScore (piu' veloce, nessun download)
    python metriche_finali.py --preds-dir ... --no-bertscore
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

TOK = re.compile(r"[a-zàèéìòóùâêîôû'\u2019\-]+")
ISTRUZIONE = {"T1": "Traduci in napoletano: ",
              "T2": "Continua il turno in napoletano: "}


def norm(s):
    return str(s).replace("\u2019", "'")


def tokenizza(s):
    return TOK.findall(norm(s).lower())


def parse_run(nome):
    """'minerva-7b-instruct-v1.0__T1__ft__nucleus' -> componenti."""
    p = nome.split("__")
    return {"modello": p[0],
            "task": p[1] if len(p) > 1 else "?",
            "tag": p[2] if len(p) > 2 else "?",
            "decoding": p[3] if len(p) > 3 else "greedy"}


def parte_variabile(prompt, task):
    """La frase italiana (T1) o il prefisso napoletano (T2) dentro il prompt."""
    m = ISTRUZIONE.get(task)
    if not m or m not in prompt:
        return ""
    return prompt.rpartition(m)[2].split("\n")[0].strip()


def blocco_contesto(prompt):
    """I turni di contesto, se il prompt ne ha. Utile da avere in colonna."""
    try:
        from Minerva7B.T1_Traduzione.training.contesto_metrica import estrai_blocco_contesto
    except ImportError:
        return ""
    t = estrai_blocco_contesto(prompt)
    return " | ".join(t[2]) if t else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds-dir", default="/kaggle/working/eval")
    ap.add_argument("--csv", default="/kaggle/working/predizioni_test.csv")
    ap.add_argument("--out", default="/kaggle/working/metriche_finali.json")
    ap.add_argument("--lessico", default=None,
                    help="output di lessico.py: abilita le metriche lessicali")
    ap.add_argument("--bert-model", default="bert-base-multilingual-cased",
                    help="encoder per BERTScore. Alternative: xlm-roberta-large, "
                         "dbmdz/bert-base-italian-xxl-cased (tokenizza meglio le "
                         "forme apostrofate, ma nessuno dei tre conosce il napoletano)")
    ap.add_argument("--no-bertscore", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import pandas as pd
    import sacrebleu

    file_preds = sorted(Path(a.preds_dir).glob("*.preds.jsonl"))
    if not file_preds:
        sys.exit(f"Nessun *.preds.jsonl in {a.preds_dir}. Esegui prima "
                 f"evaluate_task.py per i tre task.")
    print(f"Trovati {len(file_preds)} file di predizioni:")

    righe = []
    for f in file_preds:
        info = parse_run(f.name.replace(".preds.jsonl", ""))
        n = 0
        for i, linea in enumerate(f.read_text(encoding="utf-8").splitlines()):
            if not linea.strip():
                continue
            d = json.loads(linea)
            righe.append({**info, "idx": i, "id": d.get("id"),
                          "prompt": d["prompt"],
                          "contesto": blocco_contesto(d["prompt"]),
                          "ingresso": parte_variabile(d["prompt"], info["task"]),
                          "riferimento": norm(d["target"]),
                          "generato": norm(d["hyp"])})
            n += 1
        print(f"  {f.name:52s} {n:5d} item")

    df = pd.DataFrame(righe)

    # --- metriche per item ---------------------------------------------------
    df["chrf_item"] = [
        round(sacrebleu.sentence_chrf(h, [r], word_order=2).score, 2)
        for h, r in zip(df["generato"], df["riferimento"])]
    df["parole_rif"] = df["riferimento"].map(lambda s: len(s.split()))
    df["parole_gen"] = df["generato"].map(lambda s: len(s.split()))

    lex = None
    vocab_nap = None
    if a.lessico:
        from Minerva7B.T1_Traduzione.training.pesi_lessicali import carica_lessico
        lex = carica_lessico(a.lessico)
        dial = lex["dialettali"]
        df["dial_rif"] = df["riferimento"].map(lambda s: sum(w in dial for w in tokenizza(s)))
        df["dial_gen"] = df["generato"].map(lambda s: sum(w in dial for w in tokenizza(s)))

        # --- forme inedite ---------------------------------------------------
        # Parole prodotte che non compaiono da nessuna parte nel napoletano di
        # train. Cattura le forme inventate per analogia morfologica
        # ("puparuolo", "cusinajo", "pruove ccusarelle"): sono fluenti, superano
        # il discriminatore ita/nap, e nessuna delle altre metriche le vede.
        # Va TARATA: il vocabolario di train ha ~1.500 tipi, quindi anche i
        # riferimenti umani di test contengono forme fuori vocabolario. Il loro
        # tasso e' il pavimento naturale, e si calcola sulla stessa colonna.
        grezzo = json.loads(Path(a.lessico).read_text(encoding="utf-8"))
        vocab_nap = set(grezzo.get("vocabolario_nap", {}))
        if not vocab_nap:
            print("! il lessico non contiene 'vocabolario_nap': rigenera con la "
                  "versione aggiornata di lessico.py per avere le forme inedite")
        else:
            def inedite(testo):
                t = tokenizza(testo)
                return sum(1 for w in t if w not in vocab_nap), len(t)
            for col, dest in (("generato", "inedite_gen"), ("riferimento", "inedite_rif")):
                coppie = df[col].map(inedite)
                df[dest] = [c[0] for c in coppie]
                df[dest + "_su"] = [c[1] for c in coppie]

    # --- BERTScore -----------------------------------------------------------
    # Un solo passaggio su TUTTE le ipotesi e su tutte le serie di taratura:
    # caricare l'encoder una volta sola e' l'unica parte costosa.
    if not a.no_bertscore:
        try:
            from bert_score import score as bert_score
        except ImportError:
            sys.exit("bert-score non installato: pip install bert-score --quiet\n"
                     "(oppure usa --no-bertscore)")
        rnd = random.Random(a.seed)

        cand, rif, etichette = [], [], []
        for (mod, task, tag, dec), g in df.groupby(["modello", "task", "tag", "decoding"]):
            r = g["riferimento"].tolist()
            chiave = (mod, task, tag, dec)
            # sistema
            cand += g["generato"].tolist(); rif += r
            etichette += [(chiave, "sistema")] * len(r)
            # pavimento: riferimento di un altro item, stessa varieta', zero relazione
            perm = list(range(len(r)))
            rnd.shuffle(perm)
            perm = [p if p != k else (p + 1) % len(r) for k, p in enumerate(perm)]
            cand += [r[p] for p in perm]; rif += r
            etichette += [(chiave, "pavimento")] * len(r)
            # copia-italiano: solo T1, dove l'ingresso e' la frase italiana
            if task == "T1" and g["ingresso"].str.len().gt(0).all():
                cand += g["ingresso"].tolist(); rif += r
                etichette += [(chiave, "copia_italiano")] * len(r)

        print(f"\nBERTScore con {a.bert_model} su {len(cand)} coppie "
              f"(sistema + tarature)...")
        P, R, F = bert_score(cand, rif, model_type=a.bert_model,
                             batch_size=a.batch, verbose=False)
        bs = {}
        for (chiave, serie), p, r_, f_ in zip(etichette, P.tolist(), R.tolist(), F.tolist()):
            bs.setdefault((chiave, serie), []).append((p, r_, f_))
        # il valore per item del solo sistema va anche nel CSV
        per_item = {}
        for (chiave, serie), vals in bs.items():
            if serie == "sistema":
                per_item[chiave] = [v[2] for v in vals]
        colonna = []
        for (mod, task, tag, dec), g in df.groupby(["modello", "task", "tag", "decoding"]):
            colonna += list(zip(g.index, per_item[(mod, task, tag, dec)]))
        for i, v in colonna:
            df.loc[i, "bertscore_f1"] = round(v, 4)

    # --- CSV -----------------------------------------------------------------
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.csv, index=False)
    print(f"\nScritto {a.csv}: {len(df)} righe, colonne {list(df.columns)}")

    # --- riepilogo per run ---------------------------------------------------
    from Minerva7B.T1_Traduzione.training.pesi_lessicali import valuta
    riepilogo = {}
    print(f"\n{'run':40s} {'n':>5s} {'chrF++':>7s} {'BS-F1':>7s} {'BS-pav':>7s} "
          f"{'rec_d':>6s} {'ined':>6s} {'(um)':>6s} {'len':>5s}")
    for (mod, task, tag, dec), g in df.groupby(["modello", "task", "tag", "decoding"]):
        chiave = (mod, task, tag, dec)
        h, r = g["generato"].tolist(), g["riferimento"].tolist()
        voce = {
            "modello": mod, "task": task, "tag": tag, "decoding": dec, "n": len(g),
            "chrf++": round(sacrebleu.corpus_chrf(h, [r], word_order=2).score, 2),
            "bleu": round(sacrebleu.corpus_bleu(h, [r]).score, 2),
            "rapporto_lunghezza": round(g["parole_gen"].sum() / max(1, g["parole_rif"].sum()), 3),
        }
        if not a.no_bertscore:
            for serie in ("sistema", "pavimento", "copia_italiano"):
                v = bs.get((chiave, serie))
                if not v:
                    continue
                n = len(v)
                voce[f"bertscore_{serie}"] = {
                    "P": round(sum(x[0] for x in v) / n, 4),
                    "R": round(sum(x[1] for x in v) / n, 4),
                    "F1": round(sum(x[2] for x in v) / n, 4)}
            voce["_lettura_bertscore"] = (
                "confronta F1 con bertscore_pavimento (napoletano non correlato): "
                "quello e' il vero zero della scala. Su T1 confronta anche con "
                "copia_italiano: l'encoder non conosce il napoletano e premia la copia.")
        if lex:
            src = g["ingresso"].tolist() if task == "T1" else r
            voce["lessicali"] = {k: v for k, v in valuta(src, r, h, lex).items()
                                 if not k.startswith("_")}
        if vocab_nap:
            tg, sg = g["inedite_gen"].sum(), g["inedite_gen_su"].sum()
            tr_, sr = g["inedite_rif"].sum(), g["inedite_rif_su"].sum()
            voce["forme_inedite"] = {
                "tasso_generato": round(tg / max(1, sg), 4),
                "tasso_riferimenti_umani": round(tr_ / max(1, sr), 4),
                "_lettura": "il tasso dei riferimenti umani e' il pavimento "
                            "naturale (il vocabolario di train non copre tutto il "
                            "napoletano). Conta solo l'ECCESSO del generato su "
                            "quel valore: e' la quota di forme inventate."}
        riepilogo[f"{mod}__{task}__{tag}__{dec}"] = voce

        bsf = voce.get("bertscore_sistema", {}).get("F1", float("nan"))
        bsp = voce.get("bertscore_pavimento", {}).get("F1", float("nan"))
        recd = voce.get("lessicali", {}).get("recall_dialettale", float("nan"))
        ing = voce.get("forme_inedite", {})
        print(f"{mod[:14]+'__'+task+'__'+tag:40s} {len(g):5d} {voce['chrf++']:7.2f} "
              f"{bsf:7.4f} {bsp:7.4f} {recd:6.3f} "
              f"{ing.get('tasso_generato', float('nan')):6.3f} "
              f"{ing.get('tasso_riferimenti_umani', float('nan')):6.3f} "
              f"{voce['rapporto_lunghezza']:5.2f}")

    Path(a.out).write_text(json.dumps(riepilogo, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print(f"\nScritto {a.out}")
    if not a.no_bertscore:
        print("\nBS-pav e' BERTScore fra riferimenti NON correlati: e' il pavimento "
              "della scala.\nLa distanza BS-F1 meno BS-pav e' l'intervallo utile, "
              "non BS-F1 in assoluto.")
    if vocab_nap:
        print("ined = quota di parole generate fuori dal vocabolario napoletano di "
              "train; (um) e' lo\nstesso tasso sui riferimenti umani di test. Conta "
              "l'eccesso di ined su (um), non ined.")


if __name__ == "__main__":
    main()
