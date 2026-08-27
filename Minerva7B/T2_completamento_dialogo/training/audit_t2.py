#!/usr/bin/env python3
"""
audit_t2.py — controlla i tre JSON PRIMA di spendere GPU.

Non produce un giudizio estetico: ogni voce e' un controllo che, se fallisce,
rende invalido un numero a valle. Le voci sono divise in
  [OK]      va bene
  [NOTA]    da sapere quando leggi i risultati, non blocca
  [ALLARME] se non lo risolvi, i numeri finali non vogliono dire niente

    python audit_t2.py --split-dir /kaggle/working/split --out audit_t2.json
"""

import argparse
import json
import re
import statistics as st
from collections import Counter

import Minerva7B.T2_completamento_dialogo.training.t2_common as C


def audit(split_dir: str) -> dict:
    # verifica=False: si carica comunque, cosi' l'audit puo' DIRE cosa non va
    # invece di morire su un'eccezione trenta righe piu' avanti
    dati = {s: C.load_split(split_dir, s, verifica=False)
            for s in ("train", "dev", "test")}
    rep, esiti = {}, []

    def dire(livello, testo):
        esiti.append((livello, testo))
        print(f"[{livello}] {testo}")

    # -- 0. e' davvero il layout giusto? -----------------------------------
    print("\n=== 0. Layout ===")
    guasto = False
    for s, v in dati.items():
        ok, motivo = C.e_layout_t2(v)
        print(f"  {s:5s} layout={v[0].get('layout') if v else '?'} -> "
              f"{'OK' if ok else 'NO: ' + motivo}")
        guasto |= not ok
    if guasto:
        print("\n[ALLARME] I file non sono il layout2 (completamento di turno).\n"
              "  Questo notebook addestra SOLO T2. Se il tuo Dataset contiene\n"
              "  tutti e tre i layout, punta SPLIT_SRC alla sottocartella\n"
              "  layout2_completamento_turno e riesegui il Passo 2.\n"
              "  Mi fermo qui: proseguire su dati sbagliati non ha senso.")
        raise SystemExit(2)
    print("  [OK] tutti e tre i file sono layout2 / T2")

    # -- 1. schema ---------------------------------------------------------
    print("\n=== 1. Schema e integrita' ===")
    campi_attesi = {"id", "prompt", "target", "speaker", "conversazione",
                    "turn_index", "fonte", "split"}
    for s, v in dati.items():
        mancanti = set()
        for r in v:
            mancanti |= campi_attesi - set(r)
        if mancanti:
            dire("ALLARME", f"{s}: campi mancanti {mancanti}")
        vuoti = sum(1 for r in v if not r["target"].strip()
                    or C.MARCA_ISTRUZIONE not in r["prompt"])
        if vuoti:
            dire("ALLARME", f"{s}: {vuoti} item con target vuoto o prompt malformato")
    sim = {str(r.get("similarita", "")) for v in dati.values() for r in v}
    if sim == {""}:
        dire("NOTA", "il campo 'similarita' e' vuoto ovunque: e' un campo morto, "
                     "nessuno script lo deve leggere")
    if not [e for e in esiti if e[0] == "ALLARME"]:
        dire("OK", "schema coerente sui tre split")

    # -- 2. dimensioni e composizione --------------------------------------
    print("\n=== 2. Dimensioni e composizione ===")
    rep["dimensioni"] = {}
    for s, v in dati.items():
        fonti = Counter(r["fonte"] for r in v)
        conv = Counter(r["conversazione"] for r in v)
        spk = Counter(r["speaker"] for r in v)
        rep["dimensioni"][s] = {"n": len(v), "fonte": dict(fonti),
                                "conversazione": dict(conv), "speaker": dict(spk)}
        print(f"  {s:5s} n={len(v):4d} | fonti={dict(fonti)} | conv={dict(conv)}")
    n_tr = len(dati["train"])
    if n_tr < 1000:
        dire("NOTA", f"train di sole {n_tr} istanze: con batch efficace 16 sono "
                     f"~{n_tr // 16} step per epoca. eval_steps va DERIVATO da "
                     "questo numero, non messo fisso a 100 (non partirebbe "
                     "nessuna valutazione e l'early stopping resterebbe inerte)")

    # -- 3. la regola prefisso/target --------------------------------------
    print("\n=== 3. Come e' tagliato il turno ===")
    rapporti, delta = [], Counter()
    for r in dati["train"]:
        _, pref = C.parse_item(r)
        np_, nt = len(pref.split()), len(r["target"].split())
        rapporti.append(np_ / (np_ + nt))
        delta[nt - np_] += 1
    rep["taglio"] = {"rapporto_medio": round(st.mean(rapporti), 3),
                     "delta_target_meno_prefisso": dict(delta)}
    print(f"  prefisso / turno intero = {st.mean(rapporti):.3f} (mediana "
          f"{st.median(rapporti):.3f})")
    print(f"  len(target) - len(prefisso) in parole: {dict(delta)}")
    if set(delta) <= {0, 1}:
        dire("NOTA", "il taglio e' DETERMINISTICO: prefisso = meta' esatta delle "
                     "parole del turno, target = l'altra meta' (+1 se dispari). "
                     "Due conseguenze operative: (a) la lunghezza attesa della "
                     "continuazione e' nota a inferenza e va usata per fissare "
                     "max_new_tokens, invece di lasciare il modello libero; "
                     "(b) l'EOS che il modello impara NON e' una fine di frase "
                     "linguistica ma un conteggio di parole, quindi non aspettarti "
                     "che le continuazioni siano sintatticamente concluse")

    # -- 4. lunghezze e deriva fra split -----------------------------------
    print("\n=== 4. Lunghezze ===")
    rep["lunghezze"] = {}
    for s, v in dati.items():
        tl = [len(r["target"].split()) for r in v]
        rep["lunghezze"][s] = {"media": round(st.mean(tl), 2),
                               "mediana": st.median(tl),
                               "min": min(tl), "max": max(tl)}
        print(f"  {s:5s} target: media {st.mean(tl):.2f} | mediana "
              f"{st.median(tl)} | min {min(tl)} | max {max(tl)}")
    m_tr = rep["lunghezze"]["train"]["media"]
    m_te = rep["lunghezze"]["test"]["media"]
    if abs(m_tr - m_te) / m_tr > 0.15:
        dire("NOTA", f"target di train piu' lunghi del {100*(m_tr-m_te)/m_te:.0f}% "
                     f"rispetto a test ({m_tr} vs {m_te} parole). Conseguenza "
                     "diretta: il modello addestrato tendera' a generare lungo e "
                     "il chrF++, che penalizza l'eccesso, ne paghera' il conto. "
                     "Per questo la valutazione riporta anche la variante a "
                     "lunghezza controllata")
    if min(min(len(r["target"].split()) for r in v) for v in dati.values()) >= 3:
        dire("OK", "nessun target sotto le 3 parole: i turni corti sono gia' stati "
                   "filtrati a monte, non c'e' il problema che affossa i chrF su "
                   "target di una parola")

    # -- 5. contesto -------------------------------------------------------
    print("\n=== 5. Contesto conversazionale ===")
    rep["contesto"] = {}
    for s, v in dati.items():
        n_turni = Counter()
        stesso_spk = 0
        for r in v:
            ctx, _ = C.parse_item(r)
            t = C.turni_contesto(ctx)
            n_turni[len(t)] += 1
            if t and t[-1][0] == r["speaker"]:
                stesso_spk += 1
        rep["contesto"][s] = {"turni": dict(n_turni),
                              "ultimo_turno_stesso_speaker": stesso_spk}
        print(f"  {s:5s} turni di contesto: {dict(sorted(n_turni.items()))} | "
              f"ultimo turno dello stesso parlante: {stesso_spk}/{len(v)}")
    senza = sum(1 for v in dati.values() for r in v
                if not C.parse_item(r)[0].strip())
    if senza:
        dire("NOTA", f"{senza} item in tutto senza contesto (inizio conversazione): "
                     "vanno tenuti, ma la metrica ctx_acc va calcolata solo su "
                     "quelli che il contesto ce l'hanno")
    dire("NOTA", "il prompt NON dichiara chi sta parlando nel turno da completare. "
                 "Il campo 'speaker' pero' c'e': il rendering del prompt lo inietta "
                 "nell'istruzione ('Continua il turno di B'). Senza, su un contesto "
                 "come 'A: emh' il modello non ha modo di sapere se deve rispondere "
                 "ad A o proseguire A")

    # -- 6. igiene dello split --------------------------------------------
    print("\n=== 6. Igiene dello split ===")
    ids = {s: {r["id"] for r in v} for s, v in dati.items()}
    sovrap = {f"{a}/{b}": len(ids[a] & ids[b])
              for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))}
    print("  id in comune:", sovrap)
    if any(sovrap.values()):
        dire("ALLARME", f"id duplicati fra split: {sovrap}")

    finestre = {}
    for s, v in dati.items():
        for c in sorted({r["conversazione"] for r in v}):
            ti = sorted(r["turn_index"] for r in v if r["conversazione"] == c)
            finestre[f"{s}/{c}"] = (ti[0], ti[-1])
    print("  finestre di turn_index:", finestre)
    cronologico = all(
        finestre[f"train/{c}"][1] < finestre[f"dev/{c}"][0] <
        finestre[f"dev/{c}"][1] < finestre[f"test/{c}"][0]
        for c in ("KPN001", "KPN003") if f"train/{c}" in finestre)
    rep["split_cronologico"] = cronologico

    # Uno split non cronologico non e' di per se' sbagliato: lo e' se manca la
    # zona cuscinetto. Il controllo che conta e' diretto: nessun turno di
    # dev/test deve comparire nel CONTESTO di un item di train (e viceversa).
    # Si verifica sui turni, non sulle finestre di indici.
    turni = {s: {(r["conversazione"], r["turn_index"]) for r in v}
             for s, v in dati.items()}
    perdite = {}
    for s, v in dati.items():
        altrui = set()
        for t in ("train", "dev", "test"):
            if t != s:
                altrui |= turni[t]
        n = 0
        for r in v:
            for k in (1, 2, 3):
                if (r["conversazione"], r["turn_index"] - k) in altrui:
                    n += 1
                    break
        perdite[s] = n
    print("  item il cui contesto tocca un turno di un altro split:", perdite)
    rep["contesto_fra_split"] = perdite

    if cronologico:
        dire("OK", "lo split e' CRONOLOGICO: blocchi contigui di turni, "
                   "train prima, poi dev, poi test. Nessun turno di test puo' "
                   "comparire nel contesto di un item di train")
    elif not any(perdite.values()):
        dire("OK", "lo split NON e' cronologico ma la zona cuscinetto tiene: "
                   "zero item hanno nel proprio contesto un turno di un altro "
                   "split. E' la divisione a blocchi interlacciati, che copre "
                   "tutti gli argomenti in tutti e tre gli split invece di "
                   "lasciarne un terzo fuori dal train")
    else:
        dire("ALLARME", f"split non cronologico E senza cuscinetto: {perdite} "
                        "item hanno nel contesto un turno di un altro split. "
                        "Il modello li ha letti in addestramento")

    testi_train = "\n".join(r["prompt"] for r in dati["train"])
    for s in ("dev", "test"):
        n = sum(1 for r in dati[s] if r["target"] in testi_train)
        print(f"  target di {s} presenti nei prompt di train: {n}/{len(dati[s])}")
        if n > 0.05 * len(dati[s]):
            dire("ALLARME" if cronologico else "NOTA",
                 f"{n} target di {s} ({100*n/len(dati[s]):.0f}%) compaiono "
                 "alla lettera nei prompt di train. Su un corpus di due sole "
                 "conversazioni le formule brevi si ripetono ('sì, ce sta', "
                 "'nun 'o saccio'): non e' lo stesso turno, e' la stessa "
                 "frase detta altrove. rebuild_t2_data.py li elimina con "
                 "--anti-eco (attivo di default)")

    for s, v in dati.items():
        auto = sum(1 for r in v if r["target"] in C.parse_item(r)[0])
        if auto:
            dire("NOTA", f"{s}: {auto} item hanno il proprio target gia' dentro il "
                         "proprio contesto (il parlante si ripete). Sono gratis per "
                         "il modello e vanno esclusi dalla lettura qualitativa")
        dup = len(v) - len({r["prompt"] for r in v})
        if dup:
            dire("ALLARME", f"{s}: {dup} prompt duplicati")

    # -- 7. sorgente dei target -------------------------------------------
    print("\n=== 7. Qualita' dei riferimenti ===")
    tipi = C.costruisci_lessico(dati["train"])
    rep["tipi_dialettali"] = len(tipi)
    print(f"  tipi dialettali estratti dal solo train: {len(tipi)}")
    for s, v in dati.items():
        for f in sorted({r["fonte"] for r in v}):
            sub = [r["target"] for r in v if r["fonte"] == f]
            d = C.densita_dialettale(sub, tipi)
            print(f"  {s:5s} fonte={f:8s} n={len(sub):4d} densita' dialettale={d:.3f}")
    sint = {s: sum(1 for r in v if r["fonte"] != "golden") for s, v in dati.items()}
    if sint["dev"] or sint["test"]:
        dire("NOTA", f"dev e test contengono target sintetici (fonte != golden): "
                     f"{sint['dev']} e {sint['test']}. Non sono riferimenti umani: "
                     "le metriche vanno riportate anche sul solo sottoinsieme "
                     "'golden', altrimenti stai misurando quanto il modello imita "
                     "un altro modello")

    # -- 8. forma della superficie ----------------------------------------
    print("\n=== 8. Forma dei target ===")
    maiusc = sum(1 for r in dati["train"] if r["target"][:1].isupper())
    punteg = sum(1 for r in dati["train"] if r["target"][:1] in ",.;:!?")
    print(f"  target che iniziano per maiuscola: {maiusc}/{n_tr} | "
          f"per punteggiatura: {punteg}/{n_tr}")
    dire("OK" if maiusc / n_tr < 0.02 else "NOTA",
         "i target iniziano quasi sempre in minuscola e si saldano al prefisso "
         "con un solo spazio: e' esattamente la giunzione che il rendering del "
         "prompt deve riprodurre carattere per carattere")

    # -- 9. fondo scala ----------------------------------------------------
    print("\n=== 9. Fondo scala delle metriche (leggere prima dei risultati) ===")
    rep["baseline"] = {}
    for s in ("dev", "test"):
        b = C.baseline_taratura(dati[s], tipi)
        rep["baseline"][s] = b
        print(f"  {s}: {json.dumps(b, ensure_ascii=False)}")
    forbice = (rep["baseline"]["test"]["ripeti_il_prefisso"]
               - rep["baseline"]["test"]["pavimento_target_mescolati"])
    dire("NOTA", f"fra il caso puro (target mescolati, chrF++ "
                 f"{rep['baseline']['test']['pavimento_target_mescolati']}) e la "
                 f"baseline banale (ripeti il prefisso, "
                 f"{rep['baseline']['test']['ripeti_il_prefisso']}) ci sono "
                 f"{forbice:.1f} punti. Su T2 il chrF++ ha una forbice "
                 "strettissima perche' il riferimento e' UNO fra molte "
                 "continuazioni valide: da solo non basta a dire se il modello "
                 "ha imparato. Per questo si misurano anche densita' dialettale, "
                 "tasso di copia, rapporto di lunghezza e ctx_acc")

    rep["esiti"] = [{"livello": l, "testo": t} for l, t in esiti]
    print("\n=== Riepilogo ===")
    c = Counter(l for l, _ in esiti)
    print(f"  OK: {c['OK']} | NOTA: {c['NOTA']} | ALLARME: {c['ALLARME']}")
    if c["ALLARME"]:
        print("  -> ci sono ALLARMI: risolvili prima di addestrare.")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out", default="audit_t2.json")
    a = ap.parse_args()
    r = audit(a.split_dir)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print("\nreport scritto in", a.out)
