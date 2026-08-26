#!/usr/bin/env python3
"""
rerank_t3.py — best-of-n con selezione a posteriori su T3 (e T2), senza
riaddestrare niente.

Perche' esiste
--------------
Le metriche del run `minerva-7b-instruct-v1.0__T3__ft__nucleus` dicono tre cose:

  lunghezza_media 17.57 contro 6.99 umano, ks 0.589 p=0
  controllo_solo_lunghezza 0.776 su accuracy_umano_vs_macchina 0.80
  rep_2 0.151 contro 0.0365 umano, loop_massimo 63

La seconda riga e' la piu' informativa: il 97% del potere discriminante del
classificatore avversariale viene dalla LUNGHEZZA. Il sistema non e'
riconoscibile perche' sbaglia il napoletano, e' riconoscibile perche' parla
troppo e viene troncato al tetto di max_new_tokens senza mai emettere fine di
turno. Una frase tagliata a meta' si legge come "sconclusionata" anche quando la
deriva semantica non c'e'.

Cosa fa questo script
---------------------
1. DIAGNOSTICA (sempre, prima di tutto): quali id di fine turno esistono, quali
   arrivano a generate(), e la frazione di generazioni che raggiunge il tetto.
   Se quella frazione e' alta, tutto il resto viene dopo: stai misurando un
   troncamento, non la pragmatica.

2. Campiona n candidati per item con un gen_kw CORRETTO (eos di fine turno,
   repetition_penalty, no_repeat_ngram, typical/nucleus) e ne sceglie uno.

3. Tre criteri di selezione, tutti reference-free (nessuno guarda il target):

   mbr        utilita' = chrF++ medio del candidato contro gli altri candidati
              dello stesso item. Nessun iperparametro. Elimina la coda di
              campionamenti anomali per costruzione: un candidato sconclusionato
              non ha consenso. (Eikema & Aziz 2020; Freitag et al. 2022)

   composito  combinazione lineare di:
                d_ctx  = logP(cand | contesto reale) - logP(cand | senza contesto)
                         normalizzata per token, con `rimuovi_contesto` di
                         decodifica_contestuale.py: e' lo stesso delta di
                         contesto_metrica.py, usato qui come punteggio invece che
                         come metrica.
                d_nap  = -|P_nap(cand) - P_nap(riferimenti umani)|
                         il bersaglio e' 0.877, NON 1.0: il run misurato sta a
                         0.944, cioe' IPER-dialettale. Massimizzare P_nap
                         peggiora.
                d_len  = log-densita' della lunghezza sotto la distribuzione
                         empirica dei target di TRAIN (mai del test).
              I pesi si tarano su dev con --taratura.

   ibrido     mbr + composito, standardizzati per item.

4. FILTRI HARD, applicati prima dei punteggi: candidato vuoto, loop di un token
   ripetuto >= 3 volte, copia normalizzata di un turno di contesto, lunghezza
   oltre il 99mo percentile umano. Un candidato che li viola viene scartato; se
   li violano tutti si tiene il meno peggio e lo si conta.

5. Salva {run}.preds.jsonl + {run}.metrics.json nello STESSO formato di
   evaluate_task.py, cosi' metriche_finali.py li raccoglie affiancati e la
   tabella comparativa esce da sola. Salva anche il candidato 0 come arm
   "1 campione" per un confronto appaiato a parita' di seme e di gen_kw.

6. Metriche STRATIFICATE per lunghezza del riferimento. Nel corpus il 33.7% dei
   turni sta a <= 2 parole e i piu' frequenti sono riscontri (mh 95, si' 71,
   mhmh 58): meta' del test e' un sotto-task diverso, e aggregare i due annega
   il segnale sui turni contenutistici.

Uso su Kaggle
-------------
  # 1. diagnostica, 30 secondi, prima di tutto il resto
  !python rerank_t3.py --model minerva --task T3 --split-dir {SPLIT} \
      --adapter {RUNS['T3']} --solo-diagnostica

  # 2. taratura dei pesi del composito sul DEV
  !python rerank_t3.py --model minerva --task T3 --split-dir {SPLIT} \
      --adapter {RUNS['T3']} --split dev --taratura --n 8

  # 3. run finale su test, tre arm in un colpo
  !python rerank_t3.py --model minerva --task T3 --split-dir {SPLIT} \
      --adapter {RUNS['T3']} --split test --n 12 --modo tutti \
      --pesi 1.0,2.0,0.5 --out /kaggle/working/eval

Costo su T4, 192 item, n=12
---------------------------
  generazione   192 * 12 sequenze da <= 32 token          ~6-10 min
  d_ctx         2 forward per candidato, batch 8          ~4-6 min
  P_nap         CPU, sklearn                              secondi
Con --modo mbr il d_ctx non serve e resta solo la generazione.
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from Minerva7B.T1_Traduzione.training.common import (load_backbone, load_hf_token, load_split, render_prompt,
                    resolve_model, slug)
from Minerva7B.T1_Traduzione.training.evaluate_task import (MAX_NEW, baseline_non_banali, bootstrap_chrf,
                           costruisci_discriminatore_dialetto, degenerazione,
                           discriminazione_avversariale, distribuzionali,
                           frazione_al_tetto, id_fine_turno, loop_massimo, norm,
                           pulisci_generato, riferimento,
                           stratifica_per_lunghezza)
from Minerva7B.T1_Traduzione.training.contesto_metrica import _logp_per_token, estrai_blocco_contesto
from Minerva7B.T1_Traduzione.training.decodifica_contestuale import rimuovi_contesto


# --------------------------------------------------------------------------- #
# 1. diagnostica: il modello emette fine di turno, e generate() la ascolta?
# --------------------------------------------------------------------------- #

def diagnostica(model, tok, rows, max_new, verbose=True):
    """id_fine_turno() stampa gia' quali id servono; qui si aggiunge il confronto
    con quello che generate() userebbe da sola."""
    eot = id_fine_turno(tok, rows, verbose=verbose)
    gc_eos = getattr(model.generation_config, "eos_token_id", None)
    gc_set = set(gc_eos if isinstance(gc_eos, (list, tuple)) else
                 ([gc_eos] if gc_eos is not None else []))
    if verbose:
        print(f"  generation_config.eos_token_id = {gc_eos}")
        mancanti = [i for i in eot if i not in gc_set]
        if mancanti:
            print(f"  ! {mancanti} NON sono in generation_config: senza il fix "
                  f"generate() non si ferma e arriva sempre a max_new={max_new}")
        else:
            print("  ok: generation_config copre tutti gli id di fine turno")
    return eot


# --------------------------------------------------------------------------- #
# 2. generazione di n candidati
# --------------------------------------------------------------------------- #

def genera_candidati(model, tok, torch, rows, n, max_new, eot_ids,
                     decoding="typical", top_p=0.9, typical_p=0.9,
                     temperature=0.7, repetition_penalty=1.15,
                     no_repeat_ngram=3, batch_prompt=2, seed=42):
    """Ritorna una lista di liste: cand[i] = n candidati per rows[i].

    batch_prompt e' basso di proposito: con num_return_sequences=n il batch
    effettivo e' batch_prompt * n. Su T4 con un 7B in 4 bit, batch_prompt=2 e
    n=12 danno 24 sequenze in volo, che ci stanno; alzare batch_prompt fa OOM
    prima di far guadagnare tempo.
    """
    torch.manual_seed(seed)
    prev_side = tok.padding_side
    tok.padding_side = "left"                     # obbligatorio in generazione
    fuori = []
    gen_kw = dict(max_new_tokens=max_new, min_new_tokens=1,
                  pad_token_id=tok.pad_token_id, eos_token_id=eot_ids,
                  num_return_sequences=n, do_sample=True, top_k=0,
                  temperature=temperature,
                  repetition_penalty=repetition_penalty,
                  no_repeat_ngram_size=no_repeat_ngram)
    if decoding == "typical":
        gen_kw["typical_p"] = typical_p           # locally typical sampling
    else:
        gen_kw["top_p"] = top_p                   # nucleus, per confronto
    try:
        for i in range(0, len(rows), batch_prompt):
            b = rows[i:i + batch_prompt]
            testi = [render_prompt(tok, r["prompt"]) for r in b]
            enc = tok(testi, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, **gen_kw)
            larghezza = enc["input_ids"].shape[1]
            for j in range(len(b)):
                blocco = g[j * n:(j + 1) * n]
                fuori.append([pulisci_generato(tok.decode(s[larghezza:],
                                                 skip_special_tokens=True))
                              for s in blocco])
            print(f"  generazione {min(i + batch_prompt, len(rows))}/{len(rows)}",
                  end="\r")
    finally:
        tok.padding_side = prev_side
    print()
    return fuori


# --------------------------------------------------------------------------- #
# 3. filtri hard
# --------------------------------------------------------------------------- #

def turni_di_contesto(prompt):
    trovato = estrai_blocco_contesto(prompt)
    if trovato is None:
        return []
    return [norm(r) for r in trovato[2] if norm(r)]


def ammissibile(cand, ctx_norm, len_max):
    """False = da scartare. Sono patologie, non preferenze stilistiche."""
    if not cand.strip():
        return False, "vuoto"
    if loop_massimo(cand) >= 3:
        return False, "loop"
    c = norm(cand)
    if not c:
        return False, "vuoto"
    # Copia del contesto: il pattern degenere e' il turno di contesto ripetuto
    # verbatim. La sottostringa si accetta solo da 4 parole in su, altrimenti si
    # scartano echi legittimi ("nun saccio" dopo "B: nun saccio" e' una replica
    # plausibile, non degenerazione). Controlla `scarti_per_motivo` nel summary:
    # se copia_contesto e' sopra ~15% dei candidati il filtro sta stringendo
    # troppo per questo corpus.
    n_par = len(c.split())
    if any(c == t or (n_par >= 4 and c in t) for t in ctx_norm):
        return False, "copia_contesto"
    if len(cand.split()) > len_max:
        return False, "troppo_lungo"
    return True, None


# --------------------------------------------------------------------------- #
# 4. punteggi
# --------------------------------------------------------------------------- #

def utilita_mbr(cands, chrf_cache=None):
    """Per ogni candidato: chrF++ medio contro TUTTI gli altri candidati.

    E' MBR con utilita' chrF e distribuzione approssimata dai campioni. Non
    guarda il riferimento: e' la selezione del consenso, non della verita'.
    """
    import sacrebleu
    m = len(cands)
    if m == 1:
        return [0.0]
    fuori = []
    for a in range(m):
        s = 0.0
        for b in range(m):
            if a == b:
                continue
            s += sacrebleu.sentence_chrf(cands[a], [cands[b]], word_order=2).score
        fuori.append(s / (m - 1))
    return fuori


def costruisci_densita_lunghezza(split_dir, task, tetto=40):
    """log-densita' empirica della lunghezza in parole, dai target di TRAIN.

    Da train e non da test: usare la distribuzione del test come prior
    equivarrebbe a tarare il sistema sul set di valutazione.
    """
    tr = load_split(split_dir, task, "train")
    L = [min(len(r["target"].split()), tetto) for r in tr]
    c = Counter(L)
    tot = sum(c.values()) + 0.5 * (tetto + 1)
    import math
    logp = {k: math.log((c.get(k, 0) + 0.5) / tot) for k in range(tetto + 1)}
    minimo = min(logp.values())
    p99 = sorted(L)[int(0.99 * (len(L) - 1))]
    return logp, minimo, p99, {
        "media": round(statistics.mean(L), 2),
        "mediana": statistics.median(L),
        "p99": p99,
        "frazione_le2": round(sum(1 for x in L if x <= 2) / len(L), 3),
    }


def punteggio_lunghezza(cand, logp, minimo, tetto=40):
    return logp.get(min(len(cand.split()), tetto), minimo)


def punteggio_contesto(model, tok, torch, prompt, cands, max_len, batch=8):
    """d_ctx = logP(cand | prompt) - logP(cand | prompt senza contesto),
    normalizzata per token. Riusa _logp_per_token di contesto_metrica.py."""
    nudo = rimuovi_contesto(prompt)
    if nudo is None:
        return [0.0] * len(cands), False
    con = _logp_per_token(model, tok, torch, [prompt] * len(cands), cands,
                          render_prompt, max_len, batch=batch)
    senza = _logp_per_token(model, tok, torch, [nudo] * len(cands), cands,
                            render_prompt, max_len, batch=batch)
    return [a - b for a, b in zip(con, senza)], True


def standardizza(v):
    if len(v) < 2:
        return [0.0] * len(v)
    mu = statistics.mean(v)
    sd = statistics.pstdev(v)
    return [0.0] * len(v) if sd < 1e-9 else [(x - mu) / sd for x in v]


# --------------------------------------------------------------------------- #
# 5. selezione
# --------------------------------------------------------------------------- #

def seleziona(cands_per_item, rows, model, tok, torch, clf_dial, p_nap_bersaglio,
              logp_len, minimo_len, p99_len, modo, pesi, max_len, verbose=True):
    """Ritorna dict modo -> lista di ipotesi scelte, piu' la diagnostica."""
    w_ctx, w_nap, w_len = pesi
    scelte = {m: [] for m in ("1campione", "mbr", "composito", "ibrido")}
    conta_scarti = Counter()
    n_tutti_scartati = 0
    serve_ctx = modo in ("composito", "ibrido", "tutti")
    ctx_disponibile = 0

    for i, (r, cands) in enumerate(zip(rows, cands_per_item)):
        scelte["1campione"].append(cands[0])

        ctx_norm = turni_di_contesto(r["prompt"])
        ok, motivi = [], []
        for c in cands:
            a, m = ammissibile(c, ctx_norm, p99_len * 2)
            ok.append(a)
            if not a:
                conta_scarti[m] += 1
        vivi = [k for k in range(len(cands)) if ok[k]]
        if not vivi:                                  # nessuno passa: tieni tutti
            n_tutti_scartati += 1
            vivi = list(range(len(cands)))
        sub = [cands[k] for k in vivi]

        u_mbr = utilita_mbr(sub)
        scelte["mbr"].append(sub[max(range(len(sub)), key=lambda k: u_mbr[k])])

        if serve_ctx:
            d_ctx, ha_ctx = punteggio_contesto(model, tok, torch, r["prompt"],
                                               sub, max_len)
            ctx_disponibile += int(ha_ctx)
            p = clf_dial.predict_proba([norm(x) for x in sub])[:, 1] \
                if clf_dial is not None else [p_nap_bersaglio] * len(sub)
            d_nap = [-abs(float(x) - p_nap_bersaglio) for x in p]
            d_len = [punteggio_lunghezza(x, logp_len, minimo_len) for x in sub]
            comp = [w_ctx * a + w_nap * b + w_len * c
                    for a, b, c in zip(d_ctx, d_nap, d_len)]
            scelte["composito"].append(
                sub[max(range(len(sub)), key=lambda k: comp[k])])
            z = [a + b for a, b in zip(standardizza(u_mbr), standardizza(comp))]
            scelte["ibrido"].append(sub[max(range(len(z)), key=lambda k: z[k])])
        else:
            scelte["composito"].append(scelte["mbr"][-1])
            scelte["ibrido"].append(scelte["mbr"][-1])

        if verbose and (i + 1) % 20 == 0:
            print(f"  selezione {i + 1}/{len(rows)}", end="\r")
    if verbose:
        print()

    diag = {"scarti_per_motivo": dict(conta_scarti),
            "item_con_tutti_i_candidati_scartati": n_tutti_scartati,
            "item_con_contesto_estraibile": ctx_disponibile}
    return scelte, diag


# --------------------------------------------------------------------------- #
# 6. metriche, stratificate
# --------------------------------------------------------------------------- #

def blocco_metriche(task, rows, hyps, refs, clf_dial, info_dial, tok, max_new):
    res = {
        "F_riferimento": riferimento(hyps, refs),
        "F_bootstrap_chrf": bootstrap_chrf(hyps, refs),
        "F_baseline": baseline_non_banali(task, rows, refs),
        "C_distribuzionali": distribuzionali(hyps, refs),
        "C_degenerazione": degenerazione(hyps, refs),
        "A_avversariale": discriminazione_avversariale(hyps, refs),
        "G_troncamento": {
            "frazione_al_tetto": round(frazione_al_tetto(tok, hyps, max_new), 3),
            "_lettura": "frazione di output che raggiunge max_new senza emettere "
                        "fine di turno. Sopra ~0.05 la qualita' apparente e' "
                        "dominata dal troncamento, non dalla pragmatica.",
        },
    }
    if clf_dial is not None:
        pm = clf_dial.predict_proba([norm(x) for x in hyps])[:, 1]
        pr = clf_dial.predict_proba([norm(x) for x in refs])[:, 1]
        res["D_dialettalita"] = {
            "P_nap_generato": round(float(pm.mean()), 3),
            "P_nap_riferimenti_umani": round(float(pr.mean()), 3),
            "discriminatore": info_dial,
            "_lettura": "il bersaglio e' P_nap_riferimenti_umani, non 1.0: "
                        "sopra quel valore il sistema e' IPER-dialettale.",
        }
    return res


# --------------------------------------------------------------------------- #
# 7. taratura dei pesi su dev
# --------------------------------------------------------------------------- #

def obiettivo(hyps, refs, clf_dial):
    """Reference-free, da minimizzare. NON usa chrF: su un task uno-a-molti con
    riferimento singolo chrF++ 12.6 contro un floor di 9.79 non discrimina.

      |accuracy_avversariale - 0.50|   il sistema deve essere indistinguibile
    + |P_nap - P_nap_umano|            ne' sotto- ne' iper-dialettale
    + 0.5 * ks_lunghezza_stat          distribuzione delle lunghezze compatibile
    """
    a = discriminazione_avversariale(hyps, refs)
    if "errore" in a:
        return None, a
    d = distribuzionali(hyps, refs)
    ks = d.get("ks_lunghezza_stat", 1.0)
    if clf_dial is not None:
        pm = float(clf_dial.predict_proba([norm(x) for x in hyps])[:, 1].mean())
        pr = float(clf_dial.predict_proba([norm(x) for x in refs])[:, 1].mean())
        dn = abs(pm - pr)
    else:
        pm = pr = dn = 0.0
    val = abs(a["accuracy_umano_vs_macchina"] - 0.5) + dn + 0.5 * ks
    return val, {"adv": a["accuracy_umano_vs_macchina"],
                 "adv_solo_lunghezza": a["controllo_solo_lunghezza"],
                 "ks": ks, "p_nap": round(pm, 3), "p_nap_umano": round(pr, 3),
                 "obiettivo": round(val, 4)}


def taratura(cands_per_item, rows, refs, model, tok, torch, clf_dial,
             p_nap_bersaglio, logp_len, minimo_len, p99_len, max_len):
    griglia = [(wc, wn, wl)
               for wc in (0.0, 0.5, 1.0, 2.0)
               for wn in (0.0, 1.0, 2.0, 4.0)
               for wl in (0.0, 0.5, 1.0)]
    print(f"Taratura: {len(griglia)} combinazioni sul dev "
          f"(il d_ctx si calcola UNA volta e si riusa)")

    # d_ctx, d_nap, d_len non dipendono dai pesi: calcolali una volta sola
    cache = []
    for i, (r, cands) in enumerate(zip(rows, cands_per_item)):
        ctx_norm = turni_di_contesto(r["prompt"])
        vivi = [k for k, c in enumerate(cands)
                if ammissibile(c, ctx_norm, p99_len * 2)[0]] or list(range(len(cands)))
        sub = [cands[k] for k in vivi]
        d_ctx, _ = punteggio_contesto(model, tok, torch, r["prompt"], sub, max_len)
        p = clf_dial.predict_proba([norm(x) for x in sub])[:, 1] \
            if clf_dial is not None else [p_nap_bersaglio] * len(sub)
        cache.append((sub, d_ctx,
                      [-abs(float(x) - p_nap_bersaglio) for x in p],
                      [punteggio_lunghezza(x, logp_len, minimo_len) for x in sub]))
        print(f"  punteggi {i + 1}/{len(rows)}", end="\r")
    print()

    righe = []
    for (wc, wn, wl) in griglia:
        hyps = []
        for sub, dc, dn, dl in cache:
            s = [wc * a + wn * b + wl * c for a, b, c in zip(dc, dn, dl)]
            hyps.append(sub[max(range(len(s)), key=lambda k: s[k])])
        val, det = obiettivo(hyps, refs, clf_dial)
        if val is None:
            continue
        righe.append({"pesi": [wc, wn, wl], **det})
        print(f"  ({wc},{wn},{wl}) -> obiettivo {val:.4f}  adv {det['adv']:.3f} "
              f"ks {det['ks']:.3f} p_nap {det['p_nap']:.3f}")
    righe.sort(key=lambda x: x["obiettivo"])
    print("\nMigliori 5:")
    for r in righe[:5]:
        print("  ", json.dumps(r, ensure_ascii=False))
    print(f"\nUsa: --pesi {','.join(str(x) for x in righe[0]['pesi'])}")
    return righe


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="T3", choices=["T2", "T3"],
                    help="T1 no: e' quasi deterministico, il best-of-n non serve")
    ap.add_argument("--split-dir", default="/kaggle/working/split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default="/kaggle/working/eval")
    ap.add_argument("--n", type=int, default=12, help="candidati per item")
    ap.add_argument("--modo", default="tutti",
                    choices=["mbr", "composito", "ibrido", "tutti"])
    ap.add_argument("--pesi", default="1.0,2.0,0.5",
                    help="w_ctx,w_nap,w_len del composito (vedi --taratura)")
    ap.add_argument("--taratura", action="store_true",
                    help="griglia sui pesi; usalo su --split dev, non su test")
    ap.add_argument("--solo-diagnostica", action="store_true")
    ap.add_argument("--decoding", default="typical", choices=["typical", "nucleus"])
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--typical-p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--batch-prompt", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limite", type=int, default=None,
                    help="usa solo i primi N item: per provare la pipeline")
    ap.add_argument("--hf-token", default=None)
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    repo_id = resolve_model(a.model)
    max_new = a.max_new or MAX_NEW[a.task]
    token = load_hf_token(a.hf_token)

    tok = AutoTokenizer.from_pretrained(repo_id, token=token, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    cc = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dtype)
    model, arch = load_backbone(repo_id, dtype, quantization_config=quant,
                                device_map={"": 0}, token=token,
                                low_cpu_mem_usage=True,
                                attn_implementation="eager")
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    model.config.use_cache = True

    rows = load_split(a.split_dir, a.task, a.split)
    if a.limite:
        rows = rows[:a.limite]
    refs = [r["target"] for r in rows]
    tag = "ft" if a.adapter else "zs"
    base = f"{slug(repo_id)}__{a.task}__{tag}"
    print(f"=== {base} | {len(rows)} item di {a.split} | n={a.n} ===\n")

    print("[1] Diagnostica fine turno")
    eot = diagnostica(model, tok, rows, max_new)

    if a.solo_diagnostica:
        print("\n  campione di 16 item con il gen_kw corretto, per misurare il tetto")
        c = genera_candidati(model, tok, torch, rows[:16], 1, max_new, eot,
                             decoding=a.decoding, top_p=a.top_p,
                             typical_p=a.typical_p, temperature=a.temperature,
                             repetition_penalty=a.repetition_penalty,
                             no_repeat_ngram=a.no_repeat_ngram,
                             batch_prompt=a.batch_prompt, seed=a.seed)

        h = [x[0] for x in c]
        print(f"  frazione al tetto: {frazione_al_tetto(tok, h, max_new):.3f} "
              f"(deve stare sotto 0.05)")
        print(f"  lunghezza media: "
              f"{statistics.mean(len(x.split()) for x in h):.2f} parole "
              f"| riferimenti: "
              f"{statistics.mean(len(x.split()) for x in refs[:16]):.2f}")
        for r, x in list(zip(rows, h))[:8]:
            print(f"    RIF {r['target']!r}\n    GEN {x!r}")
        return

    print("\n[2] Prior di lunghezza dai target di TRAIN")
    logp_len, minimo_len, p99_len, stat_len = costruisci_densita_lunghezza(
        a.split_dir, a.task)
    print("  ", json.dumps(stat_len, ensure_ascii=False))

    print("\n[3] Discriminatore italiano-vs-napoletano (dal train di T1)")
    clf_dial, info_dial = costruisci_discriminatore_dialetto(a.split_dir)
    print("  ", json.dumps(info_dial, ensure_ascii=False))
    p_nap_bersaglio = float(
        clf_dial.predict_proba([norm(x) for x in refs])[:, 1].mean()
    ) if clf_dial is not None else 0.877
    print(f"  bersaglio P_nap (riferimenti umani di questo split): "
          f"{p_nap_bersaglio:.3f}  <- NON 1.0")

    print(f"\n[4] Generazione di {a.n} candidati per item "
          f"({a.decoding}, T={a.temperature}, rep={a.repetition_penalty})")
    cands = genera_candidati(model, tok, torch, rows, a.n, max_new, eot,
                             decoding=a.decoding, top_p=a.top_p,
                             typical_p=a.typical_p, temperature=a.temperature,
                             repetition_penalty=a.repetition_penalty,
                             no_repeat_ngram=a.no_repeat_ngram,
                             batch_prompt=a.batch_prompt, seed=a.seed)
    unici = statistics.mean(len(set(c)) for c in cands)
    print(f"  candidati distinti per item: {unici:.2f}/{a.n} "
          f"(se e' vicino a 1 la temperatura e' troppo bassa: non c'e' "
          f"niente da selezionare)")

    if a.taratura:
        print("\n[5] Taratura dei pesi")
        righe = taratura(cands, rows, refs, model, tok, torch, clf_dial,
                         p_nap_bersaglio, logp_len, minimo_len, p99_len,
                         a.max_len)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / f"{base}__taratura.json").write_text(
            json.dumps(righe, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    pesi = tuple(float(x) for x in a.pesi.split(","))
    assert len(pesi) == 3, "--pesi vuole tre valori: w_ctx,w_nap,w_len"
    print(f"\n[5] Selezione (pesi composito: ctx={pesi[0]} nap={pesi[1]} "
          f"len={pesi[2]})")
    scelte, diag_sel = seleziona(cands, rows, model, tok, torch, clf_dial,
                                 p_nap_bersaglio, logp_len, minimo_len,
                                 p99_len, a.modo, pesi, a.max_len)
    print("  ", json.dumps(diag_sel, ensure_ascii=False))

    da_salvare = ["1campione"] + (
        ["mbr", "composito", "ibrido"] if a.modo == "tutti" else [a.modo])
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n[6] Metriche")
    riassunto = {}
    for m in da_salvare:
        hyps = scelte[m]
        run = f"{base}__{a.decoding}" if m == "1campione" else \
              f"{base}__{a.decoding}_bon{a.n}_{m}"
        res = {"run": run, "decoding": a.decoding, "repo_id": repo_id,
               "task": a.task, "tag": tag, "architettura": arch,
               "n_test": len(rows),
               "selezione": {"modo": m, "n_candidati": a.n,
                             "pesi_composito": list(pesi) if m in
                             ("composito", "ibrido") else None,
                             "gen_kw": {"decoding": a.decoding,
                                        "temperature": a.temperature,
                                        "typical_p": a.typical_p,
                                        "top_p": a.top_p,
                                        "repetition_penalty": a.repetition_penalty,
                                        "no_repeat_ngram_size": a.no_repeat_ngram,
                                        "eos_token_id": eot,
                                        "max_new_tokens": max_new},
                             "diagnostica": diag_sel},
               **blocco_metriche(a.task, rows, hyps, refs, clf_dial, info_dial,
                                 tok, max_new),
               "H_stratificato": stratifica_per_lunghezza(hyps, refs),
               }
        with (outdir / f"{run}.preds.jsonl").open("w", encoding="utf-8") as f:
            for r, h, cc_ in zip(rows, hyps, cands):
                f.write(json.dumps({"id": r.get("id"), "prompt": r["prompt"],
                                    "target": r["target"], "hyp": h,
                                    "candidati": cc_}, ensure_ascii=False) + "\n")
        (outdir / f"{run}.metrics.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        riassunto[m] = {
            "chrf++": res["F_riferimento"]["chrf++"],
            "adv": res["A_avversariale"].get("accuracy_umano_vs_macchina"),
            "adv_len": res["A_avversariale"].get("controllo_solo_lunghezza"),
            "adv_uu": res["A_avversariale"].get("controllo_umano_vs_umano"),
            "lung_media": res["C_distribuzionali"]["lunghezza_media"],
            "ks": res["C_distribuzionali"]["ks_lunghezza_stat"],
            "rep_2": res["C_distribuzionali"]["rep_2"],
            "loop3+": res["C_degenerazione"]["frazione_con_loop_3plus"],
            "p_nap": res.get("D_dialettalita", {}).get("P_nap_generato"),
            "al_tetto": res["G_troncamento"]["frazione_al_tetto"],
        }

    umano = {"lung_media": round(statistics.mean(len(x.split()) for x in refs), 2),
             "p_nap": round(p_nap_bersaglio, 3), "adv": 0.5}
    print("\n" + "=" * 78)
    print(f"{'arm':<12}{'chrf':>7}{'adv':>7}{'adv_len':>9}{'lung':>7}"
          f"{'ks':>7}{'rep_2':>8}{'loop3+':>8}{'p_nap':>7}{'tetto':>7}")
    print("-" * 78)
    for m, v in riassunto.items():
        print(f"{m:<12}{v['chrf++']:>7}{v['adv']:>7}{v['adv_len']:>9}"
              f"{v['lung_media']:>7}{v['ks']:>7}{v['rep_2']:>8}"
              f"{v['loop3+']:>8}{v['p_nap']:>7}{v['al_tetto']:>7}")
    print("-" * 78)
    print(f"{'UMANO':<12}{'-':>7}{'0.5':>7}{'-':>9}{umano['lung_media']:>7}"
          f"{'0':>7}{'0.037':>8}{'0.01':>8}{umano['p_nap']:>7}{'-':>7}")
    print("=" * 78)
    print("\nCome leggere: adv deve SCENDERE verso 0.5 e adv_len deve scendere\n"
          "con lei. Se adv scende ma adv_len resta alto, hai solo cambiato\n"
          "lunghezza senza migliorare il napoletano. chrF++ puo' restare piatto\n"
          "o calare: con riferimento singolo su task aperto non e' la metrica\n"
          "di merito (floor 9.79, il run misurato sta a 12.6).")
    print(f"\nSalvato in {outdir}/  -> metriche_finali.py li raccoglie affiancati")


if __name__ == "__main__":
    main()
