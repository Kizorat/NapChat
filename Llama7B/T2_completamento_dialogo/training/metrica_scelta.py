#!/usr/bin/env python3
"""
metrica_scelta.py — accuratezza@N e MRR su continuazioni distrattrici.

Perche' serve. Su T2 il chrF++ ha una forbice di 2,8 punti fra il caso puro
(target mescolati: 9,1) e la baseline banale (ripeti il prefisso: 11,9). Dentro
quella forbice non si distingue un modello che ha imparato da uno che non ha
imparato: il riferimento e' UNA continuazione fra molte valide, e un
completamento perfettamente sensato ma diverso dal riferimento prende lo stesso
punteggio di uno sbagliato.

Questa metrica cambia domanda. Invece di "quanto assomiglia il testo generato al
riferimento", chiede: "fra la continuazione vera e N-1 continuazioni prese da
altri turni, il modello sceglie quella giusta?". Il fondo scala e' 1/N, esatto e
non discutibile, e la sensibilita' e' molto piu' alta perche' la scelta si gioca
sui token che dipendono dal contesto.

Costa N forward per item e zero generazioni: piu' veloce della generazione che
gia' si fa.

Uso dentro eval_t2.py, dopo il blocco di teacher forcing:

    import metrica_scelta as MS
    ris["scelta"] = MS.accuratezza_scelta(model, tok, dati, a.stile,
                                          a.max_seq_len, n_distrattori=9)
    print("scelta:", json.dumps(ris["scelta"], indent=2))
"""

from __future__ import annotations

import random

import t2_common as C


def _distrattori(records, i, n, rng, solo_stessa_conv=True):
    """Continuazioni prese da altri turni, di lunghezza in parole simile a
    quella vera. Se fossero di lunghezza qualsiasi, il modello potrebbe
    scegliere per lunghezza invece che per contenuto e il numero salirebbe
    senza che abbia capito niente."""
    r = records[i]
    n_par = len(r["target"].split())
    pool = [q for j, q in enumerate(records)
            if j != i
            and abs(len(q["target"].split()) - n_par) <= 1
            and (not solo_stessa_conv or q["conversazione"] == r["conversazione"])]
    if len(pool) < n:                      # ripiego: tutta la lista
        pool = [q for j, q in enumerate(records) if j != i]
    if len(pool) < n:
        return None
    return [q["target"] for q in rng.sample(pool, n)]


def accuratezza_scelta(model, tok, records, stile="prefill", max_seq_len=512,
                       n_distrattori=9, batch=4, seed=0,
                       normalizza_lunghezza=True) -> dict:
    """{'accuratezza@N', 'mrr', 'fondo_scala', 'n'}.

    normalizza_lunghezza=True usa la log-prob MEDIA per token invece della
    somma: senza, vincerebbe sistematicamente il candidato piu' corto.
    """
    rng = random.Random(seed)
    corrette = 0.0
    rr = 0.0
    usati = 0

    for i, r in enumerate(records):
        dis = _distrattori(records, i, n_distrattori, rng)
        if dis is None:
            continue
        candidati = [r["target"]] + dis
        finti = [dict(r, target=t) for t in candidati]
        p = C.punteggia(model, tok, finti, stile, max_seq_len, batch)
        if 0 not in p or len(p) < len(candidati):
            continue                       # qualche candidato e' stato scartato
        punteggi = []
        for k in range(len(candidati)):
            lp, n_tok, _ = p[k]
            punteggi.append(lp / max(1, n_tok) if normalizza_lunghezza else lp)
        vero = punteggi[0]
        # rango del vero fra tutti (1 = migliore). Gli ex aequo contano a meta'.
        migliori = sum(1 for s in punteggi[1:] if s > vero)
        pari = sum(1 for s in punteggi[1:] if s == vero)
        rango = migliori + pari / 2 + 1
        corrette += 1.0 if migliori == 0 and pari == 0 else 0.0
        rr += 1.0 / rango
        usati += 1

    N = n_distrattori + 1
    return {"accuratezza@N": round(corrette / max(1, usati), 4),
            "mrr": round(rr / max(1, usati), 4),
            "N": N,
            "fondo_scala": round(1.0 / N, 4),
            "n_item": usati}
