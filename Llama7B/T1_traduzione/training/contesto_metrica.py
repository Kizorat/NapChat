#!/usr/bin/env python3
"""
contesto_metrica.py — misura se il modello USA il contesto conversazionale,
durante il training e come metrica di selezione del checkpoint.

Premessa onesta
---------------
Una metrica non fa fare niente al modello: non c'e' nessuna metrica che "lo
costringa" a stare sul senso. Cio' che il modello fa lo decidono l'obiettivo e i
dati. Quello che una metrica decide e' **quale checkpoint tieni**, e su questo
task e' una leva vera: selezionare su chrF premia il checkpoint che indovina le
parole del riferimento, e su T2/T3, dove il riferimento e' uno fra molti validi,
quel checkpoint e' spesso quello generico o quello che ricopia il prefisso.
Selezionare su una metrica di coerenza tiene un checkpoint diverso.

La misura: delta di verosimiglianza contesto vero vs contesto sbagliato
----------------------------------------------------------------------
Per ogni item di dev si calcola la log-verosimiglianza del target per token, due
volte:

    L_vero  = log P(target | contesto REALE)
    L_falso = log P(target | contesto di un ALTRO punto della conversazione)

    delta = L_vero - L_falso        (per token, quindi non premia i target corti)
    acc   = quota di item con L_vero > L_falso

Se il modello ignora il contesto, delta ~ 0 e acc ~ 0.50: sta producendo un
napoletano plausibile in astratto, slegato da cio' che e' stato detto prima. E'
esattamente il "spara termini a caso" nella sua forma misurabile. Se delta > 0 e
acc sale, il contesto sta entrando nella predizione.

E' l'immagine speculare della contrastiva che hai gia' in evaluate_task.py: la'
i distrattori stanno sul lato del TARGET (quale replica e' quella vera), qui
stanno sul lato del CONTESTO (quale conversazione ha prodotto questa replica).
Le due cose non sono ridondanti: un modello puo' riconoscere la replica giusta
perche' e' l'unica in napoletano fluente, senza usare il contesto.

Costo: due forward per item, nessuna generazione. Su 64 item e' l'ordine di
grandezza di una eval in teacher forcing, quindi si puo' tenere accesa a ogni
valutazione.

Cosa NON risolve
----------------
Con ~450-750 istanze di training la coerenza pragmatica non si insegna: viene dal
modello di base. Questa metrica serve a **non distruggerla** — a scegliere il
checkpoint e a fermarsi al punto giusto — non a crearla.

Integrazione in common.py
-------------------------
Nella lista dei callback, PRIMA di EarlyStoppingCallback (che legge
metrics[metric_for_best_model] e se non lo trova si disattiva):

    if args.ctx_metric_n > 0:
        from contesto_metrica import build_context_callback
        callbacks.append(build_context_callback(
            TrainerCallback, torch, tokenizer, dev_rows, task,
            n=args.ctx_metric_n, seed=args.seed))
    callbacks.append(EarlyStoppingCallback(...))

piu' l'argomento:

    ap.add_argument("--ctx-metric-n", type=int, default=64,
                    help="item di dev su cui misurare la sensibilita' al contesto "
                         "(0 disattiva). Aggiunge eval_ctx_delta e eval_ctx_acc")

Da quel momento `--metric ctx_acc` seleziona il checkpoint sulla coerenza.

Uso consigliato
---------------
    # T3: seleziona sulla coerenza invece che su chrF
    python finetune_t3_replica.py --model minerva --split-dir SPLIT --metric ctx_acc

    # T1: lascia chrF come metrica di selezione (la traduzione HA un modo) ma
    # leggi ctx_acc nei log: se resta a 0.50, il blocco di contesto e' decorativo
    python finetune_t1_traduzione.py --model minerva --split-dir SPLIT
"""

from __future__ import annotations

import random

INTESTAZIONI_CONTESTO = ("Conversazione finora", "Conversazione", "Contesto")


def _e_separatore(riga):
    """Fine del blocco di contesto.

    I layout non usano tutti la stessa convenzione: T1 chiude il contesto con una
    riga vuota, T2 e T3 con una riga di trattini. Cercare solo la riga vuota fa
    inghiottire l'intero prompt, istruzione compresa, e la misura diventa priva
    di senso senza dare errore. Si accetta quindi entrambe le forme.
    """
    t = riga.strip()
    return (not t) or (len(t) >= 3 and set(t) <= set("-=*_"))


def estrai_blocco_contesto(prompt):
    """Ritorna (indice_intestazione, indice_fine, righe_del_blocco).

    Il blocco e' delimitato dall'intestazione e dalla prima riga vuota. Se non
    c'e' intestazione l'item non ha contesto e va escluso dalla misura: senza
    contesto il delta non e' definito, e includerlo come 0 diluirebbe la metrica
    verso 0.50 facendola sembrare cieca quando invece manca l'input.
    """
    righe = prompt.split("\n")
    i = next((k for k, r in enumerate(righe)
              if any(h in r for h in INTESTAZIONI_CONTESTO)), None)
    if i is None:
        return None
    j = next((k for k in range(i + 1, len(righe)) if _e_separatore(righe[k])),
             len(righe))
    if j <= i + 1:
        return None                              # intestazione senza turni
    return i, j, righe[i + 1:j]


def sostituisci_contesto(prompt, nuovi_turni):
    """Rimpiazza i turni di contesto tenendo intestazione, istruzione e suffissi."""
    trovato = estrai_blocco_contesto(prompt)
    if trovato is None:
        return None
    i, j, _ = trovato
    righe = prompt.split("\n")
    return "\n".join(righe[:i + 1] + list(nuovi_turni) + righe[j:])


def prepara_coppie(rows, seed=0):
    """Per ogni item con contesto: (prompt_vero, prompt_falso, target).

    Il contesto falso viene da un ALTRO item, non da turni sintetici: cosi' il
    confronto e' fra due contesti entrambi plausibili e ben formati, e il delta
    misura la pertinenza, non la stranezza del testo.
    """
    con_ctx = []
    for r in rows:
        t = estrai_blocco_contesto(r["prompt"])
        if t is not None:
            con_ctx.append((r, t[2]))
    if len(con_ctx) < 2:
        return []
    rnd = random.Random(seed)
    coppie = []
    for k, (r, _) in enumerate(con_ctx):
        # si pesca un donatore diverso da se stesso; con 2+ item il ciclo termina
        while True:
            d = rnd.randrange(len(con_ctx))
            if d != k:
                break
        falso = sostituisci_contesto(r["prompt"], con_ctx[d][1])
        if falso and falso != r["prompt"]:
            coppie.append((r["prompt"], falso, r["target"]))
    return coppie


def _logp_per_token(model, tokenizer, torch, prompt_testi, target_testi,
                    render, max_len, batch=8):
    """log P(target | prompt) diviso per il numero di token di target."""
    fuori = []
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"             # scoring, non generazione
    try:
        for i in range(0, len(prompt_testi), batch):
            pp = prompt_testi[i:i + batch]
            tt = target_testi[i:i + batch]
            interi = [render(tokenizer, p, t) for p, t in zip(pp, tt)]
            n_prompt = [len(tokenizer(render(tokenizer, p), add_special_tokens=False)
                             ["input_ids"]) for p in pp]
            enc = tokenizer(interi, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_len, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                logits = model(**enc).logits.float()
            lp = torch.log_softmax(logits[:, :-1, :], dim=-1)
            ids = enc["input_ids"][:, 1:]
            att = enc["attention_mask"][:, 1:]
            scelti = lp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            for b in range(ids.size(0)):
                maschera = att[b].clone().bool()
                maschera[:max(0, n_prompt[b] - 1)] = False   # solo i token di target
                n = int(maschera.sum())
                fuori.append(float(scelti[b][maschera].sum() / n) if n else 0.0)
    finally:
        tokenizer.padding_side = prev_side
    return fuori


def valuta_sensibilita(model, tokenizer, torch, rows, render, max_len=512,
                       n=64, batch=8, seed=0):
    """Ritorna {'ctx_delta', 'ctx_acc', 'n'} oppure None se manca il contesto."""
    coppie = prepara_coppie(rows, seed=seed)[:n]
    if not coppie:
        return None
    veri = [c[0] for c in coppie]
    falsi = [c[1] for c in coppie]
    targ = [c[2] for c in coppie]
    lv = _logp_per_token(model, tokenizer, torch, veri, targ, render, max_len, batch)
    lf = _logp_per_token(model, tokenizer, torch, falsi, targ, render, max_len, batch)
    delta = [a - b for a, b in zip(lv, lf)]
    return {"ctx_delta": sum(delta) / len(delta),
            "ctx_acc": sum(1 for d in delta if d > 0) / len(delta),
            "logp_vero": sum(lv) / len(lv),
            "logp_falso": sum(lf) / len(lf),
            "n": len(coppie)}


def build_context_callback(TrainerCallback, torch, tokenizer, rows, task,
                           n=64, batch=8, seed=0, render=None):
    """Aggiunge eval_ctx_delta e eval_ctx_acc al dict delle metriche.

    Mutare `metrics` dentro on_evaluate propaga alla selezione del checkpoint,
    quindi --metric ctx_acc funziona. Va inserito PRIMA di EarlyStoppingCallback.
    """
    if render is None:
        from common import render_prompt
        render = render_prompt

    max_len = getattr(task, "max_seq_len", 512)
    coppie_disponibili = len(prepara_coppie(rows, seed=seed))
    if coppie_disponibili == 0:
        print("! sensibilita' al contesto non misurabile: nessun item di dev ha un "
              "blocco di contesto. Metrica disattivata (non e' un errore per un "
              "layout senza contesto).")

    class ContextSensitivityCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
            if model is None or metrics is None or coppie_disponibili == 0:
                return
            era_training = model.training
            model.eval()
            try:
                res = valuta_sensibilita(model, tokenizer, torch, rows, render,
                                         max_len=max_len, n=n, batch=batch, seed=seed)
                if res is None:
                    return
                metrics["eval_ctx_delta"] = res["ctx_delta"]
                metrics["eval_ctx_acc"] = res["ctx_acc"]
                print(f"  [contesto su {res['n']} item] delta {res['ctx_delta']:+.4f} "
                      f"nat/token | acc {res['ctx_acc']:.3f} "
                      f"(0.50 = contesto ignorato)")
            finally:
                if era_training:
                    model.train()

    return ContextSensitivityCallback()
