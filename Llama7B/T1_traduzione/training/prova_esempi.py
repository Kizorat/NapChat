#!/usr/bin/env python3
"""
prova_esempi.py — prova un adapter addestrato su frasi tue, con prompt
BYTE-IDENTICI a quelli del training.

Perche' non basta scrivere il prompt a mano
-------------------------------------------
I tre layout hanno formati precisi (blocco di contesto, istruzione, separatori) e
il fine-tuning ha legato il comportamento a quel formato. Un prompt ricostruito
"a occhio" produce output peggiori del modello reale, e non sai se stai vedendo
un limite del modello o un errore di formattazione. Qui il prompt viene preso da
una riga vera di test.json e si sostituisce SOLO la parte variabile dopo
l'istruzione, che e' l'unico punto in cui il tuo input entra.

Cosa aspettarsi per ciascun task
--------------------------------
T1  Traduci in napoletano: <frase italiana>
    Funziona sul formato. Ma il corpus e' parlato conversazionale con mediana 4
    parole: frasi lunghe e di dominio assente (meteo, oggetti mai citati) daranno
    resa parziale. Non e' un bug, e' copertura lessicale.

T2  Continua il turno in napoletano: <prima meta' del turno, IN NAPOLETANO>
    Intra-lingua: il prefisso deve essere GIA' in napoletano. Passare un prefisso
    italiano e' un task mai addestrato. Se non hai un prefisso napoletano, usa
    --da-test per vedere il task come e' stato addestrato.

T3  <3 turni di contesto in napoletano> + istruzione a rispondere
    Non e' instruction-following: non risponde a "parlami di qualcosa". Serve
    contesto conversazionale in napoletano. Con --contesto lo passi tu, altrimenti
    si usa quello di una riga di test.

Uso
---
    # traduzione di frasi tue
    python prova_esempi.py --model minerva --task T1 \\
        --adapter /kaggle/working/runs/Minerva-7B-instruct-v1.0__T1/adapter_final \\
        --frase "Oggi e' proprio una giornata nuvolosa, dovevo portare l'ombrello" \\
        --frase "Non lo so, forse domani"

    # completamento (prefisso napoletano)
    python prova_esempi.py --model minerva --task T2 --adapter ... \\
        --frase "Nun 'o saccio, pecche'"

    # replica con contesto tuo
    python prova_esempi.py --model minerva --task T3 --adapter ... \\
        --contesto "A: Comme staje?" --contesto "B: Nun c'e' male, e tu?"

    # il task come e' stato addestrato, su righe vere di test
    python prova_esempi.py --model minerva --task T3 --adapter ... --da-test 5

Diagnostica lessicale
---------------------
Con --lessico stampa, per ogni frase italiana di T1, quali parole hanno una resa
dialettale attestata nel corpus e quali no. Le seconde non possono essere
tradotte: il modello non le ha mai viste in napoletano. E' la risposta a "perche'
mi ha lasciato 'ombrello' in italiano".
"""

import argparse
import json
import re
import sys

from common import (MODEL_REGISTRY, load_backbone, load_hf_token, load_split,
                    render_prompt, resolve_model, slug)

ISTRUZIONE = {"T1": "Traduci in napoletano: ",
              "T2": "Continua il turno in napoletano: "}
MAX_NEW = {"T1": 64, "T2": 48, "T3": 64}


def splicing(prompt_modello, task, testo):
    """Sostituisce la parte variabile di un prompt reale, tenendo il resto.

    Per T1 e T2 l'istruzione delimita esattamente il punto di innesto. Se il
    marker non c'e', il layout e' cambiato: meglio fermarsi che tirare a indovinare.
    """
    marker = ISTRUZIONE[task]
    if marker not in prompt_modello:
        sys.exit(f"ERRORE: '{marker.strip()}' non compare nel prompt di test di "
                 f"{task}. Il layout e' cambiato: rigenera lo split o aggiorna "
                 f"ISTRUZIONE in questo script.")
    testa, _, coda = prompt_modello.rpartition(marker)
    # la coda oltre la frase (eventuale suffisso del layout) va conservata
    resto = coda.split("\n", 1)
    suffisso = "\n" + resto[1] if len(resto) > 1 else ""
    return testa + marker + testo + suffisso


def prompt_t3_con_contesto(prompt_modello, turni):
    """Riscrive il blocco di contesto di un prompt T3 con i turni forniti."""
    righe = prompt_modello.split("\n")
    # il blocco di contesto e' delimitato dall'intestazione e dalla riga vuota
    try:
        i = next(k for k, r in enumerate(righe) if "Conversazione" in r)
    except StopIteration:
        sys.exit("ERRORE: nessun blocco di contesto nel prompt T3 di test.")
    from contesto_metrica import _e_separatore
    j = next((k for k in range(i + 1, len(righe)) if _e_separatore(righe[k])),
             len(righe))
    return "\n".join(righe[:i + 1] + list(turni) + righe[j:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=f"alias ({'/'.join(MODEL_REGISTRY)}) o repo_id")
    ap.add_argument("--task", required=True, choices=["T1", "T2", "T3"])
    ap.add_argument("--adapter", default=None, help="senza adapter = modello di base")
    ap.add_argument("--split-dir", default="/kaggle/working/split")
    ap.add_argument("--frase", action="append", default=[],
                    help="ripetibile. T1: frase italiana. T2: prefisso NAPOLETANO")
    ap.add_argument("--contesto", action="append", default=[],
                    help="ripetibile, per T3: turni di contesto in napoletano, es. \"A: Comme staje?\"")
    ap.add_argument("--da-test", type=int, default=0,
                    help="quante righe vere di test provare, per vedere il task come e' addestrato")
    ap.add_argument("--lessico", default=None, help="output di lessico.py, per la diagnostica")
    ap.add_argument("--decoding", choices=["greedy", "nucleus"], default=None,
                    help="default: greedy su T1/T2, nucleus su T3 (come in evaluate_task.py)")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--hf-token", default=None)
    a = ap.parse_args()

    if not a.frase and not a.contesto and not a.da_test:
        sys.exit("Passa almeno --frase, --contesto o --da-test.")

    righe_test = load_split(a.split_dir, a.task, "test")
    if not righe_test:
        sys.exit(f"Nessuna riga di test per {a.task} in {a.split_dir}.")
    modello_prompt = righe_test[0]["prompt"]

    # --- costruzione dei prompt --------------------------------------------
    casi = []                                     # (etichetta, prompt, riferimento)
    for i, r in enumerate(righe_test[:a.da_test]):
        casi.append((f"test[{i}]", r["prompt"], r["target"]))
    if a.task == "T3":
        if a.contesto:
            casi.append(("contesto tuo",
                         prompt_t3_con_contesto(modello_prompt, a.contesto), None))
        for f in a.frase:
            print(f"! T3 ignora --frase ({f!r}): serve --contesto. T3 continua una "
                  f"conversazione, non risponde a un'istruzione.")
    else:
        for f in a.frase:
            casi.append((f"tua: {f[:40]}", splicing(modello_prompt, a.task, f), None))

    # --- diagnostica lessicale (prima di caricare il modello) ---------------
    if a.lessico and a.task == "T1" and a.frase:
        from pesi_lessicali import carica_lessico, tokenizza
        lex = carica_lessico(a.lessico)
        print("\n=== copertura lessicale delle tue frasi ===")
        print("Una parola senza resa attestata NON puo' essere tradotta: quella "
              "forma napoletana non compare nel corpus di train.\n")
        for f in a.frase:
            noti, ignoti = [], []
            for w in tokenizza(f):
                (noti if w in lex["it2nap"] else ignoti).append(w)
            print(f"  {f}")
            print(f"    resa attestata ({len(noti)}): {' '.join(noti) or '-'}")
            print(f"    nessuna resa  ({len(ignoti)}): {' '.join(ignoti) or '-'}")
            tot = len(noti) + len(ignoti)
            print(f"    copertura: {len(noti)/tot:.0%}\n" if tot else "")

    # --- modello ------------------------------------------------------------
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    repo_id = resolve_model(a.model)
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
                                low_cpu_mem_usage=True, attn_implementation="eager")
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
        print(f"\nAdapter: {a.adapter}")
    else:
        print("\n! nessun adapter: stai provando il modello di BASE (baseline zero-shot)")
    model.eval()
    model.config.use_cache = True

    dec = a.decoding or ("nucleus" if a.task == "T3" else "greedy")
    max_new = a.max_new or MAX_NEW[a.task]
    torch.manual_seed(a.seed)
    print(f"{slug(repo_id)} | {a.task} | arch={arch} | decoding={dec} | max_new={max_new}")

    for etichetta, prompt, riferimento in casi:
        testo = render_prompt(tok, prompt)
        enc = tok(testo, return_tensors="pt", add_special_tokens=False).to(0)
        kw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        if dec == "nucleus":
            kw.update(do_sample=True, top_p=a.top_p, temperature=a.temperature)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(**enc, **kw)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        gen = gen.strip().split("\n")[0].strip()      # come in evaluate_task.py

        print("\n" + "=" * 70)
        print(f"[{etichetta}]")
        print("-" * 70)
        print(prompt)
        print("-" * 70)
        print(f"GENERATO:    {gen}")
        if riferimento:
            print(f"RIFERIMENTO: {riferimento}")


if __name__ == "__main__":
    main()
