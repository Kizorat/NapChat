#!/usr/bin/env python3
"""
t3_eval.py — valutazione di T3 centrata sulla domanda giusta.

La domanda non e' "quanto assomiglia al turno realmente pronunciato" (riferimento
singolo su un task open-ended: chrF misura la fortuna) ma "l'uscita cambia se
cambia il contesto". Da qui la metrica primaria:

  ABLAZIONE — si genera lo stesso item due volte, col contesto vero e con un
  contesto preso da un altro punto, e si misura il chrF++ FRA LE DUE USCITE.
      chrF alto / identiche       -> il modello e' cieco al contesto
      chrF basso e uscite sensate -> il contesto sta entrando nella risposta

Modalita' di decodifica (--modo):
  greedy      riferimento
  nucleus     top_p 0.9, T 0.8
  cad         decodifica contestuale: logit = (1+g)*logit(y|ctx) - g*logit(y|no ctx).
              Amplifica in inferenza la differenza fra le due distribuzioni.
              Non riaddestra niente e agisce esattamente sul difetto osservato.
  bon         best-of-n: n candidati con nucleus, si tiene quello che massimizza
              logP(cand|ctx) - logP(cand|senza ctx) normalizzato per lunghezza.
              Criterio reference-free: nessuno guarda il target.

Uso:
    python t3_eval.py --model minerva --adapter /kaggle/working/runs/..._T3/adapter_final \
        --split-dir /kaggle/working/split_t3 --modo greedy cad bon --gamma 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics

from t3_train import (base_ambiente, carica_split, carica_token,
                      controlla_finito, id_fine_turno, render, resolve_model,
                      scegli_dtype, slug, sostituisci_contesto)

# Marcatori dialettali: proxy grezzo e dichiarato tale, non un classificatore.
MARCATORI = re.compile(
    r"\b(nun|ca|'a|'o|'e|'nu|na|nnu|chill|chest|accuss|pecch|aggi|sta[cm]|mo'|"
    r"tenimm|simm|jamm|facimm|vulit|vede|guagli|assaje|overo|bbuon|nzomma)",
    re.IGNORECASE)


def dialettalita(testo: str) -> float:
    parole = testo.split()
    if not parole:
        return 0.0
    return sum(1 for p in parole if MARCATORI.match(p)) / len(parole)


def senza_contesto(prompt: str) -> str:
    """Stesso prompt con il blocco di contesto svuotato: e' il ramo 'no ctx'
    della decodifica contestuale."""
    testa, _, istr = prompt.partition("\n---\n")
    return testa.split("\n")[0] + "\n(nessun contesto disponibile)\n---\n" + istr


# --------------------------------------------------------------------------- #
# Generazione
# --------------------------------------------------------------------------- #

def genera_hf(model, tok, prompt, eot_ids, max_new, campiona, temp, top_p, seed=None):
    import torch
    if seed is not None:
        torch.manual_seed(seed)
    inp = tok(render(tok, prompt), add_special_tokens=False,
              return_tensors="pt").to(model.device)
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=max_new, do_sample=campiona,
                           temperature=temp if campiona else None,
                           top_p=top_p if campiona else None,
                           eos_token_id=eot_ids, pad_token_id=tok.pad_token_id,
                           repetition_penalty=1.15, no_repeat_ngram_size=3)
    return tok.decode(o[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def genera_cad(model, tok, prompt, eot_ids, max_new, gamma, temp=0.8, top_p=0.9,
               campiona=False):
    """logit = (1+gamma)*logit(y|ctx) - gamma*logit(y|senza ctx), passo per passo."""
    import torch
    import torch.nn.functional as F

    ids_c = tok(render(tok, prompt), add_special_tokens=False,
                return_tensors="pt")["input_ids"].to(model.device)
    ids_n = tok(render(tok, senza_contesto(prompt)), add_special_tokens=False,
                return_tensors="pt")["input_ids"].to(model.device)
    kv_c = kv_n = None
    prod = []
    with torch.no_grad():
        for passo in range(max_new):
            o_c = model(input_ids=ids_c, past_key_values=kv_c, use_cache=True)
            o_n = model(input_ids=ids_n, past_key_values=kv_n, use_cache=True)
            kv_c, kv_n = o_c.past_key_values, o_n.past_key_values
            lg = (1 + gamma) * o_c.logits[:, -1, :] - gamma * o_n.logits[:, -1, :]
            if campiona:
                p = F.softmax(lg / temp, dim=-1)
                ordinati, idx = p.sort(descending=True)
                taglia = ordinati.cumsum(-1) - ordinati > top_p
                ordinati[taglia] = 0.0
                scelto = idx.gather(-1, ordinati.multinomial(1))
            else:
                scelto = lg.argmax(-1, keepdim=True)
            if scelto.item() in eot_ids:
                break
            prod.append(scelto.item())
            ids_c = ids_n = scelto
    return tok.decode(prod, skip_special_tokens=True).strip()


def punteggio_contesto(model, tok, prompt, candidato, max_len=512):
    """logP(cand|ctx) - logP(cand|senza ctx), normalizzato per token."""
    import torch
    from t3_train import logp_per_sequenza
    valori = []
    with torch.no_grad():
        for prm in (prompt, senza_contesto(prompt)):
            ids = tok(render(tok, prm, candidato), add_special_tokens=False,
                      return_tensors="pt")["input_ids"][:, :max_len].to(model.device)
            n_p = len(tok(render(tok, prm), add_special_tokens=False)["input_ids"])
            lab = ids.clone()
            lab[:, :n_p] = -100
            valori.append(logp_per_sequenza(model(input_ids=ids).logits, lab)[2].item())
    return valori[0] - valori[1]


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="minerva")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--modo", nargs="+", default=["greedy", "cad"],
                    choices=["greedy", "nucleus", "cad", "bon"])
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--n-bon", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--ablazione-n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    a = ap.parse_args()
    a.out = a.out or os.path.join(base_ambiente(), "eval")

    import sacrebleu
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    repo = resolve_model(a.model)
    token = carica_token()
    tok = AutoTokenizer.from_pretrained(repo, token=token, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype, _ = scegli_dtype(repo, a.dtype)
    print(f"precisione {str(dtype).replace('torch.', '')}")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            repo, dtype=dtype, quantization_config=quant, device_map={"": 0},
            token=token, attn_implementation="eager")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=dtype, quantization_config=quant, device_map={"": 0},
            token=token, attn_implementation="eager")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
        print("adapter:", a.adapter)
    else:
        print("NESSUN adapter: e' il modello di base (arm di confronto)")
    model.eval()
    model.config.use_cache = True

    righe = carica_split(a.split_dir, a.split, a.limite or None)
    eot = id_fine_turno(tok, righe[0]["prompt"], righe[0]["target"])
    eot_ids = sorted({eot, tok.eos_token_id} - {None})
    print(f"{len(righe)} item | fine turno {eot_ids}")

    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    base_tag = a.tag or ("ft" if a.adapter else "base")
    riepilogo = {}

    for modo in a.modo:
        print(f"\n===== {modo} =====")
        preds = []
        for i, r in enumerate(righe):
            if modo == "greedy":
                g = genera_hf(model, tok, r["prompt"], eot_ids, a.max_new, False, None, None)
            elif modo == "nucleus":
                g = genera_hf(model, tok, r["prompt"], eot_ids, a.max_new, True, 0.8, 0.9,
                              seed=a.seed + i)
            elif modo == "cad":
                g = genera_cad(model, tok, r["prompt"], eot_ids, a.max_new, a.gamma)
            else:                                    # best-of-n
                cand = {genera_hf(model, tok, r["prompt"], eot_ids, a.max_new, True,
                                  0.9, 0.95, seed=a.seed + i * 100 + k)
                        for k in range(a.n_bon)}
                cand = [c for c in cand if c] or [""]
                g = max(cand, key=lambda c: punteggio_contesto(model, tok, r["prompt"], c))
            preds.append({"id": r["id"], "prompt": r["prompt"], "riferimento": r["target"],
                          "generato": g, "fonte": r.get("fonte", "")})
            if i % 25 == 0:
                print(f"  {i}/{len(righe)}  {g[:70]!r}")

        ipo = [p["generato"] for p in preds]
        rif = [p["riferimento"] for p in preds]
        m = {
            "n": len(preds),
            "chrf": round(sacrebleu.corpus_chrf(ipo, [rif], word_order=2).score, 3),
            "lunghezza_media": round(statistics.mean(len(h.split()) for h in ipo), 2),
            "lunghezza_umana": round(statistics.mean(len(t.split()) for t in rif), 2),
            "vuote": sum(1 for h in ipo if not h.strip()),
            "dialettalita": round(statistics.mean(dialettalita(h) for h in ipo), 3),
            "dialettalita_umana": round(statistics.mean(dialettalita(t) for t in rif), 3),
            "uscite_distinte": len(set(ipo)),
        }
        m["rapporto_lunghezza"] = round(m["lunghezza_media"] / m["lunghezza_umana"], 3)

        # --- ABLAZIONE: la metrica primaria ---------------------------------
        n_abl = min(a.ablazione_n, len(righe))
        campione = righe[:n_abl]
        veri = [p["generato"] for p in preds[:n_abl]]
        falsi = []
        for i, r in enumerate(campione):
            altro = campione[(i + 1 + rng.randrange(n_abl - 1)) % n_abl]["prompt"]
            p_falso = sostituisci_contesto(r["prompt"], altro)
            if modo == "cad":
                falsi.append(genera_cad(model, tok, p_falso, eot_ids, a.max_new, a.gamma))
            else:
                falsi.append(genera_hf(model, tok, p_falso, eot_ids, a.max_new,
                                       modo != "greedy", 0.8, 0.9, seed=a.seed + i))
        m["ablazione_chrf_vero_vs_falso"] = round(
            sacrebleu.corpus_chrf(falsi, [veri], word_order=2).score, 3)
        m["ablazione_identiche"] = round(
            sum(1 for v, f in zip(veri, falsi) if v.strip() == f.strip()) / n_abl, 3)
        m["ctx_delta"] = round(statistics.mean(
            punteggio_contesto(model, tok, r["prompt"], v)
            for r, v in zip(campione, veri)), 4)

        print(json.dumps(m, indent=2, ensure_ascii=False))
        print("  lettura: ablazione_chrf ALTA (>60) o identiche alto = cieco al contesto;")
        print("           ctx_delta <= 0 = il contesto non aumenta la verosimiglianza.")

        nome = f"{slug(repo)}__T3__{base_tag}__{modo}"
        with open(os.path.join(a.out, nome + ".preds.jsonl"), "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        json.dump(m, open(os.path.join(a.out, nome + ".metrics.json"), "w"),
                  indent=2, ensure_ascii=False)
        riepilogo[modo] = m

    print("\n===== riepilogo =====")
    colonne = ["chrf", "rapporto_lunghezza", "ablazione_chrf_vero_vs_falso",
               "ablazione_identiche", "ctx_delta", "dialettalita"]
    print(f"{'modo':10s}" + "".join(f"{c[:14]:>16s}" for c in colonne))
    for modo, m in riepilogo.items():
        print(f"{modo:10s}" + "".join(f"{m[c]:>16}" for c in colonne))


if __name__ == "__main__":
    main()
