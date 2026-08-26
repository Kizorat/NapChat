#!/usr/bin/env python3
"""
pesi_lessicali.py — due cose che vanno dentro common.py:

  1. PESATURA DELLA LOSS sui token dialettali.
     In un turno di 12 parole di cui 3 dialettali, la cross-entropy media
     premia chi indovina le 9 parole condivise con l'italiano: copiare porta
     gia' molto lontano. Pesare i token dialettali (peso --peso-dial contro 1.0)
     cambia il gradiente dove il dialetto vive davvero. E' la versione
     "termine per termine" applicata dentro l'SFT, senza cambiare i task.

  2. METRICHE LESSICALI.
     chrF sale anche solo copiando l'italiano (baseline copia-italiano: 35,92)
     perche' le due varieta' condividono gran parte dei caratteri. Serve una
     misura che risponda a "ha imparato le parole?":

       recall_dialettale   quanti dei tipi dialettali del riferimento compaiono
                           nell'ipotesi (multiset, non solo presenza)
       precisione_dial.    quanti dei tipi dialettali prodotti sono attestati
                           nel riferimento: sanziona chi spara 'o e nun a caso
       tasso_italianismi   fra le parole italiane per cui il lessico conosce una
                           resa dialettale, quante restano in forma italiana
                           nell'ipotesi. E' la metrica piu' diagnostica: misura
                           esattamente lo scivolamento verso l'italiano
       tasso_copia         ipotesi identiche alla frase italiana di partenza

Uso come libreria, dentro common.py
-----------------------------------
    from pesi_lessicali import (ChatDatasetPesato, make_collate_pesato,
                                TrainerPesato, carica_lessico)

    lex = carica_lessico(args.lessico)                      # None se non passato
    Dataset = ChatDatasetPesato if lex else ChatDataset
    ds_tr = Dataset(rows_tr, tok, cfg.max_seq_len, lessico=lex, peso_dial=args.peso_dial)
    collate = make_collate_pesato(pad_id) if lex else make_collate_fn(pad_id)
    Tr = TrainerPesato if lex else Trainer

I pesi viaggiano come colonna del batch, quindi TrainingArguments deve avere
remove_unused_columns=False (nel tuo common.py c'e' gia').

Autotest
--------
    python pesi_lessicali.py --csv dataset_finale.csv --lessico lessico_train.json
Valuta la baseline copia-italiano e il riferimento umano con le stesse metriche:
serve a vedere il fondo scala prima di leggere i numeri di un modello.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

TOK = re.compile(r"[a-zàèéìòóùâêîôû'\u2019\-]+")


def norm(s):
    return str(s).replace("\u2019", "'")


def tokenizza(s):
    return TOK.findall(norm(s).lower())


def carica_lessico(path):
    """Ritorna {'dialettali': set, 'it2nap': {ita: set(nap dialettali)}}."""
    if not path:
        return None
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d["_meta"].get("solo_csv"):
        raise SystemExit("Lessico estratto con --solo-csv: contiene dev/test, "
                         "non usarlo ne' per pesare la loss ne' per valutare.")
    it2nap = {}
    for v in d["lessico"]:
        if v["dialettale"]:
            it2nap.setdefault(v["italiano"], set()).add(v["napoletano"])
    return {"dialettali": set(d["tipi_dialettali"]), "it2nap": it2nap}


# --------------------------------------------------------------------------- #
# 1. Pesatura della loss
# --------------------------------------------------------------------------- #

def _span_dialettali(target, dialettali):
    """Span di caratteri (dentro `target`) occupati da parole dialettali."""
    return [(m.start(), m.end()) for m in TOK.finditer(norm(target).lower())
            if m.group() in dialettali]


class ChatDatasetPesato:
    """Come ChatDataset, piu' una colonna `pesi` per-token.

    Attribuzione via offset_mapping del tokenizer fast: e' l'unico modo esatto
    di sapere quali token coprono quali caratteri, e quindi quali sottoparole
    appartengono a una forma dialettale. Con un tokenizer slow gli offset non
    esistono: si ricade su pesi uniformi e lo si dichiara, invece di indovinare
    i confini e pesare i token sbagliati.
    """

    def __init__(self, rows, tokenizer, max_len, lessico, peso_dial=3.0,
                 render=None):
        if render is None:                      # import locale: evita cicli
            from Minerva7B.T1_Traduzione.training.common import render_prompt
            render = render_prompt
        self.rows, self.tok, self.max_len = rows, tokenizer, max_len
        self.dial = lessico["dialettali"]
        self.peso_dial = peso_dial
        self.render = render
        self.fast = bool(getattr(tokenizer, "is_fast", False))
        self.n_troncati = 0
        self.n_prefix_mismatch = 0
        self.n_token_pesati = 0
        self.n_token_target = 0
        if not self.fast:
            print("! tokenizer non-fast: nessun offset_mapping, pesi uniformi "
                  "(la pesatura lessicale e' DISATTIVATA per questo run)")
        for k, r in enumerate(rows[:200]):
            p = self.tok(self.render(self.tok, r["prompt"]),
                         add_special_tokens=False)["input_ids"]
            f = self.tok(self.render(self.tok, r["prompt"], r["target"]),
                         add_special_tokens=False)["input_ids"]
            if len(f) > max_len:
                self.n_troncati += 1
            if f[:len(p)] != p:
                self.n_prefix_mismatch += 1

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        testo_p = self.render(self.tok, r["prompt"])
        testo_f = self.render(self.tok, r["prompt"], r["target"])
        n_prompt = len(self.tok(testo_p, add_special_tokens=False)["input_ids"])

        enc = self.tok(testo_f, add_special_tokens=False,
                       return_offsets_mapping=self.fast)
        ids = enc["input_ids"][:self.max_len]
        labels = list(ids)
        for j in range(min(n_prompt, len(labels))):
            labels[j] = -100

        pesi = [1.0] * len(ids)
        if self.fast:
            off = enc["offset_mapping"][:len(ids)]
            base = testo_f.rfind(norm(r["target"]))
            if base >= 0:
                span = [(base + s, base + e)
                        for s, e in _span_dialettali(r["target"], self.dial)]
                for j, (a, b) in enumerate(off):
                    if labels[j] == -100 or a == b:
                        continue
                    self.n_token_target += 1
                    if any(a < fe and b > fs for fs, fe in span):
                        pesi[j] = self.peso_dial
                        self.n_token_pesati += 1
        return {"input_ids": ids, "attention_mask": [1] * len(ids),
                "labels": labels, "pesi": pesi}

    def riepilogo_pesi(self):
        if not self.n_token_target:
            return "pesatura: nessun token osservato (dataset non ancora iterato)"
        q = self.n_token_pesati / self.n_token_target
        return (f"pesatura: {self.n_token_pesati}/{self.n_token_target} token di "
                f"target dialettali ({q:.1%}), peso {self.peso_dial}")


def make_collate_pesato(pad_id):
    import torch

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids, attn, lab, pesi = [], [], [], []
        for b in batch:
            k = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_id] * k)
            attn.append(b["attention_mask"] + [0] * k)
            lab.append(b["labels"] + [-100] * k)
            pesi.append(b["pesi"] + [0.0] * k)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(lab, dtype=torch.long),
                "pesi": torch.tensor(pesi, dtype=torch.float)}
    return collate


def _trainer_pesato():
    """Costruito a runtime: importare Trainer al top del modulo costerebbe il
    caricamento di transformers anche a chi usa solo le metriche."""
    import torch
    from transformers import Trainer

    class TrainerPesato(Trainer):
        """Loss pesata sui token dialettali, con la normalizzazione CORRETTA
        rispetto all'accumulo di gradiente.

        Il punto delicato: transformers guarda se il forward del modello accetta
        **kwargs (un PeftModel lo accetta sempre) e in quel caso NON divide la
        loss per gli step di accumulo, perche' si aspetta una loss gia'
        normalizzata su `num_items_in_batch` (il numero di token supervisionati
        nell'intera finestra di accumulo). Restituire una media per micro-batch
        e ignorare num_items_in_batch gonfia i gradienti di un fattore
        grad_accum: si vede in grad_norm a tre cifre e in max_grad_norm che
        clippa a ogni step, quindi il lr effettivo non e' quello impostato.

        Qui i pesi vengono RINORMALIZZATI a media 1 sui token supervisionati
        del micro-batch, poi si somma e si divide per num_items_in_batch. Cosi':
          * la scala resta identica a quella di un run non pesato (media 1),
            quindi il lr non va ritoccato e i log sono confrontabili;
          * la semantica dell'accumulo e' quella attesa da transformers;
          * il RAPPORTO fra token dialettali e non dialettali resta peso_dial.
        """

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None, **kw):
            pesi = inputs.pop("pesi", None)
            labels = inputs.get("labels")
            out = model(**inputs)
            # shift standard del causal LM: la posizione t predice t+1
            lg = out.logits[:, :-1, :].contiguous()
            lb = labels[:, 1:].contiguous()
            perdite = torch.nn.functional.cross_entropy(
                lg.view(-1, lg.size(-1)).float(), lb.view(-1),
                ignore_index=-100, reduction="none").view(lb.shape)

            maschera = (lb != -100).float()
            n_sup = maschera.sum().clamp(min=1.0)
            if pesi is None:
                w = maschera
            else:
                w = pesi[:, 1:].contiguous() * maschera
                w = w * (n_sup / w.sum().clamp(min=1.0))   # media 1 sui supervisionati

            if num_items_in_batch is None:
                # nessun accumulo dichiarato: media ponderata, come prima
                loss = (perdite * w).sum() / n_sup
            else:
                # transformers sommera' i micro-batch senza dividere
                loss = (perdite * w).sum() / num_items_in_batch
            return (loss, out) if return_outputs else loss

    return TrainerPesato


class _Lazy:
    def __call__(self, *a, **k):
        return _trainer_pesato()(*a, **k)

    def __mro_entries__(self, bases):
        return (_trainer_pesato(),)


TrainerPesato = _Lazy()


# --------------------------------------------------------------------------- #
# 2. Metriche lessicali
# --------------------------------------------------------------------------- #

def valuta(fonti_ita, riferimenti, ipotesi, lessico):
    """Metriche lessicali su una lista di triple (italiano, riferimento, ipotesi)."""
    dial, it2nap = lessico["dialettali"], lessico["it2nap"]
    rec_num = rec_den = pre_num = pre_den = 0
    ital_num = ital_den = 0
    copie = 0
    for src, rif, ipo in zip(fonti_ita, riferimenti, ipotesi):
        t_src, t_rif, t_ipo = tokenizza(src), tokenizza(rif), tokenizza(ipo)
        c_rif = Counter(w for w in t_rif if w in dial)
        c_ipo = Counter(w for w in t_ipo if w in dial)
        rec_num += sum((c_rif & c_ipo).values()); rec_den += sum(c_rif.values())
        pre_num += sum((c_rif & c_ipo).values()); pre_den += sum(c_ipo.values())
        set_ipo = set(t_ipo)
        for w in t_src:
            if w not in it2nap:
                continue                      # nessuna resa dialettale attestata
            ital_den += 1
            # italianismo: la forma italiana e' rimasta e nessuna resa dialettale
            # nota compare al suo posto
            if w in set_ipo and not (it2nap[w] & set_ipo):
                ital_num += 1
        if norm(ipo).strip().lower() == norm(src).strip().lower():
            copie += 1
    n = max(1, len(ipotesi))
    r = rec_num / rec_den if rec_den else 0.0
    p = pre_num / pre_den if pre_den else 0.0
    # La precisione e' illeggibile quando il denominatore e' minuscolo: la
    # baseline copia-italiano produce 13 forme dialettali per caso (omografie) e
    # ne "azzecca" 12, quindi precisione 0.92 accanto a un recall di 0.002. Il
    # numero non e' sbagliato, e' privo di significato: va letto solo insieme al
    # conteggio, e sotto 30 forme prodotte si dichiara non interpretabile.
    avvertenza = None
    if pre_den < 30:
        avvertenza = (f"precisione_dialettale calcolata su sole {pre_den} forme "
                      f"dialettali prodotte: non interpretabile")
    return {
        "recall_dialettale": round(r, 4),
        "precisione_dialettale": round(p, 4),
        "f1_dialettale": round(2 * p * r / (p + r), 4) if p + r else 0.0,
        "tasso_italianismi": round(ital_num / ital_den, 4) if ital_den else None,
        "tasso_copia": round(copie / n, 4),
        "_conteggi": {"tipi_dial_riferimento": rec_den, "tipi_dial_ipotesi": pre_den,
                      "occasioni_lessicali": ital_den, "n": len(ipotesi)},
        "_avvertenza": avvertenza,
    }


def main():
    ap = argparse.ArgumentParser(description="autotest delle metriche lessicali")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--lessico", required=True)
    ap.add_argument("--escludi-fonte", nargs="*", default=[])
    a = ap.parse_args()

    import pandas as pd
    lex = carica_lessico(a.lessico)
    df = pd.read_csv(a.csv)
    if a.escludi_fonte:
        df = df[~df["fonte"].isin(a.escludi_fonte)]
    ita, nap = df["italiano"].tolist(), df["napoletano"].tolist()

    print("Baseline COPIA-ITALIANO (l'ipotesi e' la frase italiana):")
    print(json.dumps(valuta(ita, nap, ita, lex), ensure_ascii=False, indent=1))
    print("\nTetto: RIFERIMENTO UMANO come ipotesi (controllo di sanita'):")
    print(json.dumps(valuta(ita, nap, nap, lex), ensure_ascii=False, indent=1))
    for f, g in df.groupby("fonte"):
        print(f"\nSolo fonte={f} ({len(g)} righe), baseline copia:")
        print(json.dumps(valuta(g["italiano"].tolist(), g["napoletano"].tolist(),
                                g["italiano"].tolist(), lex),
                         ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
