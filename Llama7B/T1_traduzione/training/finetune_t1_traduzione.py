#!/usr/bin/env python3
"""
FASE 3 — T1: traduzione italiano -> napoletano con contesto conversazionale.

Layout 1 (layout1_traduzione_con_contesto): il prompt contiene i 3 turni
precedenti già in napoletano più la frase italiana da tradurre; il target è la
resa napoletana del turno corrente.

Perché una configurazione propria e non gli stessi valori degli altri due task:
  * è il layout con più istanze (~1134 in train) -> più step per epoca, quindi
    meno epoche sono sufficienti e l'eval può essere meno frequente
  * il prompt è il più lungo dei tre (3 turni di contesto + istruzione + frase):
    serve max_seq_len ampio, altrimenti il TARGET viene troncato e la loss si
    calcola su un riferimento mutilato
  * esiste un unico riferimento corretto -> chrF è una metrica di selezione
    legittima, non un proxy debole

Uso su Kaggle:
    !python finetune_t1_traduzione.py --model minerva \
        --split-dir /kaggle/input/napoletano-split/split
"""

from common import TaskConfig, run

TASK = TaskConfig(
    layout="T1",
    nome="traduzione con contesto",
    descrizione="italiano + 3 turni di contesto napoletano -> napoletano",
    max_seq_len=512,          # contesto lungo: sotto i 512 token si troncano i target
    epochs=4,
    lr=1e-4,
    metric="chrf",            # riferimento unico: chrF è appropriato
    greater_is_better=True,
    gen_max_new_tokens=64,
    repetition_penalty=1.15,
    no_repeat_ngram_size=0,   # su una traduzione breve un trigramma ripetuto può essere corretto
    patience=4,
    evals_per_epoch=2,        # ~71 step/epoca -> eval ogni ~35 step, ~8 eval nel run
    note=[
        "Metrica primaria del paper: chrF++ su test con generate() reale, non questo proxy.",
        "Baseline obbligatoria: copia dell'italiano invariato (le coppie identiche nel corpus "
        "la rendono tutt'altro che banale).",
    ],
)

if __name__ == "__main__":
    run(TASK)
