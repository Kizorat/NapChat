#!/usr/bin/env python3
"""
STADIO A2 — iniezione lessicale, fra il continued pretraining (stadio A) e
l'SFT sui tre task (stadio B).

Cosa addestra
-------------
Gli item prodotti da dati_lessicali.py: glossa (corrispondenza it->nap nuda),
cloze in contesto (il termine dialettale va prodotto dentro la frase e dentro
la conversazione), turno_breve (turni corti ad alta densita' dialettale).
Target corti - da 1 parola a poche - quindi il segnale lessicale non e' diluito
su decine di token come nell'SFT sui turni interi.

Posizione nella catena
----------------------
    python lessico.py          --csv dataset_finale.csv --split-dir SPLIT --out lessico_train.json
    python dati_lessicali.py   --csv dataset_finale.csv --split-dir SPLIT \\
                               --lessico lessico_train.json --out-dir SPLIT/stadio_a2_lessico
    python pretrain_dialect.py --model minerva --split-dir SPLIT                       # A
    python finetune_a2_lessico.py --model minerva --split-dir SPLIT \\
        --init-adapter /kaggle/working/cpt/<slug>/adapter_final                         # A2
    python finetune_t1_traduzione.py --model minerva --split-dir SPLIT \\
        --init-adapter /kaggle/working/runs/<slug>-A2/adapter_final                     # B
    (idem T2 e T3, tutti e tre dallo STESSO adapter A2)

Un solo run di A2 per modello, condiviso dai tre task: 3 run in piu' su 9, non 9.

Perche' lr basso e una sola epoca in piu' non serve
---------------------------------------------------
lr=5e-5 e 3 epoche: gli item sono ~4.000 su un lessico di ~300 tipi, quindi ogni
tipo viene visto molte volte. Spingere di piu' fa memorizzare la coppia
(italiano, napoletano) come lookup e degrada la fluenza a valle - si vede nel
dev di A2 che continua a scendere mentre il chrF di T1 peggiora. Se succede,
taglia a 2 epoche invece di alzare la patience.

metric="exact" non e' disponibile in common.py: si seleziona su chrF, che su
target di una parola coincide di fatto con l'accuratezza a livello di carattere.
"""

from Minerva7B.T1_Traduzione.training.common import TaskConfig, run

TASK = TaskConfig(
    layout="A2",
    nome="iniezione lessicale",
    descrizione="glossa + cloze in contesto + turni brevi ad alta densita' dialettale",
    max_seq_len=512,          # i cloze portano 3 turni di contesto
    epochs=3,
    lr=5e-5,
    metric="chrf",
    greater_is_better=True,
    gen_max_new_tokens=24,    # target corti: 64 token sono solo occasioni di divagare
    repetition_penalty=1.0,   # su un target di una parola la penalita' e' dannosa
    no_repeat_ngram_size=0,
    patience=3,
    evals_per_epoch=3,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    note=[
        "Stadio intermedio: i numeri di questo run NON vanno nel confronto "
        "cross-modello, servono solo a verificare che l'iniezione abbia preso.",
        "Richiede LAYOUT_DIRS['A2'] = 'stadio_a2_lessico' in common.py.",
        "Controllo di sanita': il recall dialettale su T1/T2/T3 deve salire "
        "rispetto agli stessi task partiti dal solo stadio A. Se non sale, il "
        "problema non e' l'iniezione ma il formato dei prompt.",
    ],
)

if __name__ == "__main__":
    run(TASK)
