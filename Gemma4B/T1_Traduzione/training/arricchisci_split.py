#!/usr/bin/env python3
"""
arricchisci_split.py — aggiunge INFORMAZIONE ai prompt, e costruisce il mix
multi-task. Nessun dato nuovo.

Il punto di partenza
--------------------
`diagnosi_cecita.py` mostra che nel training di T3 l'istruzione e' la STESSA
stringa in tutti i 748 esempi. Una feature costante ha informazione mutua zero
col target: il modello non puo' usarla per abbassare la loss, quindi impara a
ignorarla. Dopo l'SFT l'istruzione non e' piu' letta come un'istruzione, e'
diventata un DELIMITATORE - un pattern che dice "qui inizia il pezzo dove devo
produrre testo tipo-parlato-trascritto".

Da cui la conseguenza che va detta chiaramente: riscrivere meglio l'istruzione
NON serve, perche' il modello non la sta leggendo. Prima va resa informativa.

Le quattro modifiche (--variazioni, --parlanti, --ancora, e igiene_split
--finestre per la quarta) aggiungono informazione che ora NON c'e'. Non sono
riformulazioni.

1. --variazioni  L'istruzione viene campionata fra N parafrasi. Smette di essere
                 costante, quindi smette di essere ignorabile: il modello deve
                 leggere il prompt per capire cosa produrre. In DEV e TEST si usa
                 SEMPRE la parafrasi canonica (indice 0), non una campionata: il
                 training vede varieta', la valutazione vede una forma fissa,
                 cosi' due arm restano confrontabili e nessun run e' fortunato.

2. --parlanti    Etichetta del parlante sui turni di contesto + istruzione che
                 nomina il destinatario con lo STESSO identificativo. Senza,
                 il modello non puo' sapere che i turni si alternano ne' a chi
                 tocca. Il CSV ha la colonna `speaker`: l'informazione esiste e
                 non e' usata. Se i turni sono gia' etichettati nello split,
                 l'opzione aggiunge solo il riferimento nell'istruzione.

3. --ancora      Una riga con l'argomento del segmento, ricavata via TF-IDF dalle
                 parole di contenuto piu' distintive dei turni attorno. Attacca
                 direttamente il "non sa di cosa si sta parlando". NON e' dato
                 nuovo: e' informazione derivata dai turni che hai, resa
                 esplicita invece di lasciata implicita in ~21 parole di parlato
                 frammentario.

4. --multitask   Scrive un layout "MT" con T1+T2+T3 MESCOLATI, per addestrare UN
                 SOLO adapter invece di tre.

Perche' il multi-task e' la modifica strutturale
------------------------------------------------
Ora addestri tre adapter separati. Ognuno vede una sola istruzione, sempre
identica -> costante -> ignorata. Mescolando i tre task, l'istruzione diventa
l'UNICA cosa che li distingue: il modello DEVE leggerla, per necessita' e non
per esortazione.

Due effetti che si sommano:

    istanze viste dall'adapter che poi usi su T3
        ora            748
        multi-task     1134 + 683 + 748 = 2565

E il secondo, che conta di piu': T1 fa da ANCORA SEMANTICA. La traduzione e'
uno-a-uno e obbliga a una mappatura che preserva il significato - non puoi
tradurre bene ignorando cosa dice la frase. Addestrando insieme, quel vincolo
tiene il modello agganciato al contenuto anche su T3, dove la loss da sola non
lo richiede. E' l'opposto di quello che succede ora, dove l'SFT su T3 isolato
spinge verso la superficie del parlato.

Richiede una riga in common.py: LAYOUT_DIRS["MT"] = "multitask". La cella del
notebook la applica.

Uso
---
    python arricchisci_split.py --split-dir /kaggle/working/split \\
        --out-dir /kaggle/working/split_arr \\
        --variazioni --parlanti --ancora --multitask

    # compone con igiene_split.py: prima igiene, poi arricchimento
    python igiene_split.py --split-dir SPLIT --out-dir /kaggle/working/split_ig \\
        --task T3 T2 --tipo-turno --sottocampiona-duplicati --finestre 1,2,3,4,5
    python arricchisci_split.py --split-dir /kaggle/working/split_ig \\
        --out-dir /kaggle/working/split_arr --variazioni --parlanti --ancora \\
        --multitask

ATTENZIONE: i prompt devono essere BYTE-IDENTICI fra training e inferenza.
Qualunque cartella usi per addestrare, usa la stessa in evaluate_task.py,
rerank_t3.py, finetune_t3_dpo.py e pipeline_due_stadi.py.
"""

import argparse
import json
import random
import re
import shutil
import statistics
from collections import Counter
from pathlib import Path

from common import LAYOUT_DIRS, load_split
from contesto_metrica import _e_separatore, estrai_blocco_contesto

# Parafrasi dell'istruzione, per layout. L'indice 0 e' la CANONICA: e' quella
# usata in dev/test, quindi deve restare la formulazione piu' neutra.
# Le altre non sono sinonimi decorativi: variano il modo in cui il compito e'
# descritto (imperativo, ruolo, obiettivo), cosi' il modello impara a mappare
# INTENTI diversi sullo stesso comportamento invece di riconoscere una stringa.
VARIAZIONI = {
    "T1": [
        "Traduci in napoletano la frase italiana.",
        "Rendi in napoletano quello che segue, restando fedele al significato.",
        "Qual e' la versione napoletana di questa frase italiana?",
        "Riscrivi la frase in napoletano senza cambiare cio' che dice.",
        "Volgi in napoletano, mantenendo il senso e il registro parlato.",
        "Da italiano a napoletano: traduci.",
        "Come si dice in napoletano?",
        "Trasponi in napoletano il contenuto della frase italiana.",
    ],
    "T2": [
        "Completa il turno.",
        "Continua e concludi il turno iniziato, restando coerente col discorso.",
        "Come finisce questo turno?",
        "Porta a termine la frase del parlante in napoletano.",
        "Il turno e' interrotto: completalo in modo coerente col contesto.",
        "Prosegui il turno fino alla fine.",
        "Scrivi la parte che manca del turno.",
        "Termina l'enunciato cominciato sopra.",
    ],
    "T3": [
        "Rispondi come il parlante successivo.",
        "Cosa dice adesso l'altro parlante? Rispondi nel merito di quello che e' "
        "stato detto.",
        "Tocca all'altro parlante: scrivi il suo turno, pertinente al discorso.",
        "Prosegui la conversazione con il turno successivo, in napoletano.",
        "Sei il parlante successivo: replica a quello che hai appena sentito.",
        "Scrivi la replica che viene ora, coerente con l'argomento della "
        "conversazione.",
        "Che cosa risponde l'interlocutore?",
        "Continua il dialogo: il prossimo turno tocca a te.",
    ],
}

# Parole troppo frequenti nel parlato per essere distintive di un argomento.
# Non e' una stoplist italiana generica: e' tarata sui riscontri e sui
# segnalidiscorsivi che dominano questo corpus (mh 95, si' 71, mhmh 58, no 32).
STOP = set("""
e ma pe' po' ca che chi cu nun no si' se mh mhmh eh ah oh uh ehm mm cioe' pero'
allora quindi insomma tipo ecco vabbuo' okay ok gia' pure comme quanno quando
dove chesta chesto chella chello chisto chillo stu sta 'o 'a 'e 'nu 'na nu na
me te ce ve se ne ci vi mi ti lo la li le gli il un una uno dei delle del della
a ai al alla da dai dal dalla di in con su per tra fra non piu' meno molto poco
tutto tutta tutti tutte cosa cose fatto fatta essere stato stata avere aveva
sono era erano ho hai ha abbiamo avete hanno faccio fa fai famo facimmo
grazie prego scusa senti sai vedi guarda dimmi niente nulla qualcosa
""".split())

MIN_LEN_PAROLA = 4


def normalizza(s):
    return re.sub(r"[^\w'\s]", " ", (s or "").lower())


def parole_contenuto(testo):
    return [w for w in normalizza(testo).split()
            if len(w) >= MIN_LEN_PAROLA and w not in STOP and not w.isdigit()]


def costruisci_ancore(rows, k=4, finestra=8):
    """Per ogni item: le k parole piu' distintive del suo intorno.

    TF-IDF fatto a mano invece che con sklearn: il documento e' l'intorno di un
    item, il corpus sono tutti gli intorni. Cosi' l'ancora non contiene le parole
    frequenti in TUTTA la conversazione (che non distinguono un segmento
    dall'altro) ma quelle caratteristiche di QUESTO punto.

    L'intorno include il contesto e i turni vicini, MAI il target: altrimenti
    l'ancora conterrebbe le parole della risposta e sarebbe leakage - il modello
    imparerebbe a copiarle e i numeri sarebbero gonfiati senza che nulla di
    reale sia migliorato.
    """
    import math
    intorni = []
    for i, r in enumerate(rows):
        trovato = estrai_blocco_contesto(r["prompt"])
        turni = trovato[2] if trovato else []
        vicini = []
        for j in range(max(0, i - finestra // 2), min(len(rows), i + finestra // 2)):
            t = estrai_blocco_contesto(rows[j]["prompt"])
            if t:
                vicini += t[2]
        intorni.append(parole_contenuto(" ".join(turni + vicini)))

    df = Counter()
    for doc in intorni:
        df.update(set(doc))
    n_doc = max(1, len(intorni))

    ancore = []
    for doc in intorni:
        tf = Counter(doc)
        punteggi = {w: c * math.log(n_doc / (1 + df[w])) for w, c in tf.items()}
        top = [w for w, _ in sorted(punteggi.items(), key=lambda x: -x[1])[:k]]
        ancore.append(top)
    return ancore


def scomponi(prompt):
    """(intestazione+turni, separatori, istruzione) come liste di righe."""
    trovato = estrai_blocco_contesto(prompt)
    if trovato is None:
        return None
    i0, i1, _ = trovato
    righe = prompt.split("\n")
    k = i1
    while k < len(righe) and _e_separatore(righe[k]):
        k += 1
    return righe[:i1], righe[i1:k], righe[k:]


def etichette_presenti(turni):
    n = 0
    for t in turni:
        p = t.strip().split(":", 1)
        if len(p) == 2 and 0 < len(p[0]) <= 12:
            n += 1
    return n / len(turni) if turni else 0.0


def etichetta_turni(turni, nomi=("A", "B")):
    """Etichetta alternata. Non e' la vera identita' dei parlanti - quella sta
    nella colonna `speaker` del CSV e andrebbe propagata in split_dataset.py -
    ma rende esplicita l'ALTERNANZA, che e' l'informazione di cui il modello ha
    bisogno per sapere a chi tocca. Riportalo come approssimazione."""
    return [f"{nomi[i % len(nomi)]}: {t.strip()}" for i, t in enumerate(turni)]


def prossimo_parlante(turni, nomi=("A", "B")):
    return nomi[len(turni) % len(nomi)]


def riscrivi(rows, layout, split, a, rng, ancore=None):
    fuori = []
    n_var, n_lab, n_anc, n_salti = 0, 0, 0, 0
    for i, r in enumerate(rows):
        sc = scomponi(r["prompt"])
        if sc is None:
            n_salti += 1
            fuori.append(dict(r))
            continue
        testa, sep, istr = sc
        trovato = estrai_blocco_contesto(r["prompt"])
        turni = trovato[2]
        intest = testa[:len(testa) - len(turni)]
        q = dict(r)

        # 1. etichette dei parlanti
        if a.parlanti:
            if etichette_presenti(turni) < 0.9:
                turni = etichetta_turni(turni)
                n_lab += 1
            q["parlante_target"] = prossimo_parlante(turni)

        # 2. istruzione: variata (train) o canonica (dev/test)
        if a.variazioni:
            varianti = VARIAZIONI.get(layout) or []
            if varianti:
                testo = varianti[0] if split != "train" else rng.choice(varianti)
                if split == "train":
                    n_var += 1
                if a.parlanti and layout == "T3":
                    testo += f" Rispondi a {turni[-1].split(':')[0].strip()}."
                # L'istruzione va ULTIMA, adiacente al punto di generazione:
                # le altre righe (ancora, tipo di turno di igiene_split) sono
                # contesto, non comando. Le righe preesistenti che sono la
                # vecchia istruzione vengono rimosse, non duplicate.
                resto = [x for x in istr if x.strip()
                         and not _somiglia_istruzione(x, varianti)]
                istr = resto + [testo]
                q["variante_istruzione"] = testo

        # 3. ancora tematica: prima di tutto il resto dell'istruzione
        if a.ancora and ancore and ancore[i]:
            istr = [f"Si sta parlando di: {', '.join(ancore[i])}."] + istr
            q["ancora"] = ancore[i]
            n_anc += 1

        q["prompt"] = "\n".join(intest + turni + sep + istr)
        q["layout"] = layout
        fuori.append(q)
    return fuori, {"istruzioni_variate": n_var, "turni_etichettati": n_lab,
                   "ancore_aggiunte": n_anc, "prompt_non_scomponibili": n_salti}


def _somiglia_istruzione(riga, varianti):
    """Riconosce la vecchia istruzione per non lasciarne due nel prompt.

    Confronto sulle prime parole di contenuto: le formulazioni originali
    ("Rispondi come il parlante successivo.") e le parafrasi condividono il
    verbo iniziale, e un match esatto non basterebbe.
    """
    r = normalizza(riga).split()
    if not r:
        return False
    teste = {normalizza(v).split()[0] for v in varianti if v.strip()}
    return r[0] in teste


def statistiche(rows, etichetta):
    n_istr = Counter()
    for r in rows:
        sc = scomponi(r["prompt"])
        if sc:
            n_istr["\n".join(sc[2])] += 1
    L = [len(r["prompt"].split()) for r in rows]
    return {"nome": etichetta, "n": len(rows),
            "istruzioni_distinte": len(n_istr),
            "parole_prompt_medie": round(statistics.mean(L), 1) if L else 0}


def main():
    ap = argparse.ArgumentParser(
        description="Aggiunge informazione ai prompt + mix multi-task",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", default="/kaggle/working/split")
    ap.add_argument("--out-dir", default="/kaggle/working/split_arr")
    ap.add_argument("--task", default=["T1", "T2", "T3"], nargs="+",
                    choices=["T1", "T2", "T3"])
    ap.add_argument("--variazioni", action="store_true")
    ap.add_argument("--parlanti", action="store_true")
    ap.add_argument("--ancora", action="store_true")
    ap.add_argument("--ancora-k", type=int, default=4)
    ap.add_argument("--multitask", action="store_true",
                    help="scrive anche il layout MT con i task mescolati")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    tasks = a.task if isinstance(a.task, list) else [a.task]
    if not (a.variazioni or a.parlanti or a.ancora or a.multitask):
        print("Nessuna modifica richiesta. Usa almeno una fra --variazioni, "
              "--parlanti, --ancora, --multitask.")
        return

    src, out = Path(a.split_dir), Path(a.out_dir)
    rng = random.Random(a.seed)
    rapporto = {"origine": str(src), "opzioni": vars(a), "task": {}}
    mix = {"train": [], "dev": [], "test": []}

    for t in tasks:
        d = out / LAYOUT_DIRS[t]
        d.mkdir(parents=True, exist_ok=True)
        rapporto["task"][t] = {}
        for split in ("train", "dev", "test"):
            rows = load_split(str(src), t, split)
            ancore = (costruisci_ancore(rows, a.ancora_k) if a.ancora else None)
            nuove, info = riscrivi(rows, t, split, a, rng, ancore)
            json.dump(nuove, open(d / f"{split}.json", "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            rapporto["task"][t][split] = {
                "prima": statistiche(rows, f"{t}/{split} prima"),
                "dopo": statistiche(nuove, f"{t}/{split} dopo"), **info}
            print(f"  {t}/{split}: {len(rows)} item | istruzioni distinte "
                  f"{rapporto['task'][t][split]['prima']['istruzioni_distinte']}"
                  f" -> {rapporto['task'][t][split]['dopo']['istruzioni_distinte']}"
                  + (f" | {info['ancore_aggiunte']} ancore" if a.ancora else "")
                  + (f" | ! {info['prompt_non_scomponibili']} non scomponibili"
                     if info["prompt_non_scomponibili"] else ""))
            if a.multitask:
                for r in nuove:
                    q = dict(r)
                    q["task_origine"] = t
                    q["layout"] = "MT"      # load_split filtra su questo campo
                    mix[split].append(q)

    if a.multitask:
        dm = out / "multitask"
        dm.mkdir(parents=True, exist_ok=True)
        for split in ("train", "dev", "test"):
            rng.shuffle(mix[split])
            json.dump(mix[split], open(dm / f"{split}.json", "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        comp = {s: dict(Counter(r["task_origine"] for r in mix[s]))
                for s in mix}
        rapporto["multitask"] = {"n": {s: len(v) for s, v in mix.items()},
                                 "composizione": comp}
        print(f"\n  MT: train {len(mix['train'])} | dev {len(mix['dev'])} | "
              f"test {len(mix['test'])}")
        print(f"      composizione train: {comp['train']}")
        print("      ricorda la riga in common.py: "
              'LAYOUT_DIRS["MT"] = "multitask"')

    # gli altri layout (A2) vanno copiati intatti
    for k, sub in LAYOUT_DIRS.items():
        if k in tasks or k == "MT":
            continue
        s = src / sub
        if s.exists():
            shutil.copytree(s, out / sub, dirs_exist_ok=True)
            print(f"  {k}: copiato intatto")

    (out / "arricchimento.json").write_text(
        json.dumps(rapporto, ensure_ascii=False, indent=2), encoding="utf-8")

    # un prompt di esempio per far vedere il risultato
    esempio = load_split(str(out), tasks[-1], "train")[0]
    print(f"\n  --- un prompt di {tasks[-1]}/train dopo l'arricchimento ---")
    for riga in esempio["prompt"].split("\n"):
        print(f"  | {riga}")
    print(f"  | TARGET: {esempio['target']}")

    print(f"\nScritto {out}")
    print(f"Usa --split-dir {out} in TUTTI gli script, training e valutazione: "
          f"i prompt devono essere byte-identici da entrambe le parti.")


if __name__ == "__main__":
    main()
