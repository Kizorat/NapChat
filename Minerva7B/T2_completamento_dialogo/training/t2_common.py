#!/usr/bin/env python3
"""
t2_common.py — nucleo condiviso del notebook T2 (completamento di turno in
napoletano). Lo importano train_t2.py, eval_t2.py e audit_t2.py.

Contiene, in una copia sola:
  * il registry dei modelli e il caricamento QLoRA tarato su T4 (fp16, no bf16)
  * il parsing degli item T2 (contesto / prefisso / target)
  * il rendering del prompt nei due stili: "prefill" (default) e "chat"
  * il dataset con masking della loss sui soli token di target
  * le metriche: chrF++, densita' dialettale, tasso di copia, rapporto di
    lunghezza, accuratezza token-level, ctx_acc (uso del contesto)

Niente qui addestra o valuta da solo.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# 1. Modelli
# --------------------------------------------------------------------------- #

MODEL_REGISTRY = {
    "minerva":       "sapienzanlp/Minerva-7B-instruct-v1.0",
    "llama":         "meta-llama/Llama-2-7b-chat-hf",
    "gemma":         "google/gemma-3-4b-it",
    # Smoke test: servono solo a verificare percorsi, template e masking, non
    # la lingua. Vanno tenuti NON gated: Minerva-1B-base e' ad accesso ristretto
    # e su un account senza autorizzazione il primo controllo del notebook
    # fallisce con un 403 che sembra un bug del codice e non lo e'.
    "smoke":        "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smoke-qwen":   "Qwen/Qwen2.5-0.5B-Instruct",
    "minerva-small": "sapienzanlp/Minerva-1B-base-v1.0",   # gated: serve accesso
}

# Istruzione corta di proposito: nel few-shot compare una volta per esempio, e
# piu' testo di impalcatura c'e' nel prompt, piu' e' probabile che il modello lo
# ricopi invece di continuare il turno (e' esattamente quello che succedeva con
# la versione lunga: le generazioni contenevano "Non ricominciare da capo:...").
ISTRUZIONE = "Continua 'o turno 'n napulitano, senza ricumincia' da capo."

SYSTEM = ("Si' nu parlante napoletano. Rispunne sempe 'n napulitano, "
          "cu 'o stesso riggistro parlato d' 'a cunversazione.")


# --------------------------------------------------------------------------- #
# 1b. Compatibilita' fra versioni di transformers
# --------------------------------------------------------------------------- #
# Kaggle aggiorna le immagini senza preavviso e due nomi di parametro sono
# cambiati sotto i piedi negli ultimi rilasci. Meglio scoprirlo qui che con un
# TypeError dopo venti minuti di download del modello.

def kw_strategia_eval() -> str:
    import inspect
    from transformers import TrainingArguments
    p = set(inspect.signature(TrainingArguments.__init__).parameters)
    return "eval_strategy" if "eval_strategy" in p else "evaluation_strategy"


def kwargs_accettati(cls, **kw) -> tuple[dict, list]:
    """(kwargs validi, nomi scartati). Su Kaggle l'immagine puo' saltare a una
    major nuova di transformers da un giorno all'altro e qualche parametro di
    TrainingArguments sparisce: meglio scartarlo dichiarandolo che morire con un
    TypeError dopo il download del modello."""
    import inspect
    validi = set(inspect.signature(cls.__init__).parameters)
    if "kwargs" in validi:
        return kw, []
    return ({k: v for k, v in kw.items() if k in validi},
            sorted(k for k in kw if k not in validi))


def kw_dtype(dtype) -> dict:
    import transformers
    v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if v >= (4, 56) else {"torch_dtype": dtype}


# --------------------------------------------------------------------------- #
# 2. Caricamento dello split
# --------------------------------------------------------------------------- #

def e_layout_t2(records: list[dict]) -> tuple[bool, str]:
    """(va bene, motivo). Il controllo guarda il CONTENUTO, non il nome della
    cartella: e' l'unico modo di accorgersi che qualcuno ha passato il layout
    sbagliato prima che il traceback arrivi trenta righe piu' in la'."""
    if not records:
        return False, "file vuoto"
    lay = {r.get("layout") for r in records[:50]}
    marca = sum(1 for r in records[:50] if MARCA_ISTRUZIONE in r.get("prompt", ""))
    if marca == 0:
        return False, (f"nessuno dei primi 50 prompt contiene "
                       f"'{MARCA_ISTRUZIONE}' (layout dichiarato: {lay}). "
                       "Quasi certamente e' il layout1 (traduzione) o il "
                       "layout3 (replica), non il layout2")
    if marca < len(records[:50]):
        return False, f"solo {marca}/50 prompt hanno la marca di T2"
    if lay - {"T2", None}:
        return False, f"campo 'layout' incoerente: {lay}"
    return True, "ok"


def load_split(split_dir: str, subset: str, verifica: bool = True) -> list[dict]:
    """Accetta sia una cartella con train/dev/test.json, sia la cartella padre
    che contiene layout2_completamento_turno/."""
    cand = [
        os.path.join(split_dir, f"{subset}.json"),
        os.path.join(split_dir, "layout2_completamento_turno", f"{subset}.json"),
    ]
    for p in cand:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                dati = json.load(f)
            if verifica:
                ok, motivo = e_layout_t2(dati)
                if not ok:
                    raise ValueError(
                        f"\n\n{p}\nNON e' il layout2 (completamento di turno): "
                        f"{motivo}.\nQuesto notebook addestra SOLO T2. Punta "
                        f"SPLIT alla cartella layout2_completamento_turno.\n")
            return dati
    raise FileNotFoundError(f"{subset}.json non trovato: ho provato {cand}")


# --------------------------------------------------------------------------- #
# 3. Parsing di un item T2
# --------------------------------------------------------------------------- #

def prefisso_comune(a: list[int], b: list[int]) -> int:
    """Lunghezza del prefisso comune fra due sequenze di id."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


MARCA_ISTRUZIONE = "Continua il turno in napoletano:"
RIGA_SEPARATORE = re.compile(r"^\s*[-=_*]{3,}\s*$", re.M)
RIGA_TURNO = re.compile(r"^\s*([A-Z]{1,3})\s*:\s*(.*)$")


def parse_item(rec: dict) -> tuple[str, str]:
    """(contesto, prefisso). Il contesto e' il blocco di turni precedenti,
    gia' ripulito da 'Conversazione finora:' e dalla riga di trattini.
    Stringa vuota se l'item non ha contesto (succede in 1-2 casi per split)."""
    p = rec["prompt"]
    if MARCA_ISTRUZIONE not in p:
        raise ValueError(
            f"l'item {rec.get('id')} (layout dichiarato: {rec.get('layout')}) "
            f"non contiene '{MARCA_ISTRUZIONE}'. Stai passando dati di un altro "
            f"layout a un notebook che addestra T2.\nPrompt: {p[:120]!r}")
    testa, prefisso = p.split(MARCA_ISTRUZIONE, 1)
    prefisso = prefisso.strip()

    pezzi = RIGA_SEPARATORE.split(testa)
    ctx = pezzi[0] if len(pezzi) > 1 else ""
    ctx = ctx.replace("Conversazione finora:", "").strip()
    return ctx, prefisso


def turni_contesto(ctx: str) -> list[tuple[str, str]]:
    """[(speaker, testo)] dal blocco di contesto."""
    out = []
    for riga in ctx.splitlines():
        m = RIGA_TURNO.match(riga)
        if m:
            out.append((m.group(1), m.group(2).strip()))
    return out


# --------------------------------------------------------------------------- #
# 4. Rendering del prompt
# --------------------------------------------------------------------------- #
# Due stili, per poterli confrontare come ablation.
#
#   prefill (default) — il turno da completare sta DENTRO il messaggio
#     dell'assistente: il prompt di generazione finisce con il prefisso e il
#     modello prosegue letteralmente il proprio testo. Nessun problema di
#     "incollatura" fra prompt e continuazione, e il prefisso non compare due
#     volte.
#
#       system : Si' nu parlante napoletano...
#       user   : Conversazione finora:\nA: emh\n---\nContinua il turno...
#       assist : tu nun l'he' maje fatte sti ffoto quanno si' ghiuta...
#                └── prefisso (mascherato) ──┘└── target (loss attiva) ──┘
#
#   chat — il prefisso sta nel messaggio utente e l'assistente produce solo la
#     seconda meta'. E' la forma classica istruzione->risposta. Il modello deve
#     imparare che la sua risposta si salda al testo dell'utente: su un instruct
#     e' proprio il comportamento che il post-training scoraggia.

def messaggi_utente(rec: dict, stile: str,
                    con_istruzione: bool = True) -> list[dict]:
    """con_istruzione=False per i turni di ESEMPIO del few-shot: ripetere
    l'istruzione identica k+1 volte la rende il testo piu' probabile del
    prompt, e il modello la ricopia al posto di continuare il turno."""
    ctx, prefisso = parse_item(rec)
    spk = rec.get("speaker", "")
    righe = []
    if ctx:
        righe.append("Conversazione finora:")
        righe.append(ctx)
        righe.append("---")
    if con_istruzione:
        istr = ISTRUZIONE
        if spk:
            istr = istr.replace("Continua 'o turno",
                                f"Continua 'o turno 'e {spk}")
        righe.append(istr)
    elif spk:
        righe.append(f"Turno 'e {spk}:")
    if stile == "chat":
        righe.append(f"Turno da completare: {prefisso}")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(righe)}]


_RIPIEGO_ANNUNCIATO = set()


def _template(tok, msgs) -> str:
    """apply_chat_template con ripiego per i modelli base (Minerva-1B-base non
    ha chat_template: senza ripiego lo smoke test non parte proprio).

    Il ripiego viene ANNUNCIATO una volta per motivo. Serve per i modelli il
    cui template rifiuta il ruolo 'system' (e' il caso di alcuni Gemma): senza
    l'avviso il notebook continuerebbe a girare concatenando i messaggi a mano,
    cioe' con un formato di prompt diverso da quello che il modello ha visto in
    post-training, e il confronto fra modelli misurerebbe soprattutto quello."""
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
        except Exception as e:
            chiave = f"{type(tok).__name__}:{type(e).__name__}"
            if chiave not in _RIPIEGO_ANNUNCIATO:
                _RIPIEGO_ANNUNCIATO.add(chiave)
                print(f"! chat_template rifiutato ({type(e).__name__}: "
                      f"{str(e)[:140]}).\n"
                      f"  Ripiego sulla concatenazione dei messaggi. Se il "
                      f"motivo e' il ruolo 'system', valuta di fonderlo nel "
                      f"messaggio utente invece di accettare il ripiego: "
                      f"cambia il formato del prompt per QUESTO modello soltanto "
                      f"e i confronti fra modelli ne risentono.", flush=True)
    else:
        chiave = f"{type(tok).__name__}:nessun-template"
        if chiave not in _RIPIEGO_ANNUNCIATO:
            _RIPIEGO_ANNUNCIATO.add(chiave)
            print("! il tokenizer non ha chat_template (modello base): "
                  "i messaggi vengono concatenati a mano", flush=True)
    corpo = "\n\n".join(m["content"] for m in msgs) + "\n"
    bos = getattr(tok, "bos_token", None)      # il template ce lo mette gia';
    return (bos or "") + corpo                 # senza template va aggiunto a mano


def rendi(tok, rec: dict, stile: str = "prefill") -> tuple[str, str]:
    """(testo_prompt, testo_completo).
    testo_prompt e' esattamente cio' che il modello vede in generazione;
    testo_completo = testo_prompt + il target. La loss vive nella differenza,
    quindi qui il separatore va deciso una volta sola e per tutti e due gli usi:
    se prompt e completo divergessero anche di uno spazio, la maschera
    scivolerebbe di un token."""
    _, prefisso = parse_item(rec)
    testa = _template(tok, messaggi_utente(rec, stile))
    if stile == "prefill":
        # il turno e' gia' iniziato: il target prosegue il prefisso dopo uno spazio
        prompt = testa + prefisso
        completo = prompt + " " + rec["target"].strip()
    elif stile == "chat":
        # il target apre il messaggio dell'assistente: nessuno spazio in testa,
        # a meno che il template non finisca gia' attaccato al testo
        sep = "" if (not testa or testa[-1] in " \n\t>") else " "
        prompt = testa
        completo = prompt + sep + rec["target"].strip()
    else:
        raise ValueError(f"stile sconosciuto: {stile}")
    return prompt, completo


# --------------------------------------------------------------------------- #
# 5. Dataset con masking della loss
# --------------------------------------------------------------------------- #

class DatasetT2:
    """Tokenizza e maschera. Espone anche i contatori diagnostici che servono
    a capire, PRIMA di addestrare, se la loss sta guardando i token giusti."""

    def __init__(self, records, tok, max_seq_len=512, stile="prefill",
                 peso_prefisso=0.0):
        """peso_prefisso > 0 accende la loss anche sul PREFISSO DEL TURNO (non
        sull'impalcatura: system, contesto e istruzione restano sempre a -100).

        Perche'. Con la maschera piena il modello riceve ~12 token di
        supervisione per esempio: su 683 esempi sono ~8.500 token contro ~40M
        di parametri LoRA. Sbloccando il prefisso a peso ridotto la
        supervisione raddoppia e il modello impara la struttura INTERNA del
        turno, non solo la sua seconda meta'. 0.3 e' un punto di partenza
        ragionevole; 0.0 riproduce esattamente il comportamento precedente ed
        e' il default, cosi' `punteggia` continua a misurare il target puro."""
        self.items = []
        self.indici = []          # indice ORIGINALE di ogni item tenuto: senza,
        self.n_ingresso = len(records)   # chi scarta un item disallinea tutto
        self.troncati = 0                # cio' che viene dopo
        self.prefix_mismatch = 0
        self.token_target = []
        self.peso_prefisso = peso_prefisso
        eos = tok.eos_token_id
        for pos, rec in enumerate(records):
            prompt, completo = rendi(tok, rec, stile)
            ids_p = tok(prompt, add_special_tokens=False)["input_ids"]
            ids_c = tok(completo, add_special_tokens=False)["input_ids"]

            # Con i tokenizer SentencePiece tok(a+b) puo' non iniziare con
            # tok(a): si ritokenizza a cavallo del confine. Invece di fidarsi,
            # si misura il prefisso comune reale e si maschera fino a li'.
            n = prefisso_comune(ids_p, ids_c)
            if n != len(ids_p):
                self.prefix_mismatch += 1

            # Confine fra impalcatura e prefisso del turno. Si ricava
            # ritokenizzando la sola TESTA (tutto cio' che sta prima del
            # prefisso): con peso_prefisso=0 non serve e non si calcola.
            n_testa = n
            if peso_prefisso > 0 and stile == "prefill":
                _, pref = parse_item(rec)
                testa = prompt[:len(prompt) - len(pref)]
                n_testa = prefisso_comune(
                    tok(testa, add_special_tokens=False)["input_ids"], ids_c)

            ids = ids_c + [eos]
            labels = [-100] * n_testa + ids[n_testa:]
            pesi = ([0.0] * n_testa + [peso_prefisso] * (n - n_testa)
                    + [1.0] * (len(ids) - n))

            if len(ids) > max_seq_len:
                self.troncati += 1
                ids = ids[-max_seq_len:]
                labels = labels[-max_seq_len:]
                pesi = pesi[-max_seq_len:]
            if all(l == -100 for l in labels):
                continue                      # niente da imparare: si scarta
            self.indici.append(pos)
            self.token_target.append(sum(1 for w in pesi if w == 1.0))
            self.items.append({"input_ids": ids, "labels": labels,
                               "pesi": pesi,
                               "attention_mask": [1] * len(ids)})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def diagnosi(self) -> dict:
        tt = self.token_target
        return {"istanze": len(self.items),
                "scartati": self.n_ingresso - len(self.items),
                "troncati": self.troncati,
                "prefix_mismatch": self.prefix_mismatch,
                "token_target_medi": round(sum(tt) / max(1, len(tt)), 1),
                "token_target_max": max(tt) if tt else 0}


def collate(batch, pad_id):
    """Aggiunge 'pesi' (float) accanto a input_ids/labels/attention_mask. Il
    forward del modello non lo vede mai: lo consuma il compute_loss del
    Trainer, e `punteggia` chiama il modello con gli argomenti espliciti."""
    L = max(len(b["input_ids"]) for b in batch)
    import torch
    interi = {"input_ids": [], "labels": [], "attention_mask": []}
    pesi = []
    for b in batch:
        d = L - len(b["input_ids"])
        interi["input_ids"].append(b["input_ids"] + [pad_id] * d)
        interi["labels"].append(b["labels"] + [-100] * d)
        interi["attention_mask"].append(b["attention_mask"] + [0] * d)
        pesi.append(b.get("pesi", [1.0] * len(b["input_ids"])) + [0.0] * d)
    out = {k: torch.tensor(v, dtype=torch.long) for k, v in interi.items()}
    out["pesi"] = torch.tensor(pesi, dtype=torch.float)
    return out


# --------------------------------------------------------------------------- #
# 5b. Punteggio in teacher forcing (nessuna generazione)
# --------------------------------------------------------------------------- #
# Da qui escono tre numeri che rispondono a domande diverse:
#   ppl_target  quanto il modello trova probabile ESATTAMENTE la continuazione
#               scritta nel JSON -> "sa completarla come sta scritta"
#   acc_token   quota di token di target indovinati al primo colpo
#   ctx_acc     quota di item in cui il target e' piu' probabile col contesto
#               vero che con un contesto preso da un altro punto della
#               conversazione -> "sta usando il contesto o lo ignora"
# 0,50 di ctx_acc significa contesto ignorato. Costa due forward per item e
# zero generazioni, quindi si puo' tenere accesa a ogni valutazione.

def sostituisci_contesto(rec: dict, ctx_nuovo: str) -> dict:
    """Copia dell'item con il blocco di contesto rimpiazzato. Serve al
    controfattuale di ctx_acc: tutto identico tranne il contesto."""
    _, prefisso = parse_item(rec)
    nuovo = dict(rec)
    testa = f"Conversazione finora:\n{ctx_nuovo}\n---\n" if ctx_nuovo else ""
    nuovo["prompt"] = f"{testa}{MARCA_ISTRUZIONE} {prefisso}"
    return nuovo


def punteggia(model, tok, records, stile="prefill", max_seq_len=512, batch=4):
    """{indice_originale: (logp_totale, n_token, n_corretti)} sui SOLI token di
    target. La chiave e' l'indice nella lista `records` in ingresso, non la
    posizione nel batch: se un item viene scartato (target interamente fuori da
    max_seq_len) il chiamante se ne accorge invece di leggere il vicino."""
    import torch
    ds = DatasetT2(records, tok, max_seq_len=max_seq_len, stile=stile)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out, was_training = [], model.training
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ds), batch):
            b = collate([ds[j] for j in range(i, min(i + batch, len(ds)))], pad)
            b = {k: v.to(model.device) for k, v in b.items()}
            logits = model(input_ids=b["input_ids"],
                           attention_mask=b["attention_mask"]).logits[:, :-1]
            tgt = b["labels"][:, 1:]
            maschera = tgt != -100
            # si selezionano le SOLE posizioni di target prima del softmax:
            # un log_softmax su [batch, 512, vocab] in fp32 sono centinaia di MB
            # buttati, e su T4 durante il training quella memoria non c'e'.
            lp = torch.log_softmax(logits[maschera].float(), dim=-1)
            t = tgt[maschera]
            scelto = lp.gather(-1, t.unsqueeze(-1)).squeeze(-1)
            corretto = (lp.argmax(-1) == t)
            k = 0
            for n in maschera.sum(1).tolist():
                out.append((float(scelto[k:k + n].sum()), int(n),
                            int(corretto[k:k + n].sum())))
                k += n
    if was_training:
        model.train()
    return dict(zip(ds.indici, out))


def metriche_teacher_forcing(model, tok, records, stile="prefill",
                             max_seq_len=512, batch=4, seed=0) -> dict:
    import math
    import random
    p = punteggia(model, tok, records, stile, max_seq_len, batch)
    tot_lp = sum(x[0] for x in p.values())
    tot_n = sum(x[1] for x in p.values())
    tot_ok = sum(x[2] for x in p.values())

    # controfattuale: contesto di un altro item, stessa conversazione, lontano
    con_ctx = [(i, r) for i, r in enumerate(records)
               if parse_item(r)[0].strip() and i in p]
    rng = random.Random(seed)
    falsi, indici = [], []
    for i, r in con_ctx:
        pool = [q for j, q in con_ctx
                if q["conversazione"] == r["conversazione"]
                and abs(q["turn_index"] - r["turn_index"]) > 20]
        if not pool:
            continue
        falsi.append(sostituisci_contesto(r, parse_item(rng.choice(pool))[0]))
        indici.append(i)
    ctx_acc, delta = None, None
    if falsi:
        pf = punteggia(model, tok, falsi, stile, max_seq_len, batch)
        coppie = [(p[indici[j]], q) for j, q in pf.items()]
        if coppie:
            ctx_acc = sum(1.0 for v, f in coppie if v[0] > f[0]) / len(coppie)
            delta = sum((v[0] - f[0]) / max(1, v[1]) for v, f in coppie) / len(coppie)
    return {"ppl_target": round(math.exp(-tot_lp / max(1, tot_n)), 3),
            "acc_token": round(tot_ok / max(1, tot_n), 4),
            "ctx_acc": None if ctx_acc is None else round(ctx_acc, 4),
            "delta_logp_per_token": None if delta is None else round(delta, 4),
            "n_item_ctx": len(falsi)}


# --------------------------------------------------------------------------- #
# 6. Lessico dialettale ricavato dal SOLO train
# --------------------------------------------------------------------------- #
# Non serve il CSV parallelo: qui interessa "quanto e' denso di forme
# napoletane il testo generato", non la corrispondenza it->nap. Il set si
# costruisce dai soli turni di train (target + contesti), quindi nessuna forma
# osservata solo in dev/test entra nella metrica.

SEME_DIALETTALE = {
    "'o", "'a", "'e", "'i", "'u", "'nu", "'na", "'sta", "'sto", "'stu",
    "nun", "ca", "pecche'", "pecché", "pe'", "accussi'", "accussì", "cchiu'",
    "cchiù", "quanno", "chesta", "chesto", "chestu", "chillo", "chella",
    "chelle", "chille", "aggio", "aggia", "songo", "so'", "sto'", "stongo",
    "tene", "tenimmo", "facimmo", "simmo", "vaco", "vene", "veneno", "mo",
    "addo'", "addò", "comm'", "comme", "nc'", "ce", "'ncopp'", "'ncoppa",
    "ll'", "ê", "â", "ô", "sti", "'e", "cu", "cu'", "pure", "doppo", "maje",
    "niente", "nisciuno", "nisciuna", "vabbuo'", "vabbuò", "quacche", "quaccosa",
}

PATTERN_DIALETTALE = [
    re.compile(r"^'[a-zàèéìòù]"),            # aferesi: 'o, 'a, 'nu, 'ncopp'
    re.compile(r"[âêîôû]"),                   # vocali con circonflesso: â, ê, ô
    re.compile(r"^(ff|cc|mm|nn|pp|tt|vv|zz|gg|bb|dd|ss|rr)[aeiouàèéìòù]"),  # rafforzamento
    re.compile(r"[a-z]{2,}(mmo|nno|tte|ppe|ate)$"),
]


def impalcatura() -> set[str]:
    """4-grammi di parole del testo di servizio (system, istruzione, etichette).
    Se compaiono nella generazione, il modello sta ricopiando il prompt: la
    ripulitura taglia li'."""
    testo = " ".join([SYSTEM, ISTRUZIONE, "Conversazione finora",
                      "Turno da completare", MARCA_ISTRUZIONE,
                      "Continua 'o turno 'e A", "Continua 'o turno 'e B"])
    w = testo.lower().split()
    return {" ".join(w[i:i + 4]) for i in range(len(w) - 3)}


def normalizza(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parole(s: str) -> list[str]:
    return re.findall(r"[a-zàèéìòùâêîôûA-Z']+", normalizza(s))


def costruisci_lessico(records_train: list[dict], min_freq: int = 2) -> set[str]:
    """Tipi dialettali = seme curato + forme del train che matchano i pattern
    ortografici tipici e compaiono almeno min_freq volte."""
    freq = Counter()
    for r in records_train:
        ctx, pref = parse_item(r)
        for w in parole(" ".join([ctx, pref, r["target"]])):
            freq[w] += 1
    tipi = set(SEME_DIALETTALE)
    for w, f in freq.items():
        if f >= min_freq and any(p.search(w) for p in PATTERN_DIALETTALE):
            tipi.add(w)
    return tipi


def densita_dialettale(testi: list[str], tipi: set[str]) -> float:
    tot = dial = 0
    for t in testi:
        ws = parole(t)
        tot += len(ws)
        dial += sum(1 for w in ws if w in tipi)
    return dial / max(1, tot)


# --------------------------------------------------------------------------- #
# 7. Metriche
# --------------------------------------------------------------------------- #

def chrf(ipotesi: list[str], riferimenti: list[str], beta: int = 2) -> float:
    import sacrebleu
    return sacrebleu.corpus_chrf(ipotesi, [riferimenti], word_order=2,
                                 beta=beta).score


def tasso_copia(ipotesi: list[str], prefissi: list[str]) -> float:
    """Quota di generazioni che ricopiano il prefisso invece di continuarlo.
    E' il fallimento degenere di questo task ed e' invisibile al chrF."""
    n = 0
    for h, p in zip(ipotesi, prefissi):
        wh, wp = set(parole(h)), set(parole(p))
        if not wh:
            continue
        if normalizza(h).startswith(normalizza(p)[:20]) or \
           (wp and len(wh & wp) / len(wh) >= 0.8):
            n += 1
    return n / max(1, len(ipotesi))


def tasso_ciclo(ipotesi: list[str], n: int = 3) -> float:
    """Quota di generazioni che contengono un n-gramma di parole ripetuto.
    E' il fallimento tipico della decodifica greedy su un modello addestrato su
    poche centinaia di esempi ("cioe', cioe', cioe', ..."), e il chrF++ lo
    nasconde: un ciclo di parole plausibili conserva buona parte dei caratteri
    del riferimento e perde solo qualche punto."""
    n_loop = 0
    for h in ipotesi:
        w = parole(h)
        g = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
        if len(g) != len(set(g)):
            n_loop += 1
    return n_loop / max(1, len(ipotesi))


def parole_attese(rec: dict) -> int:
    """Quante parole ci si aspetta dalla continuazione.

    Negli split ricostruiti da rebuild_t2_data.py il taglio e' MOBILE e la
    lunghezza attesa non e' piu' deducibile dal prefisso: il campo
    'n_parole_target' la porta esplicitamente. Sui file vecchi (taglio fisso a
    meta') il campo non c'e' e si ricade sulla vecchia stima. Il valore serve
    solo a fissare il budget di generazione, non entra in nessuna metrica.
    """
    n = rec.get("n_parole_target")
    if isinstance(n, int) and n > 0:
        return n
    return max(1, len(parse_item(rec)[1].split()))


def rapporto_lunghezza(ipotesi: list[str], riferimenti: list[str]) -> float:
    a = sum(len(parole(h)) for h in ipotesi)
    b = sum(len(parole(r)) for r in riferimenti)
    return a / max(1, b)


def riepilogo_metriche(ipotesi, riferimenti, prefissi, tipi) -> dict:
    return {
        # chrF di sacrebleu usa beta=2, cioe' pesa il richiamo il doppio della
        # precisione: allungare la generazione alza il punteggio quasi gratis
        # (con un output lungo 1,8 volte il riferimento ESATTO si perdono 13
        # punti con beta=2 e 28 con beta=1). Si riportano entrambi, altrimenti
        # un sistema logorroico sembra migliore di uno preciso.
        "chrf++": round(chrf(ipotesi, riferimenti), 2),
        "chrf++_beta1": round(chrf(ipotesi, riferimenti, beta=1), 2),
        "tasso_ciclo": round(tasso_ciclo(ipotesi), 4),
        "densita_dial_gen": round(densita_dialettale(ipotesi, tipi), 4),
        "densita_dial_rif": round(densita_dialettale(riferimenti, tipi), 4),
        "tasso_copia": round(tasso_copia(ipotesi, prefissi), 4),
        "rapporto_lunghezza": round(rapporto_lunghezza(ipotesi, riferimenti), 3),
        "vuote": sum(1 for h in ipotesi if not parole(h)),
    }


def baseline_taratura(records, tipi) -> dict:
    """Fondo scala e tetto, da leggere PRIMA di qualsiasi numero del modello.
    Su T2 la forbice del chrF++ e' strettissima: dirlo con i numeri evita di
    festeggiare un 15 che vale quanto il caso."""
    import random
    rif = [r["target"] for r in records]
    pref = [parse_item(r)[1] for r in records]
    mescolati = rif[:]
    random.Random(0).shuffle(mescolati)
    ult = []
    for r in records:
        t = turni_contesto(parse_item(r)[0])
        ult.append(t[-1][1] if t else "")
    return {
        "pavimento_target_mescolati": round(chrf(mescolati, rif), 2),
        "ripeti_il_prefisso": round(chrf(pref, rif), 2),
        "ultimo_turno_di_contesto": round(chrf(ult, rif), 2),
        "densita_dial_riferimenti": round(densita_dialettale(rif, tipi), 4),
        "densita_dial_prefissi": round(densita_dialettale(pref, tipi), 4),
    }
