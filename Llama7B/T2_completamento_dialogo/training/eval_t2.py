#!/usr/bin/env python3
"""
eval_t2.py — valuta UN sistema su T2, con lo stesso prompt visto in training.

Tre modalita', per avere un confronto e non un numero isolato:
    --adapter PATH   il modello fine-tuned
    --zero-shot      il modello di base, stesso prompt
    --few-shot 4     il modello di base con 4 esempi presi dal TRAIN

Produce due famiglie di numeri:
  A) teacher forcing, nessuna generazione: ppl_target, acc_token, ctx_acc.
     Rispondono a "sa completare il turno come sta scritto nel JSON".
  B) generazione reale: chrF++, densita' dialettale, tasso di copia, rapporto
     di lunghezza. Rispondono a "cosa produce davvero".

    python eval_t2.py --model minerva --split-dir SP --adapter A --split test
"""

import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import t2_common as C

DECODIFICA_USATA = ""     # riempita da genera(): puo' differire da quella
                          # chiesta se la ricerca contrastiva non e' disponibile

APERTURA_TURNO = re.compile(r"^([A-Z]{1,3})\s*:\s*")
RUMORE_INLINE = re.compile(r"\s*(?:---+|Conversazione finora|Continua il turno|"
                           r"Turno da completare).*$")


def ripulisci(testo: str, prefisso: str, impalcatura=frozenset()) -> str:
    """La generazione grezza puo' aprire un turno nuovo, ricopiare il prefisso o
    rigurgitare pezzi di prompt. Il target e' sempre una riga sola: si tiene la
    prima riga e le si tolgono i quattro rumori tipici, in quest'ordine."""
    t = testo.strip().split("\n")[0].strip()
    t = RUMORE_INLINE.sub("", t).strip()
    m = APERTURA_TURNO.match(t)           # "B: ..." -> ha aperto un turno nuovo
    if m:
        t = t[m.end():]
    if prefisso and t.lower().startswith(prefisso.lower()):   # ha ricopiato
        t = t[len(prefisso):]             # lower() non cambia la lunghezza
    # fuga di impalcatura: la generazione prosegue con il testo di servizio del
    # prompt ("...napulitano, senza ricumincia' da capo"). Il regex sopra prende
    # solo gli inizi di riga noti; qui si taglia al primo 4-gramma che compare
    # nel testo di servizio, ovunque si trovi.
    if impalcatura:
        w = t.split()
        for i in range(max(0, len(w) - 3)):
            if " ".join(w[i:i + 4]).lower() in impalcatura:
                t = " ".join(w[:i])
                break
    return t.strip(" \t-–—")


def taglia_a_parole(testo: str, n: int) -> str:
    p = testo.split()
    return " ".join(p[:max(1, n)])


def esempi_few_shot(train, k, stile, seed=0):
    import random
    rng = random.Random(seed)
    scelti = rng.sample([r for r in train if C.parse_item(r)[0].strip()], k)
    msgs = []
    for r in scelti:
        # con_istruzione=False: negli esempi l'istruzione NON va ripetuta, o
        # diventa il testo piu' probabile del prompt e il modello ricopia quella
        u = C.messaggi_utente(r, stile, con_istruzione=False)[1]
        _, pref = C.parse_item(r)
        risposta = (f"{pref} {r['target']}" if stile == "prefill" else r["target"])
        msgs += [u, {"role": "assistant", "content": risposta}]
    return msgs


def genera(model, tok, records, stile, few_shot_msgs, max_seq_len, batch=16,
           decodifica="contrastiva", temperature=0.8, top_p=0.9, rep_pen=1.0,
           no_repeat=4, cap=48, seed=0, impalcatura=frozenset(),
           penalty_alpha=0.6, top_k=4, frazione_min=0.5):
    """Genera su tutti gli item, con barra di avanzamento.

    Due accorgimenti che qui valgono un fattore ~4 sul tempo totale, e su una
    T4 con un 7B a 4 bit la differenza fra sette minuti e mezz'ora:

    1. gli item vengono ORDINATI per lunghezza del prompt prima di essere
       raggruppati. `generate` gira per `max(budget)` passi su tutto il batch,
       quindi un batch che mescola un prefisso da 3 token con uno da 24 paga il
       budget del piu' lungo per tutti;
    2. il batch di default e' 16. Il modello a 4 bit occupa ~4,5 GB su 15,6: la
       memoria per allargare il batch c'e', e la generazione e' limitata dai
       passi sequenziali, non dalla larghezza.
    """
    import time
    try:
        from tqdm.auto import tqdm
    except ImportError:                      # senza tqdm si va avanti lo stesso
        def tqdm(x, **k):
            return x

    tok.padding_side = "left"
    model.eval()
    torch.manual_seed(seed)

    prompts, prefissi, budget, scaffold = [], [], [], []
    for r in records:
        _, pref = C.parse_item(r)
        msgs = C.messaggi_utente(r, stile)
        if few_shot_msgs:
            msgs = [msgs[0]] + few_shot_msgs + [msgs[1]]
        testa = C._template(tok, msgs)
        prompts.append(testa + pref if stile == "prefill" else testa)
        prefissi.append(pref)
        # impalcatura specifica di QUESTO item: le righe di servizio del suo
        # messaggio utente (istruzione, etichette), non il contesto. Ricavarla
        # dal prompt reale invece che da una lista fissa la rende indipendente
        # da come e' formulata l'istruzione.
        serv = msgs[-1]["content"].split("---")[-1] + " " + C.SYSTEM
        ws = serv.lower().split()
        scaffold.append(impalcatura | {" ".join(ws[k:k + 4])
                                       for k in range(max(0, len(ws) - 3))})
        # Il budget si deriva dalle PAROLE attese, non da un multiplo a caso
        # dei token del prefisso. Il taglio del turno e' a meta' esatta: se il
        # prefisso ha k parole, il target ne ha k o k+1. Si converte in token
        # col rapporto token/parola misurato sul prefisso stesso, che e' lo
        # stesso testo, stessa lingua, stesso tokenizer.
        #
        # Perche' conta: un budget largo non serve a "dare spazio", serve solo
        # a dare al modello altri trenta passi in cui avvitarsi in un ciclo
        # ("cioe', cioe', cioe', ..."). Stringerlo dimezza anche il tempo.
        n_par = C.parole_attese(r)
        n_tok = len(tok(pref, add_special_tokens=False)["input_ids"])
        rapporto = max(1.0, n_tok / max(1, len(pref.split())))
        budget.append(max(6, min(cap, int(rapporto * (n_par + 1)) + 4)))

    ordine = sorted(range(len(records)), key=lambda i: len(prompts[i]))
    fuori = [""] * len(records)

    # generation_config esplicita: senza, transformers eredita temperature/top_p
    # dal generation_config del modello e con do_sample=False stampa a ogni
    # chiamata "the following generation flags are not valid"
    from transformers import GenerationConfig
    base = dict(repetition_penalty=rep_pen, no_repeat_ngram_size=no_repeat,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    # La ricerca contrastiva e' il default: su un modello adattato con poche
    # centinaia di esempi la greedy degenera (frammenti memorizzati dal train,
    # ripetizioni), e penalty_alpha penalizza proprio i token la cui
    # rappresentazione e' troppo simile a quelle gia' emesse. Costa circa il
    # 30% in piu' della greedy e niente in memoria.
    if decodifica == "contrastiva":
        modo = dict(do_sample=False, penalty_alpha=penalty_alpha, top_k=top_k)
    elif decodifica == "campionamento":
        modo = dict(do_sample=True, temperature=temperature, top_p=top_p)
    elif decodifica == "beam":
        modo = dict(do_sample=False, num_beams=4, length_penalty=0.8,
                    early_stopping=True)
    else:                                    # greedy
        modo = dict(do_sample=False, num_beams=1)

    # Da transformers 4.56 la ricerca contrastiva non e' piu' nel core: vive
    # in un repo di codice remoto e va chiesta con custom_generate +
    # trust_remote_code. La versione installata su Kaggle cambia senza
    # preavviso e il repo puo' non essere scaricabile, quindi invece di
    # indovinare si prova: una generazione da un token, tre esiti possibili.
    extra = {}
    if decodifica == "contrastiva":
        sonda = tok(["ciao"], return_tensors="pt").to(model.device)
        prova = GenerationConfig(max_new_tokens=1, pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id, **modo)
        ultimo = None
        for etichetta, kw in (
                ("nel core di transformers", {}),
                ("via custom_generate", {
                    "custom_generate": "transformers-community/contrastive-search",
                    "trust_remote_code": True})):
            try:
                with torch.no_grad():
                    model.generate(**sonda, generation_config=prova, **kw)
            except Exception as e:
                ultimo = e
                continue
            extra = kw
            print(f"ricerca contrastiva: {etichetta}", flush=True)
            break
        else:
            print(f"! ricerca contrastiva non disponibile "
                  f"({type(ultimo).__name__}: {str(ultimo)[:120]}).\n"
                  f"  Ripiego su beam search, che e' il sostituto piu' vicino "
                  f"contro la degenerazione. Il tag del run resta quello "
                  f"chiesto: la decodifica EFFETTIVA finisce in "
                  f"metrics.json sotto 'decodifica'.", flush=True)
            decodifica = "beam"
            modo = dict(do_sample=False, num_beams=4, length_penalty=0.8,
                        early_stopping=True)

    global DECODIFICA_USATA
    DECODIFICA_USATA = decodifica

    # La ricerca contrastiva replica la cache KV top_k volte
    # (batch_repeat_interleave), quindi il batch effettivo in memoria e' top_k
    # volte quello che si chiede. Con batch=16 e top_k=4 diventano 64 sequenze e
    # su una T4 la generazione muore per OOM a meta' del corpus, dopo che la
    # sonda da un token era passata senza problemi.
    if decodifica == "contrastiva":
        batch = max(1, batch // top_k)
        print(f"batch ridotto a {batch}: la ricerca contrastiva replica la "
              f"cache KV {top_k} volte", flush=True)
    if decodifica == "beam":
        batch = max(1, batch // 2)             # 4 beam per sequenza

    def _genera_blocco(idx):
        """Un blocco di item. Restituisce False se e' finita la memoria: chi
        chiama dimezza e riprova, invece di far fallire tutta la valutazione
        per un solo batch sfortunato."""
        mx = max(budget[i] for i in idx)
        # min_new_tokens derivato dalla lunghezza attesa invece che fisso a 2:
        # con 2 il modello puo' chiudere subito e il rapporto di lunghezza
        # crolla (0,64 nella versione precedente). I batch sono ordinati per
        # lunghezza, quindi dentro un batch i budget sono simili.
        mn = max(2, min(mx - 1, int(min(budget[i] for i in idx) * frazione_min)))
        try:
            enc = tok([prompts[i] for i in idx], return_tensors="pt",
                      padding=True, truncation=True, max_length=max_seq_len,
                      add_special_tokens=False).to(model.device)
            gc = GenerationConfig(max_new_tokens=mx, min_new_tokens=mn,
                                  **base, **modo)
            with torch.no_grad():
                out = model.generate(**enc, generation_config=gc, **extra)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return False
        nuovi = out[:, enc["input_ids"].shape[1]:]
        for k, i in enumerate(idx):
            testo = tok.decode(nuovi[k][:budget[i]], skip_special_tokens=True)
            fuori[i] = ripulisci(testo, prefissi[i], scaffold[i])
        del enc, out, nuovi
        return True

    t0, fatti = time.time(), 0
    blocchi = list(range(0, len(ordine), batch))
    barra = tqdm(blocchi, desc="generazione", unit="batch")
    dimezzati = 0
    for s in barra:
        coda = [ordine[s:s + batch]]
        while coda:
            gruppo = coda.pop(0)
            if _genera_blocco(gruppo):
                fatti += len(gruppo)
                continue
            if len(gruppo) == 1:
                # un solo item non entra in memoria: si rinuncia a quello,
                # non a tutta la valutazione. Finira' fra le "vuote".
                print(f"\n! OOM anche a batch 1 sull'item {gruppo[0]}: saltato",
                      flush=True)
                fatti += 1
                continue
            meta = len(gruppo) // 2
            coda[:0] = [gruppo[:meta], gruppo[meta:]]
            dimezzati += 1
        if hasattr(barra, "set_postfix"):
            trascorso = time.time() - t0
            barra.set_postfix(item=f"{fatti}/{len(records)}",
                              stima=f"{trascorso/fatti*len(records)/60:.1f} min")
    if dimezzati:
        print(f"! {dimezzati} batch dimezzati per memoria. Se capita spesso, "
              f"abbassa --batch: il tempo perso a riprovare supera quello "
              f"guadagnato col batch largo.", flush=True)
    print(f"generazione: {len(records)} item in "
          f"{(time.time()-t0)/60:.1f} minuti", flush=True)
    tok.padding_side = "right"
    return fuori


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="minerva")
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "bf16", "fp16", "fp32"],
                    help="deve coincidere con quello del training")
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--adapter", default="")
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--few-shot", type=int, default=0)
    ap.add_argument("--stile", default="prefill", choices=["prefill", "chat"])
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16,
                    help="batch di GENERAZIONE. 16 sta in una T4 con un 7B a "
                         "4 bit; abbassalo solo se va in OOM")
    ap.add_argument("--cap", type=int, default=48,
                    help="tetto assoluto ai token generati per item")
    ap.add_argument("--no-repeat", type=int, default=4,
                    help="blocca gli n-grammi ripetuti in decodifica. 4 spezza "
                         "i cicli senza vietare le ripetizioni brevi, che nel "
                         "parlato spontaneo sono normali. 0 = spento")
    ap.add_argument("--decodifica", default="contrastiva",
                    choices=["contrastiva", "greedy", "beam", "campionamento"],
                    help="contrastiva = default. greedy riproduce la versione "
                         "precedente")
    ap.add_argument("--penalty-alpha", type=float, default=0.6)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--frazione-min", type=float, default=0.5,
                    help="min_new_tokens = questa frazione del budget atteso")
    ap.add_argument("--distrattori", type=int, default=9,
                    help="candidati falsi per l'accuratezza di scelta")
    ap.add_argument("--salta-scelta", action="store_true")
    ap.add_argument("--sampling", action="store_true",
                    help="alias storico di --decodifica campionamento")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="1.0 = spenta. Sul napoletano i clitici ('o, 'e, ca) "
                         "si ripetono per forza: penalizzarli e' un danno")
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="/kaggle/working/eval")
    ap.add_argument("--salta-teacher-forcing", action="store_true")
    a = ap.parse_args()
    if a.sampling:
        a.decodifica = "campionamento"

    repo = C.MODEL_REGISTRY.get(a.model, a.model)
    nome_prec, dtype = C.scegli_dtype(repo, a.dtype)
    print(f"precisione: {nome_prec}")

    train = C.load_split(a.split_dir, "train")
    dati = C.load_split(a.split_dir, a.split)
    if a.limite:
        dati = dati[:a.limite]
    tipi = C.costruisci_lessico(train)        # lessico dal SOLO train

    tok = AutoTokenizer.from_pretrained(a.adapter or repo,
                                        token=os.environ.get("HF_TOKEN"))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=dtype)
    attn = C.attn_impl(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, quantization_config=bnb, device_map={"": 0},
        attn_implementation=attn, token=os.environ.get("HF_TOKEN"),
        **C.kw_dtype(dtype))
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
        etichetta = "fine-tuned"
    elif a.few_shot:
        etichetta = f"few-shot-{a.few_shot}"
    else:
        etichetta = "zero-shot"
    model.config.use_cache = True             # generazione: la cache serve
    model.eval()

    tag = a.tag or f"{a.model}__T2__{etichetta}__{a.split}"
    print(f"=== {tag} | {len(dati)} item ===", flush=True)

    ris = {"tag": tag, "modello": repo, "sistema": etichetta, "split": a.split,
           "stile": a.stile, "decodifica": a.decodifica, "n": len(dati)}

    # --- A. teacher forcing ----------------------------------------------
    if not a.salta_teacher_forcing:
        ris["teacher_forcing"] = C.metriche_teacher_forcing(
            model, tok, dati, a.stile, a.max_seq_len, batch=2)
        print("teacher forcing:", json.dumps(ris["teacher_forcing"], indent=2),
              flush=True)

    # --- A2. accuratezza di scelta ---------------------------------------
    # La domanda cambia: non "quanto assomiglia il testo generato al
    # riferimento" (chrF++, forbice di 2,8 punti fra caso e baseline banale)
    # ma "fra la continuazione vera e N-1 prese da altri turni, sceglie
    # quella giusta". Fondo scala esatto 1/N, nessuna generazione.
    if not a.salta_scelta:
        try:
            import metrica_scelta as MS
            ris["scelta"] = MS.accuratezza_scelta(
                model, tok, dati, a.stile, a.max_seq_len,
                n_distrattori=a.distrattori, batch=2)
            print("scelta fra candidati:",
                  json.dumps(ris["scelta"], indent=2), flush=True)
        except Exception as e:
            print("! metrica di scelta non calcolata:", e, flush=True)

    # --- B. generazione ---------------------------------------------------
    fs = esempi_few_shot(train, a.few_shot, a.stile) if a.few_shot else []
    ipotesi = genera(model, tok, dati, a.stile, fs, a.max_seq_len,
                     batch=a.batch, decodifica=a.decodifica,
                     temperature=a.temperature, top_p=a.top_p,
                     rep_pen=a.repetition_penalty, cap=a.cap,
                     no_repeat=a.no_repeat, impalcatura=C.impalcatura(),
                     penalty_alpha=a.penalty_alpha, top_k=a.top_k,
                     frazione_min=a.frazione_min)
    rif = [r["target"] for r in dati]
    pref = [C.parse_item(r)[1] for r in dati]

    ris["decodifica"] = DECODIFICA_USATA or a.decodifica
    if ris["decodifica"] != a.decodifica:
        print(f"! decodifica chiesta '{a.decodifica}', usata "
              f"'{ris['decodifica']}'", flush=True)
    ris["generazione"] = C.riepilogo_metriche(ipotesi, rif, pref, tipi)
    # variante a lunghezza controllata: si taglia alla lunghezza del PREFISSO,
    # che a inferenza e' nota (non e' un oracolo sul riferimento)
    tagliate = [taglia_a_parole(h, C.parole_attese(r))
                for h, r in zip(ipotesi, dati)]
    ris["generazione_lunghezza_controllata"] = C.riepilogo_metriche(
        tagliate, rif, pref, tipi)
    # solo riferimenti umani
    idx = [i for i, r in enumerate(dati) if r["fonte"] == "golden"]
    if idx and len(idx) < len(dati):
        ris["generazione_solo_golden"] = C.riepilogo_metriche(
            [ipotesi[i] for i in idx], [rif[i] for i in idx],
            [pref[i] for i in idx], tipi)
    ris["baseline"] = C.baseline_taratura(dati, tipi)

    # Le due righe vanno lette INSIEME. La lunghezza controllata taglia la
    # generazione alle parole del prefisso, che a inferenza sono note: e' il
    # confronto onesto, perche' chrF con beta=2 premia il richiamo e quindi
    # regala punti a chi scrive lungo.
    print("\ngenerazione (libera)      :",
          json.dumps(ris["generazione"], indent=2), flush=True)
    print("generazione (lungh. contr.):",
          json.dumps(ris["generazione_lunghezza_controllata"], indent=2))
    if "generazione_solo_golden" in ris:
        print("solo riferimenti golden    :",
              json.dumps(ris["generazione_solo_golden"], indent=2))
    print("baseline                   :", json.dumps(ris["baseline"], indent=2))
    g = ris["generazione"]
    if g["tasso_ciclo"] > 0.15:
        print(f"\n! il {100*g['tasso_ciclo']:.0f}% delle generazioni contiene un "
              "ciclo di ripetizione. E' un problema di DECODIFICA, non di "
              "addestramento: guarda acc_token qui sopra prima di concludere "
              "che il modello non ha imparato.")
    if g["rapporto_lunghezza"] > 1.3:
        print(f"! generazioni lunghe {g['rapporto_lunghezza']:.2f} volte i "
              "riferimenti: il chrf++ libero e' gonfiato, usa la riga a "
              "lunghezza controllata.")

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"{tag}.preds.jsonl"), "w",
              encoding="utf-8") as f:
        for r, h, t, p in zip(dati, ipotesi, tagliate, pref):
            f.write(json.dumps({"id": r["id"], "speaker": r["speaker"],
                                "fonte": r["fonte"], "contesto": C.parse_item(r)[0],
                                "prefisso": p, "riferimento": r["target"],
                                "generato": h, "generato_tagliato": t},
                               ensure_ascii=False) + "\n")
    with open(os.path.join(a.out_dir, f"{tag}.metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(ris, f, ensure_ascii=False, indent=2)

    print("\n--- primi 5 (GEN = libera, TAG = tagliata alle parole attese) ---")
    for r, h, t in list(zip(dati, ipotesi, tagliate))[:5]:
        print(f"  ...{C.parse_item(r)[1][-40:]}")
        print(f"     RIF: {r['target']}")
        print(f"     GEN: {h}")
        print(f"     TAG: {t}\n")
    print("scritto in", a.out_dir)


if __name__ == "__main__":
    main()
