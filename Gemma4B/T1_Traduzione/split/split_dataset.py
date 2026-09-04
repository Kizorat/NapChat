#!/usr/bin/env python3
"""
Fase 1.2d + 1.3 + 1.4 — Esclusione turni inglesi, generazione dei 3 layout
(T1/T2/T3) e congelamento dello split train/test, a partire da dataset_finale
(l'output gia' arbitrato di scripts/build_final_dataset.py).

Il file dataset_finale/dataset_finale.json contiene ancora i 6 turni con
sorgente inglese (§1.2d della pipeline non era mai stata applicata a questo
file): questo script li esclude PRIMA di generare qualunque layout, cosi' non
finiscono ne' come esempio ne' come contesto di un altro esempio.

Per ciascuna conversazione (KPN001, KPN003) i turni superstiti vengono divisi,
in base alla loro POSIZIONE nella conversazione (non al turn_index grezzo, che
dopo l'esclusione ha dei buchi), in tre blocchi contigui train/dev/test con le
proporzioni --train-frac/--dev-frac/--test-frac (default 70/15/15):

    [-------- train --------][buf][-- dev --][buf][---- test ----]

Tra train/dev e tra dev/test viene scartato un piccolo blocco cuscinetto
(--buffer turni, default 4): serve a evitare che un turno di dev o di test
compaia come CONTESTO di un esempio di train o di dev (il contesto di T1/T2/T3
e le coppie adiacenti di T3 non attraversano mai un confine di blocco).

Le proporzioni si applicano separatamente a ciascuna conversazione (non al
totale KPN001+KPN003 unito), cosi' sia KPN001 sia KPN003 contribuiscono a
train/dev/test nella stessa proporzione.

Output, dentro <out-dir> (default: la cartella di questo script), una
sottocartella per layout con dentro train/dev/test .json/.csv:

    layout1_traduzione_con_contesto/
    layout2_completamento_turno/
    layout3_replica_conversazionale/

--------------------------------------------------------------------------
REVISIONE (due modifiche rispetto alla versione precedente)
--------------------------------------------------------------------------

1. T2 RICEVE IL CONTESTO CONVERSAZIONALE, come T1 e T3.

   Prima il prompt di T2 era solo "Continua il turno in napoletano: {prefix}",
   senza nessun contesto: era l'unica asimmetria fra T2 e gli altri due layout.
   Senza contesto il task e' quasi una lotteria - dato
       "no, è comm'a quanno pigli nu cioccolatino,"
   nessun parlante, madrelingua compreso, puo' indovinare
       "'o pigli e t' 'o magne sano".
   Il contesto e' cio' che vincola le continuazioni plausibili.

   ATTENZIONE: i prompt di T2 diventano lunghi come quelli di T1, quindi in
   fase di training serve max_seq_len 512 (non 192) o i target vengono
   troncati - e un target troncato viene trattato dalla loss come corretto.

2. T3 HA UNA SOGLIA SEPARATA PER IL TURNO DI CONTESTO.

   Prima la coppia (turno_precedente, turno_target) veniva scartata se UNO DEI
   DUE era piu' corto di --min-words-t3. Ma il turno precedente serve solo come
   contesto: scartare una coppia perche' il turno prima era un "mh" butta via
   dati utili. Ora la soglia sul target resta --min-words-t3 (default 3, cosi'
   non si addestra a produrre backchannel) mentre quella sul contesto e'
   --min-words-t3-ctx (default 1).

   Effetto misurato: T3 passa da 450 a 748 istanze di train e da 137 a 192 di
   test, il che restringe anche gli intervalli di confidenza in valutazione.

I punteggi dei tre layout NON sono confrontabili fra loro: hanno tetti metrici
diversi. Baseline misurate sul dev (chrF++): T1 copia-dell'italiano 35.92,
T2 ripeti-il-prefisso 11.67, T3 replica-da-un-altro-punto 8.83. Il confronto
valido e' fra modelli a layout fissato.

Uso tipico (dalla radice del progetto):
    py -3.12 split/split_dataset.py
    py -3.12 split/split_dataset.py --train-frac 0.8 --dev-frac 0.1 --test-frac 0.1
    py -3.12 split/split_dataset.py --min-words-t3-ctx 3   # comportamento precedente di T3
"""

import argparse, csv, json, os, sys

DEFAULT_INPUT = os.path.join("dataset_finale", "dataset_finale.json")
DEFAULT_OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- §1.2d: i 6 turni con sorgente italiano non italiana (in realta' inglese).
# Elenco preso da pipeline_progetto_v5.md; nessun file english_turns_da_rivedere.txt
# e' presente nel repo, quindi la lista e' hardcoded qui (unica fonte di verita').
EXCLUDED_ENGLISH_TURNS = {
    ("KPN001", 18),   # "la mia il mio thinking behind it was"
    ("KPN001", 824),  # "a trovare altre risorse let me know"
    ("KPN001", 826),  # "what is the approach"
    ("KPN001", 899),  # "what do you mean tu tu tu ru tu"
    ("KPN001", 904),  # "why"
    ("KPN003", 66),   # "never stop dreaming"
}
# NOTA: KPN003 #345 ("fish and chips", prestito lessicale come nome di piatto)
# NON va escluso per policy (§1.2d) — resta nel dataset.

LAYOUT_DIRS = {
    "T1": "layout1_traduzione_con_contesto",
    "T2": "layout2_completamento_turno",
    "T3": "layout3_replica_conversazionale",
}

FIELDS = ["id", "layout", "split", "conversazione", "turn_index", "speaker",
          "speaker_id", "prompt", "target", "fonte", "similarita"]


def load_samples(path):
    raw = open(path, encoding="utf-8").read().strip()
    obj = json.loads(raw)
    return obj if isinstance(obj, list) else [obj]


def nwords(text):
    return len(text.split())


def exclude_english(samples):
    """Rimuove i turni di EXCLUDED_ENGLISH_TURNS. Ritorna (conv_id -> [turni]), conteggio."""
    per_conv, rimossi = {}, 0
    for s in samples:
        turni = [t for t in s["turni"] if (s["id"], t["turn_index"]) not in EXCLUDED_ENGLISH_TURNS]
        rimossi += len(s["turni"]) - len(turni)
        per_conv[s["id"]] = turni
    return per_conv, rimossi


def split_pools(per_conv, train_frac, dev_frac, test_frac, buffer_turns):
    """Per ogni conversazione, divide i turni superstiti (in ordine) in tre
    blocchi contigui train/dev/test per POSIZIONE, con un buffer scartato ad
    ogni confine interno. Ritorna conv_id -> {"train"/"dev"/"test": [...]}."""
    pools = {}
    for conv, turni in per_conv.items():
        n = len(turni)
        n_test = round(n * test_frac)
        n_dev = round(n * dev_frac)
        n_train = n - n_test - n_dev          # il resto, cosi' non si perdono/duplicano turni per arrotondamento

        train_end = max(0, n_train - buffer_turns)          # confine train | buffer1 | dev
        dev_end = max(train_end, n_train + n_dev - buffer_turns)  # confine dev | buffer2 | test

        pools[conv] = {
            "train": turni[0:train_end],
            "dev": turni[n_train:dev_end],
            "test": turni[n_train + n_dev:n],
        }
    return pools


def anonymize_map(turni):
    """speaker_id -> 'A'/'B'/'C'... per ordine di prima apparizione nella conversazione."""
    labels, mapping = "ABCDEFGH", {}
    for t in turni:
        sp = t["speaker"]
        if sp not in mapping:
            mapping[sp] = labels[len(mapping) % len(labels)]
    return mapping


def render_context(ctx, spk_map):
    """Blocco di contesto condiviso da T1, T2 e T3: stesso formato byte per byte
    su tutti i layout, cosi' il modello vede una sola convenzione."""
    if not ctx:
        return ""
    righe = "\n".join(f"{spk_map[c['speaker']]}: {c['napoletano']}" for c in ctx)
    return f"Conversazione finora:\n{righe}\n---\n"


def gen_t1(conv, pool, spk_map, ctx_window, min_words):
    """Layout 1 — traduzione con contesto: contesto = napoletano dei turni
    precedenti nello stesso pool (finestra di ctx_window turni)."""
    out = []
    for i, t in enumerate(pool):
        if nwords(t["italiano"]) < min_words:
            continue
        prompt = render_context(pool[max(0, i - ctx_window):i], spk_map)
        prompt += f"Traduci in napoletano: {t['italiano']}"
        out.append({
            "id": f"{conv}_T1_{t['turn_index']}", "layout": "T1",
            "conversazione": conv, "turn_index": t["turn_index"],
            "speaker": spk_map[t["speaker"]], "speaker_id": t["speaker"],
            "prompt": prompt, "target": t["napoletano"],
            "fonte": t.get("fonte", ""), "similarita": t.get("similarita", ""),
        })
    return out


def gen_t2(conv, pool, spk_map, ctx_window, min_words):
    """Layout 2 — completamento di turno: taglio al 50% delle parole napoletane,
    con il contesto conversazionale dei turni precedenti (finestra ctx_window).

    Il contesto e' la modifica principale rispetto alla versione precedente:
    senza di esso la seconda meta' del turno non e' determinata da nulla di
    osservabile e il task misura fortuna, non competenza."""
    out = []
    for i, t in enumerate(pool):
        words = t["napoletano"].split()
        if len(words) < min_words:
            continue
        cut = len(words) // 2
        prefix, resto = " ".join(words[:cut]), " ".join(words[cut:])
        prompt = render_context(pool[max(0, i - ctx_window):i], spk_map)
        prompt += f"Continua il turno in napoletano: {prefix}"
        out.append({
            "id": f"{conv}_T2_{t['turn_index']}", "layout": "T2",
            "conversazione": conv, "turn_index": t["turn_index"],
            "speaker": spk_map[t["speaker"]], "speaker_id": t["speaker"],
            "prompt": prompt, "target": resto,
            "fonte": t.get("fonte", ""), "similarita": t.get("similarita", ""),
        })
    return out


def gen_t3(conv, pool, spk_map, ctx_window, min_words, min_words_ctx):
    """Layout 3 — replica conversazionale: coppie adiacenti nello stesso pool,
    cambio di parlante.

    Due soglie distinte, non una: il TARGET deve avere >= min_words parole
    (cosi' non si addestra a produrre backchannel), il turno di CONTESTO solo
    >= min_words_ctx. Con un'unica soglia si scartavano coppie perfettamente
    utili solo perche' il turno precedente era un "mh"."""
    out = []
    for i in range(len(pool) - 1):
        cur, nxt = pool[i], pool[i + 1]
        if cur["speaker"] == nxt["speaker"]:
            continue
        if nwords(cur["napoletano"]) < min_words_ctx:
            continue
        if nwords(nxt["napoletano"]) < min_words:
            continue
        prompt = render_context(pool[max(0, i - ctx_window + 1):i + 1], spk_map)
        prompt += f"Rispondi come {spk_map[nxt['speaker']]} in napoletano:"
        out.append({
            "id": f"{conv}_T3_{nxt['turn_index']}", "layout": "T3",
            "conversazione": conv, "turn_index": nxt["turn_index"],
            "speaker": spk_map[nxt["speaker"]], "speaker_id": nxt["speaker"],
            "prompt": prompt, "target": nxt["napoletano"],
            "fonte": nxt.get("fonte", ""), "similarita": nxt.get("similarita", ""),
        })
    return out


def write_layout(rows, split_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for r in rows:
        r["split"] = split_name
    json_path = os.path.join(out_dir, f"{split_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    csv_path = os.path.join(out_dir, f"{split_name}.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return json_path, csv_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help=f"dataset_finale di input (default: {DEFAULT_INPUT})")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="cartella dentro cui creare le sottocartelle per layout "
                         "(default: la cartella di questo script)")
    ap.add_argument("--context-window", type=int, default=3,
                    help="turni di contesto precedenti per T1/T2/T3 (default: 3, "
                         "come nell'esempio di §1.3 della pipeline)")
    ap.add_argument("--min-words-t1", type=int, default=3)
    ap.add_argument("--min-words-t2", type=int, default=6)
    ap.add_argument("--min-words-t3", type=int, default=3,
                    help="parole minime nel TARGET di T3 (default: 3)")
    ap.add_argument("--min-words-t3-ctx", type=int, default=1,
                    help="parole minime nel turno di CONTESTO di T3 (default: 1). "
                         "Metti 3 per riprodurre il comportamento precedente, in cui "
                         "la soglia era la stessa per target e contesto")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--buffer", type=int, default=4,
                    help="turni cuscinetto scartati ad ogni confine train/dev e dev/test (default: 4)")
    args = ap.parse_args()

    if abs(args.train_frac + args.dev_frac + args.test_frac - 1.0) > 1e-6:
        sys.exit(f"ERRORE: le tre frazioni devono sommare a 1.0 "
                 f"(train {args.train_frac} + dev {args.dev_frac} + test {args.test_frac})")
    if args.min_words_t3_ctx > args.min_words_t3:
        print(f"! --min-words-t3-ctx ({args.min_words_t3_ctx}) e' maggiore della soglia sul "
              f"target ({args.min_words_t3}): stai filtrando piu' sul contesto che su cio' "
              f"che il modello deve produrre.", file=sys.stderr)

    if not os.path.exists(args.input):
        sys.exit(f"ERRORE: input non trovato: {args.input!r}\n"
                 f"Generalo con:  py -3.12 scripts/build_final_dataset.py")

    samples = load_samples(args.input)
    tot_prima = sum(len(s["turni"]) for s in samples)

    per_conv, rimossi = exclude_english(samples)
    print(f"Turni caricati da {args.input}: {tot_prima}")
    print(f"Esclusi per sorgente inglese (par. 1.2d): {rimossi}")

    pools = split_pools(per_conv, args.train_frac, args.dev_frac, args.test_frac, args.buffer)
    for conv, p in pools.items():
        tot = len(per_conv[conv])
        buf_tot = tot - len(p["train"]) - len(p["dev"]) - len(p["test"])
        print(f"  {conv}: {tot} turni superstiti -> train {len(p['train'])}, "
              f"dev {len(p['dev'])}, test {len(p['test'])}  (buffer scartato: {buf_tot})")

    layouts = {"T1": {"train": [], "dev": [], "test": []},
               "T2": {"train": [], "dev": [], "test": []},
               "T3": {"train": [], "dev": [], "test": []}}

    for conv, p in pools.items():
        full_conv_turni = per_conv[conv]
        spk_map = anonymize_map(full_conv_turni)
        for split_name in ("train", "dev", "test"):
            pool = p[split_name]
            layouts["T1"][split_name] += gen_t1(
                conv, pool, spk_map, args.context_window, args.min_words_t1)
            layouts["T2"][split_name] += gen_t2(
                conv, pool, spk_map, args.context_window, args.min_words_t2)
            layouts["T3"][split_name] += gen_t3(
                conv, pool, spk_map, args.context_window, args.min_words_t3,
                args.min_words_t3_ctx)

    print("\nIstanze generate:")
    agg = {"train": 0, "dev": 0, "test": 0}
    for layout, dirname in LAYOUT_DIRS.items():
        out_dir = os.path.join(args.out_dir, dirname)
        counts = {}
        for split_name in ("train", "dev", "test"):
            write_layout(layouts[layout][split_name], split_name, out_dir)
            counts[split_name] = len(layouts[layout][split_name])
            agg[split_name] += counts[split_name]
        tot = sum(counts.values())
        print(f"  {layout} ({dirname}): train {counts['train']}  |  dev {counts['dev']}  |  "
              f"test {counts['test']}  |  totale {tot}")
        print(f"    -> {out_dir}{os.sep}{{train,dev,test}}.json (+.csv)")

    # lunghezza dei prompt: serve a scegliere max_seq_len in training senza
    # indovinare. Con il contesto attivo su tutti e tre i layout, T2 non sta
    # piu' in 192 token.
    print("\nLunghezza dei prompt (parole, sul train):")
    for layout in ("T1", "T2", "T3"):
        rows = layouts[layout]["train"]
        if not rows:
            continue
        L = sorted(nwords(r["prompt"]) + nwords(r["target"]) for r in rows)
        p95 = L[int(0.95 * (len(L) - 1))]
        print(f"  {layout}: mediana {L[len(L) // 2]:3d}  p95 {p95:3d}  max {L[-1]:3d} parole "
              f"-> indicativamente {p95 * 3} token al 95° percentile")

    print(f"\n  Aggregato T1+T2+T3: train {agg['train']}  |  dev {agg['dev']}  |  "
          f"test {agg['test']}  |  totale {sum(agg.values())}")
    print("  Un layout per run: finetune_t1_traduzione.py / finetune_t2_completamento.py / "
          "finetune_t3_replica.py")


if __name__ == "__main__":
    main()