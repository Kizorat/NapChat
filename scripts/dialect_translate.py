#!/usr/bin/env python3
"""
Fase 1.2 — Bozza delle traduzioni napoletane via Ollama Cloud.

Legge il dataset strutturato prodotto da build_conversation_dataset.py
(`dataset_filter/dataset_structured.json`: array di conversazioni, ciascuna con
la sua lista ordinata di `turni`), traduce ogni turno italiano in napoletano
usando un modello ospitato su Ollama Cloud, e riscrive il dataset con il campo
`napoletano` compilato, conservando struttura e formattazione dell'input.

Caratteristiche:
- CONTESTO: passa al modello i turni precedenti (italiano + napoletano gia'
  tradotto) per mantenere coerenza conversazionale.
- RIPRESA: salvataggio incrementale; se lo script si interrompe, rilanciandolo
  riprende dai turni ancora vuoti (non ritraduce quelli gia' fatti).
- RETRY: ritenta con backoff sugli errori di rete/API.

Prerequisiti:
    pip install requests
    La chiave viene letta da un file .api in formato  OLLAMA_KEY=...
    (default: .napoli/.api). In alternativa dalla variabile d'ambiente OLLAMA_KEY.

Uso tipico:
    # default: dataset_filter/dataset_structured.json -> results/napoletano.json (+ .csv)
    python dialect_translate.py
    # solo i primi 50 turni (test):
    python dialect_translate.py --limit 50
    # una sola conversazione del dataset:
    python dialect_translate.py --only KPN001
    # percorsi espliciti e chiave altrove:
    python dialect_translate.py --input dataset_filter/KPN001.json \
        --output results/KPN001_nap.json --api-file percorso/al/.api
"""

import argparse, csv, json, os, sys, time
import requests

# serializzatore condiviso col build script (stesso formato .json/.csv)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_conversation_dataset import write_json, CSV_FIELDS

DEFAULT_INPUT = "dataset_filter/dataset_structured.json"
DEFAULT_OUTPUT = "results/napoletano.json"

OLLAMA_URL = "https://ollama.com/api/chat"

SYSTEM_PROMPT = (
    "Sei un traduttore madrelingua esperto di dialetto napoletano. "
    "Traduci fedelmente dall'italiano al napoletano parlato e colloquiale, "
    "come si parla realmente in una conversazione informale a tavola. "
    "Mantieni il registro informale e la spontaneita' del parlato. "
    # code-switching: i parlanti infilano spezzoni di inglese nell'italiano
    # (\"what is the approach\", \"let me know\", \"never stop dreaming\").
    "Se la frase contiene parole o spezzoni in INGLESE, procedi in due passi: "
    "(1) rendi prima quello spezzone in italiano, "
    "(2) poi traduci in napoletano l'intera frase cosi' sistemata. "
    "Nell'output non deve restare nessuna parola inglese. "
    "ECCEZIONI da lasciare invariate: nomi propri e marchi "
    "(Pull and Bear, Happy Casa, Instagram, Machu Picchu), nomi di piatti "
    "(fish and chips, hot dog, sushi) e i prestiti ormai correnti in italiano "
    "(ok, smart working, feedback, fake, chips, weekend, computer). "
    "NON aggiungere spiegazioni, note, virgolette o testo extra: "
    "restituisci ESCLUSIVAMENTE la traduzione in napoletano."
)


def build_messages(italiano, context_turns):
    """Costruisce i messaggi per la chat, includendo il contesto se presente."""
    user = ""
    if context_turns:
        user += "Contesto della conversazione (turni precedenti):\n"
        for sp, it, nap in context_turns:
            line = f"- {it}"
            if nap:
                line += f"  ->  {nap}"
            user += line + "\n"
        user += "\n"
    user += "Traduci in napoletano SOLO la frase seguente, coerente col contesto:\n"
    user += f"\"{italiano}\""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def call_ollama(model, messages, api_key, temperature, retries=3):
    """Chiama l'endpoint chat di Ollama Cloud con retry+backoff."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(OLLAMA_URL, headers=headers, json=body, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data["message"]["content"].strip()
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"    [retry {attempt}/{retries}] errore: {e} -> attendo {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Chiamata fallita dopo {retries} tentativi: {last_err}")


def load_api_key(api_file, env_name="OLLAMA_KEY"):
    """Legge la chiave da un file .api (formato KEY=VALUE, JSON o chiave grezza).
    Fallback: variabile d'ambiente."""
    if api_file and os.path.exists(api_file):
        raw = open(api_file, encoding="utf-8").read().strip()
        if raw.startswith("{"):                       # JSON
            d = json.loads(raw)
            return d.get(env_name) or next(iter(d.values()), None)
        if "=" in raw:                                # KEY=VALUE (una o piu' righe)
            kv = {}
            for line in raw.splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    kv[k.strip()] = v.strip().strip('"').strip("'")
            return kv.get(env_name) or next(iter(kv.values()), None)
        return raw                                    # chiave grezza
    return os.environ.get(env_name)


def load_samples(path):
    """Carica il dataset in qualsiasi forma: oggetto JSON singolo, array JSON, o JSONL."""
    raw = open(path, encoding="utf-8").read().strip()
    try:                                              # JSON valido (oggetto o array)
        obj = json.loads(raw)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:                      # JSONL (un oggetto per riga)
        return [json.loads(l) for l in raw.splitlines() if l.strip()]


def save(samples, out_json):
    """Salvataggio incrementale: .json (stesso formato dell'input) + .csv gemello.

    Scrive su file temporaneo e poi rinomina: se lo script viene interrotto a
    meta' del salvataggio, il file di ripresa non resta troncato.
    """
    tmp = out_json + ".tmp"
    write_json(samples, tmp)
    os.replace(tmp, out_json)

    out_csv = os.path.splitext(out_json)[0] + ".csv"
    tmp_csv = out_csv + ".tmp"
    with open(tmp_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for s in samples:
            for t in s["turni"]:
                w.writerow({
                    "conversazione": s["id"], "regione": s.get("regione", ""),
                    "macro_regione": s.get("macro_regione", ""),
                    "languages": s.get("languages", ""),
                    "turn_index": t["turn_index"], "tu_id": t.get("tu_id", ""),
                    "speaker": t["speaker"], "italiano": t["italiano"],
                    "napoletano": t.get("napoletano", ""), "note": t.get("note", ""),
                })
    os.replace(tmp_csv, out_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help=f"Dataset strutturato di input (default: {DEFAULT_INPUT})")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help=f"JSON di output, napoletano compilato (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--only", nargs="+", default=None, metavar="CODE",
                    help="Traduci solo queste conversazioni del dataset (es. KPN001)")
    ap.add_argument("--api-file", default=".napoli/.api", help="File con la chiave (formato OLLAMA_KEY=...)")
    ap.add_argument("--model", default="gemma4:cloud", help="Modello Ollama Cloud (verifica il tag esatto)")
    ap.add_argument("--context", type=int, default=4, help="N turni precedenti passati come contesto")
    ap.add_argument("--limit", type=int, default=None, help="Traduci solo i primi N turni ancora vuoti")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sleep", type=float, default=0.0, help="Pausa (s) tra le chiamate")
    ap.add_argument("--save-every", type=int, default=10,
                    help="Salva su disco ogni N traduzioni (default: 10)")
    args = ap.parse_args()

    if not os.path.exists(args.input) and not os.path.exists(args.output):
        sys.exit(f"ERRORE: input non trovato: {args.input!r}\n"
                 f"Generalo con:  python scripts/build_conversation_dataset.py --kipasti dataset_Kipasti")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    api_key = load_api_key(args.api_file)
    if not api_key:
        sys.exit(f"ERRORE: chiave non trovata. Controlla il file {args.api_file!r} "
                 f"(formato OLLAMA_KEY=...) o la variabile d'ambiente OLLAMA_KEY.")

    # carica input; se l'output esiste gia', riprende da li' (resume)
    src = args.output if os.path.exists(args.output) else args.input
    print(f"Carico da: {src}")
    samples = load_samples(src)
    riassunto = ", ".join("{}:{} turni".format(s["id"], len(s["turni"])) for s in samples)
    print(f"Conversazioni nel dataset: {len(samples)}  ({riassunto})")

    # --only: si traducono solo alcune conversazioni, ma il file salvato resta completo
    if args.only:
        ignoti = [c for c in args.only if c not in {s["id"] for s in samples}]
        if ignoti:
            sys.exit(f"ERRORE: conversazioni non presenti nel dataset: {', '.join(ignoti)}")
        da_tradurre = [s for s in samples if s["id"] in set(args.only)]
    else:
        da_tradurre = samples

    # conta il lavoro da fare
    todo = sum(1 for s in da_tradurre for t in s["turni"] if not t.get("napoletano"))
    print(f"Turni ancora da tradurre: {todo}  |  modello: {args.model}")
    if not todo:
        print("Niente da fare: tutti i turni sono gia' tradotti.")
        return

    done = 0
    try:
        for s in da_tradurre:
            print(f"\n=== {s['id']} ===")
            turni = s["turni"]
            for i, t in enumerate(turni):
                if t.get("napoletano"):        # gia' tradotto -> salta (resume)
                    continue
                if args.limit is not None and done >= args.limit:
                    break
                # contesto: fino a N turni precedenti (italiano + napoletano se presente)
                ctx = [(turni[j]["speaker"], turni[j]["italiano"], turni[j].get("napoletano", ""))
                       for j in range(max(0, i - args.context), i)]
                messages = build_messages(t["italiano"], ctx)
                try:
                    nap = call_ollama(args.model, messages, api_key, args.temperature)
                except RuntimeError as e:
                    print(f"  ! turno {t['turn_index']} saltato: {e}", file=sys.stderr)
                    continue
                t["napoletano"] = nap
                done += 1
                print(f"  [{done}/{todo}] {s['id']} t{t['turn_index']}  "
                      f"{t['italiano'][:45]!r} -> {nap[:45]!r}")
                if done % max(1, args.save_every) == 0:
                    save(samples, args.output)      # salvataggio incrementale
                if args.sleep:
                    time.sleep(args.sleep)
            if args.limit is not None and done >= args.limit:
                break
    except KeyboardInterrupt:
        print("\nInterrotto: salvo quanto tradotto finora...", file=sys.stderr)
    finally:
        # in ogni caso (fine, Ctrl+C o errore) il progresso finisce su disco
        save(samples, args.output)

    print(f"\nFatto. Tradotti in questa sessione: {done}. Output: {args.output} (+ .csv)")


if __name__ == "__main__":
    main()