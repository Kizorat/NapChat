#!/usr/bin/env python3
"""
Patch mirata: i turni in cui e' rimasto inglese non tradotto nel napoletano.

Il SYSTEM_PROMPT di dialect_translate.py e' stato corretto per gestire il
code-switching (inglese -> italiano -> napoletano). Questo script applica la
stessa regola ai file gia' prodotti, senza rilanciare traduzioni:

    gemma   -> results/napoletano.json          (bozza LLM)
    golden  -> golden_translate/golden_validator.json  (validato a mano)

Il golden va patchato insieme alla bozza: il validatore nativo aveva seguito la
politica opposta (conservare l'inglese come tratto di code-switching del
parlato), quindi se si tocca solo la bozza i due file misurano un disaccordo
che non riguarda la qualita' del napoletano.

Restano invariati per policy, in entrambi i file:
- marchi e nomi propri   : Pull and Bear, Happy Casa, Black Fire, Machu Picchu
- nomi di piatti         : fish and chips, hot dog, sushi, wasabi
- prestiti ormai italiani: smart working, feedback, fake, flash, chips, play
- troncamenti ambigui    : "'o ball~" (l'italiano "il ball~" puo' essere
                           *ballo* o *ball*: tradurlo sarebbe inventare)

Le correzioni sono espresse come sostituzioni di SOTTOSTRINGA, non di frase
intera: si tocca solo lo spezzone inglese e il resto della resa resta identico
(anche le sue imperfezioni), per non riscrivere il lavoro altrui.

Lo script e' idempotente: se lo spezzone inglese non c'e' piu' ma c'e' gia' la
sostituzione, il turno risulta "gia' corretto"; se il testo e' inatteso, viene
saltato e segnalato senza sovrascrivere nulla.

ATTENZIONE (solo golden): golden_validator.{json,csv,xlsx} devono restare
allineati. Dopo --apply su golden, rigenera il workbook:
    py -3.8 golden_translate/script/golden_workbook.py export
    py -3.8 golden_translate/script/golden_workbook.py verify

Uso:
    python scripts/fix_english_turns.py                 # anteprima, entrambi
    python scripts/fix_english_turns.py --set golden    # anteprima, solo golden
    python scripts/fix_english_turns.py --apply         # scrive .json + .csv
"""

import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dialect_translate import load_samples, save

# (conversazione, turn_index): [(spezzone inglese, resa napoletana), ...]
# Le chiavi di ricerca sono ASCII: nessuna trascrizione a mano di ê/â/ù.

CORREZIONI_GEMMA = {
    ("KPN001", 26):   [("na functioning alcoholic", "n'alcolista funzionale")],
    ("KPN001", 171):  [("excitement", "gasata")],
    ("KPN001", 1016): [("rotisserie chicken", "pulle arrustute")],
    ("KPN003", 13):   [("basic", "'e base")],
    ("KPN003", 65):   [("motivation", "mutivazione")],
    ("KPN003", 69):   [("never stop", '"nun te fermà maje"')],
}

CORREZIONI_GOLDEN = {
    ("KPN001", 18):   [("thinking behind it was", "raggiunamento areto era")],
    ("KPN001", 26):   [("na functioning alcoholic", "n'alcolista funzionale")],
    ("KPN001", 32):   [("reservation", "prenotazione")],
    ("KPN001", 171):  [("excited", "gasata")],
    ("KPN001", 222):  [("settled", "sistemate")],
    ("KPN001", 358):  [("sad story", "storia triste")],
    ("KPN001", 423):  [("raised", "tirato fora")],
    ("KPN001", 601):  [("appealing", "attraente")],
    ("KPN001", 803):  [("submit it", "cunzignarlo")],
    ("KPN001", 811):  [("desk ba~, desk based research",
                        "ricerca 'a scri~, ricerca 'a scrivania")],
    ("KPN001", 824):  [("let me know", "famme sapé")],
    ("KPN001", 826):  [("what is the approach", "qual è ll'approccio")],
    ("KPN001", 899):  [("what do you mean", "che vuó dicere")],
    ("KPN001", 904):  [("why", "pecché")],
    ("KPN001", 977):  [("of course", "cierto")],
    ("KPN001", 1016): [("rotisserie chicken", "pullaste arrustute")],
    ("KPN003", 13):   [("basic", "'e base")],
    ("KPN003", 65):   [("motivation", "mutivazione")],
    ("KPN003", 66):   [("never stop dreaming", "nun smettere maje 'e sunnà")],
    ("KPN003", 69):   [("never stop", '"nun smettere maje"')],
    ("KPN003", 111):  [("cheers", "salute")],
    ("KPN003", 112):  [("cheers", "salute")],
    ("KPN003", 113):  [("cheers", "salute")],
    ("KPN003", 353):  [("'o fi~", "'o pi~"), ("vero fish", "vero pesce")],
    ("KPN003", 1008): [("nu fetish", "na fissazione")],
}

INSIEMI = {
    "gemma":  ("results/napoletano.json", CORREZIONI_GEMMA),
    "golden": (os.path.join("golden_translate", "golden_validator.json"),
               CORREZIONI_GOLDEN),
}


def applica(samples, correzioni):
    """Applica le sostituzioni; ritorna (turni modificati, anteprima, saltati)."""
    idx = {(s["id"], t["turn_index"]): t for s in samples for t in s["turni"]}
    modificati, anteprima, saltati = 0, [], []

    for (conv, ti), coppie in sorted(correzioni.items()):
        t = idx.get((conv, ti))
        if t is None:
            saltati.append(f"{conv} t{ti}: turno assente dal file")
            continue

        prima = t.get("napoletano", "")
        dopo = prima
        problemi = []
        for vecchio, nuovo in coppie:
            if vecchio in dopo:
                dopo = dopo.replace(vecchio, nuovo)
            elif nuovo in dopo:
                continue                                   # gia' applicata
            else:
                problemi.append(vecchio)

        if problemi:
            saltati.append(f"{conv} t{ti}: spezzone non trovato "
                           f"{problemi} -> non tocco nulla\n"
                           f"      testo: {prima!r}")
            continue
        if dopo == prima:
            saltati.append(f"{conv} t{ti}: gia' corretto")
            continue

        anteprima.append((conv, ti, t["italiano"], prima, dopo))
        t["napoletano"] = dopo
        modificati += 1

    return modificati, anteprima, saltati


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="insiemi", nargs="+", default=["gemma", "golden"],
                    choices=["gemma", "golden"],
                    help="quali file correggere (default: entrambi)")
    ap.add_argument("--apply", action="store_true",
                    help="scrive davvero i file (senza, mostra solo l'anteprima)")
    args = ap.parse_args()

    totale = 0
    for nome in args.insiemi:
        path, correzioni = INSIEMI[nome]
        print("=" * 70)
        print(f"{nome.upper()}  ->  {path}")
        print("=" * 70)
        if not os.path.exists(path):
            print(f"  ERRORE: file non trovato, salto\n")
            continue

        samples = load_samples(path)
        n, anteprima, saltati = applica(samples, correzioni)

        for conv, ti, it, prima, dopo in anteprima:
            print(f"[{conv} t{ti}]")
            print(f"   IT   : {it}")
            print(f"   prima: {prima}")
            print(f"   dopo : {dopo}")
        if saltati:
            print(f"\n  SALTATI ({len(saltati)}):")
            for s in saltati:
                print(f"    - {s}")

        if args.apply and n:
            save(samples, path)
            print(f"\n  Scritto: {path}")
            print(f"  Scritto: {os.path.splitext(path)[0] + '.csv'}")
        print(f"\n  Turni corretti: {n}\n")
        totale += n

    if not args.apply:
        print(f"Anteprima: {totale} turni da correggere in totale. "
              f"Rilancia con --apply per scrivere.")
    elif "golden" in args.insiemi and totale:
        print("Ricorda di riallineare il workbook del golden:\n"
              "  py -3.8 golden_translate/script/golden_workbook.py export\n"
              "  py -3.8 golden_translate/script/golden_workbook.py verify")


if __name__ == "__main__":
    main()
