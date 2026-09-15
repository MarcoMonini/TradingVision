# Handoff — ramo `claude/fervent-knuth-se8cw5`

Stato in una riga: **le misure sono state fatte.** La predizione non guida il prezzo; le feature
di esaurimento sì, ma di un fattore 6-170 sotto il costo di eseguirle. La predizione che ottiene
Rank IC 0,4114 contro l'etichetta swing ottiene **−0,0405 contro il rendimento forward**. Il ramo
`book-on-screen` è stato mergiato. Lo spec è aggiornato con tutti i numeri.

Base: merge di `book-on-screen` in `claude/fervent-knuth-se8cw5`.

---

## 1. Cosa è stato misurato, e su cosa

Tutte le misure di questa sessione girano su dati di mercato veri (`data/` pieno, 20 coppie USDT
da Binance, 2023-01 → 2026-09) e non su sintetico. La sessione precedente non aveva rete e non
aveva prodotto nessun numero; questa li ha prodotti tutti.

**Prerequisito che è costato metà del lavoro.** Il file `data/pred-swing-all-15m.parquet` che era
su disco veniva dal build a **stride posizionale**, precedente alla correzione che
`book-on-screen` porta con sé. Numeri: breadth 7,82, **zero** timestamp con venti simboli, 15.989
timestamp con un simbolo solo. Su un timestamp da un simbolo la neutralizzazione cross-sectional
demedia un valore contro sé stesso e restituisce **esattamente zero**, quindi ogni metrica
market-neutral letta su quel file era diluita. Erano il 7,3% delle righe — abbastanza da sporcare,
non abbastanza da ribaltare.

Lo stamp della cache ha fatto esattamente il suo lavoro e ha rifiutato sia il tensore sia
`step2.parquet` (la `features.py` mergiata aggiunge `log_dollar_volume`, 29 colonne invece di 28).
Ricostruiti entrambi. Breadth dopo: **19,95**, 10.648 timestamp su 11.010 con tutti e venti.

Tempi misurati, utili per pianificare:

| fase | tempo |
|---|---|
| `gbm --horizon` + rebuild `step2.parquet` (1,64 GB) | 16 min |
| build tensore `step3-15m-all.npy` (1,79 GB) | ~4 min |
| `gru` 4 fold × 5 seed su MPS | **~55 min** |
| `legcheck` + `swingrule` + rotation null | ~5 min |

---

## 2. Misura 0 — la baseline riproduce lo spec

```
        train      test      ic    icir  rank_ic  rank_icir
fold 1  423089   55060   0.4479  1.8360   0.4229     1.7834
fold 2  478114   55040   0.4493  1.8112   0.4317     1.8619
fold 3  533148   55019   0.3972  1.6314   0.3863     1.6734
fold 4  588234   54579   0.4294  1.8581   0.4046     1.7410
mean                     0.4310  1.7842   0.4114     1.7650
std                      0.0242  0.1037   0.0202     0.0789
```

Atteso dallo spec: Rank IC ~0,4215 ± 0,0154, ICIR ~1,74. **Torna.** Lo store e il campione sono
quelli su cui lo spec ha misurato, quindi tutto il resto è confrontabile.

---

## 3. Misura 1 — il giudice, e il suo verdetto

Il criterio era fissato prima di guardare l'output, ed è quello che il vecchio handoff aveva
scritto. Su 219.135 righe, venti simboli, rendimento market-neutral:

| decile | pred medio | fwd 6 | t | fwd 24 | t | fwd 72 | t |
|---|---|---|---|---|---|---|---|
| **0 (compra)** | −0.491 | +0.00006 | 0.68 | −0.00053 | **−3.55** | −0.00074 | **−2.91** |
| 4 | −0.051 | −0.00001 | −0.10 | +0.00014 | 0.98 | +0.00001 | 0.05 |
| 5 | +0.065 | −0.00007 | −0.95 | −0.00006 | −0.39 | +0.00008 | 0.31 |
| **9 (vendi)** | +0.506 | +0.00009 | 0.86 | +0.00004 | 0.21 | −0.00013 | −0.40 |

Rank IC contro il prezzo: **−0.0405** (6 barre), **−0.0279** (24), **−0.0145** (72).

Il criterio chiedeva decile 0 positivo con |t| > 2 e decile 9 negativo con |t| > 2. **Esce
l'opposto**: il decile 0 è negativo e significativo a 24 e 72 barre, il decile 9 non si distingue
da zero a nessun orizzonte, e tutto il resto della tabella è piatto senza monotonia.

Il controllo causale: `corr_pred_control` **0,34** con `elapsed_position`, e togliendolo non
migliora niente — `ic_residual` −0,0363 contro `ic_raw` −0,0405. Non c'è una metà buona da isolare.

**Questo chiude la tesi dell'utente.** Il problema non era la regola di trading: non c'è niente su
cui una regola possa agire.

---

## 4. Misura 2 — la regola long-only, e il controllo nuovo

Venti simboli, 2025-09 → 2026-09, 25 bp/lato, banda dinamica:

| quantile | in mercato | trade/anno | lordo | commissioni | netto | win rate | trade |
|---|---|---|---|---|---|---|---|
| 0.80 | 0.482 | 78.5 | −0.359 | 0.391 | **−0.750** | 0.545 ± 0.013 | 1577 |
| 0.90 | 0.473 | 46.8 | −0.494 | 0.233 | −0.728 | 0.538 ± 0.016 | 941 |
| 0.95 | 0.447 | 27.8 | −0.540 | 0.138 | −0.678 | 0.505 ± 0.021 | 558 |
| 0.99 | 0.358 | 8.3 | −0.271 | 0.041 | −0.312 | 0.455 ± 0.039 | 167 |

Buy and hold sulle stesse righe: **−81,1%** l'anno.

### Il win rate sopra 0,5 non è un edge — e il controllo che lo dimostra

Il win rate è 0,545 ± 0,013 su 1577 trade, tre errori standard sopra la moneta, con trade mediano
positivo. Sembra abilità. Il trade *medio* è −0,0096: tante vincite piccole, poche perdite grandi.

`buy_and_hold` era l'unico controllo che la regola aveva e risponde alla domanda sbagliata: tiene
ogni barra, quindi confonde l'esposizione col timing, e contro un paniere sceso dell'81% stare
flat metà del tempo somiglia a bravura. **`swingrule.rotation_null`** (aggiunto in questo ramo)
tiene l'esposizione fissa per costruzione — ruota il vettore di posizione di ogni simbolo di uno
sfasamento casuale, stesse barre tenute, stesso numero di trade, stessa struttura delle durate — e
distrugge solo la fase.

| | lordo reale | null | z | p |
|---|---|---|---|---|
| banda 0.80, 500 rotazioni | −0.3588 | −0.4019 ± 0.0923 | 0.47 | **0.334** |
| banda 0.99, 500 rotazioni | −0.2711 | −0.3039 ± 0.0728 | 0.45 | **0.310** |

Marginalmente meglio del caso, lontanissimo dalla significatività. Tutto il lordo è esposizione;
le commissioni lo portano poi da −0,359 a −0,750.

---

## 5. Le feature di esaurimento — la strada del punto aperto 4, percorsa

`exhaustcheck` isola le sei colonne di `legs.exhaustion` contro il **rendimento forward** e non
contro un'etichetta. È l'unica strada che poteva cambiare il *segno* del risultato, e in parte
lo cambia.

**Quattro su sei tengono il segno su tutti e quattro i fold**, nella direzione che l'esaurimento
prevede: `stretch` −0.0433, `divergence` −0.0255 (24 barre), `streak` −0.0229, `rejection`
+0.0185. `volume_climax` e `deceleration` sono rumore (2 fold su 4).

**Non è microstruttura.** Ritardando ogni feature di una barra intera — il test che separa un lead
da un rimbalzo bid-ask — `divergence` e `rejection` non perdono niente (−0.0253, +0.0186).
`stretch` tiene l'82%, `streak` metà. Il composito equipesato delle quattro colonne firmate:
**+0.0376** a 24 barre, **+0.0325 ± 0.0100** con il ritardo, positivo su 4 fold su 4.

Per la prima volta un segnale causale punta dalla parte **giusta** del prezzo, dove il modello
fittato sull'etichetta punta all'indietro (−0.0279 alle stesse 24 barre).

### E non si può tradare — `exhaustcheck --price`

| cadenza | trade/anno | lordo hedged | netto 25bp | fee break-even |
|---|---|---|---|---|
| 1h | 2413.7 | +0.4087 | −11.660 | 0.85 bp |
| 1D | 200.5 | +0.0163 | −0.986 | 0.41 bp |
| **3D** | 70.2 | +0.0247 | −0.326 | **1.76 bp** |
| 7D | 29.7 | +0.0009 | −0.148 | 0.15 bp |
| 30D | 8.4 | +0.0007 | −0.041 | 0.40 bp |

La leva che aveva salvato il composito fattoriale del punto 1 — rallentare il ribilanciamento —
qui non salva niente, e la forma della tabella dice perché: **il lordo cala alla stessa velocità
delle commissioni**. Il segnale è veloce e decade a +0.0009 entro sette giorni. Ogni cadenza è la
stessa operazione in perdita a taglie diverse.

La cadenza migliore chiede **1,76 bp per lato**. Alpaca taker è 25 bp, un maker realistico ~10.

**Limiti, dichiarati.** Letture univariate, quindi una colonna che funziona solo in combinazione
qui sembra piatta; e l'orizzonte è un forward grezzo, non condizionato all'essere vicino a una
svolta, che è il regime per cui le feature sono pensate. Un negativo qui pesa meno di quanto
avrebbe pesato un positivo.

---

## 6. Cosa NON è stato misurato

**La Misura 3 (`--weight rank`, `--weight excursion`, `--finetune 20`) non è stata lanciata.**
Sono quattro run da ~55 min ciascuna, ~3,7 ore di GPU, e il motivo per fermarsi è la Misura 1:
ottimizzano la Rank ICIR **contro l'etichetta swing**, che è la quantità appena misurata a
−0,04 contro il prezzo. Migliorare 0,4114 non avvicina di un passo un rendimento.

Chi riprende decida esplicitamente: è lavoro legittimo sulla qualità del modello, ma non è lavoro
sul P&L, e il vecchio handoff lo elencava quando la Misura 1 non era ancora stata fatta.

Non misurata nemmeno la **Misura 4** (le leve mai tunate: `STEPS`, `H`, rifare `selection` contro
swing, loss di ranking). Stessa obiezione, con una sfumatura: `STEPS = 24` è l'unica di quelle che
potrebbe cambiare *cosa* il modello vede e non solo quanto bene fitta, perché una finestra da 6 ore
non arriva all'inizio di gambe lunghe fino a 754 barre.

---

## 7. Cosa è stato aggiunto al codice

- **Merge di `book-on-screen`** (12 commit): il libro fattoriale cross-sectional (`factor.py`), la
  regola swing tradabile (`swing.py`, `legs.py`), la pagina chart, e — la parte che è servita
  subito — il campionamento sull'orologio che corregge la breadth. I due conflitti erano
  entrambi registri (`CLAUDE.md`, `tests/test_selfchecks.py`) e sono stati risolti a unione.
- **`swingrule.rotation_null`** — il controllo a esposizione fissa descritto sopra, col suo
  self-check e la voce `--rotations` nella CLI.
- **`exhaustcheck.py`** — l'isolamento delle sei colonne di esaurimento, il controllo `--lag` che
  separa il lead dal rimbalzo, il composito firmato, e `--price` che lo prezza come libro con la
  macchina di `factor`. Registrato in `tests/test_selfchecks.py`.
- **Spec aggiornato**: sezione 9 ha ora la regola long-only prezzata, la rotation null, e la
  tabella per decile di `legcheck` col verdetto.

`uv run pytest -q` → **37 passed, 0 skipped**. I tre test che vogliono lo store girano, perché
`data/` è pieno.

### Una trappola di ambiente, per chi ripete le misure

`gru` stampa i risultati **solo alla fine** e le righe per epoca solo con `--verbose`. In più
stdout è block-buffered quando rediretto su file: un log vuoto per quaranta minuti non vuol dire
che la run sia bloccata. Usare `python -u` e `--verbose` se serve vedere l'avanzamento. `py-spy`
su macOS richiede root, quindi non è una via d'uscita.

---

## 8. Cosa non rifare

Tutto quello che il vecchio handoff elencava resta chiuso. Si aggiunge:

- **Cercare la colpa nella regola di trading.** La Misura 1 è senza regola e senza commissioni:
  la tabella per decile è piatta e dove è significativa punta dalla parte sbagliata.
- **Leggere un win rate come un edge.** 0,545 ± 0,013 su 1577 trade è tre sigma sopra la moneta e
  non vale niente, perché il trade medio è negativo e il lordo non batte il suo null.
- **Confrontare una regola long-only col solo buy-and-hold** su un periodo in discesa. Serve un
  controllo a esposizione fissa; ora c'è.
- **Ripartire dalle feature di esaurimento sperando in una soglia migliore.** Il lead c'è ed è
  reale; manca un fattore 6-170 sul costo di esecuzione, che nessuna regola recupera.
- **Fidarsi di un file `pred-*.parquet` senza controllarne la breadth.** Una riga:
  `d.groupby(d.index.get_level_values(0)).size().mean()`. Se non è ~20, il file è precedente alla
  correzione del campionamento e ogni metrica cross-sectional letta su di esso è diluita.
