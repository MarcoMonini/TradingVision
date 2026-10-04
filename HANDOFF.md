# Handoff — ramo `claude/fervent-knuth-se8cw5`

Stato in una riga: **le misure sono state fatte.** La predizione non guida il prezzo; le feature
di esaurimento sì, ma di un fattore 6-170 sotto il costo di eseguirle. La predizione che ottiene
Rank IC 0,4114 contro l'etichetta swing ottiene **−0,0405 contro il rendimento forward**. Il ramo
`book-on-screen` è stato mergiato. Lo spec è aggiornato con tutti i numeri.

Base: merge di `book-on-screen` in `claude/fervent-knuth-se8cw5`.

**Aggiornamento 2026-09-19** — dopo la PR #14 è stata fatta una passata di revisione su tutto il
codice (nessuna misura nuova, nessun numero spostato): sezione 13. Contiene un crash della pagina
deployata, tre cache che non invalidavano, e un'ottimizzazione da 61x su `features`.

**Aggiornamento 2026-10-03** — ramo `claude/retrospective-only`: il progetto lavora solo su
etichette retrospettive. Tutto il lavoro predittivo e la prima pipeline (`dataset` → `gru` →
`legsweep`) sono in `OLD/`, congelati al tag `archive-predictive`: sezione 16.

**Aggiornamento 2026-10-03, più tardi** — ramo `claude/strategy-study`: lo studio delle regole di
trading sulla predizione v2 (`strategy.py`) e tutte quelle regole sulla pagina chart: sezione 17.

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

---

## 9. La regola always-in, prezzata a soglia fissa (ramo `always-in-at-a-number`)

La regola chiesta — sotto −0.4 long, sopra +0.4 si chiude il long e si va short, mai flat — **era
già in `threshold.py`**, che è esattamente quel modello di stato con `sign=-1`. Mancava solo il
modo di prezzarla a un numero grezzo invece che a un quantile dell'output: ora c'è `--at`.

```bash
uv run python -m tradingvision.threshold --pred data/pred-swing-all-15m.parquet --at 0.4
```

Venti simboli, 219.698 righe, 2025-06 → 2026-09, 25 bp/lato. `--at 0.4` cade sul quantile 0,763 di
|pred| (il 23,7% delle righe supera 0.4 in valore assoluto).

| soglia | trade/anno | lordo long | lordo short | lordo | commissioni | netto | fee di pareggio |
|---|---|---|---|---|---|---|---|
| ±0.3 | 325.1 | −0.234 | +0.250 | +0.015 | 1.624 | −1.608 | 0.24 bp |
| **±0.4** | 195.8 | −0.203 | +0.286 | +0.083 | 0.977 | **−0.894** | 2.13 bp |
| ±0.5 | 92.7 | −0.249 | +0.252 | +0.004 | 0.461 | −0.458 | 0.19 bp |
| q 0.70 | 237.6 | −0.160 | +0.327 | +0.167 | 1.186 | −1.019 | **3.52 bp** |
| q 0.95 | 65.9 | −0.235 | +0.260 | +0.024 | 0.328 | −0.303 | 1.85 bp |

Buy and hold sulle stesse righe: −0.490 l'anno.

**Il netto è negativo ovunque** e il punto migliore di tutta la griglia chiede 3,52 bp per lato,
contro 25 bp taker e ~10 bp maker. Allargare la banda taglia commissioni e lordo insieme, che è
la stessa forma di tabella di `exhaustcheck --price`.

**La gamba long perde a ogni soglia** (−0.203 a ±0.4) e tutto il lordo è la gamba short (+0.286)
su un paniere sceso del 49% l'anno. È la stessa colonna di sinistra che `swingrule` documenta e
che la Misura 1 spiega: il decile 0 della predizione ha rendimento forward negativo a 24 e 72
barre. Comprare sotto −0.4 è comprare dove il prezzo scende.

Nessun `rotation_null` qui: con un lordo di +0.083 contro 0.977 di commissioni non c'è una fase da
testare. Servirebbe solo se il lordo coprisse il costo.

`threshold._selfcheck` non era registrato in `tests/test_selfchecks.py` — ora lo è.

### La regola sulla pagina chart

La stessa regola è ora disegnata su `chart.py` sopra l'etichetta **swing leg position**, con il suo
riquadro di metriche e la scomposizione per gamba. Legge la **predizione** e mai l'etichetta: il
label retrospettivo nasce da una finestra centrata, quindi una regola che lo tradasse leggerebbe 24
barre di futuro e sarebbe un oracolo travestito da strategia. Per questo il toggle compare solo
quando un modello è acceso.

Sulle candele le due cose si distinguono per forma, non per colore: **quadratini vuoti** sono i
pivot dell'oracolo (che il futuro lo leggono davvero), **triangoli pieni** sono i fill della regola.
Le percentuali di ampiezza che stavano accanto ai pivot sono state tolte — annotavano le gambe
dell'oracolo, non i trade, e con entrambe le serie sulla stessa riga rendevano il grafico
illeggibile. L'ampiezza mediana resta nella didascalia in fondo.

Il libro della regola passa da `threshold.on_one`, che solleva la singola coppia nel MultiIndex a
un simbolo che tutte le funzioni del modulo raggruppano: la regola disegnata è la stessa che
`--at` prezza sul pannello, senza una seconda implementazione che possa divergere.

---

## 10. ±0.5 su venti coppie, e la tesi "lo ingannano i movimenti grandi"

```bash
uv run python -m tradingvision.threshold --pred data/pred-swing-all-15m.parquet \
    --at 0.5 --by-symbol --by-move 5 --horizon 0 24 72 168
```

**Il pannello.** ±0.5 sta sul quantile 0,917 di |pred|. 92,7 flip l'anno per coppia, lordo +0.004,
commissioni 0.461, netto **−0.458**. Per coppia: **4 su 20** positive, media −0.458 ± 0.121 (errore
standard fra coppie) — 3,8 sigma sotto zero. Lo Spearman fra il lordo della gamba short e il
buy-and-hold della coppia è **−0,734**: le quattro che guadagnano sono le quattro scese di più
(AVAX −84%, SUSHI −89%, XTZ −78%). Esposizione, non selezione.

**La tesi, misurata.** Sul hold la tabella le dà ragione: quintile dei movimenti grandi hit rate
0,397 contro 0,74 del quintile medio, e da solo vale −24.1 contro i +12.6 di tutti gli altri.

**Ma la lettura è circolare**, e la colonna `median_bars` lo dice: 100 barre nel quintile alto
contro 50 nel quintile 2. Una regola a isteresi esce solo quando la predizione raggiunge la banda
opposta, quindi un hold su cui ha ragione viene *chiuso* dal movimento e uno su cui ha torto resta
aperto mentre il movimento corre. Dimensione del movimento e durata del hold sono la stessa
variabile. Il meccanismo produce quella tabella senza alcun segnale.

**Il controllo** (`--horizon`, ogni ingresso giudicato su N barre fisse, uscita fuori dalla misura):

| orizzonte | hit rate min–max sui 5 quintili | pendenza alto − basso |
|---|---|---|
| 24 bar | 0,464 – 0,544 | +0,023 |
| 72 bar | 0,489 – 0,558 | **+0,069** |
| 168 bar | 0,464 – 0,519 | +0,002 |

Errore standard di un hit rate: 0,023. **Piatto.** A 72 bar la pendenza è positiva e il quintile
dei movimenti più grandi è l'unico sopra la moneta.

**Verdetto.** Il segnale non prevede i movimenti grandi — a nessun orizzonte batte la moneta, come
`legcheck` già diceva per decile — ma nemmeno ci sbatte contro più che contro gli altri. Quello che
perde sui movimenti grandi è l'**uscita**. La leva che questa tabella indica non è una soglia né un
modello: è uno stop. Da misurare, con la solita avvertenza: si parte da un lordo di +0.004 contro
0.461 di commissioni, quindi lo stop dovrebbe guadagnare due ordini di grandezza, non qualche
punto.

**Non rifare:** leggere un hit rate per dimensione del movimento *sul hold* di una regola a
isteresi senza la colonna delle durate accanto. È la variabile stessa, ordinata.

---

## 11. Le uscite della regola always-in (`stops.py`) — strumento pronto, **numeri non presi**

La sezione 10 chiude dicendo che la leva non è una soglia né un modello, è **uno stop**. Questo
ramo la scrive. Non la misura: lo store (`data/*.parquet`) non era presente in questa sessione,
quindi la griglia sullo store non è mai stata lanciata e **in questo documento non c'è un solo
numero nuovo sul P&L**. Chi riprende parte da qui:

```bash
uv run python -m tradingvision.stops --pred data/pred-swing-all-15m.parquet --at 0.5 --grid
uv run python -m tradingvision.stops --pred data/pred-swing-all-15m.parquet --at 0.5 \
    --sl atr:2 --tp atr:4 --after-stop reverse
uv run python -m tradingvision.stops --pred data/pred-swing-all-15m.parquet --at 0.5 \
    --sl fee:2 --tp fee:4 --tie take     # l'altra lettura della stessa barra
```

### Cosa è parametrizzato, e perché ognuno è una domanda e non un'impostazione

- **Dove vanno le barriere.** Distanza in log dal prezzo d'ingresso, **fissata all'ingresso**
  (trailing non c'è, è un'altra regola). Tre unità: `atr:k` (la colonna che `features` porta,
  quindi la stessa unità degli input del modello), `fee:k` (multipli dell'andata e ritorno, l'unica
  distanza con un significato aritmetico — `fee:1` incassa esattamente zero) e `pct:x` piatta.
  Take e stop prendono specifiche separate: un take largo con uno stop stretto è la forma che la
  tabella per movimento indica, e simmetrico è una scommessa sulla simmetria del *prezzo*, che
  l'etichetta ha e il prezzo no.
- **Se lo stop trascina** (`--trail`). Lo stop si aggancia al massimo raggiunto dal hold invece che
  al prezzo d'ingresso, stessa larghezza: da "quanto sono disposto a perdere" a "quanto di ciò che
  ho guadagnato sono disposto a restituire". È l'unica uscita che può chiudere un hold **in
  profitto** senza un take profit. Il cricchetto va in una direzione sola e viene alzato solo da
  barre già controllate e sopravvissute — mai da quella su cui è in prova, che sarebbe la stessa
  ipotesi sul percorso che `--tie` isola. Uno spike può portare lo stop sopra il prezzo corrente e
  la barra dopo riempie alla sua apertura: è quello che fa uno stop trailing vero, non un artefatto.
- **Cosa si tiene dopo.** `reverse` entra subito dall'altra parte al prezzo di uscita, `opposite`
  sta flat fino al segnale opposto, `rearm` sta flat fino a un qualsiasi attraversamento fresco.
  Una politica per barriera: invertire su uno stop è una scommessa di momentum, invertire su un
  take è una di mean reversion.
- **L'attraversamento fresco.** Il lato appena stoppato resta sbarrato finché la predizione non
  rientra nella banda e non ne riesce. È il motivo per cui `threshold.signals` è stato separato da
  `positions` — una posizione forward-filled ripete l'ultimo tocco per sempre e non può dire se il
  segnale ha parlato *dopo* lo stop. Senza, `rearm` ricompra alla barra successiva.
- **La candela che tocca entrambi i livelli.** Una barra dà un massimo e un minimo ma non l'ordine
  in cui sono arrivati, quindi una barra che raggiunge sia lo stop sia il take ha due letture.
  `--tie stop` (default, prende la perdita) contro `--tie take`. La distanza fra le due **è** la
  dimensione dell'ipotesi intrabarra. Un risultato che vive solo su `--tie take` è un risultato sul
  percorso. Sulla pagina il controllo si chiama ora *"If one candle reaches the stop and the take
  profit"* con le due opzioni scritte per esteso — il vecchio *"When one bar holds both levels"* non
  diceva cosa si stesse decidendo.

### Il controllo

Senza barriere `stops` **è** `threshold`: stesse posizioni, stesso lordo, stesse commissioni,
stesso win rate, asserito riga per riga in `stops._selfcheck` contro `threshold.pnl`. `--grid`
stampa quella riga per prima e ogni altra deve batterla. Se nessuna la batte, l'uscita non è dove
la regola perde. E il punto di partenza è quello di sezione 9: a ±0.5 il lordo è +0.004 contro
0.461 di commissioni, quindi **lo stop deve guadagnare due ordini di grandezza, non qualche punto**.

### Sulla pagina

`chart.py` apre ora su `swing_leg_target` e su ±0.5, e la regola disegnata passa da `stops` anche
quando nessuna barriera è accesa — una sola strada di codice, non un ramo dietro il toggle.
Controlli: stop loss e take profit (unità + moltiplicatore, widget separati perché "3" non vuol
dire niente finché non dice tre di cosa), la politica di ognuna, e la lettura della barra
ambigua. Sulle candele una X rossa è uno stop e una stella verde un take, **al prezzo di
riempimento** e non alla chiusura della barra: una barra che gappa oltre il livello riempie
all'apertura, e la distanza fra il segno e il livello è esattamente ciò che va visto. Il riquadro
metriche guadagna "closed by a barrier", che è la lettura che dice se la barriera sta facendo
qualcosa: una quota di stop vicina a zero vuol dire che la barriera è più larga dei hold della
regola e i numeri accanto sono quelli di `threshold` con passaggi in più.

### Un bug trovato strada facendo, fuori tema ma bloccante

`gru.restore` e `swing.restore` chiamavano `torch.load` senza `map_location`. Un checkpoint porta
con sé il device del processo che l'ha salvato, quindi quelli addestrati su Mac dicono `mps` e la
pagina **non si apriva affatto** su Linux — cioè nel caso deployato, che è l'artefatto che questo
progetto spedisce. Una riga per file.

### Trappole

- `ta` riempie il warm-up dell'ATR con **zeri**, non con NaN. Una barriera larga zero sta sul
  prezzo d'ingresso e scatta alla prima barra che si muove. `stops.width` dà a un ATR non
  strettamente positivo nessuna barriera, con l'assert accanto.
- Una posizione aperta *dentro* una barra da un `reverse` è controllata contro le proprie barriere
  solo dalla barra dopo: il percorso dentro la barra in cui è nata non è noto.

### Il bug segnalato: era il marcatore, non la macchina a stati

Segnalato come "un BUY mai chiuso e poi un altro BUY". La macchina a stati è a posto — 1944
combinazioni di barriera, politica, tie e trail su sei serie, e nessuna invariante rotta: le
posizioni stanno in {−1, 0, +1}, i hold non si sovrappongono mai, e il lato di ogni hold è la
posizione tenuta alla sua barra d'ingresso.

Erano le **etichette dei triangoli**. `chart.py` le derivava dal *segno della variazione*: ogni
salita "buy", ogni discesa "sell". Con la regola always-in era corretto per costruzione, perché non
c'è stato flat e la posizione va da +1 a −1 e ritorno, quindi le parole si alternavano da sole. Con
un'uscita in mezzo c'è il flat, e **chiudere uno short (−1 → 0) e aprire un long (0 → +1) sono
entrambi una salita**: due "buy" di fila senza "sell" in mezzo, che sul grafico si legge come una
posizione aperta due volte e mai chiusa. Il trade era giusto, la didascalia no.

Ora il marcatore è indicizzato sullo **stato in cui la regola entra** — `long`, `short`, `flat` —
quindi due marcatori consecutivi uguali sono strutturalmente impossibili. Il libro fattoriale
graduato tiene la vecchia lettura: non ha stati da nominare, solo più e meno.

### Un secondo bug trovato mentre si verificava

`atr_pct` andava in `IndexError` dentro `ta` su un frame più corto della finestra ATR: la libreria
scrive `atr[window - 1]` in un array più corto di così. Sulla pagina significa che con una barriera
in ATR e poche giornate di storia la pagina cadeva. Ora un simbolo con meno barre della finestra
resta NaN, che `width` traduce in nessuna barriera.

### Un terzo bug, in produzione: `ModuleNotFoundError: No module named 'scipy'`

La pagina è caduta su una didascalia, `pred.corr(target, method='spearman')`. `Series.corr` con
`method="spearman"` **importa scipy pigramente**, dentro `pandas.core.nanops`: l'import non si vede
a import-time e non fallisce in nessun venv che scipy ce l'ha. E scipy in questo progetto arriva
**solo come dipendenza transitiva di lightgbm**, che sta nel gruppo dev di proposito — quindi la
riga funzionava ovunque tranne che sull'unico host che conta. Era lì da mesi.

`metrics.spearman(a, b)` è il rimpiazzo: Pearson sui ranghi, che *è* Spearman. Il `dropna` va prima
del `rank` e non dopo — `Series.corr` scarta le coppie dove uno dei due è NaN, quindi rankare ogni
serie sulle sue righe valide darebbe un numero diverso. Verificato su 300 forme casuali con pattern
di NaN diversi fra le due serie: **identico a pandas a piena precisione**, non "vicino".

Convertiti `chart.py` (il crash) e i quattro call site di `legcheck.py` (stessa violazione, modulo
non deployato ma la regola è di progetto). I numeri di `legcheck` nello spec non cambiano.

`tests/test_deploy.py` è la guardia, in due metà che si coprono a vicenda:
- **runtime** — il grafo di import della pagina, letto dal sorgente di `chart.py` con l'AST così si
  aggiorna da solo, girato in un sottoprocesso con `sys.modules["scipy"] = None`. È l'host di
  deploy in miniatura, ed è ciò che sorveglia `metrics`, l'unico modulo autorizzato a chiamare
  quella di pandas.
- **sorgente** — la chiamata bandita ovunque tranne `metrics`, letta dall'albero sintattico e non
  dal testo (altrimenti i commenti che spiegano la regola la fanno scattare). Serve perché un
  import pigro sta su un ramo che nessun test percorre.

Entrambe verificate per mutazione: rimesso `method="spearman"` dentro `metrics.spearman`, il test
runtime fallisce.

`scipy>=1.14` è ora **dichiarato** nel gruppo dev. `selection.py` importa `scipy.cluster.hierarchy`
direttamente e nessuno lo dichiarava: il giorno che lightgbm smette di tirarselo dietro, `selection`
ne ha ancora bisogno. Dev e mai runtime.

### Perché la predizione appariva a tratti, e tre bug che stavano dietro

Segnalato su DOGE/USD: la riga arancione della predizione compare solo a pezzi. Non è stato
possibile riprodurlo sui dati veri — da questa sessione la rete verso Alpaca è bloccata — quindi
sotto c'è quello che è stato **stabilito dal codice e misurato su serie sintetiche**, non dedotto.

**Il meccanismo.** `gru.predict_frame` marca una barra scoribile solo se **tutte** le
`steps × len(keep)` celle della sua finestra sono finite: una rete ricorrente non ha modo di essere
avvisata che una cella manca. Una sola feature non finita annulla quindi le 24 barre che la
leggono, ed è per questo che i buchi sono larghi e a blocchi invece che sparsi. Sul percorso reale
(checkpoint `data/gru.pt`, serie sintetica in stile DOGE — prezzo basso, quantizzato al tick, con e
senza barre mancanti) la copertura è **97,5%**: il percorso funziona, quindi la causa sta nei dati
della coppia o nella lunghezza della finestra, non nel modello.

**Quello che la pagina non diceva.** Un buco disegnato e basta è indistinguibile da un modello
rotto. `gru.coverage` ora separa le tre cause — warm-up in testa, buchi interni con il nome delle
colonne responsabili, e copertura totale — e la pagina scrive un avviso quando la copertura scende
sotto il 90%. Legge lo **stesso** tensore di `predict_frame` (entrambi passano da `_inputs`), così
la diagnosi non può descrivere un frame diverso da quello sullo schermo.

**Bug 1 — `features` andava in IndexError sotto 2·EXTREMA_WINDOW+1 barre.** `ta` scrive
`adx[window]` in un array che ha già tagliato di `window` righe. Raggiungibile dalla UI in un clic:
History 1 giorno + timeframe 4h fa **sei barre**, e la pagina moriva con un traceback. Ora torna
tutto NaN con indice e colonne intatti, che è ciò che ogni consumatore a valle già sa leggere.
`MIN_BARS = 2 * EXTREMA_WINDOW + 1`, misurato e non derivato dal sorgente della libreria: 48 barre
sollevano, 49 no. Verificato l'intero percorso della pagina su 1, 3, 6, 40, 49, 120 e 900 barre.

**Bug 2 — i buchi di Alpaca erano tappati in `panel` e non in `get_candles`.** Il modulo li
documenta e li misura (su 60 giorni a 15m: DOGE 63 barre mancanti, ETH 76, LTC 48, SOL 25) e li
riempiva solo per il percorso cross-sectional. Il grafico disegna l'altro. Ora `candles.complete`
mette una coppia su una griglia senza buchi e `get_candles` ci passa: 24 barre da 15m che
silenziosamente coprono nove ore sono un input diverso da quello su cui i pesi sono stati
addestrati, e niente nel frame lo diceva. Un bucket senza scambi ha open = high = low = close
precedente e volume **zero** — non si fa `ffill` di high e low separatamente, perché inventerebbe
uno stoppino che nessuno ha stampato e `stops` legge esattamente quei massimi e minimi per le
barriere. Il volume zero è anche il marcatore con cui si contano le barre riempite.

**Bug 3 — la combo box mostrava cinque coppie.** Ora sono le **venti dello studio**
(`data.binance.SYMBOLS` quotate in USD, stesso ordine: un pair sul grafico dovrebbe essere un pair
su cui i numeri dello spec sono stati misurati) e la casella accetta anche una coppia digitata.
Quali di esse Alpaca elenchi davvero non è una cosa che questo progetto possa sapere — la copertura
del venue è più stretta di quella di Binance e cambia — quindi la lista è un punto di partenza e
non un'affermazione sul venue: una coppia che Alpaca non serve fa scattare l'avviso che c'era già.

---

## 12. La griglia smoothing × leg window, misurata tutta (`legsweep.py`)

**Domanda.** Le due manopole di `swing_leg_target` non erano mai state misurate contro il modello:
la finestra dei pivot (24, calibrata in `oracle` sul P&L di indietro penalizzato dal ritardo — una
domanda sulla struttura del mercato, non su cosa una rete ricorrente riesce a ordinare) e il
`peso_tempo` (0,7, documentato in `data/target.py` come "a starting value, tunable", e mai tunato).
Griglia completa: 9 smoothing × 13 finestre = **117 celle, tutte addestrate**, 45 minuti.

**Cosa si muove e cosa no.** Le feature restano su `EXTREMA_WINDOW = 24` in ogni cella e il tensore
`step3-15m-all.npy` è sempre lo stesso file mappato: si muove solo il target. È il contratto di
`dataset.relabel`, scritto lì — ricalcolare l'etichetta sulle righe che lo step 2 ha già scelto
tiene il campione identico, quindi l'unica differenza fra due celle è la domanda posta. `next_pivot`
**viene** ricalcolato per finestra, cosa che `relabel` non fa: le gambe non sono più le stesse
gambe, quindi l'orizzonte di purging non è più lo stesso orizzonte, e uno split non purgato a
finestra lunga perde attraverso il taglio che esiste per proteggerlo.

**Protocollo veloce, e dichiarato tale.** Uno split temporale invece dei quattro fold walk-forward,
un seed, griglia di campionamento a 4h invece che oraria (un quarto delle righe, cross-section
intere — si assottiglia per *timestamp* e mai per riga, perché la metrica è cross-sectional),
15 epoche con patience 3. La cella corrente riproduce il riferimento: **IC 0,4188 / Rank IC 0,4083**
contro 0,4310 / 0,4114 del protocollo pieno a 4 fold e 5 seed. Un seed vale 0,001 (misurato su tre).
I numeri di questa griglia **non** sono confrontabili con quelli dello spec: è un ordinatore, non
una misura.

### Il risultato

| cella | IC | Rank IC | `rsi_centered` | edge |
|---|---|---|---|---|
| 0,7 / 24 — attuale | 0,4188 | 0,4083 | 0,3785 | 0,0298 |
| **0,2 / 12 — massimo** | **0,5033** | **0,4753** | 0,4424 | 0,0330 |
| 0,9 / 60 — minimo | 0,3439 | 0,3254 | 0,2232 | **0,1022** |

IC e Rank IC hanno lo stesso massimo e **la superficie non ha nessun ottimo interno**: sale
monotonamente verso finestre corte e pesi-tempo bassi, in entrambe le direzioni, su tutte e 117 le
celle. Su tre seed alla finestra 12: 0,4741 ± 0,0006 a smoothing 0,2, 0,4731 ± 0,0009 a 0,1,
0,4708 ± 0,0010 a 0,3 — quindi 0,2 batte davvero 0,3 ed è testa o croce contro 0,1. La risposta
alla domanda posta è **finestra 12, peso tempo ≤ 0,3**, e vale +0,067 di Rank IC.

### Perché non basta, e il controllo che lo dice

Ogni cella misura anche `rsi_centered_15m` — un indicatore grezzo — sulla **stessa** etichetta e
sulle stesse righe. Serve a rispondere a "quanto di questo Rank IC è l'etichetta che diventa più
facile da ordinare": una finestra più larga fa un'etichetta più lenta, e un'etichetta più lenta si
ordina meglio a parità di modello.

`edge = rank_ic − base_rank_ic` sale monotonamente **nella direzione opposta**, fino a 0,1022 a
0,9/60. Le due superfici sono angoli opposti della stessa griglia e nessuna delle due ha un massimo
interno. Alla cella migliore un RSI grezzo legge 0,4424 dei 0,4753 del modello: **il 93% del vincitore
è un indicatore per cui il modello non serve**, contro il 92,7% della cella attuale. I +0,067 di
Rank IC valgono **+0,003 di edge**, cioè tre seed.

**Quindi: non spostare il progetto su 0,2/12 su questa base.** Comprerebbe un numero più grande e
non un grammo di segnale in più, ed è esattamente la distinzione che la sezione 3 di questo handoff
ha già pagato — la predizione che fa 0,4114 contro l'etichetta swing fa **−0,0405 contro il
rendimento forward**. Se la griglia va rifatta, va rifatta su un giudice che sia il prezzo
(`legcheck`, `swingrule`), non la correlazione con l'etichetta.

### Cosa c'è a terra

- `data/legsweep.csv` — una riga per cella: `ic, icir, rank_ic, rank_icir, base_rank_ic, delta, rows, seconds`.
  `python -m tradingvision.legsweep --table` stampa le tre matrici (Rank IC, IC, edge).
- `data/gru-swing-s<smoothing>-w<finestra>.pt` — **117 checkpoint**, uno per cella. Il nome porta i
  parametri perché la pagina deve decidere *prima* di aprire qualcosa quale dei 117 file caricare.
- `data/legsweep-labels/w<finestra>.parquet` — cache delle etichette, 13 file da ~50 MB. Usa e getta:
  si ricostruisce in pochi secondi per finestra.
- La pagina: lo slider dello smoothing ora ha passo 0,1 (ogni posizione è una cella addestrata) e
  c'è **Leg window (label pivots)** sulle 13 finestre. I due insieme scelgono il checkpoint, il
  label disegnato, i pivot, e i due oracoli. Su una cella che nessuno ha addestrato la pagina **non**
  ripiega su un altro modello — scrive quale comando la addestra — perché un modello adattato
  all'etichetta a uno smoothing disegna una riga che contro l'etichetta a un altro smoothing
  significa un'altra cosa. `gru --save data/gru-swing-s0.20-w12.pt` sovrascrive una cella col fit
  pieno a quattro fold, che è ciò che il vincitore si merita se e quando lo si sposta.

### Non rifare

- **Non rifare la griglia muovendo anche la finestra delle feature.** Costa step 2 e il tensore
  ricostruiti per finestra (16 + 4 minuti × 13) e confonde "etichetta migliore" con "input migliori".
- **Non leggere `rank_ic` da solo.** L'argmax di Rank IC è la cella in cui l'etichetta somiglia di
  più a un RSI. `--table edge` è la colonna che lo dice.

---

## 13. Passata di revisione post-PR #14 — bug, cache, prestazioni

Nessuna misura nuova e **nessun numero di questo documento o dello spec cambia**: le correzioni
sono di codice e di infrastruttura, e quella sulle prestazioni è verificata identica bit per bit.
`uv run pytest -q` → **77 passed** (prima: 76 passed, 1 failed).

### I due che rompevano qualcosa

**La pagina andava in crash sul proprio percorso principale.** In `chart.py` il libro del modello
swing legava `per`, che è lo stesso nome del frame del libro fattoriale cinquanta righe più sotto.
L'etichetta retrospettiva non passa mai dal ramo cross-sectional, quindi `per` restava quello dello
swing e il blocco `if per is not None` in fondo alla pagina ripartiva: ridisegnava il libro swing
sotto la didascalia del fattore e poi moriva su `factor.swap(0)` → `ZeroDivisionError`, con `pos`
mai definito dietro. Si attiva con etichetta retrospettiva + timeframe 4h + `models/swing.pt`, che
è esattamente la configurazione che il checkpoint committato serve. Rinominato in `swing_per`;
verificato sulla pagina vera (BTC/USD 4h, 175 giorni) che ora arriva in fondo senza traceback.

**`test_gru_selfcheck` falliva su qualunque macchina con Metal.** Il self-check costruiva `Net` su
CPU e gli passava tensori `.to(DEVICE)`. In CI `DEVICE` è `cpu`, quindi era verde lì e rosso sulla
macchina che addestra — il caso peggiore per un test. Una `.to(DEVICE)`.

### Le tre cache che non invalidavano

- **Il modello sulla pagina non si aggiornava mai dopo un riaddestramento.** `load_prediction` e
  `load_coverage` portano `_mtime` nella chiave proprio per quello, ma chiamavano `load_model(path)`,
  che era cachata sul solo path: ricalcolavano la predizione sui pesi vecchi. Risposta fresca da un
  modello vecchio, che è peggio di una risposta vecchia. `load_swing_model` aveva già la forma
  giusta; `_mtime` ora passa anche di lì.
- **`legsweep.cached_labels` non aveva timbro.** Rileggeva il parquet controllando solo le colonne,
  e `rows_of` allinea quel frame a `df` **posizionalmente** (`lab.iloc[:, 1].notna().to_numpy()`).
  Un file di etichette costruito contro uno `step2.parquet` precedente non avrebbe sollevato niente:
  avrebbe rietichettato l'intera griglia contro le barre sbagliate. Ora confronta anche l'indice,
  che qui *è* la provenienza (`labels` finisce su `reindex(index)`), quindi non serve un JSON
  accanto come per `dataset.cached` e `gru.cached_sequences`.
- **`stops.run` calcolava l'ATR del pannello due volte** — stessa colonna, stessa finestra, una per
  barriera. `--grid` ne pagava quattordici passate invece di una.

### `features` è 4,6x più veloce, e il PSAR è identico

Profilando `features` su 93.696 barre: **PSAR era 9,4s di 10,8s**. `ta.trend.PSARIndicator` esegue
la ricorsione di Wilder con `.iloc` scalare di pandas in get *e* in set. Portato a un loop su
array numpy in `features._psar`, con i default `step`/`max_step` di `ta`.

Verificato **bit per bit** contro `ta` su tutte le 20 coppie × 3 timeframe dal 2023 — 60 serie,
zero differenze. L'uguaglianza è ora un assert nel self-check del modulo, quindi `ta` resta la
definizione e il porting non può derivare in silenzio. Serve che sia esatto e non "vicino": la
ricorsione è path-dependent, una barra di scarto si propagherebbe fino in fondo alla serie e ogni
numero già misurato su quella colonna sarebbe un altro numero.

| | prima | dopo |
|---|---|---|
| PSAR su tutto lo store (60 serie) | 304s | 5s |
| `features`, 93.696 barre | 3,11s | 0,67s |
| `uv run pytest -q` | 192s | 105s |

### Un numero sbagliato in una didascalia della pagina

`chart.py` scriveva che la regola nuda fa **−89% l'anno**. Quel numero non esiste da nessuna parte:
la misura è **−0.458 log l'anno** a ±0.5 (sezione 9, e §9 dello spec). Il −89% è il *drawdown* di
SUSHI, che nello spec sta nella frase accanto. Corretto alla cifra misurata.

### Cosa è stato guardato e lasciato stare

Tre cose sono sbagliate ma correggerle sposterebbe un numero già pubblicato, quindi sono decisioni
di chi riprende e non correzioni silenziose:

- **`swing.baselines` taglia le candele dalla prima riga di *test***, quindi ogni regola a una
  colonna sta piatta durante il proprio warm-up delle feature (~49 barre) mentre il modello no. È
  conservativo — penalizza la baseline, che vince lo stesso — ma sposterebbe il +0.116 della tabella.
- **`nearpivot.rank_ic` riporta un `mean/std*sqrt(n)` ingenuo** su date sovrapposte, cioè
  esattamente ciò che `metrics.blocked` esiste per vietare. Impatto basso (la banda è < 24 barre),
  ma contraddice la regola di progetto.
- **`simulation.sweep` campiona per posizione** (`np.arange(len(close)) % step == 0`) dove
  `factor.hourly` campiona sull'orologio. Sicuro sull'indice-unione, incoerente con la regola che
  `dataset.build` documenta.

---

## 14. Swing Leg Position v2 — pivot e feature a 12, peso tempo 0,5, 48 barre (2026-09-28)

**Configurazione, decisa e non misurata.** `swing --timeframe 15m --window 12 --smoothing 0.5
--inputs reduced --steps 48 --stage label --test-start 2025-06`, sulle 15 coppie di `SYMBOLS` dal
2021. La finestra 12 è sia quella dei pivot dell'etichetta sia quella delle feature: ogni colonna
che derivava da `N` è ricalcolata a 12 (ATR, ADX, RSI, TSI, massimi/minimi di finestra, volatilità,
EMA e sigma di `legs.exhaustion`). Lo stato della gamba segue la finestra per proporzione,
`swing.scales(w) = (w/4, w/2, w)`: a 24 resta (6, 12, 24), a 12 diventa (3, 6, 12), quindi
`reduced(12)` ha `signed_move_3/_6/_12` al posto di `_6/_12/_24` e resta a 15 colonne. La
significatività dell'etichetta legge ancora 96 barre di volatilità (`SIGNIFICANCE_LOOKBACK`): è
parte della definizione dell'etichetta e non una feature, e la pagina disegna la stessa.
3.013.026 righe × 48 × 15, tensore di righe da 181 MB.

### I numeri (4 fold da 2025-06, 1 seed, fuori campione)

| lettura | valore |
|---|---|
| ρ con l'etichetta, per fold | 0,652 / 0,663 / 0,624 / 0,640 — pooled **0,644** |
| `rsi_centered` a 12, stesse righe | 0,596 → il modello aggiunge 0,048, **il 92,5% è un RSI** |
| ρ con il rendimento forward neutrale a 4 / 12 / 48 barre | **−0,026 / −0,026 / −0,023** |
| decili del forward a 12 barre | forma a U, estremi +1,4 / +2,6 bps (t 1,3 / 1,7) contro 50 bps di round trip |
| banda long-only scelta in validazione, 13 tradabili | lordo +0,033, **netto −0,275** log/anno, 81 trade/anno, buy and hold −0,263 |
| stadio policy (`--stage both`), 4 fold su 4 | converge su **zero trade** |
| always-in / long-flat, 30 configurazioni, soglia scelta sui fold precedenti | netto −0,09 / −0,28 / −1,02 sui fold 2-4, lordo ≈ 0 |
| `rsi_centered` a 12, regole a una colonna | −2,5 … −10 netti: turnover di 670-2500 trade/anno |
| l'etichetta conosciuta perfettamente, always-in | +11,7 / +19,5 log/anno — il tetto sta tutto nel futuro che l'etichetta legge |

**Le gambe a 12, test period, 13 tradabili:** ~1.400 gambe/anno per coppia, gamba mediana
**2,37%** contro lo 0,50% di round trip (BTC 1,05%), 18-21 barre. Oracolo col senno di poi +15,8
log/anno; lo stesso oracolo che entra 12 barre dopo il pivot, il primo momento in cui lo si può
sapere, fa **+0,11**, ed è negativo su BTC, ETH, SOL, XRP e LTC. È il soffitto di qualunque regola
che reagisca ai pivot confermati, e a 12 non paga il costo.

**Perché 0,644 non vale niente.** La predizione non è in ritardo sull'etichetta: la correlazione di
`pred(t)` con `target(t−k)` ha il picco a k = 0 (0,651), e anzi è più alta con i 6 bar successivi
(0,466) che con i 6 precedenti (0,307). Il modello sa *dove sta sulla gamba* e un po' dove andrà
l'etichetta — ma a peso tempo 0,5 metà dell'avanzamento dell'etichetta è un orologio, e prevedere
l'orologio non dice nulla sul prezzo. Il rendimento forward è la lettura che conta, ed è zero.

**Il checkpoint.** `data/swing-v2.pt` (copiato in `models/`): stadio label, fittato fino al
2025-06, con la calibrazione e — novità — lo scaler di ogni simbolo di train. La pagina lo disegna
dietro il checkbox **Swing Leg Position v2** sotto l'etichetta retrospettiva: blocca smoothing e
finestra ai valori del checkpoint, porta la finestra delle feature a 12, spegne GRU e swing v1, e
la regola always-in legge la predizione v2 calibrata.

### Leak e difetti trovati, e corretti

1. **Il purging sul pivot successivo perde.** L'etichetta interpola verso un pivot che è
   definitivo solo quando la sua serie di estremi dello stesso tipo si chiude, cioè `window` barre
   dopo il primo estremo grezzo di tipo opposto. `legs.label_reach` sostituisce `legs.next_pivot`:
   a 12 l'etichetta legge p50 46 / p95 104 / p99 153 barre avanti, contro 13 / 60 / 100 fino al
   pivot. Il purge vecchio lasciava nel train 350-700 righe per taglio etichettate con prezzi del
   test. `legs._selfcheck` fissa la proprietà per troncamento, e mostra che il vecchio orizzonte la
   viola. **Il modello A (`dataset`, `legsweep`) purga ancora sul pivot** — task separato.
2. **Lo scaler leggeva il primo mese di test.** `f.loc[:"2025-06"]` con una stringa parziale
   finisce a *fine* giugno. Ora il confronto è stretto.
3. **La pagina scalava sulla finestra a schermo.** Trenta giorni di quartili al posto di quattro
   anni, ricentrati sul regime locale e calcolati anche sulle barre *dopo* quella predetta: la
   linea sulla pagina non era la predizione misurata nel walk-forward. Ora `save` spedisce lo
   scaler di train di ogni simbolo e `predict_frame(..., symbol)` lo usa; `live_scaler` resta solo
   per una coppia mai vista, e la pagina lo dice.
4. **Un checkpoint di stadio label veniva prenotato sulla linea calibrata** con una banda scelta
   sull'uscita grezza (±0,4 contro ±1): un'altra regola. Ora la regola legge l'uscita grezza e la
   calibrazione serve solo a disegnare.
5. **`cached` con `with_suffix`** metteva `-m0.50` e `-m0.60` sotto lo stesso `-m0`: ora aggiunge
   l'estensione. Lo stamp registra `smoothing` e `build = 2`, quindi i tensori `swing` costruiti
   prima delle correzioni 1-2 vengono rifiutati e non riletti.

### Cosa serve per ottenere di più a valle — in ordine di quanto sposta

1. **Il costo, prima del modello.** A 12 barre di 15m la gamba mediana è 4,7 round trip e
   l'oracolo raggiungibile è +0,11: nessun modello che reagisce può pagare lo 0,25% per lato. Le
   leve sono il costo (ordini maker, un venue più economico) o gambe più lunghe per trade (1h/4h,
   dove `rsi_centered` a 4h fa +0,116 netto).
2. **Un'etichetta che sia già il trade.** Il 92,5% di ρ è RSI e l'altra parte è l'orologio del
   peso tempo. Ciò che rende il tetto enorme sono le 46 barre di futuro che l'etichetta legge; un
   target sul rendimento netto dei costi (barriera tripla, o `move_balance` sui rendimenti oltre
   il round trip) misura direttamente la sola cosa che paga.
3. **Scegliere la regola sulla validazione del fold, non sul test.** Sulle soglie scelte sui fold
   precedenti il netto è negativo in 3 fold su 3; la stessa griglia letta su tutto il test ha righe
   meno negative, nessuna positiva, e sceglierne una a posteriori sarebbe la promozione per test.
4. **Seed ed ensemble non servono qui.** Riducono la varianza di un segnale che non c'è: il
   forward è −0,026 a ogni orizzonte.

### Non rifare

- Non leggere il ρ con l'etichetta come un risultato: 0,644 contro 0,596 di un RSI grezzo.
- Non ripetere la run policy a 12/15m senza cambiare il costo: converge su zero trade, che è la
  risposta giusta alla domanda che le viene posta.

---

## 15. Due target su un encoder — `--label swing+balance` (2026-09-28)

**Domanda.** Se l'encoder impara la struttura delle gambe (etichetta swing) *e* il movimento che
viene dopo (`move_balance` a 48 barre) nello stesso momento, la seconda testa legge il prezzo
meglio di un modello che vede solo il secondo?

**Come.** `swing.Net(aux=True)` ha tre teste sull'encoder: `head` impara `move_balance` — è
`target`, quindi la leggono la regola, la calibrazione e l'early stopping —, `aux` impara
`swing_leg_target` (`target2`), `policy` resta quella di prima. Loss = Huber₁/δ₁ + `aux` ·
Huber₂/δ₂, ognuna divisa per il suo δ perché nessuna scala decida il mix. Purging sul più lontano
dei due orizzonti. Tutto il resto è v2: 15m, finestra 12, peso tempo 0,5, `reduced(12)`, 48 barre,
4 fold da 2025-06, 1 seed. Il riferimento è **la stessa run con `--aux 0`**: stessi input, stesse
righe, stesso purging, la testa ausiliaria senza gradiente — l'unica differenza è il compito
ausiliario.

| | aux 1 (congiunto) | aux 0 (solo forward) |
|---|---|---|
| validazione, correlazione con `move_balance`, per fold | 0,020 / 0,038 / 0,016 / 0,016 — **0,022** | 0,048 / 0,032 / 0,014 / 0,003 — **0,024** |
| test, ρ con `move_balance` per fold | +0,013 / −0,016 / −0,027 / +0,046 | +0,006 / −0,022 / −0,038 / +0,054 |
| test, ρ pooled | +0,008 | −0,017 |
| testa ausiliaria contro l'etichetta swing | **0,640** (v2 da solo: 0,644) | −0,101 (non addestrata) |
| ρ col forward neutrale a 12 / 48 barre | −0,003 / −0,007 | −0,000 / −0,002 |
| banda long-only scelta in validazione, 13 tradabili | lordo +0,006, **netto −0,253** | lordo −0,524, netto −0,718 |
| regola scelta sui fold precedenti, fold 2 / 3 / 4 | −1,34 / −1,46 / +0,70 | −0,00 / −0,65 / +0,05 |
| buy and hold, fold 2 / 3 / 4 | −1,52 / −1,21 / +0,95 | |

**Risposta: no.** L'encoder condiviso impara l'etichetta quanto v2 da solo (0,640), quindi la
struttura delle gambe ce l'ha; non passa alla testa del prezzo. La correlazione di validazione non
distingue i due pesi (0,022 contro 0,024: scelto sulla validazione vincerebbe `aux 0`), quella di
test cambia segno fra un fold e l'altro in entrambe le run, e il forward neutrale è piatto. Il netto
migliore della run congiunta (−0,253, 25 punti sopra il buy and hold) è dentro la dispersione fra
coppie (sd 0,48) con un lordo di +0,006; il +0,70 del fold 4 è un'esposizione di 2 trade l'anno in
un fold dove il mercato fa +0,95.

**Cosa resta in piedi.** Il multi-task è implementato e testato (`_selfcheck` fa imparare a ogni
testa il suo target sul giocattolo), e i checkpoint salvati prima si caricano ancora: la terza
testa esiste solo nei modelli congiunti. Con gli stessi 15 input a 12/48, il prezzo delle 12 ore
dopo non si legge né da solo né con l'etichetta come maestra; la leva, se c'è, è negli input o nel
costo, non nella loss.

---

## 16. Pulizia: solo etichette retrospettive (2026-10-03)

**Decisione.** Le etichette predittive sono state misurate contro il prezzo tre volte, e il
risultato è sempre nullo o non pagabile:

| Etichetta | Contro il prezzo |
|---|---|
| `remaining_excursion` | −0,033 |
| `move_balance` | +0,0015, t 0,27 |
| `move_balance` con lo swing come compito ausiliario | da −0,027 a +0,046 fra i fold |

Il progetto lavora quindi solo su `swing_leg_target` e sulle strategie che leggono la sua
previsione. Tutto il resto è in `OLD/`, congelato al tag `archive-predictive` (`983c4eb`).
`OLD/README.md` elenca ogni file con il numero che lo ha chiuso e spiega come rieseguirlo.

**Cosa è uscito, in breve:**

- **Le tre etichette predittive.** `data/target.py` → `OLD/.../data/target_predictive.py`.
- **La prima pipeline:** `dataset`, `split`, `linear`, `gbm`, `selection`, `nearpivot`,
  `crosscheck`, `gru`, `legsweep`, `legcheck`, `exhaustcheck`.
  - `gru.pt` e le celle di `legsweep` erano sull'etichetta swing. Escono lo stesso, per tre ragioni:
    - il loro campione è quello di `remaining_excursion`;
    - il loro purge legge il pivot successivo, non `legs.label_reach`;
    - `swing.py` fa lo stesso lavoro con il purge corretto.
  - Il task che chiedeva di correggere quel purge è decaduto.
- **Il fattore cross-sectional e la simulazione** (`factor`, `simulation`). Il fattore **non** ha
  fallito: è l'unico netto positivo del progetto, +0,238 l'anno a 25 bp, misurato su `STUDY`. Esce
  per concentrazione, e il README di `OLD/` lo dichiara come eccezione.
- **Nei moduli vivi:**
  - `swing.py` perde `--label balance`, `--label swing+balance`, `--aux` e la terza testa;
  - `features.SELECTED` (la coppia del fattore) è rimossa;
  - `lightgbm` esce dal gruppo dev.
- **Dalla pagina** escono:
  - il selettore dell'etichetta;
  - le heatmap e il libro del fattore;
  - la linea GRU e le celle della griglia;
  - il modello `move_balance`.

  La pagina disegna l'etichetta swing, il modello a 4h dello step 7, v2, la regola always-in con
  le uscite e l'oracolo. Spenta *All 29 candidates*, mostra le colonne di candela di `REDUCED`.

**Verificato.**

- ruff, black e 46 test passano. Erano 77: la differenza sono i test dei moduli archiviati.
- La pagina gira su BTC/USD in due configurazioni, senza errori in console né nel server:
  - 15m con v2: previsione, regola, Spearman 0,71 sulla finestra;
  - 4h con `swing.pt`: book, oracoli, regola.
- L'asserzione del Dockerfile ora controlla `swing.pt` e `swing-v2.pt`.

**Cache locali.** Il tag dei tensori di `swing` non contiene più l'etichetta. Per esempio
`swing-15m-reduced-swing-s48-…` diventa `swing-15m-reduced-s48-…`, e lo stamp non registra più
`label`. Le cache in `data/` costruite prima vengono quindi ricostruite al primo run; quelle
congiunte non servono più.

**Cosa resta vero.**

- Il retrospettivo non ha ancora guadagnato soldi. v2 predice l'etichetta a 0,644, ma il forward
  neutrale è −0,026 e la regola long-only perde contro l'hold (§14).
- Delle leve di §14 la 2, un target sul rendimento netto, è predittiva: è fuori strada.
- Restano le altre:
  1. il costo: ordini maker, gambe più lunghe per trade;
  2. la regola scelta sulla validazione del fold.

**Aperto, in ordine.**

1. **Le CLI di `threshold`, `stops` e `swingrule` leggono `pred-swing-*.parquet` nel formato di
   `gru`**, che ora non scrive più nessuno. Le loro funzioni le usa la pagina e sono testate, ma i
   `__main__` vanno ricollegati a `pos-swing-*.parquet` di `swing.py` prima di prezzare una
   strategia sulla v2.
2. **L'analisi di `legcheck` per `swing.py`.** Decili del forward per previsione e controllo
   parziale sull'età della gamba. Oggi sta in uno script fuori dal repo: va promossa a modulo.
3. **Il fattore, se si torna a una strategia di portafoglio**: va rimisurato su `SYMBOLS` (15),
   non su `STUDY`.

---

## 17. Lo studio delle regole sulla v2 (`strategy.py`, ramo `claude/strategy-study`, 2026-10-03)

**Domanda.** Una regola di trading sulla predizione v2 di `swing_leg_target` che guadagni, costruita
un passo alla volta: punto di ingresso, uscite, filtri, regimi. Senza commissioni, guardando il
lordo per trade contro l'andata e ritorno (50 bp taker, ~20 maker).

**Protocollo, fissato prima.** Predizioni fuori campione della v2
(`pos-swing-15m-reduced-swing-s48-t2025-06-w12-m0.50-label.parquet`). Asset: i tre tradabili con il
Rank IC più alto contro l'etichetta sui fold 1-2, ETH, BTC, SOL. Sviluppo sui fold 1-2, hold-out
sui fold 3-4. Regimi sulle date del ciclo BTC: ribasso dal massimo del 2025-10-06, rialzo dal minimo
del 2026-07-01.

| soglia 0,40, bp per trade (errore) | sviluppo | hold-out |
|---|---|---|
| banda, sempre in posizione | +1,7 (9,8) | −25,2 (13,9) |
| rientro nel range | +15,0 (10,1) | −42,9 (13,9) |
| rientro + stop 6 ATR | +12,4 (6,7) | −23,0 (7,6) |
| rientro + BTC sotto media 200 giorni | +41,0 (16,1) | −40,5 (14,9) |
| rientro + stop 6 ATR, poi solo segnale opposto | +26,1 (7,9) | −26,5 (8,9) controllo |
| svolte della predizione col senno di poi, finestra 12 | +180,4 | +164,4 |
| svolte della predizione alla conferma, finestra 12 | +4,8 | −3,3 |
| svolte col senno di poi, finestra 12, eseguite 2 / 4 / 6 barre dopo | +99,4 / +68,0 / +41,3 | +88,9 / +54,9 / +34,5 |
| rivelatore bayesiano (Shiryaev) sulle svolte, soglia 0,5 / 0,9 | −1,0 / +6,7 | +0,1 / −13,3 |
| zigzag (CUSUM) sulla predizione, h 0,2 | −0,3 | −1,5 |

**Risultati.**

- **Nessuna regola si ripete fra i fold.** Il guadagno dello sviluppo era il fold 2: il rientro fa
  −1,8 / +32,8 / −44,8 / −40,9 bp per fold. Nello sviluppo il ribasso coincideva con il fold 2, e
  l'hold-out ha detto che era il modello, non il regime.
- **Lo stop a 6 ATR è l'unico effetto con lo stesso segno nei due periodi:** riduce perdite e
  drawdown, non crea guadagno.
- **Le svolte della predizione sono nel posto giusto e arrivano tardi.** Col senno di poi valgono il
  96% del lordo dell'oracolo e battono quelle di `rsi_centered` in ogni fold e a ogni finestra (6, 12,
  24, 48). Alla conferma valgono circa zero, come quelle del prezzo e dell'RSI.
- **Il loro valore si consuma in poche barre** (`--hindsight ... --delay`). A finestra 12 resta il
  70% dopo una barra di ritardo, il 55% dopo due, il 23% dopo sei; a metà finestra il 20-25% a ogni
  finestra. Contro 50 bp pagano fino a 4 barre di ritardo a finestra 12 e 6 a finestra 24. Il
  vantaggio sull'RSI sta nella barra di svolta e nella successiva e sparisce a ritardo 3.
- **Quel valore è la geometria del rumore** (`--null`). Su prezzi ricostruiti con il segno di ogni
  rendimento estratto a caso le svolte valgono di più (+216-219 bp a finestra 12 contro +201) e la
  quota rimasta a ogni ritardo è la stessa entro 0,01-0,03: segue 1 − √(d/w), la legge di un
  percorso browniano dopo un estremo.
- **Riconoscerle in tempo reale non rende** (`detect.py`). Zigzag e rivelatore di Shiryaev (a priori
  dal livello della predizione, che alle svolte è 0,37 di mediana, e dall'età della gamba) trovano
  l'80% delle svolte con 3 barre di ritardo mediano e 0,35 falsi allarmi per svolta, ma rendono
  zero: le rilevazioni giuste fanno +31 bp, i falsi allarmi −80, come vuole il teorema d'arresto
  opzionale. Con soglie alte, positivo sui fold 1-2 e negativo sui 3-4, come ogni altra regola.
- **Combinarli riduce i falsi allarmi e non il risultato** (`--both`). Shiryaev 0,5 più un
  ritracciamento di 0,4 porta i falsi allarmi da 0,35 a 0,07 per svolta, ma le rilevazioni scendono
  da +31 a +16 bp e i falsi rimasti salgono da −80 a −138: sviluppo +2,8, hold-out −9,7. Sulle
  dodici combinazioni provate, sviluppo da +0,6 a +6,5 e hold-out da −0,8 a −15,6.
- **Separare gli allarmi veri dai falsi si può, e non rende** (`--features`). Sedici variabili note
  all'allarme, logistica sullo sviluppo: AUC 0,637, hold-out 0,629, dal 53% al 83% di allarmi veri
  fra il quintile peggiore e il migliore. Ma salendo di quintile gli allarmi veri guadagnano meno
  (+49 → +25 bp) e i falsi perdono di più (−68 → −99): ogni quintile fa fra −6,6 e +6,5.
- **Funding, open interest, posizionamento, flusso dei taker e book non aggiungono niente
  all'allarme** (`--futures`, `data/futures.py`, dump dei futures Binance). Da soli AUC 0,546 /
  0,525, con le sedici di prima 0,639 / 0,624; ogni quintile fra −11 e +6 bp. Contro il rendimento
  futuro due colonne hanno lo stesso segno nei quattro fold: l'open interest dietro al movimento
  contro il rendimento a 48 barre (+0,075 / +0,039) e lo squilibrio del book entro il 5% contro
  quello a 12 (+0,042 / +0,024). Piccole quanto l'IC della v2, e non ancora una strategia.
- **Open interest dietro al movimento** (`--oi`). Il segno del movimento delle ultime k barre per la
  variazione dell'open interest ha IC positivo col rendimento a 48 barre in tutti i fold (k = 24:
  +0,057 / +0,125 / +0,057 / +0,045); il solo momentum no. Dopo un movimento a 24 barre con open
  interest in salita le 48 barre dopo vanno nella sua direzione (+10 / +16 / +19 / +20 bp), con open
  interest in calo tornano indietro (−22 / −13 / −2 / −6). Letto fra 64 varianti, hold-out incluso;
  come regola (segui o contrasta oltre |z|, esci dopo 48 barre) fa da −6,9 a +0,7 bp sullo sviluppo
  e il fold 2 è negativo in tutte e dodici le varianti.
- **Filtro sul livello dei segnali** (`--gate`): tiene un long solo sotto −L e uno short solo sopra
  +L. Porta lo zigzag da 5.288 a 93 trade sullo sviluppo a L = 0,5, ma il lordo per trade resta
  intorno a zero (chiudendo sui segnali scartati: da −3,2 a +4,2 sullo sviluppo, da −9,2 a +0,9
  sull'hold-out, salvo L = 0,5 con +12,5 / +19,2 su 160 trade, che il rivelatore bayesiano non
  conferma). Sulla pagina è un'opzione dei due rivelatori.
- **Lo stop loss taglia falsi e veri insieme** (`--sl`). A 1 ATR i falsi allarmi passano da −80 a
  −42 bp, le rilevazioni giuste da +31 a +17; nessuno stop da 1 a 6 ATR esce da −1,4 / +1,6 bp.
- Le altre prove (media breve, take profit, soglie alte, filtro di volatilità, momentum, inversa
  della peggiore) sono nella docstring di `strategy.py`, con i numeri.

**L'hold-out è usato.** Letto il 2026-10-03 sulle cinque `CANDIDATES` e poi, come controlli
dichiarati, sulla regola "solo segnale opposto", sull'inversa della banda a 0,55 e sulle svolte.
Ogni regola nuova sui fold 3-4 sarebbe scelta su dati già visti.

**Cosa è cambiato nel codice.**

- `strategy.py`: il modulo dello studio, con `--candidates`, `--tp/--sl/--trail/--after`,
  `--filter`, `--invert`, `--smooth`, `--path`, `--hindsight [--causal | --delay [--null]]` e un `_selfcheck`
  registrato in `tests/test_selfchecks.py`. `play` esegue qualunque regola su qualunque
  predizione; `walked` è `stops.walk` su qualunque segnale, con filtro.
- `detect.py`: zigzag (CUSUM) e rivelatore di Shiryaev sulle svolte, stimati sullo sviluppo, con
  `match` (trovate, ritardo, falsi allarmi), `book` e `split`; self-check registrato.
- `chart.py`, 2026-10-04: **le regole leggono l'uscita grezza della v2** (`swing.predict_frame`
  espone `raw`), le unità dello studio. Prima la pagina metteva le soglie sulla linea calibrata,
  dove 0,40 vale 0,28 grezzo: il "rientro a 0,40" della pagina non era quello dello studio. Nuove
  regole: zigzag e rivelatore di Shiryaev (`detect.V2_FIT`), con le righe di ciò che la regola legge,
  dei segnali chiesti (pieni se dalla parte giusta della gamba, vuoti se falsi, giudicati sulle
  svolte col senno di poi) e della probabilità o del ritracciamento su cui il rivelatore decide.
- `chart.py`: la sezione **Trading rule** passa da `strategy.play` invece che da `stops.run`.
  Regole: banda, rientro, momentum, svolte alla conferma, svolte col senno di poi (sotto un avviso).
  Poi soglia o finestra delle svolte, media della predizione, inversione, filtro BTC sui giornalieri
  di Alpaca (scaricati con le candele), uscite. Nuove metriche: bp per trade con la commissione di
  pareggio, e la curva del capitale della regola. La pagina si apre sulla combinazione dello
  sviluppo: v2, rientro a 0,40, filtro BTC, stop 6 ATR e poi solo segnale opposto (+37,2 bp sullo
  sviluppo, −22,8 sull'hold-out). Con BTC sopra la media a 200 giorni il filtro è spento e la
  regola resta flat. **Con la v2 l'ATR degli stop è ora a 12 barre,
  la finestra del modello, invece di 24**: gli stessi multipli di ATR possono dare stop diversi
  da prima.
- Un bug trovato e corretto durante lo studio: `--filter` veniva ignorato dalla CLI con `--tp`. Le
  misure con filtro passavano da `--candidates`, che lo applicava, e non ne sono toccate.

**Aperto.**

1. **Il walk-forward della v2 dal 2023-01** (`--test-start 2023-01`): circa 3,7 anni fuori campione e
   due cicli, per provare regole nuove con selezione a rotazione sui fold precedenti. Non è stato
   cronometrato.
2. **Un'informazione che preveda la gamba successiva, non la posizione in quella corrente.**
   Riconoscere prima la svolta è stato provato (`detect.py`) e rende zero: il valore delle svolte è
   la geometria del rumore. Le colonne di esaurimento sono le uniche col segno giusto sul rendimento
   futuro in tutti i fold. Funding, open interest e book sono stati provati e non bastano. Una
   scala più lunga (gambe a 24-48 barre) richiede di riaddestrare la v2.
3. Le CLI di `threshold`, `stops` e `swingrule` leggono ancora il formato di `gru` (§16, punto 1).
   `strategy` legge già quello di `swing`.

