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
