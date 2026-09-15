# Handoff — ramo `claude/fervent-knuth-se8cw5`

Stato in una riga: **tre commit di sola strumentazione, self-check verdi, zero numeri nuovi.**
Nessuna delle misure che questo lavoro esiste per produrre è stata eseguita, perché la sessione
non aveva accesso ai dati. Il prossimo agente deve scaricare lo store e lanciare le misure
elencate in "Cosa misurare", in quell'ordine.

Base: `origin/main` a `9bdeb9f`. Diff: 915 inserzioni su 5 file, nessuna riga rimossa dalla
logica esistente.

---

## 1. Perché non c'è nessun numero

La policy di egress della sessione ha risposto **403 al CONNECT** su ogni host di dati di mercato:

```
data.binance.vision:443      403      data.alpaca.markets:443   403
api.binance.com:443          403      api.alpaca.markets:443    403
data-api.binance.vision:443  403
```

`data/` è gitignored e vuoto. Senza candele non c'è `step2.parquet`, senza quello non c'è
tensore, senza tensore non c'è modello, senza modello non ci sono predizioni da prezzare.
Il README del proxy dice di non aggirare un 403 di policy e di riportarlo: fatto.

Nota di ambiente, locale a quel container e probabilmente irrilevante altrove: anche
`download.pytorch.org` era bloccato, quindi `uv sync` falliva sul gruppo dev. Aggirato con
`uv sync --no-default-groups` più `uv pip install --index-url https://pypi.org/simple torch
lightgbm ruff black pytest`, che installa la build CUDA invece della CPU pinnata in
`pyproject.toml`. **Se `uv sync` funziona, usare quello e ignorare questa nota.**

### Conseguenza da tenere a mente leggendo il codice nuovo

Ogni numero citato nei docstring dei moduli nuovi viene da una di due fonti, mai da una misura
fatta in questa sessione:

- **citato dallo spec o da un docstring esistente** — es. la scomposizione long/short di
  `threshold`, il `-0.019` dello specialista sulla banda;
- **misurato su dati sintetici dentro un `_selfcheck`** — es. lo 0.28 Spearman in
  `legcheck._selfcheck`, che gira su un random walk generato lì.

Non c'è una terza categoria. Se un numero in un docstring nuovo sembra una misura su dati reali,
è un errore di scrittura: segnalalo e correggilo.

---

## 2. Cosa è stato aggiunto

### `3f8412f` — `src/tradingvision/swingrule.py`

La regola long-only che l'etichetta swing descrive: compra vicino al minimo della gamba, tieni,
vendi vicino al massimo, **poi aspetta** (flat, non short). `threshold.py` prezza la regola
simmetrica che è sempre long o sempre short; questa è l'altra.

- `positions(pred, buy, sell, sign=-1)` — macchina a stati 1/0 via forward fill sui due marker.
- `band(pred, q, window=0)` — soglia dinamica: quantile di `|pred|` sulle righe passate del
  simbolo, espandente per default. **Scale-free**, che è il punto: un riscalamento della
  predizione non sposta nessuna barra attraverso una soglia riscalata con essa.
- `spread(pred, target)` — misura la compressione (`sd_ratio`, `implied_rescale`).
- `pnl` riporta `win_rate` **accanto al suo `win_rate_se`**, perché trenta trade su un simbolo
  non sono una misura.
- `sweep(..., fixed=True)` congela le soglie sull'intero slice: il gap fra le due tabelle dice
  quanto costa impegnarsi su una costante.

### `027742b` — `src/tradingvision/legcheck.py`

Lo strumento senza regola dentro, così nessun risultato piatto può essere attribuito a una
soglia o a una commissione.

- `by_bucket(pred, fwd)` — rendimento forward medio per decile di predizione, con SE e `t`.
- `elapsed_position(pred)` — il controllo causale: viaggio dal pivot **già confermato**, che
  costa `EXTREMA_WINDOW` barre di ritardo (6 ore su 15m).
- `partial(pred, fwd, control)` — Rank IC prima e dopo aver proiettato via il controllo, in
  rank space. Proiezione lineare = controllo più debole possibile.
- `forward(close, h, neutral=True)` — market-neutral quando c'è un cross-section; su un simbolo
  solo ricade sul grezzo **e lo dichiara** invece di restituire zeri.

### `1f8345e` — `gru.py`, tre flag nuovi

| flag | cosa fa | rischio |
|---|---|---|
| `--weight {none,excursion,rank}` | pesa la loss per `\|remaining_excursion\|` | nessuno strutturale |
| `--finetune N --finetune-q Q` | seconda fase sul top-(1−Q) per ampiezza, da modello fittato, a `LEARNING_RATE * 0.1` | distribution shift |
| `--by-move` | splitta la Rank IC di test per ampiezza del movimento, a bucket quantilici | — |

`weights_of(move, scheme)` normalizza a **media 1**: senza, uno schema con pesi medi 3
misurerebbe un learning rate più grande quanto un'enfasi diversa.

La loss è passata a `reduction="none"` più media esplicita. **`--weight none` è bit-per-bit il
percorso di prima** — `_selfcheck` asserisce che stesso seed, `weight=None` e un vettore di uno
atterrano sugli stessi pesi della testa. Tutti i numeri già misurati dal progetto continuano a
leggersi.

---

## 3. Come ripartire

```bash
uv sync
uv run pytest -q                                  # atteso: 30 passed, 3 skipped
uv run ruff check . && uv run black --check .

uv run python -m tradingvision.data.binance       # riempie data/, ore la prima volta
uv run python -m tradingvision.gbm --horizon      # costruisce data/step2.parquet — gru NON lo costruisce
```

`gru.meta()` legge `step2.parquet` e basta: se non esiste, `gbm` va lanciato prima. Il tensore
(`data/step3-15m-all.npy`) lo costruisce `gru` da solo alla prima esecuzione, ~1.7 GB per ramo.

I 3 test skipped sono quelli che vogliono lo store: **dopo aver riempito `data/` devono girare
anche quelli.** Se restano skipped, lo store non è dove il codice lo cerca.

---

## 4. Cosa misurare, in ordine

### Misura 0 — il baseline, senza cui nessun confronto esiste

```bash
uv run python -m tradingvision.gru --label swing --branches 15m --features all \
    --seeds 5 --horizon --by-move
```

Serve per due cose: produce `data/pred-swing-all-15m.parquet` che tutto il resto legge, e fissa
la riga di paragone. Atteso dallo spec: Rank IC ~0.4215 ± 0.0154, ICIR ~1.74. **Se esce molto
diverso, fermati e capisci perché prima di andare avanti** — vuol dire che lo store o il sample
non sono quelli su cui lo spec ha misurato.

### Misura 1 — la domanda dell'utente, e la più decisiva

```bash
uv run python -m tradingvision.legcheck --pred data/pred-swing-all-15m.parquet --symbols BTC
uv run python -m tradingvision.legcheck --pred data/pred-swing-all-15m.parquet   # tutti i simboli
```

**Criterio di decisione, da fissare prima di guardare l'output:**

- Il decile più basso di `pred` ha `fwd_mean > 0` con `|t| > 2`, e il più alto `< 0` con
  `|t| > 2` → **il segnale guida davvero**, la tesi dell'utente regge, e il problema è la regola
  di trading, non l'etichetta. Vai alla Misura 2 con aspettative alte.
- Tabella piatta (tutti i `|t| < 2`, nessuna monotonia) → non c'è niente su cui una regola possa
  agire, e nessuna regola migliore lo cambia.
- `ic_residual` ≈ 0 mentre `ic_raw` è sensibile → l'informazione era la metà causale
  (`elapsed`), cioè aritmetica sul passato.

Riportare **entrambi** i run: BTC da solo è un percorso solo; venti simboli danno la
cross-section che le metriche del progetto richiedono.

### Misura 2 — la strategia long-only

```bash
uv run python -m tradingvision.swingrule --pred data/pred-swing-all-15m.parquet \
    --symbols BTC --since 2025-09 --fixed 0.4 0.8
```

Produce quello che l'utente ha chiesto: **WR, netto per trade, netto annuo** sull'ultimo anno
BTC, più `spread()` che quantifica la compressione [-0.5, +0.5], la tabella a soglie congelate e
`buy_and_hold` sulle stesse righe.

**Controllo da non saltare:** `buy_and_hold` è la riga che una regola long-only deve battere.
Un netto positivo sotto il buy-and-hold non è un risultato.

Prior dallo spec, da tenere accanto all'output: la gamba long della regola simmetrica è negativa
a **ogni** soglia della griglia (−0.198, −0.224, −0.065, −0.125 l'anno) mentre lo short fa da
+0.269 a +0.410. La strategia long-only entra ed esce negli stessi due punti di quella gamba.

### Misura 3 — pesatura e fine-tune

```bash
B="--label swing --branches 15m --features all --seeds 5 --by-move"
uv run python -m tradingvision.gru $B                        # = Misura 0, il baseline
uv run python -m tradingvision.gru $B --weight rank          # sicuro: nessun distribution shift
uv run python -m tradingvision.gru $B --weight excursion     # crede alla coda, clippata a 5 mediane
uv run python -m tradingvision.gru $B --finetune 20          # la versione forte, col rischio
```

**Criterio:** ciò che promuove è la **Rank ICIR aggregata con la dispersione sui fold**, come per
ogni altra scelta in questo progetto. `--by-move` è diagnostico, non decisionale: uno schema che
alza il bucket alto e abbassa gli altri non ha migliorato il modello, ha spostato dove sbaglia —
e l'aggregata lo dirà.

Cinque seed e quattro fold sono il minimo: la dispersione fra fold nel progetto è ~0.014, quindi
una differenza sotto quella soglia non è una differenza.

### Misura 4 — le leve mai toccate

Gli iperparametri sono ai valori di partenza dello spec e **nessuno è mai stato tunato**
(commento esplicito a `gru.py:82`). In ordine di rapporto valore/costo:

1. **`STEPS = 24`** — la leva teoricamente più forte *per questa etichetta*. La finestra è 6 ore
   su 15m, ma l'etichetta è dominata da `elapsed` (distanza dal pivot precedente) e le gambe sono
   spesso più lunghe: il purging misura distanze fino a 754 barre. Se la finestra non arriva
   all'inizio della gamba, il modello stima `elapsed` senza vederne l'origine.
2. **`H = 32`** — 32 unità nascoste per una sequenza 24×28. Prova 64 e 128.
3. **Rifare `selection` contro `swing`** — le 12 colonne attuali sono state scelte con un GBM
   contro `remaining_excursion`. È il motivo per cui `--features all` batte `selected`.
4. **Loss di ranking** — oggi ottimizzi Huber e selezioni su Rank IC. Il codice lo dice:
   *"Model selection and early stopping read Rank IC and never the loss"*. Una loss pairwise o
   Spearman soft ottimizzerebbe la metrica che riporti.

Una cosa alla volta, come dice lo spec.

---

## 5. Ragionamenti e conclusioni di questa sessione

Nessuna di queste è nuova evidenza. Sono letture del codice e dello spec, messe in fila perché
la conversazione le ha richieste tre volte.

### La compressione a [-0.5, +0.5] non è un bug

`Net` ha testa lineare e **nessuna attivazione in uscita** — la tanh è stata rimossa perché
saturava sui pivot. È shrinkage verso la media condizionata, ottimo sotto perdita quadratica:
`sd(E[y|x]) = corr(pred, y) · sd(y)`. Una predizione che arriva a metà del range dell'etichetta
è un modello con correlazione ≈ 0.5. Il numero **è** la skill.

Corollario operativo: **raddoppiare la predizione è una trasformazione monotona**, non sposta
nessuna barra attraverso una soglia riscalata con essa. "Raddoppio e compro a −0.8" = "compro a
−0.4". Conta solo dove sta la soglia dentro la distribuzione, ed è per questo che
`swingrule.band` usa un quantile. `_selfcheck` lo asserisce in codice.

### L'etichetta è per il 70% un orologio

`SMOOTHING = 0.7` (`data/target.py:48`) è il peso del termine **temporale**. L'etichetta è
≈ `elapsed / (elapsed + remaining)`: `elapsed` è passato e noto, `remaining` è futuro, e lo spec
ha misurato che il denominatore è quasi costante nel cross-section. Il docstring di
`remaining_excursion` lo chiude: *"Misura dove sta la barra, non dove sta andando il prezzo"*,
con Rank IC 0.38 di un OLS **piatta a ogni distanza dal pivot**, il che esclude un difetto di
pipeline e lascia aritmetica.

`crosscheck.py` è l'esperimento che ha già risposto all'obiezione, e il suo docstring la cita
parola per parola prima di smontarla: 0.3821 sull'etichetta retrospettiva → **0.0753** sulla
predittiva. Quattro quinti evaporano. Piccolo, non zero.

### "Senza lag" è precisamente il lookahead

Un pivot è `argrelextrema` con `order = 24`: estremo solo se batte 24 barre **su entrambi i
lati**. Un minimo a `p` non è identificabile prima di `p + 24` barre — sei ore su 15m — e può
essere **revocato** dopo, quando il merge trova una barra più estrema nella stessa run.
L'etichetta non ha lag perché le è concesso guardare lì. Causale e senza lag non coesistono.

### Il numero che chiude la strategia a soglie

Punto aperto 4 dello spec: *"quella distanza è prevedibile appena — **0.044 di Rank IC al
meglio, contro 0.157 sul target**"*. La strategia richiede di sapere che la gamba sta per
finire; quella quantità è stata misurata a 0.044.

Corredo: `close_position_in_window_15m` fa **+0.157** su tutto il train e **−0.063** dentro le 24
barre — inverte segno esattamente dove si vorrebbe tradare, e 66 colonne su 112 fanno lo stesso.
Lo specialista addestrato **solo** sulla banda, con tutte le 28 colonne, arriva a **−0.019** e non
passa lo zero: esclude la risposta architetturale, perché ha già tutta la capacità e l'early
stopping sulla banda.

### La carta migliore dell'ipotesi avversa, da non liquidare

Sul modello swing le fasce vicino al pivot sono **positive e forti** — +0.378 sotto le 6 barre,
+0.587 fra 12 e 24 — cosa che nessun modello sull'altra etichetta era riuscito a fare. Il limite
è che quelle fasce sono valutate **contro l'etichetta**, non contro il rendimento forward, e la
variabile che seleziona la fascia non è nota in tempo reale. Ma lo spec stesso dice che
l'informazione in banda esiste e generalizza (0.924 di correlazione per colonna fra train e
test) e che *"quello che manca è la variabile che dice al modello in quale dei due regimi si
trova"*. La strada indicata: **feature di esaurimento** — divergenza, climax di volume,
asimmetria dei wick — invece che di momentum.

**La Misura 1 è il giudice di tutto questo.** Se il decile basso esce positivo con `t > 2`,
l'argomento qui sopra è sbagliato e va riscritto.

### Reinforcement learning: non c'è, e non è lo strumento

Grep su tutto `src/`: nessun `reward`, `policy`, `actor`, `critic`, `PPO`. Il training è
supervisionato. RL ottimizza un reward: se è il P&L stai ottimizzando l'obiettivo di trading (che
ha misurato negativo), se è l'accuratezza è supervised learning con più varianza. Su ~200k righe
rumorose e un solo percorso storico il policy gradient overfitta prima di imparare. Quello che
l'utente descriveva — *rafforzare su un set ridotto* — è curriculum/fine-tuning, ed è `--finetune`.

---

## 6. Rischi e cose non verificate

- **Nessun percorso di `swingrule` e `legcheck` ha mai visto un file di predizioni reale.** I
  `_selfcheck` girano su serie sintetiche costruite nel test. Il primo run su `data/pred-*.parquet`
  può inciampare su forme dell'indice che il sintetico non riproduce — in particolare il
  MultiIndex sfasato fra simboli (punto aperto 5: 27.812 timestamp distinti, ~8 simboli per
  istante invece di 20, ~5% di righe su timestamp che nessun altro simbolo condivide).
- **`legcheck.elapsed_position` ignora la revoca dei pivot.** Il merge può eliminare a posteriori
  un pivot già confermato; il controllo non ne tiene conto ed è quindi, se mai, **troppo generoso
  verso la predizione**. Va bene come controllo debole, non va bene come simulazione real-time.
- **`--finetune` non è mai stato eseguito su dati veri.** Il `_selfcheck` verifica solo che la
  seconda fase parta dal modello fittato e cambi le predizioni. Il numero che conta non c'è.
- **`--by-move` usa `pd.qcut` con `duplicates="drop"`**: se `move` è molto degenere escono meno
  bucket di quelli richiesti, senza errore. Controlla `len(table)` prima di leggerla.
- **La sessione ha installato torch CUDA da PyPI** invece della build CPU pinnata. Se qualcosa di
  numerico non torna rispetto allo spec, questa è la prima cosa da escludere.
- **Lo spec `swing_dataset_schema.html` non è stato toccato.** Giusto così: nessuna misura è
  atterrata. **Appena la Misura 1 o la Misura 3 producono un numero, lo spec va aggiornato** —
  è la convenzione del progetto e la memoria del lavoro.

---

## 7. Cosa non rifare

Strade già chiuse con un numero, tutte nello spec:

- Quattro rami contro uno: il singolo 15m vince su tutti e quattro i fold.
- Encoder condiviso: 0.0007 di differenza contro 0.0140 fra fold. Irrilevante, non vincente.
- Specialista sulla banda: −0.019, e con esso ogni condizionamento sul regime più debole (loss
  pesata, testa multi-task, mixture of experts).
- Indicatori smussati a monte: una GRU su 24 step impara già qualunque filtro lineare della
  finestra, quindi a monte si aggiunge solo lag. Il filtro sta a valle (`threshold.smoothed`).
- Standardizzare ogni simbolo sul periodo per fissare una soglia: legge il periodo che sta
  tradando, cioè esattamente il leakage per cui i fold vengono purgati.
