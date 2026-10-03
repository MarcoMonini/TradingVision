# OLD — il lavoro predittivo, archiviato il 2026-10-03

Questa cartella contiene tutto ciò che il progetto ha costruito per **predire il prezzo**, più la
prima pipeline su cui era montato. Da qui in avanti il progetto lavora soltanto su etichette
**retrospettive** (`swing_leg_target`) e sulle strategie che le leggono. Il motivo è misurato e sta
nelle tabelle sotto: ogni etichetta predittiva, contro il prezzo, ha dato un segnale nullo o non
pagabile, con un'unica eccezione dichiarata.

**È un archivio congelato.** Nessun modulo vivo importa niente da qui. I file non girano dove si
trovano: importano moduli che nel frattempo sono stati spostati anche loro. Sono fuori da ruff,
black e pytest (`pyproject.toml`). Per eseguirli si torna al tag in cui girano tutti:

```bash
git worktree add ../TradingVision-archive archive-predictive
cd ../TradingVision-archive && uv sync
uv run python -m tradingvision.gru --seeds 5      # e ogni altro comando del CLAUDE.md di allora
```

Il tag `archive-predictive` punta a `983c4eb`, l'ultimo commit prima della pulizia. I file qui sono
identici a quelli del tag, spostati con `git mv`: `git log --follow` ne conserva la storia.

## Le etichette

`src/tradingvision/data/target_predictive.py` raccoglie le tre etichette predittive, copiate
verbatim da `data/target.py` insieme ai loro self-check. Nel modulo vivo restano
`swing_leg_target` e le funzioni di volatilità che lo pesano.

| Etichetta | Che cosa chiedeva | Il numero che l'ha chiusa |
|---|---|---|
| `remaining_excursion` | quanta escursione resta prima che la gamba finisca, in sigma | Rank IC 0,158 su sé stessa e **−0,033 contro il rendimento futuro neutrale al mercato**. Lo 0,91 del P&L veniva dal lato short, in dodici mesi in cui il paniere è sceso del 51%: era beta, non conoscenza. |
| `cross_sectional_return` | rendimento a 72h in eccesso sul paniere, standardizzato nella data | La GRU arriva a 0,106 e perde contro due colonne sommate (0,110). **Il fattore che la predice è positivo**: vedi *L'eccezione* sotto. |
| `move_balance` | salita meno discesa nelle 48 barre successive (12h su 15m) | GRU su 15 colonne × 96 barre: **Rank IC +0,0015, t 0,27**. Come compito principale accanto all'etichetta swing: rho da −0,027 a +0,046 fra i fold, e la validazione non distingue il peso ausiliario 1 dallo 0. |

## I moduli

| File | Ruolo | Perché è qui |
|---|---|---|
| `dataset.py` | una riga per barra 5m, quattro rami di timeframe allineati | Il campione è scelto da `remaining_excursion` (step2.parquet). Il purge legge il pivot successivo, che a finestra 12 sta 13 barre avanti, mentre l'etichetta ne legge 46 (`legs.label_reach`). |
| `split.py` | purge e walk-forward su indice (timestamp, simbolo) | Lo usava solo la prima pipeline. `swing.purge` / `swing.folds` lo sostituiscono. |
| `linear.py` | step 1, OLS come allarme di leakage | Rank IC 0,124 su `remaining_excursion`. |
| `gbm.py` | step 2, LightGBM di riferimento | 0,156 ± 0,015 su `remaining_excursion`. Con lui esce anche la dipendenza dev `lightgbm`. |
| `selection.py` | taglio delle feature: 28 → 12, poi 2 colonne | Selezioni fatte su etichette predittive. La prima non si trasferiva nemmeno alla GRU (−0,005 di Rank IC). La seconda è la coppia del fattore, ed era `features.SELECTED`, mostrata di default dalla pagina. |
| `nearpivot.py` | segnale per colonna vicino al pivot | Entro 24 barre il segnale c'è ma cambia segno: 66 colonne su 112 si invertono. Lo specialista addestrato sulla sola banda arriva a −0,019 e non passa lo zero (`remaining_excursion`). |
| `crosscheck.py` | l'esperimento che cambiò etichetta | Il retrospettivo a 0,3821 diventa 0,0753 tradotto sull'etichetta predittiva. |
| `gru.py` | steps 3–6: GRU per ramo, multi-ramo, cross-sectional | Quattro rami perdono contro uno su 4 fold su 4. Sul cross-sectional perde contro un'addizione. Allena anche `--label swing`, ma sul campione e sul purge di `dataset`. |
| `legsweep.py` | griglia 9 × 13 di smoothing × finestra sull'etichetta swing | Retrospettivo, ma dipende da `gru` e `dataset`. Il risultato resta nella spec: nessun ottimo interno, e il 93% del vincitore è un RSI. v2 (12 / 0,5) è già andato dove la griglia puntava. |
| `legcheck.py` | la previsione swing anticipa il prezzo o lo riassume? | Rank IC 0,4114 contro l'etichetta, **−0,0405 contro il rendimento futuro**. Legge il formato di `gru` e gli helper di `dataset`. Per `swing.py` la stessa analisi non è ancora un modulo. |
| `exhaustcheck.py` | le sei feature di esaurimento contro il prezzo | Anticipano davvero (+0,0325 ± 0,0100 a 24 barre). Ma per pagarsi chiedono **1,76 bp per lato contro 25**. |
| `factor.py` | il composito cross-sectional e il suo libro | L'eccezione, sotto. |
| `simulation.py` | quanta abilità serve per pagare la commissione | Tabella di break-even della regola cross-sectional. |
| `tests/test_dataset.py` | la regola di allineamento dei rami, per troncamento | Testava `dataset`. |
| `models/gru.pt` | la GRU sull'etichetta swing, cella 0,7 / 24 | La pagina la disegnava. È retrospettiva come etichetta, ma è costruita sulla prima pipeline, con il campione e il purge sopra. |
| `move_balance_label.html` | statistiche di `move_balance` | Documento dell'etichetta archiviata. |

## Cosa è stato tolto senza spostare un file

Queste parti vivevano dentro moduli che restano. Il codice com'era sta al tag.

- `swing.py`:
  - `--label balance`;
  - `--label swing+balance` con `--aux`, la terza testa (`Net.aux`, `auxiliary`, `target2`);
  - `ahead()`.

  Il modulo ora allena solo l'etichetta swing. I checkpoint continuano a scrivere `"label": "swing"`.
- `app/chart.py`:
  - il selettore dell'etichetta, con le tre predittive;
  - il pannello cross-sectional: heatmap, libro fattoriale, Rank IC a schermo;
  - la linea GRU e le celle di `legsweep`;
  - il modello `move_balance`.

  Restano l'etichetta swing, il modello a 4h dello step 7, la casella v2, la regola always-in con le
  uscite e l'oracolo.
- `features.SELECTED`: la coppia del fattore. Spenta la casella *All 29 candidates*, la pagina ora
  mostra le colonne di candela di `swing.REDUCED`, cioè quelle che i modelli swing leggono davvero.
- `pyproject.toml`: `lightgbm` esce dal gruppo dev; `scipy` resta per il self-check di `metrics`.

## L'eccezione: il fattore non ha fallito

`factor.py` è predittivo e **non** è archiviato perché fallito: è l'unico risultato netto positivo
del progetto. Il composito è −rank(volatilità a 30 giorni) + rank(volume in dollari).

- **Segnale:** Rank IC 0,110 ± 0,012, positivo su 4 fold su 4 e su 15 trimestri su 15.
- **Soldi:** il libro pesato per rango rende +0,238 log l'anno netti a 25 bp, Sharpe 1,33.
- **Limite:** è un portafoglio *tilt* lento, long/short, che ribilancia circa due volte l'anno. Il
  90,9% della varianza è un livello fisso per simbolo, cioè l'ordine di capitalizzazione.
- **Dove è stato misurato:** sulle 20 coppie di `STUDY`, mai sull'universo attuale di 15.

È archiviato perché il progetto ha scelto di concentrarsi sulle etichette retrospettive, non per un
numero. Riprenderlo vuol dire ripartire dal tag e rimisurarlo su `SYMBOLS`.

## Cosa non dice questo archivio

Che il retrospettivo funzioni in soldi: non è ancora stato mostrato.

- v2 predice l'etichetta a 0,644.
- Contro il rendimento futuro neutrale al mercato legge −0,026.
- La regola long-only perde (−0,275 contro −0,263 dell'hold, sulle 13 coppie tradabili).

L'etichetta descrive la gamba. Una strategia costruita sopra deve ancora battere il prezzo, e
`HANDOFF.md` §14–16 dice da dove partire.
