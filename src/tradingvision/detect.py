"""Recognising a turn of the v2 prediction while it happens: delay, false alarms, and what they earn.

`strategy --hindsight --delay` found that a centred turn of the prediction at 12 bars is worth
+180 bp a trade on the turn bar and +41 six bars later. This module asks how close a reader that
sees only the past can get, and what being close is worth. Same predictions, same three assets,
same folds as `strategy`: the detectors are fitted on development (folds 1-2), and the hold-out is
read with them as a check, since it has been used before.

**The problem is quickest change detection.** A detector is a stopping time T: its alarm uses
nothing after T. A centred turn τ is not one, it needs `window` bars of future. Delay is T - τ, a
false alarm is an alarm raised inside the wrong leg. With a false alarm every ARL bars, the least
delay any detector can have is about log(ARL) / I (Lorden; CUSUM attains it, Moustakides), where I
is the information a bar carries about the change. Inside the prediction's legs at 12 bars, on
development, the increments are +0.035 a bar going up and -0.036 coming down, sd 0.078: I = 0.42
nats a bar. On the price the same legs give +9.2 / -9.9 bp on 31 bp, I = 0.19. Both promise turns
recognised within a few bars.

**Two detectors.** `zigzag` alarms when the series has come back `h` from its extreme since the
last alarm: CUSUM with no drift allowance. `shiryaev` is the Bayesian one: the probability that
the leg has already turned, updated every bar from a prior hazard and the bar's evidence, with an
alarm above a threshold. The hazard is a logistic of the prediction's level and the leg's age, fitted
on development (`fit`), because the prediction is shrunk towards zero and rarely reaches its ends:
at its turns at 12 bars the level is 0.37 at the median, above 0.5 on 27% of them and above 0.75
on 2.4%. The hazard a bar is 4% at a level of 0-0.25, 7% at 0.25-0.4, 10% at 0.4-0.5, 14% at
0.5-0.6, 25% at 0.6-0.7, and under 1% in the first three bars of a leg. The evidence is the bar's
increment, Gaussian before the turn (+0.027, sd 0.077, from mid-leg bars) and after it (-0.049,
sd 0.076, from the first six bars of the next leg).

**Both detect what they promise and earn nothing.** On development against the prediction's own
turns at 12 bars:

    detector             found   mean delay   false alarms a turn   bp a trade dev / hold-out
    zigzag h=0.2          97%     5.1 bars          0.31                  -0.3 / -1.5
    zigzag h=0.3          87%     7.1               0.14                  -0.2 / -4.9
    shiryaev 0.5          79%     4.7 (median 3)    0.35                  -1.0 / +0.1
    shiryaev 0.9          65%     7.1               0.10                  +6.7 / -13.3
    shiryaev 0.99         43%    10.2               0.04                 +17.7 / -23.5
    shiryaev 0.9, flat    79%     8.0               0.10                  +0.2 / -5.6

The level prior trades found turns for delay and does not move the frontier. Every zigzag from h
= 0.05 to 0.8 makes between -1 and +9 bp a trade; every Shiryaev threshold makes about zero, or the
pattern of every rule in `strategy`, positive on folds 1-2 and negative on 3-4. An RSI at 12 through
the same detector does the same. Where the money goes (`--split`, Shiryaev at 0.5, development):
the 2,765 trades opened by a right detection make +30.9 bp each, the 1,234 opened by a false alarm
lose -79.8, held 20 bars against the leg, and the two cancel. At 0.9: +21.9 on 2,284 against -89.0
on 359; the zigzag at 0.2, +19.8 on 3,403 against -79.5 on 1,081.

**Combining the two moves the parts and not the sum** (`--both`): the alarm waits for Shiryaev's
posterior and for a retracement of h from the extreme. At 0.5, h = 0.2 cuts false alarms from 0.35
to 0.20 a turn and even finds more turns (86%); h = 0.4 cuts them to 0.07. Each step makes the right
detections later and worth less (+30.9, +22.6, +16.4 bp a trade) and the false alarms that survive
dearer (-79.8, -100.4, -138.2): one that outlasts a deeper retracement was entered after a deeper
move against the leg, so the leg's return to its course costs more. Over the twelve combinations of
0.5 / 0.7 / 0.9 and h = 0.1-0.4, development makes +0.6 to +6.5 bp a trade and the hold-out -0.8 to
-15.6. A filter on the same information is one more stopping time on it: telling a true turn from a
false one at the alarm takes information about the next leg, which neither detector has.

**Telling a true alarm from a false one is possible, and pays nothing** (`--features`, `at_alarm`,
`separate`). At Shiryaev 0.5, sixteen columns known at the alarm, each signed into the leg the
alarm closes: from the prediction, from the price, the six exhaustion columns at 12 bars, the BTC
filter. The retracement from the leg's extreme separates best (AUC 0.592 on development, 0.600 on
the hold-out), then the price's retracement in ATR (0.580 / 0.603), the leg's move in units of its
noise (0.579 / 0.599) and `stretch` (0.573 / 0.582). A logistic on all of them reaches 0.637 /
0.629, and the share of true alarms runs from 53% in its bottom quintile to 83% in its top on
development, 60% to 86% on the hold-out. No column correlates with the gross of the trade beyond
|0.06| in either period, and up the quintiles the true alarms make less as they become likelier
(+49 to +25 bp on development, +40 to +9 on the hold-out) while the false ones lose more (-68 to
-99, -65 to -102): every quintile makes between -6.6 and +6.5. What says a turn has happened is how
far it has gone, which is how much of it is already spent.

**A stop loss cuts the false alarms and the true ones alike** (`--sl`). At 1 ATR (at 12 bars) it
closes 96% of the false-alarm trades and halves their loss, -79.8 to -41.6 bp, and closes 51% of the
right detections, which fall from +31.0 to +17.4; at 2 ATR -68.3 and +27.0. The book makes +0.6 /
+1.1 bp a trade at 1 ATR and -0.5 / +1.3 at 2, and no stop from 1 to 6 ATR, fixed or trailing,
leaves -1.4 to +1.6. The trailing stop at 1 ATR is positive in all four folds, +0.6 to +2.0, on
trades of two to four bars: a thirtieth of a 50 bp round trip. Right after an alarm the price is as
noisy around a true turn as around a false one, so a level that closes the false ones closes the
true ones too.

**Funding, open interest, positioning, taker flow and the book add nothing at the alarm**
(`--futures`, `data.futures`). Fifteen futures columns at Shiryaev 0.5's alarms, signed into the
leg: on their own a logistic reaches AUC 0.546 on development and 0.525 on the hold-out, with the
sixteen columns above 0.639 / 0.624 against their 0.637 / 0.629 alone, and every quintile still
makes -11 to +6 bp a trade. Against the forward return, unconditionally and on every bar, one family
keeps its sign on all four folds: the book's imbalance within 1, 2 and 5% against the 12-bar return
(within 5%: +0.028 / +0.039 / +0.020 / +0.056 by fold; its last hour's mean +0.023 / +0.042 / +0.020
/ +0.050). It is the size of v2's own IC (-0.026 / -0.039 / -0.022 / -0.063 at 12 bars) and not a
strategy: at an IC of 0.05 a trade on the 48-bar return is worth of the order of 10 bp, against a
20-50 bp round trip. Until 2026-10-07 the IC was read on one bar in h, from the first, and the open
interest's 12-bar change signed by the price's move looked stable too, +0.111 / +0.039 / +0.077 /
+0.001 against the 48-bar return: on every bar it is +0.020 / -0.012 / +0.008 / +0.007.

**Open interest behind the move, in depth** (`--oi`, `open_interest`). Signed by the direction of the
last k bars' move, the open interest's k-bar change (in units of its month's dispersion), against the
48-bar forward return on every bar: +0.029 / +0.007 / +0.007 / +0.016 by fold at k = 24, -0.008 in fold
2 at k = 12 and -0.015 at k = 4; the move alone, momentum, +0.008 / -0.049 / +0.022 / +0.000. Split by
sign: after a 24-bar move made with open interest rising the next 48 bars go its way by +16.1 /
-19.3 / +6.5 / +16.9 bp, after one made with it falling they go -4.9 / -1.1 / +10.9 / -0.8. No
sign holds across the folds. **Corrected on 2026-10-07**: until then the IC and the split were read
on one bar in 48, from the first, and said +0.057 / +0.125 / +0.057 / +0.045, +10.4 / +16.4 / +18.7 /
+19.8 and -22.0 / -13.0 / -2.4 / -6.1, "the cleanest fold-by-fold sign of the study". That was one
phase of 48: fold 2's +0.125 is the second highest of them, and the IC moves by 0.03-0.07 from phase
to phase (route 2, `--conditional`, found it). As a rule, which never subsampled, it does not hold:
following the move when open interest rose and fading it when it fell, entering when |z| reaches 0.5
to 2 and out after 48 bars, makes -6.9 to +0.7 bp a trade on development and +0.8 to +17.2 on the
hold-out, with fold 2 negative in all twelve variants (-6.4 to -41.5).

**Keeping only signals past a level** (`--gate`, `gate`). A long only where the prediction is at or
under -L, a short only at or over +L, read at the signal's bar or at the extreme of the leg it
closes; a dropped signal either closes the position or is ignored. It cuts the zigzag's 5,288
development trades to 93 at L = 0.5, and the gross a trade stays near zero: closing on a dropped
signal, -3.2 to +4.2 bp on development and -9.2 to +0.9 on the hold-out for both detectors and
every L from 0.1 to 0.5 but one; ignoring it, the always-in shape comes back, positive on folds 1-2 and down to -47 on
the hold-out. The one row positive on all four folds, the zigzag at L = 0.5 read at the signal and
closing, makes +12.5 +/- 14.7 and +19.2 +/- 16.6 on 160 trades, and Shiryaev's detector through the
same gate makes -3.2 / -3.5. Fewer trades pay fewer fees; they do not pay more each. Chosen on
development alone, for the page (`chart.LEVEL`): of h 0.1-0.5 and p 0.5-0.99 by L 0.4 / 0.5 / 0.6,
closing, the best gross a trade with folds 1 and 2 both positive is that zigzag row and Shiryaev 0.5
at L = 0.6, +15.8 on 81 trades and +28.4 on 40 on the hold-out with a 6 ATR stop, every fold
positive and every number under the 50 bp round trip. Ignoring a dropped signal holds whatever side
the last kept one took, for days, and the result is the trend of those days: Shiryaev 0.5 with the
stop makes -25 / +52 / -91 bp on development at L 0.50 / 0.55 / 0.60.

**Open interest as a confirmation does not confirm** (`--confirm`, `confirmation`). Each signal times
open interest behind the last k bars' move, positive where the column says the next bars go the
signal's way. The gross of the trade a signal opens does not rise with it: across its quintiles
the detectors stay between -9 and +7 bp. Keeping only confirmed signals (a rejected one closes)
moves development down and the hold-out up by a few bp, Shiryaev 0.5 at k = 12 and a confirmation
of 1 making -1.6 +/- 7.5 and +12.3 +/- 7.0, with fold 2 negative and fold 4 positive almost
everywhere: the pattern of the open-interest rule itself, the period and not the signal. The
column is too small and too unstable at these horizons to move a signal it is laid on.

**Why: optional stopping.** If the log price is a martingale, E[p_T - p_S | F_S] = 0 for any two
stopping times S <= T, so a trade a causal detector opens and closes has zero expected gross
whatever its delay and false alarms. The hindsight table does not contradict it because τ + d,
d < window, is not a stopping time; τ + window is, and there the turns are worth -10 to +16
(`strategy --causal`).

**And the turn's value is the geometry of noise** (`strategy --hindsight ... --delay ... --null`).
On prices rebuilt from their own returns with every sign drawn at random, volatility clusters and
tails kept and any direction gone, the turns of the price at 12 bars make +216 to +219 bp a trade
against +201 on the real one, an RSI's +169 to +172 against +164, and the share left at each delay
agrees with the real one to 0.01-0.03. The share follows 1 - sqrt(d / window): after an extreme a
path moves away as sigma sqrt(d), lost once at the entry and once at the exit (0.71, 0.59, 0.42, 0.29
at d = 1, 2, 4, 6 of 12 against 0.73, 0.59, 0.40, 0.27 measured), and the price around the
prediction's turns moves 27 bp in one bar, 56 in four and 82 in nine. The I above is inflated by
the same selection; the information that matters is the prediction's correlation with the forward
return, and at rho ~ 0.03 it is I ~ rho^2 / 2, of the order of 1e-4 to 1e-3, against which Lorden's
bound is thousands of bars. The legs v2 was trained on are 12 bars, so 12 is also the most a turn
can be late before the window confirms it: the whole question lives in those 12 bars, and in them
the price behaves as a coin would.

The price's own zigzag at 3-8% retracements is the one row with a positive number, +11 to +63 bp a
trade, all 16 months; against the same zigzag on 20 sign-randomised paths it is within 0.7-1.6
null standard deviations and changes sign between folds.

**The false-alarm rate is a free parameter, and three diagnostics say so** (added 2026-10-06, **not
yet run on the store**). Optional stopping holds for any filter on the past, so where a filter
raises the share P of true alarms it lowers what a true one makes (W) and raises what a false one
costs (L) until P W = (1 - P) L again. The quintiles of `--features` above already obey it: 53% true
with L/W = 68/49 = 1.39 against P/(1-P) = 1.13, 83% with 99/25 = 3.96 against 4.88, on development.
Precision can be bought, and its price is the payoff ratio. A filter is worth what it moves the
gross, never what it moves the AUC or the false alarms, and these ask that question:

- `conservation` (printed by `--features`) sets L/W beside P/(1-P) by quintile of the logistic and
  reduces them to `kept`, the share of the precision's face value that reaches the gross: 0 under
  a martingale, 1 when W and L do not move.
- `--null` runs the separation on prices with every return's sign drawn at random, through an RSI
  at 12 since v2 needs candles. A simulation (a GARCH random walk, zigzag 0.3 on its RSI at 12, a
  logistic on the eight columns of `geometry`; its code is in the appendix of `false_alarms.html`)
  gave an AUC of 0.69, above the market's 0.63 here, and every quintile at zero: what the real
  path adds is its AUC above the random ones.
- `--residual MARKET` runs the detectors on each asset less beta times BTC, or the equal-weighted
  market (`ew`). A filter can only pay where the path is not a martingale. In the same simulation,
  with an AR(1) component holding 30% of the variance, keeping the alarms whose leg reached 0.6
  made +10.3 bp (error 2.8) where the random walk made -0.3 (2.6), at the same 96-97% of true
  alarms. The common move is
  most of each asset's variance, and the reversal documented at short horizons is in the part it
  leaves out. The hedged trade pays two legs, `fee_bp`.

**The residual does not revert on the history no one chose on** (`--residual MARKET --period
dev|holdout|2021`, measured 2026-10-07 on the store to 2026-09-26). The RSI at 12 on each asset less
beta times the market, zigzag 0.2 and Shiryaev 0.5 through the gate read at the alarm and closing,
bp a trade gross with its trade-level error; `2021` is 2021-01 to 2025-06 in four folds, with
Shiryaev's hazard and the logistic behind `kept` fitted on 2020 and h, p and L frozen from
development:

    residual: rsi 12                  L 0.4          L 0.5          L 0.6          kept          fee_bp
    against BTC, ETH and SOL
      dev       zigzag 0.2          -4.2 (8.7)   -14.1 (23.2)     -7 (2 trades)   +0.16 (0.12)    49.6
                shiryaev 0.5        +1.0 (10.1)   +0.4 (12.6)   -3.2 (17.2)     -0.07 (0.29)
      hold-out  zigzag 0.2         +14.6 (8.9)   +40.8 (22.4)  +41.9 (7 trades) +0.13 (0.12)    43.8
                shiryaev 0.5       +10.7 (7.8)   +12.8 (11.3)  +54.4 (15.8)     +0.09 (0.33)
      2021      zigzag 0.2         -15.6 (7.2)   -10.1 (18.6)  -45.5 (34.0)     -0.14 (0.06)    44.4
                shiryaev 0.5       -19.3 (7.2)   -20.6 (11.8)  -45.5 (21.4)     -0.19 (0.15)
    against ew, ETH, BTC and SOL
      dev       zigzag 0.2         +10.5 (7.7)   +29.8 (22.6)   +220 (5 trades) +0.28 (0.12)    34.6
                shiryaev 0.5       +17.4 (9.7)   +31.5 (19.6)  +70.1 (36.9)     -0.01 (0.32)
      hold-out  zigzag 0.2          -2.4 (6.1)    +8.0 (17.5)    -38 (6 trades) -0.05 (0.10)    36.7
                shiryaev 0.5       -13.2 (6.3)    -8.7 (11.6)   +2.3 (16.9)     +0.12 (0.34)
      2021      zigzag 0.2         -11.1 (4.5)    -9.9 (7.5)   -21.7 (18.7)     -0.11 (0.05)    36.6
                shiryaev 0.5        -8.3 (4.3)   -12.8 (6.8)   -25.4 (12.6)     -0.31 (0.13)
    every TRADABLE pair: the 12 alts against BTC, the 13 against ew
      BTC dev   zigzag 0.2         +13.5 (9.4)   +23.1 (23.4)  +10.2 (37.4)     +0.11 (0.05)    52.2
                shiryaev 0.5       +12.4 (7.9)   +28.2 (9.2)   +38.3 (15.9)     +0.14 (0.14)
      BTC 2021  zigzag 0.2          -9.1 (3.4)    -5.0 (8.3)    +8.8 (28.5)     -0.06 (0.02)    44.1
                shiryaev 0.5       -14.2 (3.3)   -20.7 (5.5)   -34.4 (9.9)      -0.04 (0.07)
      ew dev    zigzag 0.2          +7.3 (5.2)   +29.8 (12.5)  +80.5 (47.7)     +0.17 (0.05)    39.9
                shiryaev 0.5        +6.4 (5.4)   +11.1 (10.0)  +25.1 (14.8)     +0.11 (0.18)
      ew 2021   zigzag 0.2          -7.1 (2.9)    -5.0 (7.6)   -10.9 (25.6)     -0.04 (0.02)    39.4
                shiryaev 0.5        -4.0 (2.6)    -8.6 (4.6)   -25.6 (9.2)      -0.15 (0.07)

Development against ew is what made the route look open: a gross growing with the level, both
folds positive, and read literally the card's criterion passes once, the zigzag at L = 0.6, on five
trades. Against BTC the hold-out showed the same shape. On 2021-2025 neither survives: every gated
row is negative and lower at L = 0.6 than at 0.4 instead of higher, on the residual as on the price,
and `kept` is negative, beyond two errors on three of the four rows: the more a turn looks
confirmed, the less it pays. The extremes keep going, as v2's did in `strategy`'s step 1. The
residual takes part of that momentum out (the price's RSI makes -144 and -57 at L = 0.6 on the same
three assets against ew's -22 and -25) and leaves no reversal to pay one leg, let alone two.

The wider universe says it with more trades. On development every TRADABLE pair against ew passes
the card with the zigzag, +80.5 on 26 trades at L = 0.6 with both folds positive and `kept` 0.17
(0.05); on 2021 the same row makes -10.9 on 144 trades, `kept` -0.04 (0.02), its folds 1-2
positive and 3-4 negative from L = 0.5 up. Against BTC on 2021 the zigzag's gross grows with the
level, -9.1 / -5.0 / +8.8, and never comes near its 44 bp fee; the price's RSI on the same pairs
makes -105 at 0.6. The ew development pattern was one period's.

**BTC does not lead the alts at 15 minutes** (`--lag 1`). On BTC's bar before, the alts' beta is
about 0.05 (`fee_bp` 20.9-21.0), so the lagged residual is the price to a few bp and makes the
price's numbers: zigzag -8.5 / -15.7 / +55.4 (7 trades) at L 0.4 / 0.5 / 0.6 on development and
-24.0 / -53.2 / -175.7 on 2021, Shiryaev +5.5 / +1.1 / +21.6 and -23.8 / -40.0 / -47.4.

**The futures columns know no more at v2's pivots than anywhere** (`--conditional`, route 2 of
`false_alarms.html`, `conditional_ic`; added 2026-10-07). The two-stage idea in its right form: a
first stage is worth something only if it picks the bars where a second model, on columns that are
not the price, knows more about the held period's return. Per column (the futures columns, open
interest behind the move at k = 4 / 12 / 24, v2 and the RSI at 12 as yardsticks), horizon (12 / 24
/ 48 bars) and fold: the rank IC on the bars at the pivot against the rank IC on every bar, ratio and
difference. "At the pivot" is |v2| >= 0.4 / 0.5 / 0.6 (8,803 / 3,887 / 1,177 bars of 34,758 in fold
1, 8,296 / 4,044 / 1,359 in fold 2) and Shiryaev 0.5's alarm bars (2,177 / 2,055). The IC is the mean
product of standardised ranks, ranked within asset, fold and subset and pooled bar by bar, so a
bar weighs one; its error, the ratio's and the difference's are the delta method over the clock
blocks of h bars (`_delta`), which on full blocks is `metrics.blocked`'s error and matches the spread
of 200 simulated draws (0.092 against 0.084). Not `metrics.blocked`'s mean, a mean of block means:
on a subset that comes in runs it weighs a block holding one bar like one holding 36, and since the
short runs are the ones that ended, fading v2 at |v2| >= 0.4 for 12 bars makes -1.4 bp a bar on fold
1 and +17.3 as a mean of blocks. The criterion (ratio above 1 by two errors in both development
folds, |IC| on the subset above rho_min = c / (0.8 sigma sqrt h) with one sign) passes on 0 of 288
rows on development and 0 on the hold-out, at spot and at perpetual; the placebos, each subset rolled
in time inside its fold, pass 0 of 864. sigma on development is 0.21% (BTC), 0.34% (ETH), 0.40% (SOL),
0.32% pooled, so rho_min is 0.23 / 0.16 / 0.11 at 12 / 24 / 48 bars on spot and half on perpetuals.
At the alarm no column gains: the median ratio over the columns is 0.72-0.89 against 0.91-1.08 for
the placebos. Past |v2| >= 0.6 at 12 bars the book's imbalance within 5% has an IC of 0.149 / 0.121
on development and 0.182 / 0.220 on the hold-out against 0.028 / 0.039 / 0.020 / 0.056 on every bar
(difference +0.12 +/- 0.05, +0.08 +/- 0.05, +0.16 +/- 0.08, +0.16 +/- 0.07), above the perpetual's
0.114 in every fold, and still fails: the ratio is 5.3 +/- 3.2 and 3.1 +/- 1.6, because an IC of 0.03
on every bar is barely measured and its error is the ratio's. That row is mostly v2's own side: there
book5 correlates 0.42-0.74 with long-at-a-bottom against short-at-a-top, v2's IC on the same bars is
-0.18 / -0.15 / -0.17 / -0.15, and within one tail book5's runs from -0.00 to +0.27. Nor is a
rank IC money: fading v2 at |v2| >= 0.6 for 12 bars grosses +13.5 / +10.0 / +9.5 / -3.4 bp a bar
(errors 6.5-16). `--power` plants a column with the store's subsets, returns and blocks: at an IC of
0.03 on every bar, the futures columns' size, no ratio up to 3 is seen more than 22% of the time; at
0.06 a ratio of 2 at the alarms is seen 78% / 55% / 24% of the time at 12 / 24 / 48 bars, past
|v2| >= 0.6 a ratio of 3 50% / 9% / 2%; a ratio of 1 passes 0-0.5% everywhere. The ratio is the
wrong statistic for columns whose IC on every bar is not distinct from zero, and the test cannot rule
out a ratio of 2 on them: what it rules out is a pivot that turns a 0.06 column into a 0.12 one at
12-24 bars. On the card, that sends the columns to every bar, as timing columns (route 6).
Read on every bar, open interest behind the move at k = 24 has an IC with the 48-bar return of
+0.029 / +0.007 / +0.007 / +0.016 by fold, the mean of the 48 phases a one-in-48 sampling can start
on: `--oi`'s +0.057 / +0.125 / +0.057 / +0.045 is phase 0, and fold 2's is the second highest of
the 48 (phase-to-phase sd 0.03-0.07). `forward_ic` sampled the same way; both read every bar now.

    uv run python -m tradingvision.detect --zigzag 0.05 0.1 0.15 0.2 0.3 0.5 0.8
    uv run python -m tradingvision.detect --shiryaev 0.5 0.7 0.8 0.9 0.95 0.98 0.99 [--flat]
    uv run python -m tradingvision.detect --shiryaev 0.5 0.9 --zigzag 0.2 --split
    uv run python -m tradingvision.detect --shiryaev 0.5 0.7 0.9 --both 0.1 0.2 0.3 0.4 --split
    uv run python -m tradingvision.detect --features 0.5
    uv run python -m tradingvision.detect --sl 1 2 3 4 6
    uv run python -m tradingvision.detect --futures   # needs `python -m tradingvision.data.futures` first
    uv run python -m tradingvision.detect --oi
    uv run python -m tradingvision.detect --gate 0.1 0.2 0.3 0.4 0.5
    uv run python -m tradingvision.detect --confirm
    uv run python -m tradingvision.detect --null 0.5 [--seeds 0 1 2]
    uv run python -m tradingvision.detect --residual BTC [--gate 0.4 0.5 0.6] [--period dev|holdout|2021]
    uv run python -m tradingvision.detect --residual ew [--period 2021] [--tradable] [--lag 1]
    uv run python -m tradingvision.detect --conditional [--period dev|holdout] [--power] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from tradingvision import legs, metrics, stops, strategy
from tradingvision.data import futures
from tradingvision.data.binance import SYMBOLS, TRADABLE
from tradingvision.data.binance import load as candles
from tradingvision.oracle import FEE
from tradingvision.strategy import fold_of, hold, plain, turn_events

POST = 6  # the first bars of a leg that make the post-turn increment distribution
# The beta `residual` hedges with: a month of 15m bars, the window `_zscore` already reads. Chosen,
# not measured; a beta that moves slower than the legs is the only requirement.
BETA_WINDOW = 96 * 30
# OKX's perpetual taker fee, 0.05% a side (10 bp round trip): from secondary sources, not measured
# and not checked against OKX's own schedule (`false_alarms.html`, #soglia). Spot is `oracle.FEE`.
PERP_FEE = 0.0005
# `fit` on v2's walk-forward predictions, development folds only (ETH, BTC, SOL, 2025-06-01 to
# 2026-01-28), in the head's raw units. Fixed here so the chart page, which has no store, runs the
# detector the study measured; `main` refits it and refuses to run if the two have drifted apart.
V2_FIT = {
    "beta": np.array([-3.3566, 3.2917, 0.9579, -0.106, -2.131]),
    "m0": 0.0265,
    "s0": 0.0773,
    "m1": -0.0489,
    "s1": 0.0758,
    "base": 0.0505,
}


def zigzag(v: np.ndarray, h: float, trace: bool = False) -> np.ndarray | pd.DataFrame:
    """+1 when `v` has risen `h` above its low since the last alarm, -1 when it fell `h` below its high.

    `trace` returns, bar by bar, the alarm, the leg the detector believed it was in before the bar
    (+1 up, -1 down, 0 not yet known) and how far `v` had come back from that leg's extreme.
    """
    out, leg, back = np.zeros(len(v)), np.zeros(len(v)), np.zeros(len(v))
    side, hi, lo = 0, v[0], v[0]
    for t in range(len(v)):
        hi, lo = max(hi, v[t]), min(lo, v[t])
        leg[t], back[t] = side, (hi - v[t]) if side > 0 else (v[t] - lo) if side < 0 else max(hi - v[t], v[t] - lo)
        if side >= 0 and hi - v[t] >= h:
            out[t], side, lo = -1, -1, v[t]
        elif side <= 0 and v[t] - lo >= h:
            out[t], side, hi = 1, 1, v[t]
    return pd.DataFrame({"alarm": out, "leg": leg, "retrace": back}) if trace else out


def _design(z: np.ndarray, age: np.ndarray) -> np.ndarray:
    z = np.clip(z, -1, 1)
    return np.column_stack([np.ones_like(z), z, z**2, np.log1p(age), (age <= 3).astype(float)])


def _logit(X: np.ndarray, y: np.ndarray, iters: int = 25) -> np.ndarray:
    """Logistic regression by Newton's method: five columns do not need a library."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        b += np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(len(b)), X.T @ (y - p))
    return b


def _runs(side: pd.Series) -> np.ndarray:
    """Bars since `side` last changed, per asset: 0 on the bar it changes."""
    run = (side != side.groupby(level=1).shift()).groupby(level=1).cumsum()
    return side.groupby([side.index.get_level_values(1), run]).cumcount().to_numpy()


def fit(x: pd.Series, cut: pd.Timestamp, window: int = strategy.WINDOW) -> dict:
    """The prior hazard and the two increment distributions, from development's centred turns of `x`.

    The hazard of bar t is P(the turn was at t-1 | the leg had not turned before), a logistic of the
    level at t-1 in the leg's direction and of the leg's age. Before the turn the increment is read
    on bars past the first `POST` of a leg, after it on those first bars, in the old leg's direction.
    """
    dev = x.index.get_level_values(0) < cut
    e = turn_events(x, window)
    leg = hold(e)
    side = leg.groupby(level=1).shift(2).fillna(0.0)  # the leg bar t was in, before any turn at t-1
    turned = (e.groupby(level=1).shift(1) == -side).to_numpy().astype(float)
    z = (x.groupby(level=1).shift(1) * side).to_numpy()
    ok = dev & (side != 0).to_numpy() & np.isfinite(z)
    beta = _logit(_design(z[ok], _runs(side)[ok].astype(float)), turned[ok])
    u = (x.groupby(level=1).diff() * leg).to_numpy()  # the bar's move in the direction of its leg
    k = _runs(leg)
    good = dev & np.isfinite(u) & (leg != 0).to_numpy()
    pre, post = u[good & (k > POST)], -u[good & (k >= 1) & (k <= POST)]
    return {
        "beta": beta,
        "m0": pre.mean(),
        "s0": pre.std(),
        "m1": post.mean(),
        "s1": post.std(),
        "base": turned[ok].mean(),
    }


def shiryaev(
    v: np.ndarray, p: dict, threshold: float, flat: bool = False, h: float = 0.0, trace: bool = False
) -> np.ndarray | pd.DataFrame:
    """+1 / -1 alarms on one asset: alarm when P(the leg has turned | the bars so far) >= `threshold`.

    Shiryaev's recursion with a hazard that moves: the prior is last bar's posterior plus the
    hazard of a turn at the last bar, the evidence is this bar's increment under the two Gaussians.
    After an alarm the side flips and the leg starts again at age 0. `flat` holds the hazard at its
    mean, which leaves the evidence alone to decide. `h` asks the zigzag's question as well: the
    alarm also waits for the series to have come back `h` from its extreme since the last alarm.
    `trace` returns, bar by bar, the alarm, the leg the detector believed it was in, the posterior,
    the prior hazard and the retracement from the leg's extreme.
    """
    out = np.zeros(len(v))
    rows = np.full((len(v), 4), np.nan)
    side, pi, age, ext = 1.0, 0.0, 0.0, v[0]
    b, m0, s0, m1, s1 = p["beta"], p["m0"], p["s0"], p["m1"], p["s1"]
    for t in range(1, len(v)):
        z = min(max(side * v[t - 1], -1.0), 1.0)
        if flat:
            rho = p["base"]
        else:
            rho = 1 / (1 + math.exp(-(b[0] + b[1] * z + b[2] * z * z + b[3] * math.log1p(age) + b[4] * (age <= 3))))
        prior = pi + (1 - pi) * rho
        u = side * (v[t] - v[t - 1])
        l1 = math.exp(-0.5 * ((u - m1) / s1) ** 2) / s1
        l0 = math.exp(-0.5 * ((u - m0) / s0) ** 2) / s0
        pi = prior * l1 / (prior * l1 + (1 - prior) * l0)
        age += 1
        ext = max(ext, side * v[t])
        rows[t] = side, pi, rho, ext - side * v[t]
        if pi >= threshold and ext - side * v[t] >= h:
            out[t], side, pi, age = -side, -side, 0.0, 0.0
            ext = side * v[t]
    if trace:
        return pd.DataFrame(rows, columns=["leg", "posterior", "hazard", "retrace"]).assign(alarm=out)
    return out


def alarms(x: pd.Series, detector, *args) -> pd.Series:
    """`detector` run on each asset of `x` separately, back on `x`'s index."""
    parts = [
        pd.Series(detector(x.xs(s, level=1).to_numpy(), *args), index=x.xs(s, level=1, drop_level=False).index)
        for s in x.index.get_level_values(1).unique()
    ]
    return pd.concat(parts).reindex(x.index)


def match(found: pd.Series, truth: pd.Series, cut: pd.Timestamp) -> dict:
    """Share of `truth`'s turns found before the next one, their delay in bars, false alarms a turn.

    An alarm belongs to the last true turn at or before it: the turn's first alarm of its kind is its
    detection, one of the other kind is false, a later one of the same kind is a re-entry after a
    false alarm. Development only.
    """
    delays, false, turns = [], 0, 0
    for s in found.index.get_level_values(1).unique():
        a, t = found.xs(s, level=1), truth.xs(s, level=1)
        a, t = a[(a != 0) & (a.index < cut)], t[(t != 0) & (t.index < cut)]
        turns += len(t)
        j = t.index.searchsorted(a.index, side="right") - 1
        seen = set()
        for when, kind, i in zip(a.index, a.to_numpy(), j):
            if i < 0 or t.iloc[i] != kind:
                false += 1
            elif i not in seen:
                seen.add(i)
                delays.append((when - t.index[i]) / pd.Timedelta("15min"))
    d = np.array(delays)
    return {"found": len(d) / turns, "delay": d.mean(), "delay_med": np.median(d), "false": false / turns}


def book(found: pd.Series, close: pd.Series, cut: pd.Timestamp) -> dict:
    """bp a trade of the always-in rule that flips at each alarm, development and hold-out, and by fold."""
    return _periods(plain(hold(found), close)[2], cut)


def _periods(held: pd.DataFrame, cut: pd.Timestamp) -> dict:
    """bp a trade with its error and count on development and on the hold-out, and by fold."""
    when = pd.DatetimeIndex(held.entry)
    row = {}
    for period, keep in (("dev", when < cut), ("holdout", when >= cut)):
        g = held.gross[keep]
        row |= {f"{period}_bp": g.mean() * 1e4, f"{period}_se": g.std() / np.sqrt(len(g)) * 1e4, f"{period}_n": len(g)}
        if "why" in held:
            row[f"{period}_stopped"] = (held.why[keep] == "stop").mean()
    return row | (held.gross.groupby(fold_of(when)).mean() * 1e4).rename(lambda k: f"fold {k}").to_dict()


def kinds(held: pd.DataFrame, truth: pd.Series) -> list[str]:
    """What opened each trade: a detection, a false alarm, or a re-entry after a false alarm."""
    kind, seen = [], set()
    for s, row in held.iterrows():
        t = truth.xs(s, level=1)
        t = t[t != 0]
        i = t.index.searchsorted(row.entry, side="right") - 1
        if i < 0 or t.iloc[i] != row.side:
            kind.append("false alarm")
        else:
            kind.append("re-entry" if (s, i) in seen else "detection")
            seen.add((s, i))
    return kind


def split(held: pd.DataFrame, truth: pd.Series, cut: pd.Timestamp) -> pd.DataFrame:
    """Development's trades by what opened them, with the share a stop closed."""
    held = held[pd.DatetimeIndex(held.entry) < cut]
    g = held.assign(kind=kinds(held, truth), stopped=held.get("why", pd.Series("", index=held.index)) == "stop")
    g = g.groupby("kind")
    return pd.DataFrame(
        {"trades": g.size(), "bp": g.gross.mean() * 1e4, "bars": g.bars.mean(), "stopped": g.stopped.mean()}
    )


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    """P(a true alarm scores above a false one), from the ranks (Mann-Whitney)."""
    r = pd.Series(score).rank().to_numpy()
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def at_alarm(found: pd.Series, x: pd.Series, close: pd.Series, truth: pd.Series, window: int = strategy.WINDOW):
    """One row per alarm: whether it was true, the gross of the trade it opened, and what was known then.

    Every column is signed into the leg the alarm says is over and reads nothing after the alarm
    bar: from the prediction its level, the leg's extreme, the retracement from it, the last move,
    the leg's age; from the price the leg's move in units of its noise, the retracement from the
    extreme in ATR, the volatility, the volume against its day; the six exhaustion columns at the
    model's window; and the BTC filter of `strategy`.
    """
    rows = []
    for sym in x.index.get_level_values(1).unique():
        xv, a, t = x.xs(sym, level=1), found.xs(sym, level=1), truth.xs(sym, level=1)
        t = t[t != 0]
        bars = candles(sym, "15m")
        ex = legs.exhaustion(bars, window).reindex(xv.index)
        sigma = np.log(bars.close).diff().rolling(4 * window).std().reindex(xv.index)
        c = close.xs(sym, level=1)
        high, low = bars.high.reindex(xv.index), bars.low.reindex(xv.index)
        atr = (np.maximum(high, c.shift()) - np.minimum(low, c.shift())).rolling(window).mean() / c
        volume = (bars.volume / bars.volume.rolling(96).mean()).reindex(xv.index)
        below = strategy.below(pd.MultiIndex.from_arrays([xv.index, [sym] * len(xv)], names=x.index.names)).to_numpy()
        v, lc = xv.to_numpy(), np.log(c.to_numpy())
        at = np.flatnonzero(a.to_numpy() != 0)
        for k, i in enumerate(at):
            kind = a.iloc[i]
            side, start = -kind, at[k - 1] if k else 0
            leg_x, leg_c = side * v[start : i + 1], side * lc[start : i + 1]
            j = t.index.searchsorted(xv.index[i], side="right") - 1
            end = at[k + 1] if k + 1 < len(at) else len(v) - 1
            rows.append(
                {
                    "symbol": sym,
                    "when": xv.index[i],
                    "side": side,
                    "true": int(j >= 0 and t.iloc[j] == kind),
                    "gross": kind * (lc[end] - lc[i]),
                    "level": side * v[i - 1],
                    "extreme": leg_x.max(),
                    "retrace": leg_x.max() - side * v[i],
                    "move": side * (v[i] - v[i - 1]),
                    "age": i - start,
                    "leg_move": (leg_c[-1] - leg_c[0]) / (sigma.iloc[i] * np.sqrt(max(i - start, 1))),
                    "price_retrace": (leg_c.max() - leg_c[-1]) / atr.iloc[i],
                    "volatility": sigma.iloc[i],
                    "volume": volume.iloc[i],
                    "btc_below": float(below[i]),
                }
                | {f"ex_{col}": side * ex[col].iloc[i] for col in legs.EXHAUSTION}
            )
    return pd.DataFrame(rows).dropna()


def _columns(f: pd.DataFrame) -> list[str]:
    """The columns of an alarm table that were known at the alarm: everything but who, when and what came."""
    return [c for c in f.columns if c not in ("symbol", "when", "side", "true", "gross")]


def _score(f: pd.DataFrame, cut: pd.Timestamp) -> np.ndarray:
    """A logistic of `true` on every known column, standardised and fitted on development, at every alarm."""
    cols, dev = _columns(f), (f.when < cut).to_numpy()
    z = ((f[cols] - f[cols][dev].mean()) / f[cols][dev].std()).clip(-5, 5).to_numpy()
    X = np.column_stack([np.ones(len(z)), z])
    return X @ _logit(X[dev], f.true.to_numpy()[dev].astype(float))


def separate(f: pd.DataFrame, cut: pd.Timestamp) -> tuple[pd.DataFrame, tuple[float, float], pd.DataFrame]:
    """How well each column, and a logistic on all of them, tells true alarms from false ones.

    Each column's direction and the logistic are fitted on development; the AUC is read on both
    periods, and so is the rank correlation with the trade's gross, which is what pays. The
    quintiles of the logistic's score, cut on development, give the share of true alarms and what
    a true and a false alarm each make there.
    """
    dev = (f.when < cut).to_numpy()
    cols = _columns(f)
    y = f.true.to_numpy()
    rows = []
    for col in cols:
        a = _auc(f[col].to_numpy()[dev], y[dev])
        sign = 1 if a >= 0.5 else -1
        rows.append(
            {
                "column": col,
                "auc_dev": max(a, 1 - a),
                "auc_holdout": _auc(sign * f[col].to_numpy()[~dev], y[~dev]),
                "ic_gross_dev": np.corrcoef(f[col][dev].rank(), f.gross[dev])[0, 1],
                "ic_gross_holdout": np.corrcoef(f[col][~dev].rank(), f.gross[~dev])[0, 1],
            }
        )
    one = pd.DataFrame(rows).set_index("column").sort_values("auc_dev", ascending=False)
    score = _score(f, cut)
    both = (_auc(score[dev], y[dev]), _auc(score[~dev], y[~dev]))
    q = np.searchsorted(np.quantile(score[dev], [0.2, 0.4, 0.6, 0.8]), score) + 1
    frame = f.assign(period=np.where(dev, "dev", "holdout"), q=q)
    keys = ["period", "q"]
    g = frame.groupby(keys)
    by = pd.DataFrame(
        {
            "alarms": g.size(),
            "share_true": g.true.mean(),
            "true_bp": frame[frame.true == 1].groupby(keys).gross.mean() * 1e4,
            "false_bp": frame[frame.true == 0].groupby(keys).gross.mean() * 1e4,
            "all_bp": g.gross.mean() * 1e4,
        }
    )
    return one, both, by


def conservation(f: pd.DataFrame, score: np.ndarray, cut: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """The precision a filter buys, and how much of it reaches the gross: `(by quintile, kept)`.

    If the log price is a martingale given what `score` reads, every quintile of it grosses zero, so
    P W = (1 - P) L in each: where the true alarms are likelier they are worth less and the false
    ones cost more, by the ratio that cancels, and `L/W` tracks `P/(1-P)`. `naive_bp` is what a
    quintile would gross if W and L stayed at their period's means, the face value of its precision.
    `kept` is the least-squares slope of the gross on that face value across the five quintiles, per
    period, with its error from the quintiles' (`dev_se`, `holdout_se`): 0 when the martingale takes
    the whole gain back, 1 when the precision is worth what it says. It is the number a filter has
    to move, and AUC is not: a logistic on a random walk's alarms separates the true from the false
    as well as one on the market's. Quintiles cut on development.
    """
    dev = (f.when < cut).to_numpy()
    q = np.searchsorted(np.quantile(score[dev], [0.2, 0.4, 0.6, 0.8]), score) + 1
    frame = f.assign(period=np.where(dev, "dev", "holdout"), q=q)
    rows, kept = [], {}
    for period, g in frame.groupby("period"):
        w, loss = g.gross[g.true == 1].mean(), -g.gross[g.true == 0].mean()
        part = []
        for k, h in g.groupby("q"):
            p = h.true.mean()
            win, lose = h.gross[h.true == 1].mean(), -h.gross[h.true == 0].mean()
            part.append(
                {
                    "period": period,
                    "q": k,
                    "alarms": len(h),
                    "share_true": p,
                    "true_bp": win * 1e4,
                    "false_bp": -lose * 1e4,
                    "L/W": lose / win,
                    "P/(1-P)": p / (1 - p),
                    "all_bp": h.gross.mean() * 1e4,
                    "se": h.gross.std() / np.sqrt(len(h)) * 1e4,
                    "naive_bp": (p * w - (1 - p) * loss) * 1e4,
                }
            )
        t = pd.DataFrame(part)
        face = t.naive_bp - t.naive_bp.mean()
        kept[period] = float((face * (t.all_bp - t.all_bp.mean())).sum() / (face**2).sum())
        kept[f"{period}_se"] = float(np.sqrt((face**2 * t.se**2).sum()) / (face**2).sum())
        rows += part
    return pd.DataFrame(rows).set_index(["period", "q"]), kept


def geometry(found: pd.Series, x: pd.Series, close: pd.Series, truth: pd.Series, window: int = strategy.WINDOW):
    """`at_alarm`'s columns that need only the series and its close: the ones a rebuilt path has too.

    From `x` the level, the leg's extreme, the retracement from it, the last move and the leg's age;
    from the price the leg's move in units of its noise, the retracement from the leg's extreme in
    units of the bar's volatility (`at_alarm` reads it in ATR, which needs the highs and lows a
    sign-randomised close does not have) and the volatility itself. `close` is on `x`'s index, so
    the first `4 * window` bars have no volatility and their alarms are dropped.
    """
    rows = []
    for sym in x.index.get_level_values(1).unique():
        xv, a, t = x.xs(sym, level=1), found.xs(sym, level=1), truth.xs(sym, level=1)
        t = t[t != 0]
        lc = np.log(close.xs(sym, level=1).reindex(xv.index).to_numpy())
        sigma = pd.Series(lc).diff().rolling(4 * window).std().to_numpy()
        v = xv.to_numpy()
        at = np.flatnonzero(a.to_numpy() != 0)
        for k, i in enumerate(at):
            kind = a.iloc[i]
            side, start = -kind, at[k - 1] if k else 0
            leg_x, leg_c = side * v[start : i + 1], side * lc[start : i + 1]
            j = t.index.searchsorted(xv.index[i], side="right") - 1
            end = at[k + 1] if k + 1 < len(at) else len(v) - 1
            rows.append(
                {
                    "symbol": sym,
                    "when": xv.index[i],
                    "side": side,
                    "true": int(j >= 0 and t.iloc[j] == kind),
                    "gross": kind * (lc[end] - lc[i]),
                    "level": side * v[i - 1],
                    "extreme": leg_x.max(),
                    "retrace": leg_x.max() - side * v[i],
                    "move": side * (v[i] - v[i - 1]),
                    "age": i - start,
                    "leg_move": (leg_c[-1] - leg_c[0]) / (sigma[i] * np.sqrt(max(i - start, 1))),
                    "price_retrace": (leg_c.max() - leg_c[-1]) / sigma[i],
                    "volatility": sigma[i],
                }
            )
    return pd.DataFrame(rows).dropna()


def null_test(pred: pd.Series, close: pd.Series, cut: pd.Timestamp, p: float = 0.5, seeds=(0, 1, 2)) -> pd.DataFrame:
    """Does telling a true alarm from a false one read the market, or the definition of a turn?

    Shiryaev at `p` with its own `fit`, judged against each series' own centred turns, on v2 and on
    an RSI at 12 of the real price, then on the RSI of prices rebuilt with every return's sign drawn
    at random (`strategy.signflip`), where no reader can know the next leg. The logistic reads the
    eight `geometry` columns on every path. Whatever AUC the random paths reach is the geometry of
    an extreme of noise: a turn that has gone further is likelier to be a turn on any path. What
    the market adds is the real RSI's AUC above them, and it pays only if `kept` is above zero too.
    The RSI stands in for v2 on the random paths because v2 needs candles a rebuilt close has not.
    """
    paths = [("v2", "real", pred, close), ("rsi 12", "real", strategy.rsi(pred.index, close=close), close)]
    for k in seeds:
        c = strategy.signflip(close, k)
        paths.append(("rsi 12", f"random signs #{k}", strategy.rsi(pred.index, close=c), c))
    rows = []
    for name, path, x, c in paths:
        f = geometry(alarms(x, shiryaev, fit(x, cut), p), x, c, turn_events(x, strategy.WINDOW))
        dev, y, score = (f.when < cut).to_numpy(), f.true.to_numpy(), _score(f, cut)
        kept = conservation(f, score, cut)[1]
        rows.append(
            {
                "series": name,
                "path": path,
                "alarms": len(f),
                "share_true": y.mean(),
                "auc_dev": _auc(score[dev], y[dev]),
                "auc_holdout": _auc(score[~dev], y[~dev]),
                "kept_dev": kept["dev"],
                "kept_dev_se": kept["dev_se"],
                "kept_holdout": kept["holdout"],
                "kept_holdout_se": kept["holdout_se"],
                "dev_bp": f.gross[dev].mean() * 1e4,
                "holdout_bp": f.gross[~dev].mean() * 1e4,
            }
        )
    return pd.DataFrame(rows).set_index(["series", "path"])


def gate(
    found: pd.Series, x: pd.Series, level: float, where: str = "alarm", close: bool = True
) -> tuple[pd.Series, pd.Series]:
    """`(kept, on)`: a long alarm kept only where `x` is at or under -`level`, a short at or over +`level`.

    `where` reads `x` at the alarm bar (`alarm`) or at the extreme of the leg the alarm closes
    (`extreme`), the lowest since the last alarm for a long and the highest for a short. A rejected
    alarm is dropped and the rule holds what it held (`close=False`), or it closes the position and
    the rule stands flat until the next kept alarm (`close=True`): `on` is off from the bar after
    it to that alarm, which is how `strategy.walked` stands a rule flat, closing at the rejected
    alarm's own close.
    """
    kept, on = pd.Series(0.0, index=found.index), pd.Series(True, index=found.index)
    for sym in found.index.get_level_values(1).unique():
        i = np.flatnonzero(found.index.get_level_values(1) == sym)
        a, v = found.to_numpy()[i], x.to_numpy()[i]
        keep, live = np.zeros(len(i)), np.ones(len(i), dtype=bool)
        off, last = False, 0
        for t in range(len(i)):
            rejected = False
            if a[t] != 0:
                lvl = v[t] if where == "alarm" else (v[last : t + 1].min() if a[t] > 0 else v[last : t + 1].max())
                if (a[t] > 0 and lvl <= -level) or (a[t] < 0 and lvl >= level):
                    keep[t], off = a[t], False
                else:
                    rejected = close
                last = t
            live[t] = not off
            off = off or rejected
        kept.iloc[i], on.iloc[i] = keep, live
    return kept, on


def residual(
    asset: pd.Series, market: pd.Series, window: int = BETA_WINDOW, lag: int = 0
) -> tuple[pd.Series, pd.Series]:
    """`(spread, beta)`: the asset's price with the market's move taken out, bar by bar.

    The spread's 15m log return is the asset's less beta times the market's, beta the slope of the
    first on the second over the `window` bars that closed before the bar: the hedge a bar is paid
    on was set at the close before it. Held long, the spread is one unit of the asset against beta
    units of the market, rebalanced every bar, which a month's beta makes a rounding error.

    `lag` takes the market's return `lag` bars before instead (counted on the asset's bars): with
    1, what is left is the asset's move past what the market's last bar foretold, the test of
    whether BTC leads the alts. That spread is not a position anyone can hold, its hedge leg is a
    bar in the past, so its gross reads whether the part BTC does not foretell reverts.
    """
    ra, rm = np.log(asset).diff(), np.log(market).reindex(asset.index).diff().shift(lag)
    n = window // 2
    beta = (ra.rolling(window, min_periods=n).cov(rm) / rm.rolling(window, min_periods=n).var()).shift()
    e = (ra - beta * rm).fillna(0.0)
    return np.exp(np.log(asset.iloc[0]) + e.cumsum()), beta


def spreads(
    index: pd.MultiIndex, market: str = "BTC", window: int = BETA_WINDOW, lag: int = 0
) -> tuple[pd.Series, pd.Series]:
    """`residual` of each asset of `index` against `market`, over the whole store: `(spread, beta)`.

    `market` is a symbol of the store, or `ew` for the equal-weighted 15m return of `SYMBOLS` other
    than the asset: no one trades it, but it is the common move itself rather than one pair's. The
    spread spans the store, so an RSI on it has its warm-up behind it; `beta` is on `index`.
    """
    closes: dict[str, pd.Series] = {}

    def close_of(s: str) -> pd.Series:
        if s not in closes:
            closes[s] = candles(s, "15m").close
        return closes[s]

    parts, betas = [], []
    for sym in index.get_level_values(1).unique():
        a = close_of(sym)
        if market == "ew":
            r = pd.concat({s: np.log(close_of(s)).diff() for s in SYMBOLS if s != sym}, axis=1).reindex(a.index)
            m = np.exp(r.mean(axis=1).fillna(0.0).cumsum())
        else:
            m = close_of(market)
        s, b = residual(a, m, window, lag)
        parts.append(s.set_axis(pd.MultiIndex.from_arrays([s.index, [sym] * len(s)], names=index.names)))
        betas.append(b.set_axis(pd.MultiIndex.from_arrays([b.index, [sym] * len(b)], names=index.names)))
    return pd.concat(parts), pd.concat(betas).reindex(index)


# Before the confirmation period, the year Shiryaev's hazard and the logistic behind `kept` are fitted
# on: the RSI's own turns of 2020, nothing of the period itself.
FIT_YEAR = pd.DateOffset(years=1)


def history(assets, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """The store's 15m closes of `assets` from `start` to before `end`, on the (open_time, symbol) index."""
    parts = []
    for sym in assets:
        c = candles(sym, "15m").close
        c = c[(c.index >= start) & (c.index < end)]
        parts.append(c.set_axis(pd.MultiIndex.from_arrays([c.index, [sym] * len(c)], names=["open_time", "symbol"])))
    return pd.concat(parts)


def _folds(held: pd.DataFrame, period: str) -> dict:
    """bp a trade, its error and count over `period` and in each of its folds (`strategy.fold_in`).

    `se` is the trade-level error `_periods` reads; `bse` the error over weekly blocks of entries
    (`metrics.blocked`), since the assets' trades overlap in time and move together. A fold with no
    trade is NaN, which no criterion counts as positive.
    """
    when = pd.DatetimeIndex(held.entry)
    k = strategy.fold_in(when, period)
    g = pd.Series(held.gross.to_numpy()[k > 0] * 1e4, index=when[k > 0])
    row = {"bp": g.mean(), "se": g.std() / np.sqrt(len(g)), "n": len(g)}
    row["bse"] = metrics.blocked(g, pd.Timedelta("7D"))["se"] if len(g) > 1 else np.nan
    first = 3 if period == "holdout" else 1
    for f in range(first, first + len(strategy.edges(period)) - 1):
        x = g[k[k > 0] == f]
        row |= {f"fold {f}": x.mean(), f"se {f}": x.std() / np.sqrt(len(x)), f"n {f}": len(x)}
    return row


def residual_study(
    market: str = "BTC",
    levels=(0.4, 0.5, 0.6),
    h: float = 0.2,
    p: float = 0.5,
    period: str = "dev",
    assets=None,
    lag: int = 0,
) -> pd.DataFrame:
    """The detectors on each asset's price and on its residual against `market`, over one `period`.

    Three series on v2's periods: v2 and an RSI at 12 on the price, and an RSI at 12 on the spread
    (`spreads`). v2 cannot be run on a spread, it reads candles; it is 92.5% an RSI at 12 (§14 of
    the handoff), so the fair comparison is the RSI on the price against the RSI on the spread, and
    v2 is the yardstick. On `2021` there is no v2, and the two RSIs run on the store's 15m bars from
    a year before the period (`FIT_YEAR`) to its end. Zigzag `h` and Shiryaev `p`, alone and
    through `gate` at each level, read at the alarm as the page runs it and at the leg's extreme as
    the simulation in the module's docstring did, closing on a rejected alarm. A trade on the spread
    is the hedged trade, and it pays two legs: `fee_bp` is the round trip at `oracle.FEE` times
    1 + |beta| at the period's alarms. AUC and `kept` are `null_test`'s, at each detector's alarms.

    What decides something is fitted before or beside the period, never on it: Shiryaev's hazard
    and the logistic behind `kept` on development for `dev` (in sample, as `null_test`) and
    `holdout`, on the year before for `2021`; `h`, `p` and the levels are development's. The
    levels are in each series' own units, and v2's are shrunk towards zero, so a level is not the
    same selection on v2 as on an RSI. `assets` defaults to `strategy.ASSETS`, less the market.
    """
    assets = [s for s in (assets or strategy.ASSETS) if s != market]
    if period == "2021":
        start, end = strategy.CONFIRM
        close, cut, series = history(assets, start - FIT_YEAR, end), start, {}
    else:
        pred, close, cut = strategy.load(assets=assets)
        series = {"price: v2": (pred, close, None)}
    spread, beta = spreads(close.index, market, lag=lag)
    series |= {
        "price: rsi 12": (strategy.rsi(close.index), close, None),
        "residual: rsi 12": (strategy.rsi(close.index, close=spread), spread.reindex(close.index), 1 + beta.abs()),
    }
    when = close.index.get_level_values(0)
    inside = strategy.fold_in(when, period) > 0
    side = "dev" if period == "dev" else "holdout"  # which side of `cut` the period is on
    rows = []
    for name, (x, c, hedge) in series.items():
        truth, params = turn_events(x, strategy.WINDOW), fit(x, cut)
        bars = pd.DataFrame({col: c for col in stops.OHLC})  # flat bars: no barrier is asked of them
        for label, found in (
            (f"zigzag {h:g}", alarms(x, zigzag, h)),
            (f"shiryaev {p:g}", alarms(x, shiryaev, params, p)),
        ):
            at = (found != 0).to_numpy() & inside
            fee = 2 * FEE * (1.0 if hedge is None else float(hedge[at].mean())) * 1e4
            row = {"series": name, "detector": label, "fee_bp": fee}
            f = geometry(found, x, c, truth)
            score = _score(f, cut)
            kept = conservation(f, score, cut)[1]
            read = strategy.fold_in(pd.DatetimeIndex(f.when), period) > 0
            stats = {
                "auc": _auc(score[read], f.true.to_numpy()[read]),
                "kept": kept[side],
                "kept_se": kept[f"{side}_se"],
            }
            m = match(found[inside], truth[inside], when[inside].max() + strategy.BAR)
            rows.append(row | {"gate": "none"} | m | stats | _folds(plain(hold(found), c)[2], period))
            for where in ("alarm", "extreme"):
                for level in levels:
                    kept_sig, on = gate(found, x, level, where, True)
                    held = strategy.walked(kept_sig, bars, on=on)[2]
                    rows.append(row | {"gate": f"{where}, close {level:g}"} | _folds(held, period))
    return pd.DataFrame(rows).set_index(["series", "detector", "gate"])


def residual_verdict(t: pd.DataFrame, levels) -> list[str]:
    """The card's criterion on each detector of `residual_study`'s table, read and not applied.

    At the gate read at the alarm: a level whose gross is above `fee_bp` with every fold positive,
    a gross that grows with the level, and `kept` beyond two of its errors. The price's RSI is
    judged the same way, as the control.
    """
    folds = [c for c in t.columns if c.startswith("fold ")]
    out = []
    for (series, detector), d in t.groupby(level=[0, 1], sort=False):
        if series == "price: v2":
            continue
        d = d.droplevel([0, 1])
        g = d.loc[[f"alarm, close {L:g}" for L in levels]]
        pays = [
            f"{L:g} ({r.n:.0f} trades)"
            for L, (_, r) in zip(levels, g.iterrows())
            if r.bp > r.fee_bp and (r[folds] > 0).all()
        ]
        grows = bool((np.diff(g.bp.to_numpy()) > 0).all())
        none = d.loc["none"]
        kept = none.kept > 2 * none.kept_se
        ok = bool(pays) and grows and kept
        out.append(
            f"{series}, {detector}: over fee_bp {none.fee_bp:.1f} with every fold positive at L ="
            f" {', '.join(pays) or 'none'}; grows with L: {'yes' if grows else 'no'}"
            f" ({' / '.join(f'{v:+.1f}' for v in g.bp)}); kept {none.kept:+.2f} +/- {none.kept_se:.2f},"
            f" {'' if kept else 'not '}two errors above 0 -> {'PASSES' if ok else 'fails'}"
        )
    return out


def _zscore(v: pd.Series, n: int = 96 * 30) -> pd.Series:
    """`v` against its own trailing month: the level of a ratio that drifts, made comparable over time."""
    return (v - v.rolling(n, min_periods=n // 3).mean()) / v.rolling(n, min_periods=n // 3).std()


def futures_columns(symbol: str, close: pd.Series) -> pd.DataFrame:
    """`data.futures`' columns as signals at each 15m bar of one symbol, every one causal.

    Funding and the two long/short ratios as levels against their month, the open interest as its
    change over 12 and 48 bars and signed by the price's 12-bar move (open interest rising with the
    move is new positions behind it), the taker flow over 12 bars, the futures-spot basis, the book's
    imbalance within 1, 2 and 5% and its 4-bar mean, the liquidity within 1%.
    """
    f = futures.load(symbol)
    c = close.reindex(f.index)
    oi, basis = np.log(f.oi.where(f.oi > 0)), np.log(f.fut_close / c)
    out = pd.DataFrame(
        {
            "funding": f.funding,
            "funding_z": _zscore(f.funding),
            "oi_12": oi.diff(12),
            "oi_48": oi.diff(48),
            "oi_with_price_12": np.sign(np.log(c).diff(12)) * oi.diff(12),
            "top_ls_z": _zscore(np.log(f.top_ls)),
            "top_ls_12": np.log(f.top_ls).diff(12),
            "ls_z": _zscore(np.log(f.ls)),
            "taker_ls_12": f.taker_ls.rolling(12).mean(),
            "taker_buy_12": f.taker_buy.rolling(12).mean() - 0.5,
            "basis_z": _zscore(basis),
            "basis_12": basis.diff(12),
            "depth1_z": _zscore(f.depth1),
        },
        index=f.index,
    )
    for k in futures.LEVELS:
        out[f"book{k}"] = f[f"book{k}"]
        out[f"book{k}_4"] = f[f"book{k}"].rolling(4).mean()
    return out


def forward_ic(columns: dict, pred: pd.Series, close: pd.Series, horizons=(12, 48)) -> pd.DataFrame:
    """Rank IC of every column with the forward log return, by fold, mean over the assets.

    On every bar; v2's prediction and an RSI at 12 are rows too, the yardstick a new column has to
    clear. Until 2026-10-07 it sampled one bar in `h`, from the first, so that no two returns
    overlapped: the IC of one phase, which moves by 0.03-0.07 from phase to phase at 48 bars. Every
    bar is the mean of the `h` phases; no error is printed here, so the overlap inflates nothing.
    """
    rows = []
    rsi = strategy.rsi(pred.index)
    for sym, cols in columns.items():
        c = np.log(close.xs(sym, level=1))
        x = cols.reindex(c.index).assign(**{"v2 prediction": pred.xs(sym, level=1), "rsi 12": rsi.xs(sym, level=1)})
        fold = fold_of(c.index)
        for h in horizons:
            fwd = c.shift(-h) - c
            for col in x.columns:
                for k in range(1, strategy.FOLDS + 1):
                    m = (fold == k) & x[col].notna().to_numpy() & fwd.notna().to_numpy()
                    ic = np.corrcoef(x[col][m].rank(), fwd[m].rank())[0, 1]
                    rows.append({"column": col, "h": h, "fold": k, "ic": ic})
    ic = pd.DataFrame(rows).groupby(["column", "h", "fold"]).ic.mean().unstack("fold")
    return ic.assign(
        dev=ic[[1, 2]].mean(axis=1), holdout=ic[[3, 4]].mean(axis=1), same_sign=np.sign(ic).nunique(axis=1) == 1
    )


def behind_the_move(symbol: str, close: pd.Series, k: int) -> pd.DataFrame:
    """The last `k` bars' move and the open interest's change over them, at each 15m bar.

    `move` is the sign of the price's `k`-bar change, `oi_z` the open interest's `k`-bar log change
    against its trailing month's dispersion: positive when positions were opened behind the move.
    """
    f = futures.load(symbol)
    c = np.log(close.reindex(f.index))
    doi = np.log(f.oi.where(f.oi > 0)).diff(k)
    return pd.DataFrame(
        {"c": c, "move": np.sign(c.diff(k)), "oi_z": doi / doi.rolling(96 * 30, min_periods=96 * 10).std()}
    )


def forward(close: pd.Series, h: int) -> pd.Series:
    """The log return from each bar's close to the close `h` bars later, per asset. The bar's own
    return is not in it: a column known at the bar's close is read against what comes after."""
    lc = np.log(close)
    return lc.groupby(level=1).shift(-h) - lc


def _shifted(mask: np.ndarray, sym: pd.Index, fold: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A placebo of `mask`: within each asset and fold the same mask rolled in time by a random offset
    between a tenth and nine tenths of the fold. Same size, same runs, any tie to the market gone."""
    out = np.zeros(len(mask), dtype=bool)
    for s in sym.unique():
        for f in np.unique(fold[fold > 0]):
            i = np.flatnonzero((sym == s) & (fold == f))
            out[i] = np.roll(mask[i], rng.integers(len(i) // 10, 9 * len(i) // 10))
    return out


def conditional_frame(pred: pd.Series, cut: pd.Timestamp, period: str = "dev", seeds=(0, 1, 2), horizons=(12, 24, 48)):
    """`(x, fwd, subsets, fold)` on v2's rows of `period`: what `conditional_ic` and `power` read.

    `x`: every `futures_columns` column, open interest behind the move (`move * oi_z`) at k = 4, 12
    and 24, and v2's prediction and the RSI at 12 as the yardstick rows. The futures columns are
    built on the store's close, so their trailing windows are warm by v2's first bar. `fwd`: the
    `forward` return at each horizon, on the store's close too. `subsets`: the bars at the pivot,
    |v2| >= 0.4 / 0.5 / 0.6 in its raw units and the alarm bars of Shiryaev 0.5 with `fit` on
    development, and for each seed a `_shifted` placebo of each.
    """
    when, sym = pred.index.get_level_values(0), pred.index.get_level_values(1)
    fold = strategy.fold_in(when, period)
    rsi, parts, closes = strategy.rsi(pred.index), [], []
    for s in sym.unique():
        c = candles(s, "15m").close
        cols = futures_columns(s, c)
        for k in (4, 12, 24):
            d = behind_the_move(s, c, k)
            cols[f"oi behind the move {k}"] = d.move * d.oi_z
        idx = pred.xs(s, level=1).index
        cols = cols.reindex(idx).assign(**{"v2 prediction": pred.xs(s, level=1), "rsi 12": rsi.xs(s, level=1)})
        on = pd.MultiIndex.from_arrays([idx, [s] * len(idx)], names=pred.index.names)
        parts.append(cols.set_axis(on))
        closes.append(c.reindex(idx).set_axis(on))
    x, close = pd.concat(parts).reindex(pred.index), pd.concat(closes).reindex(pred.index)
    fwd = pd.DataFrame({h: forward(close, h) for h in horizons})
    found = alarms(pred, shiryaev, fit(pred, cut), 0.5)
    real = {f"|v2| >= {L:g}": (pred.abs() >= L).to_numpy() for L in (0.4, 0.5, 0.6)}
    real["shiryaev 0.5 alarm"] = (found != 0).to_numpy()
    subsets = pd.DataFrame(real, index=pred.index)
    for k in seeds:
        rng = np.random.default_rng(k)
        for name, m in real.items():
            subsets[f"placebo #{k}: {name}"] = _shifted(m, sym, fold, rng)
    keep = fold > 0
    return x[keep], fwd[keep], subsets[keep], fold[keep]


def _zrank(v: np.ndarray) -> np.ndarray | None:
    """Ranks, ties averaged as Spearman takes them, standardised with ddof 0: the mean product of two
    of these is Pearson on the ranks. None when the ranks cannot be standardised."""
    r = pd.Series(v).rank().to_numpy()
    return (r - r.mean()) / r.std() if len(r) > 2 and r.std() > 0 else None


def _sums(p: np.ndarray, block: np.ndarray, nb: int) -> np.ndarray:
    """`p` (..., n) summed over each clock block, `block` the sorted block codes of its n rows."""
    out = np.zeros(p.shape[:-1] + (nb,))
    if len(block):
        start = np.flatnonzero(np.r_[True, block[1:] != block[:-1]])
        out[..., block[start]] = np.add.reduceat(p, start, axis=-1)
    return out


def _pair_sums(xv: np.ndarray, yv: np.ndarray, keep: np.ndarray, assets: list, block: np.ndarray, nb: int):
    """`(S, N)` per clock block: the products of standardised ranks of `xv` and `yv` on the rows `keep`
    where both are known, ranked within each asset (`assets` are their rows), and their count."""
    S, N = np.zeros(nb), np.zeros(nb)
    for i in assets:
        ok = keep[i] & np.isfinite(xv[i]) & np.isfinite(yv[i])
        zx, zy = _zrank(xv[i][ok]), _zrank(yv[i][ok])
        if zx is not None and zy is not None:
            S += _sums(zx * zy, block[i][ok], nb)
            N += _sums(np.ones(len(zx)), block[i][ok], nb)
    return S, N


def _delta(Sa: np.ndarray, Na: np.ndarray, Sc: np.ndarray, Nc: np.ndarray) -> dict:
    """IC on every bar (`a`) and on the subset (`c`), their ratio and difference, each with its error.

    Each IC is Σ products / Σ bars over the clock blocks. A block moves it by its residual sum over
    the total (`ea`, `ec`), the ratio by ec / a - c ea / a², the difference by ec - ea; the variance
    is the sum of their squares over the blocks, times nb / (nb - 1). With full blocks it is
    `metrics.blocked`'s error of the mean (the self-check asserts it). Arrays (..., nb) to (...).
    """
    nb = (Na > 0).sum(-1)
    k = nb / (nb - 1)
    na, nc = Na.sum(-1), Nc.sum(-1)
    a, c = Sa.sum(-1) / na, Sc.sum(-1) / nc
    ea = (Sa - a[..., None] * Na) / np.asarray(na)[..., None]
    ec = (Sc - c[..., None] * Nc) / np.asarray(nc)[..., None]
    er = ec / a[..., None] - c[..., None] * ea / a[..., None] ** 2
    se = {name: np.sqrt(k * (e**2).sum(-1)) for name, e in (("a", ea), ("c", ec), ("r", er), ("d", ec - ea))}
    return {
        "ic_all": a,
        "ic_all_se": se["a"],
        "ic_cond": c,
        "ic_cond_se": se["c"],
        "ratio": c / a,
        "ratio_se": se["r"],
        "diff": c - a,
        "diff_se": se["d"],
    }


def _blocks(when: pd.DatetimeIndex, fold: np.ndarray, f: int, h: int) -> tuple[np.ndarray, int]:
    """Codes of the clock blocks of `h` bars (`metrics.blocked`'s floor) of fold `f`'s rows, -1 elsewhere."""
    block, on = np.full(len(fold), -1), fold == f
    block[on] = pd.factorize(when[on].floor(h * strategy.BAR), sort=True)[0]
    return block, int(block.max()) + 1


def conditional_ic(x: pd.DataFrame, fwd: pd.DataFrame, subsets: pd.DataFrame, fold: np.ndarray) -> pd.DataFrame:
    """One row per (subset, h, column, fold): the rank IC with the h-bar forward return on every bar of
    the fold and on the subset's bars, their ratio and their difference, with errors from `_delta`.

    The ranks are taken within each asset, fold and subset, and the three assets are pooled bar by
    bar: the IC is the mean product of standardised ranks, which weighs each asset by its bars.
    """
    when, sym = x.index.get_level_values(0), x.index.get_level_values(1)
    rows = []
    for f in np.unique(fold):
        assets = [np.flatnonzero((fold == f) & (sym == s)) for s in sym.unique()]
        for h in fwd.columns:
            block, nb = _blocks(when, fold, f, h)
            yv = fwd[h].to_numpy()
            for col in x.columns:
                xv = x[col].to_numpy()
                Sa, Na = _pair_sums(xv, yv, np.ones(len(xv), dtype=bool), assets, block, nb)
                for name in subsets.columns:
                    Sc, Nc = _pair_sums(xv, yv, subsets[name].to_numpy(), assets, block, nb)
                    head = {"subset": name, "h": h, "column": col, "fold": f, "bars": Na.sum(), "bars_cond": Nc.sum()}
                    rows.append(head | {k: float(v) for k, v in _delta(Sa, Na, Sc, Nc).items()})
    return pd.DataFrame(rows)


def thresholds(close: pd.Series, horizons=(12, 24, 48)) -> pd.DataFrame:
    """ρ_min ≈ c / (0.8 σ √h) (`false_alarms.html`, #soglia): the IC with the h-bar return that pays a
    round trip c, spot taker (2 `FEE`) and perpetual taker (2 `PERP_FEE`). σ is the sd of each asset's
    15m log return on development, and `pooled` takes the mean of the three: a common IC on the three
    pays the round trip at that σ. Measured on development and applied unchanged to the hold-out."""
    dev = strategy.fold_in(close.index.get_level_values(0), "dev") > 0
    sigma = np.log(close).groupby(level=1).diff()[dev].groupby(level=1).std()
    sigma["pooled"] = sigma.mean()
    fees = {"spot": 2 * FEE, "perp": 2 * PERP_FEE}
    return pd.DataFrame(
        {"sigma": sigma} | {f"{k} {h}": c / (0.8 * sigma * np.sqrt(h)) for k, c in fees.items() for h in horizons}
    )


def _passes(ratio: np.ndarray, ratio_se: np.ndarray, ic_cond: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """The card's criterion over the last axis, the folds: `(ratio part, whole)`. The ratio above 1 by
    more than two errors in every fold; and the subset's IC above `rho` in size, with one sign in
    every fold, since a column is traded on either sign but not on both."""
    up = (ratio - 1 > 2 * ratio_se).all(-1)
    same = (np.sign(ic_cond) == np.sign(ic_cond[..., :1])).all(-1)
    return up, up & same & (np.abs(ic_cond) > rho).all(-1)


def verdict(t: pd.DataFrame, rho: pd.DataFrame) -> pd.DataFrame:
    """`_passes` on `conditional_ic`'s rows, per (subset, h, column), at the pooled ρ_min of each instrument."""
    w = t.set_index(["subset", "h", "column", "fold"])[["ratio", "ratio_se", "ic_cond"]].unstack("fold")
    out = {}
    h = w.index.get_level_values("h")
    for inst in ("spot", "perp"):
        r = rho.loc["pooled", [f"{inst} {k}" for k in h]].to_numpy()
        out["ratio"], out[inst] = _passes(w.ratio.to_numpy(), w.ratio_se.to_numpy(), w.ic_cond.to_numpy(), r[:, None])
    return pd.DataFrame(out, index=w.index)


def power(
    fwd: pd.DataFrame,
    subsets: pd.DataFrame,
    fold: np.ndarray,
    rho: pd.DataFrame,
    ratios=(1.0, 1.5, 2.0, 3.0),
    levels=(0.03, 0.06),
    reps: int = 400,
    seed: int = 0,
    chunk: int = 100,
) -> pd.DataFrame:
    """How often the card's criterion passes on a column with a known effect, at the store's sizes.

    The forward returns, the subsets and the clock blocks are the real ones of the period, so the
    sample sizes, the subsets' runs and the overlap are the store's. Only the column is drawn:
    x = ρ_t z + sqrt(1 - ρ_t²) ε, with z the standardised rank of the forward return within each
    asset and fold, ρ_t = ratio × ρ_all on the subset's bars and the value that keeps the IC on every
    bar at ρ_all elsewhere, and ε an independent moving sum of h bars, as persistent as the forward
    return (ponytail: real columns run from a few bars, the book, to weeks, funding; a slower ε
    widens the errors). The IC is Pearson of x on the ranks, which is the rank IC's sampling law at
    these sizes. ratio = 1 is the null of the ratio, so its column is the false-positive rate.
    Returns the share of `reps` passing the ratio part, and the whole criterion at spot and perpetual.
    """
    when, sym = fwd.index.get_level_values(0), fwd.index.get_level_values(1)
    folds, rng, rows = np.unique(fold), np.random.default_rng(seed), []
    for h in fwd.columns:
        y = fwd[h].to_numpy()
        data = {}
        for f in folds:
            block, nb = _blocks(when, fold, f, h)
            per = []
            for s in sym.unique():
                i = np.flatnonzero((fold == f) & (sym == s) & np.isfinite(y))
                per.append((_zrank(y[i]), block[i], {c: subsets[c].to_numpy()[i] for c in subsets.columns}))
            data[f] = (per, nb)
        hits = {}
        for start in range(0, reps, chunk):
            r = min(chunk, reps - start)
            stats = {}
            for f, (per, nb) in data.items():
                eps = []
                for z, _, _ in per:
                    walk = np.concatenate([np.zeros((r, 1)), rng.standard_normal((r, len(z) + h)).cumsum(1)], axis=1)
                    eps.append((walk[:, h : h + len(z)] - walk[:, : len(z)]) / np.sqrt(h))
                for cond in subsets.columns:
                    for level in levels:
                        for ratio in ratios:
                            Sa, Sc, Na, Nc = np.zeros((r, nb)), np.zeros((r, nb)), np.zeros(nb), np.zeros(nb)
                            for (z, b, masks), e in zip(per, eps):
                                m = masks[cond]
                                q = m.mean()
                                rho_t = np.where(m, ratio * level, (level - q * ratio * level) / (1 - q))
                                xv = rho_t * z + np.sqrt(1 - rho_t**2) * e
                                xa = (xv - xv.mean(1, keepdims=True)) / xv.std(1, keepdims=True)
                                Sa += _sums(xa * z, b, nb)
                                Na += _sums(np.ones(len(z)), b, nb)
                                xc, zc = xv[:, m], z[m]
                                xc = (xc - xc.mean(1, keepdims=True)) / xc.std(1, keepdims=True)
                                Sc += _sums(xc * ((zc - zc.mean()) / zc.std()), b[m], nb)
                                Nc += _sums(np.ones(m.sum()), b[m], nb)
                            d = _delta(Sa, Na, Sc, Nc)
                            stats.setdefault((cond, level, ratio), []).append((d["ratio"], d["ratio_se"], d["ic_cond"]))
            for key, per_fold in stats.items():
                ratio_, se_, ic_ = (np.stack(v, axis=-1) for v in zip(*per_fold))
                for inst in ("spot", "perp"):
                    up, whole = _passes(ratio_, se_, ic_, rho.loc["pooled", f"{inst} {h}"])
                    hits.setdefault(key + (inst,), []).append(whole)
                hits.setdefault(key + ("ratio",), []).append(up)
        for (cond, level, ratio, part), v in hits.items():
            rows.append(
                {
                    "subset": cond,
                    "h": h,
                    "ic_all": level,
                    "ratio": ratio,
                    "part": part,
                    "pass": np.concatenate(v).mean(),
                }
            )
    return pd.DataFrame(rows)


def open_interest(close: pd.Series, cut: pd.Timestamp, ks=(4, 12, 24), horizons=(12, 24, 48, 96), hold_bars=48):
    """Open interest behind the move, three ways: `(ic, quadrants, rule)`.

    `ic`: rank IC of `move * oi_z`, and of `move` alone (momentum, the control), with the h-bar
    forward return, by fold. `quadrants`: the next `hold_bars` in the move's direction, bp, by
    whether open interest rose or fell with it. `rule`: when flat and |oi_z| >= Z, follow the move if
    open interest rose with it or fade it if it fell, out after `hold_bars`; one position at a time.
    The IC and the quadrants read every bar, not one bar in `h` (see `forward_ic`).
    """
    ic_rows, quad_rows, rule_rows = [], [], []
    for k in ks:
        for sym in strategy.ASSETS:
            d = behind_the_move(sym, close.xs(sym, level=1), k)
            d = d[d.index >= strategy.TEST_START]
            fold = fold_of(d.index)
            for h in horizons:
                fwd = d.c.shift(-h) - d.c
                for name, x in (("oi behind the move", d.move * d.oi_z), ("momentum", d.move)):
                    for f in range(1, strategy.FOLDS + 1):
                        m = (fold == f) & x.notna().to_numpy() & fwd.notna().to_numpy()
                        ic = np.corrcoef(x[m].rank(), fwd[m].rank())[0, 1]
                        ic_rows.append({"column": name, "k": k, "h": h, "fold": f, "ic": ic})
                if h == hold_bars:
                    m = fwd.notna().to_numpy() & d.oi_z.notna().to_numpy()
                    q = pd.DataFrame({"bp": (d.move * fwd)[m] * 1e4, "oi": np.where(d.oi_z[m] > 0, "rose", "fell")})
                    q["fold"] = fold[m]
                    quad_rows += [
                        {"k": k} | r for r in q.groupby(["oi", "fold"]).bp.mean().reset_index().to_dict("records")
                    ]
            cv, mv, zv = d.c.to_numpy(), d.move.to_numpy(), d.oi_z.to_numpy()
            for Z in (0.5, 1.0, 1.5, 2.0):
                i = 0
                while i < len(cv) - hold_bars:
                    if np.isfinite(zv[i]) and abs(zv[i]) >= Z and mv[i] != 0:
                        bp = mv[i] * np.sign(zv[i]) * (cv[i + hold_bars] - cv[i]) * 1e4
                        rule_rows.append({"k": k, "Z": Z, "when": d.index[i], "bp": bp})
                        i += hold_bars
                    else:
                        i += 1
    ic = pd.DataFrame(ic_rows).groupby(["column", "k", "h", "fold"]).ic.mean().unstack("fold")
    ic = ic.assign(dev=ic[[1, 2]].mean(axis=1), holdout=ic[[3, 4]].mean(axis=1))
    quads = pd.DataFrame(quad_rows).groupby(["k", "oi", "fold"]).bp.mean().unstack("fold")
    r = pd.DataFrame(rule_rows)
    r["period"] = np.where(r.when < cut, "dev", "holdout")
    r["fold"] = fold_of(pd.DatetimeIndex(r.when))
    rule = r.groupby(["k", "Z", "period"]).bp.agg(["size", "mean", "sem"]).unstack("period")
    rule.columns = [f"{period}_{stat}" for stat, period in rule.columns]
    rule = rule.join(r.groupby(["k", "Z", "fold"]).bp.mean().unstack("fold").add_prefix("fold "))
    return ic, quads, rule


def confirmation(sig: pd.Series, close: pd.Series, k: int) -> pd.Series:
    """At each signal, the signal's side times open interest behind the last `k` bars' move.

    The column predicts the next bars' direction (a move continues when positions were opened
    behind it, reverses when they were closed), so the product is positive where it agrees with
    the signal: a long at the end of a fall is confirmed when the fall was made closing positions.
    """
    parts = []
    for sym in strategy.ASSETS:
        d = behind_the_move(sym, close.xs(sym, level=1), k)
        col = (d.move * d.oi_z).reindex(close.xs(sym, level=1).index)
        parts.append(pd.Series(col.to_numpy(), index=close.xs(sym, level=1, drop_level=False).index))
    return (sig * pd.concat(parts).reindex(sig.index)).where(sig != 0)


def futures_at_alarm(f: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """`at_alarm`'s rows with the futures columns at the alarm, signed into the leg it closes.

    Over the leg (from the alarm before to this one): the change in open interest and in the top
    traders' ratio, the mean taker flow; at the alarm: funding, the ratios, the basis, the book.
    """
    rows = []
    for r in f.itertuples():
        F, C = futures.load(r.symbol), columns[r.symbol]
        t, start = r.when, r.when - pd.Timedelta(minutes=15 * int(r.age))
        leg = F.loc[start:t]
        rows.append(
            {
                "oi_leg": np.log(F.oi.get(t, np.nan) / F.oi.get(start, np.nan)),
                "oi_4": np.log(F.oi.get(t, np.nan) / F.oi.shift(4).get(t, np.nan)),
                "funding_s": r.side * F.funding.get(t, np.nan),
                "funding_z_s": r.side * C.funding_z.get(t, np.nan),
                "top_ls_s": r.side * C.top_ls_z.get(t, np.nan),
                "top_ls_leg": r.side * np.log(F.top_ls.get(t, np.nan) / F.top_ls.get(start, np.nan)),
                "ls_s": r.side * C.ls_z.get(t, np.nan),
                "taker_leg": r.side * leg.taker_ls.mean(),
                "taker_4": r.side * F.taker_ls.rolling(4).mean().get(t, np.nan),
                "takerbuy_leg": r.side * (leg.taker_buy.mean() - 0.5),
                "basis_s": r.side * C.basis_z.get(t, np.nan),
                "book1_s": r.side * F.book1.get(t, np.nan),
                "book2_s": r.side * F.book2.get(t, np.nan),
                "book5_s": r.side * F.book5.get(t, np.nan),
                "depth1_leg": F.depth1.get(t, np.nan) - leg.depth1.mean(),
            }
        )
    return pd.concat([f.reset_index(drop=True), pd.DataFrame(rows)], axis=1).dropna()


def _selfcheck() -> None:
    """A noiseless saw with known turns: each detector finds every one, at the delay its rule implies."""
    i = np.arange(24 * 6 + 1)
    v = 0.05 * np.abs(i % 24 - 12)  # a top on bar 0, then a turn every 12 bars, 0.05 a bar
    t = pd.date_range("2025-06-01", periods=len(v), freq="15min", tz="UTC")
    x = pd.Series(v, index=pd.MultiIndex.from_arrays([t, ["A"] * len(v)], names=["open_time", "symbol"]))
    truth = pd.Series(0.0, index=x.index)
    turns = list(range(0, len(v) - 1, 12))
    truth.iloc[turns] = [-1.0 if k % 24 == 0 else 1.0 for k in turns]
    cut = t[-1] + pd.Timedelta("1D")
    # A drop of 0.05 a bar clears h = 0.12 on the third bar after the top: delay 3, no false alarm.
    z = alarms(x, zigzag, 0.12)
    m = match(z, truth, cut)
    assert m["found"] == 1.0 and m["delay"] == 3.0 and m["false"] == 0.0, m
    # With the evidence this sharp and a flat hazard, the first bar that moves back is enough.
    p = {"beta": np.zeros(5), "m0": 0.05, "s0": 0.01, "m1": -0.05, "s1": 0.01, "base": 0.05}
    m = match(alarms(x, shiryaev, p, 0.9, True), truth, cut)
    assert m["found"] == 1.0 and m["delay"] == 1.0 and m["false"] == 0.0, m
    # The gate on the leg's extreme: tops at 0.6 keep every short at 0.5, bottoms at 0 reject every
    # long, and a rejected long closes the short at its own close and stands flat to the next short.
    kept, on = gate(z, x, 0.5, "extreme", True)
    assert set(kept[kept != 0]) == {-1.0}
    assert on.iloc[turns[1] + 3] and not on.iloc[turns[1] + 4] and on.iloc[turns[2] + 3]
    assert gate(z, x, 0.5, "extreme", False)[1].all()
    # An alarm on the wrong side is false, a later one on the right side is a re-entry, not a detection.
    noisy = z.copy()
    noisy.iloc[turns[0] + 6] = 1.0
    assert match(noisy, truth, cut)["false"] == 1 / len(turns)
    # `geometry` on the saw: every alarm closing a whole leg is true, 3 bars and 0.15 off its extreme,
    # 12 bars after the last, and grosses the 0.30 the next leg travels before its own alarm. The
    # first 48 bars have no volatility yet and the last alarm's trade is cut by the data.
    g = geometry(z, x, np.exp(x), truth)
    assert len(g) == 8 and g.true.all() and np.allclose(g.retrace, 0.15) and (g.age == 12).all(), g
    assert np.allclose(g.gross.iloc[:-1], 0.30)
    # `conservation`: precision worth its face value keeps all of it, a martingale's keeps none. Five
    # score buckets from 50% to 90% true; true alarms make 30 bp; false ones lose a fixed 80 bp, or
    # whatever makes the bucket gross zero. The same rows on both sides of the cut.
    share = np.repeat([0.5, 0.6, 0.7, 0.8, 0.9], 100)
    y = (np.tile(np.arange(100), 5) < share * 100).astype(int)
    when = pd.date_range("2025-06-01", periods=2 * len(y), freq="h", tz="UTC")
    for loss, expected in ((np.full(len(y), 0.008), 1.0), (0.003 * share / (1 - share), 0.0)):
        gross = np.where(y == 1, 0.003, -loss)
        f = pd.DataFrame({"when": when, "true": np.tile(y, 2), "gross": np.tile(gross, 2)})
        table, kept = conservation(f, np.tile(np.repeat(np.arange(5.0), 100), 2), when[len(y)])
        assert np.isclose(kept["dev"], expected) and np.isclose(kept["holdout"], expected), kept
    assert np.allclose(table["L/W"], table["P/(1-P)"]) and np.allclose(table.all_bp, 0.0)
    # `residual`: beta found, the market's move gone from the spread, and the hedge a bar is paid on
    # set before it: a jump in the market moves the next bar's beta and not its own.
    rng = np.random.default_rng(0)
    t = pd.date_range("2025-01-01", periods=4000, freq="15min", tz="UTC")
    rm = rng.normal(0, 0.004, len(t))
    ra = 1.5 * rm + rng.normal(0, 0.002, len(t))
    market, asset = pd.Series(np.exp(np.cumsum(rm)), index=t), pd.Series(100 * np.exp(np.cumsum(ra)), index=t)
    spread, beta = residual(asset, market, 960)
    assert abs(beta.iloc[-1] - 1.5) < 0.05, beta.iloc[-1]
    assert abs(np.corrcoef(np.log(spread).diff()[1000:], rm[1000:])[0, 1]) < 0.05
    jump = rm.copy()
    jump[3000] += 0.1
    moved = residual(asset, pd.Series(np.exp(np.cumsum(jump)), index=t), 960)[1]
    assert moved.iloc[3000] == beta.iloc[3000] and moved.iloc[3001] != beta.iloc[3001]
    # `lag=1`: an asset that follows the market a bar late has its beta found on the market's bar
    # before and that lead taken out. A jump in the market moves the spread of the bar after, at the
    # beta set before that bar, and the beta only from the bar after that.
    late = pd.Series(100 * np.exp(np.cumsum(1.5 * np.r_[0.0, rm[:-1]] + rng.normal(0, 0.002, len(t)))), index=t)
    lagged, lbeta = residual(late, market, 960, lag=1)
    assert abs(lbeta.iloc[-1] - 1.5) < 0.05, lbeta.iloc[-1]
    assert abs(np.corrcoef(np.log(lagged).diff()[1001:], rm[1000:-1])[0, 1]) < 0.05
    jumped, jbeta = residual(late, pd.Series(np.exp(np.cumsum(jump)), index=t), 960, lag=1)
    assert jbeta.iloc[3001] == lbeta.iloc[3001] and jbeta.iloc[3002] != lbeta.iloc[3002]
    step = np.log(jumped).diff() - np.log(lagged).diff()
    assert step.iloc[3000] == 0.0 and np.isclose(step.iloc[3001], -0.1 * lbeta.iloc[3001])
    # `_folds` numbers 2021's slices 1-4 and leaves a fold with no trade NaN, never positive.
    entry = pd.DatetimeIndex(["2021-02-01", "2021-03-01", "2023-06-01"], tz="UTC")
    r = _folds(pd.DataFrame({"entry": entry, "gross": [0.001, 0.003, -0.002]}), "2021")
    assert r["n"] == 3 and np.isclose(r["fold 1"], 20.0) and r["n 1"] == 2 and np.isclose(r["fold 3"], -20.0)
    assert np.isnan(r["fold 2"]) and r["n 4"] == 0
    # Route 2. `_delta`'s error of a mean over full clock blocks is `metrics.blocked`'s: one estimator.
    # Imported here: at module level it is in sys.modules before `tests` runs `metrics` as __main__.
    from tradingvision import metrics

    t = pd.date_range("2025-06-01", periods=4800, freq="15min", tz="UTC")
    p, one = rng.normal(0.05, 1, len(t)), np.ones(len(t))
    b, nb = _blocks(t, np.ones(len(t), dtype=int), 1, 12)
    d = _delta(_sums(p, b, nb), _sums(one, b, nb), _sums(p, b, nb), _sums(one, b, nb))
    ref = metrics.blocked(pd.Series(p, index=t), 12 * strategy.BAR)
    assert np.isclose(d["ic_all"], p.mean()) and np.isclose(d["ic_all_se"], ref["se"]), (d, ref)
    # Three random walks, a subset of 15% of the bars, h = 12. A column planted at IC 0.5 on the
    # subset and 0.2 on every bar comes back at a ratio of 2.5; one at 0.2 everywhere at 1; the
    # bar's own return, known at its close, has no IC with `forward`, and the next bar's would.
    n, h = 6000, 12
    t = pd.date_range("2025-06-01", periods=n, freq="15min", tz="UTC")
    idx = pd.MultiIndex.from_product([t, ["A", "B", "C"]], names=["open_time", "symbol"])
    close = pd.Series(np.exp(rng.normal(0, 0.004, (n, 3)).cumsum(0)).ravel(), index=idx)
    fwd = pd.DataFrame({h: forward(close, h)})
    z = ((fwd[h] - fwd[h].mean()) / fwd[h].std()).to_numpy()
    sub, noise, fold = rng.random(len(idx)) < 0.15, rng.normal(size=len(idx)), np.ones(len(idx), dtype=int)
    rho_t = np.where(sub, 0.5, (0.2 - 0.15 * 0.5) / 0.85)
    own = np.log(close).groupby(level=1).diff()
    x = pd.DataFrame(
        {
            "planted": rho_t * z + np.sqrt(1 - rho_t**2) * noise,
            "flat": 0.2 * z + np.sqrt(1 - 0.2**2) * noise,
            "own bar": own,
            "next bar": own.groupby(level=1).shift(-1),
        },
        index=idx,
    )
    subsets = pd.DataFrame({"pivot": sub}, index=idx)
    got = conditional_ic(x, fwd, subsets, fold).set_index("column")
    g = got.loc["planted"]
    assert abs(g.ratio - 2.5) < 3 * g.ratio_se and g.ratio - 1 > 2 * g.ratio_se, g
    assert abs(got.loc["flat"].ratio - 1) < 3 * got.loc["flat"].ratio_se, got.loc["flat"]
    assert abs(got.loc["own bar"].ic_all) < 3 * got.loc["own bar"].ic_all_se and got.loc["next bar"].ic_all > 0.2
    # The placebo keeps each asset's count; `power` holds its false-positive rate at ratio 1 (one
    # fold, one side: 2.3% nominal) and finds a ratio of 3 on these sizes.
    sym = idx.get_level_values(1)
    placebo = _shifted(sub, sym, fold, rng)
    assert all(placebo[sym == s].sum() == sub[sym == s].sum() for s in "ABC") and (placebo != sub).any()
    rho = pd.DataFrame({"spot 12": [0.3], "perp 12": [0.3]}, index=["pooled"])
    pw = power(fwd, subsets, fold, rho, ratios=(1.0, 3.0), levels=(0.15,), reps=300).set_index(["ratio", "part"])
    assert pw.loc[(1.0, "ratio"), "pass"] < 0.06 and pw.loc[(3.0, "ratio"), "pass"] > 0.9, pw
    assert pw.loc[(1.0, "spot"), "pass"] == 0 and pw.loc[(3.0, "spot"), "pass"] > 0.9, pw
    # The alarms the subsets are cut on are causal: the bars after a cut change none before it.
    v = np.cumsum(rng.normal(0, 0.08, 3000))
    assert (shiryaev(v, V2_FIT, 0.5)[:2000] == shiryaev(v[:2000], V2_FIT, 0.5)).all()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zigzag", type=float, nargs="+", metavar="H", help="retracements, in the prediction's units")
    ap.add_argument("--shiryaev", type=float, nargs="+", metavar="P", help="posterior thresholds")
    ap.add_argument("--flat", action="store_true", help="with --shiryaev: also the flat-hazard detector")
    ap.add_argument("--both", type=float, nargs="+", metavar="H", help="with --shiryaev: also wait for a retracement")
    ap.add_argument("--split", action="store_true", help="where each detector's P&L goes, on the prediction")
    ap.add_argument("--features", type=float, metavar="P", help="what tells Shiryaev P's true alarms from false ones")
    ap.add_argument("--sl", type=float, nargs="+", metavar="ATR", help="Shiryaev 0.5's trades with these stops")
    ap.add_argument("--futures", action="store_true", help="funding, open interest, flow and book depth (data.futures)")
    ap.add_argument("--gate", type=float, nargs="+", metavar="L", help="zigzag 0.2 and Shiryaev 0.5 past these levels")
    ap.add_argument("--oi", action="store_true", help="open interest behind the move: IC, quadrants, a 48-bar rule")
    ap.add_argument("--confirm", action="store_true", help="open interest as a confirmation of the rules' signals")
    ap.add_argument("--null", type=float, metavar="P", help="Shiryaev P's true/false split on real and random paths")
    ap.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2], help="with --null: the random paths; --conditional: placebos"
    )
    ap.add_argument("--residual", metavar="MARKET", help="the detectors on each asset less beta times MARKET, or ew")
    ap.add_argument("--conditional", action="store_true", help="route 2: the futures columns' IC at v2's pivots")
    ap.add_argument("--power", action="store_true", help="with --conditional: the criterion on planted columns")
    ap.add_argument(
        "--period", choices=strategy.PERIODS, default="dev", help="with --residual or --conditional: the folds read"
    )
    ap.add_argument("--lag", type=int, default=0, help="with --residual: the market's return this many bars before")
    ap.add_argument("--tradable", action="store_true", help="with --residual: every TRADABLE pair, not ETH/BTC/SOL")
    args = ap.parse_args()
    if args.conditional and args.period == "2021":
        raise SystemExit(
            "--conditional reads the futures columns, which start on 2025-05-01 (data.futures): route 2 has no"
            " 2021-2025 reading. Run it with --period dev or --period holdout."
        )

    _selfcheck()
    pred, close, cut = strategy.load()
    series = {"prediction": pred, "rsi 12": strategy.rsi(pred.index)}
    pd.set_option("display.width", 250)
    truth = turn_events(pred, strategy.WINDOW)
    if args.conditional:
        x, fwd, subsets, fold = conditional_frame(pred, cut, args.period, args.seeds)
        rho, folds = thresholds(close), [int(f) for f in np.unique(fold)]
        real = [c for c in subsets.columns if not c.startswith("placebo")]
        print(f"route 2 on {args.period}, folds {folds}: rank IC with the h-bar forward return, ETH/BTC/SOL pooled\n")
        print(subsets[real].assign(all=True).groupby(fold).sum().T.rename(columns=lambda f: f"bars, fold {f}"))
        print("\nrho_min = round trip / (0.8 sigma sqrt(h)), sigma the sd of the 15m log return on development\n")
        print(rho.round(4).to_string())
        t = conditional_ic(x, fwd, subsets, fold)
        v = verdict(t, rho)
        shown = {
            "ic_all": "all",
            "ic_cond": "cond",
            "ratio": "ratio",
            "ratio_se": "+/-",
            "diff": "diff",
            "diff_se": "+/- ",
        }
        for name in real:
            w = t[t.subset == name].set_index(["h", "column", "fold"])[list(shown)].unstack("fold")
            w.columns = [f"f{f} {shown[s]}" for s, f in w.columns]
            w = w[[f"f{f} {s}" for f in folds for s in shown.values()]].join(v.loc[name])
            print(f"\n{name}: IC on every bar and on the subset, their ratio and difference, by fold\n")
            print(w.round(3).to_string())
        base = t.subset.str.replace(r"^placebo #\d+: ", "", regex=True)
        z = t.assign(base=base, kind=np.where(base != t.subset, "placebo", "real"))
        z = z.assign(z_ratio=(z.ratio - 1) / z.ratio_se, z_diff=z["diff"] / z.diff_se)
        g = z.groupby(["base", "kind", "h"])
        vv = v.reset_index()
        vv["base"] = vv.subset.str.replace(r"^placebo #\d+: ", "", regex=True)
        vv["kind"] = np.where(vv.base != vv.subset, "placebo", "real")
        summary = pd.DataFrame(
            {
                "rows": g.size(),
                "median ratio": g.ratio.median(),
                "z_ratio > 2": g.z_ratio.apply(lambda s: (s > 2).mean()),
                "|z_diff| > 2": g.z_diff.apply(lambda s: (s.abs() > 2).mean()),
            }
        ).join(vv.groupby(["base", "kind", "h"])[["ratio", "spot", "perp"]].sum().add_prefix("pass "))
        print(f"\nthe real subsets against their placebos ({len(args.seeds)} draws each); rows are (column, fold)\n")
        print(summary.round(3).to_string())
        rv, pv = v.loc[real], v.drop(index=real, level="subset")
        check = "" if args.period == "dev" else ", read as a check on the spent hold-out and never a choice"
        print(
            f"\nverdict (#condizionata{check}): the ratio above 1 by two errors in every fold on {rv.ratio.sum()}"
            f" of {len(rv)} (subset, h, column); with |IC_cond| above rho_min and one sign, spot taker"
            f" {rv.spot.sum()}, perpetual taker {rv.perp.sum()}. Placebos: {pv.ratio.sum()} of {len(pv)} on the"
            f" ratio, {pv.perp.sum()} on the whole at perpetual."
        )
        if rv.ratio.any():
            print(rv[rv.ratio].to_string())
        if args.power:
            pw = power(fwd, subsets[real], fold, rho)
            table = pw.pivot_table(index=["subset", "h", "ic_all"], columns=["part", "ratio"], values="pass")
            table = table[[(p, r) for p in ("ratio", "spot", "perp") for r in sorted(pw.ratio.unique())]]
            print(
                "\npower: share of 400 planted columns passing the ratio part and the whole criterion, by true ratio\n"
            )
            print(table.round(3).to_string())
            levels = pw.ic_all.unique()
            needs = {f"{i} {h}": rho.loc["pooled", f"{i} {h}"] / levels for i in ("spot", "perp") for h in fwd.columns}
            print("\nthe true ratio the whole criterion needs, rho_min / ic_all\n")
            print(pd.DataFrame(needs, index=pd.Index(levels, name="ic_all")).round(2).to_string())
        return
    if args.null is not None:
        t = null_test(pred, close, cut, args.null, args.seeds)
        print(f"shiryaev {args.null}: true and false alarms told apart on the real price and on random paths\n")
        print(t.round(3).to_string())
        return
    if args.residual:
        levels, assets = tuple(args.gate or (0.4, 0.5, 0.6)), TRADABLE if args.tradable else strategy.ASSETS
        t = residual_study(args.residual, levels, period=args.period, assets=assets, lag=args.lag)
        lag = f", its return {args.lag} bar before" if args.lag else ""
        print(f"the detectors on the price and on the residual against {args.residual}{lag}, {args.period} folds")
        print(f"{' '.join(s for s in assets if s != args.residual)}; bp a trade, no fees\n")
        print(t.round(2).to_string())
        print("\n" + "\n".join(residual_verdict(t, levels)))
        return
    if args.confirm:
        p, bars, rows = fit(pred, cut), strategy.ohlc(pred.index), []
        rules = {
            "shiryaev 0.5": alarms(pred, shiryaev, p, 0.5),
            "zigzag 0.2": alarms(pred, zigzag, 0.2),
            "reentry 0.40": strategy.signal(pred, "reentry", 0.40),
        }
        for k in (4, 12, 24):
            for name, sig in rules.items():
                conf = confirmation(sig, close, k)
                rows.append({"k": k, "signal": name, "keep": "all"} | _periods(plain(hold(sig), close)[2], cut))
                for c in (0.0, 0.5, 1.0):
                    # `gate` keeps a long where x <= -c and a short where x >= c: x = -side * conf.
                    kept, on = gate(sig, (-sig * conf).fillna(0.0), c, "alarm", True)
                    held = strategy.walked(kept, bars, on=on)[2]
                    rows.append({"k": k, "signal": name, "keep": f"confirmed >= {c:g}"} | _periods(held, cut))
        t = pd.DataFrame(rows).set_index(["k", "signal", "keep"]).filter(regex="^(?!.*stopped)")
        print("signals kept only where open interest confirms them, a rejected one closes; bp a trade, no fees\n")
        print(t.round(1).to_string())
        return
    if args.oi:
        ic, quads, rule = open_interest(close, cut)
        print("rank IC with the h-bar forward return, mean of ETH/BTC/SOL\n")
        print(ic.round(3).to_string())
        print("\nthe next 48 bars in the direction of the last k-bar move, bp, by whether open interest rose with it\n")
        print(quads.round(1).to_string())
        print("\nfollow the move when open interest rose, fade it when it fell, out after 48 bars; bp a trade\n")
        print(rule.round(1).to_string())
        return
    if args.gate:
        p, rows, bars = fit(pred, cut), [], strategy.ohlc(pred.index)
        for name, found in (
            ("zigzag 0.2", alarms(pred, zigzag, 0.2)),
            ("shiryaev 0.5", alarms(pred, shiryaev, p, 0.5)),
        ):
            rows.append({"detector": name, "gate": "none"} | book(found, close, cut))
            for where in ("alarm", "extreme"):
                for shut in (True, False):
                    for level in args.gate:
                        kept, on = gate(found, pred, level, where, shut)
                        label = f"{where}, {'close' if shut else 'ignore'} {level:g}"
                        held = strategy.walked(kept, bars, on=on)[2]
                        rows.append({"detector": name, "gate": label} | _periods(held, cut))
        t = pd.DataFrame(rows).set_index(["detector", "gate"]).filter(regex="^(?!.*stopped)")
        print("signals kept only past a level; bp a trade, no fees; n is trades over the period\n")
        print(t.round(1).to_string())
        return
    if args.futures:
        columns = {sym: futures_columns(sym, close.xs(sym, level=1)) for sym in strategy.ASSETS}
        print("rank IC with the forward log return, mean of ETH/BTC/SOL, every bar\n")
        print(forward_ic(columns, pred, close).round(3).sort_values(["h", "dev"]).to_string())
        f = at_alarm(alarms(pred, shiryaev, fit(pred, cut), 0.5), pred, close, truth)
        g = futures_at_alarm(f, columns)
        new = [c for c in g.columns if c not in f.columns]
        for label, table in (("the futures columns", g[["symbol", "when", "side", "true", "gross"] + new]), ("all", g)):
            one, (dev, holdout), by = separate(table, cut)
            print(f"\nshiryaev 0.5's alarms, {label}: logistic AUC {dev:.3f}, hold-out {holdout:.3f}\n")
            if label != "all":
                print(one.round(3).to_string() + "\n")
            print(by.round(2).to_string())
        return
    if args.features:
        f = at_alarm(alarms(pred, shiryaev, fit(pred, cut), args.features), pred, close, truth)
        one, (dev, holdout), by = separate(f, cut)
        print(f"shiryaev {args.features}: {len(f)} alarms, {f.true.mean():.0%} true\n")
        print(one.round(3).to_string())
        print(f"\nlogistic on every column, fitted on development: AUC {dev:.3f}, hold-out {holdout:.3f}\n")
        print(by.round(2).to_string())
        table, kept = conservation(f, _score(f, cut), cut)
        print(
            f"\nwhat the precision is worth: kept {kept['dev']:.2f} +/- {kept['dev_se']:.2f} on development,"
            f" {kept['holdout']:.2f} +/- {kept['holdout_se']:.2f} on the hold-out\n"
        )
        print(table.round(2).to_string())
        return
    if args.sl:
        found, bars = alarms(pred, shiryaev, fit(pred, cut), 0.5), strategy.ohlc(pred.index)
        rows, parts = [], {}
        for k in [0.0, *args.sl]:
            for trail in (False, True) if k else (False,):
                label = f"stop {k:g} ATR" + (", trailing" if trail else "") if k else "no stop"
                stop = ("atr", k) if k else None
                held = strategy.walked(found, bars, stop=stop, after="opposite", trail=trail)[2]
                rows.append({"rule": label} | _periods(held, cut))
                parts[label] = split(held, truth, cut)
        print("shiryaev 0.5 on the prediction; after a stop, flat until the next alarm\n")
        print(pd.DataFrame(rows).set_index("rule").round(2).to_string())
        for label, table in parts.items():
            print(f"\n{label}, development, by what opened the trade\n")
            print(table.round(2).to_string())
        return
    rows, splits = [], {}
    for name, x in series.items():
        truth = turn_events(x, strategy.WINDOW)
        runs = [(f"zigzag {h}", alarms(x, zigzag, h)) for h in args.zigzag or []]
        if args.shiryaev:
            p = fit(x, cut)
            if name == "prediction":
                # The page runs `V2_FIT`; if the predictions moved under it, the page and this table differ.
                assert all(np.allclose(p[k], V2_FIT[k], atol=1e-3) for k in V2_FIT), "refit V2_FIT"
            print(name, {k: np.round(v, 4) for k, v in p.items()})
            for flat in (False, True) if args.flat else (False,):
                runs += [
                    (f"shiryaev {th}" + (" flat" if flat else ""), alarms(x, shiryaev, p, th, flat))
                    for th in args.shiryaev
                ]
            runs += [
                (f"shiryaev {th} + zigzag {h}", alarms(x, shiryaev, p, th, False, h))
                for th in args.shiryaev
                for h in args.both or []
            ]
        for label, found in runs:
            rows.append({"signal": name, "detector": label} | match(found, truth, cut) | book(found, close, cut))
            if args.split and name == "prediction":
                splits[label] = split(plain(hold(found), close)[2], truth, cut)
    print("\nagainst each series' own centred turns at 12 bars; found, delay and false alarms on development\n")
    print(pd.DataFrame(rows).set_index(["signal", "detector"]).round(2).to_string())
    for label, table in splits.items():
        print(f"\n{label} on the prediction, development: trades by what opened them\n")
        print(table.round(1).to_string())


if __name__ == "__main__":
    main()
