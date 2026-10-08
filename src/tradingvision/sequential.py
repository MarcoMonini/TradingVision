"""Testing by betting: a test for paper trading that can be read after every trade.

A paper-trading account is read after every trade, and the reader stops at the first number that
looks good. A t-test read that way declares an edge that is not there in 43% of runs within 3,000
trades (`false_alarms.html`, `#conferma`). The test here keeps its false-positive rate under
alpha however often it is read and wherever the reader stops.

**The test** (Shafer and Vovk's testing by betting; Waudby-Smith and Ramdas, 2024, for means). A
virtual capital starts at 1 and stakes a fraction `lambda_i` of itself on each trade:

    W_n = prod_{i <= n} (1 + lambda_i * x_i / B)

`x_i` is the trade's net return and `B` the most it can lose, which a stop sets: every factor is
then at least 1 - lambda_i, positive for lambda_i < 1. `lambda_i` is read from the trades before
`i` only (`bets`): a Kelly bet, the mean over the second moment of `x / B`, both shrunk by 20
pseudo-trades towards zero and towards the prior `SCALE` squared, clipped to [0, 1/2]. With no
edge (E[x_i | the trades before] <= 0) each factor has mean at most 1, W is a nonnegative
supermartingale, and Ville's inequality bounds the chance it ever reaches 1/alpha by alpha. An
edge is declared at W >= 1/alpha, the first time it happens (`crossing`), at any trade.

`B` may change from trade to trade as long as it is known at the entry, as an ATR stop is:
`x_i / B_i` is then the trade in units of its own risk, and the null is that this has no positive
mean. The bet only ever sees `y = x / B`, which is why the functions here take `y`.

**E-values combine without a new hold-out.** The average of e-values is an e-value whatever their
dependence (`average`), so the per-asset processes of one rule, or the routes of the plan, can be
pooled into one number. e-BH (Wang and Ramdas, 2022, `ebh`) picks which of K routes have an edge
at a false-discovery rate alpha, again under any dependence. One process over trades that overlap
in time is not a supermartingale: two concurrent trades on ETH and SOL are correlated, and the
product of their factors can have a mean above 1. A rule that trades several assets is tested one
process per asset, where positions never overlap, and the processes are averaged.

**The simulation of the document reproduces** (`--sim`, `simulate`, 4,000 runs, seed 17): trades
with sd 150 bp and t(4) tails, losses cut at B = 300 bp, a location set so the clipped mean is
exactly mu. The fraction of runs declaring an edge at alpha = 5%:

    true edge a trade                         0 bp    7.5 bp   15 bp   30 bp
    t-test, fixed sample, 160 trades          4.8%    16.8%    37.4%    87.0%
    t-test, fixed sample, 1,000 trades        4.6%    52.0%    96.2%   100%
    t-test read after every trade, 3,000     42.9%    97.0%   100%     100%
    betting test, within 1,000 trades         1.8%    20.6%    76.9%   100%
    betting test, within 3,000 trades         2.4%    60.1%    99.9%   100%
    median trades to reach 20               (323)    1,487     587     154

Every cell is the document's table to its rounding: the port is the appendix's
`sim_sequential.py`, the same calls on the same generator in the same order, with the bet written
on `x / B` rather than on `x` (the same lambda, `m B / v`, which `_selfcheck` asserts). The 323 is
the median among the 2.4% of null runs that cross, which the document leaves out. The betting
test pays for the freedom to look with some power (77% against 96% at 1,000 trades for 15 bp) and
keeps its false positives under 5% whatever the reader does.

**On the two rows of HANDOFF §17 it is nowhere near 20** (`--trades`). The two rules positive on
all four folds, on v2's predictions for ETH, BTC and SOL over the sixteen months, with the 6 ATR
stop and flat after it until the opposite signal, at OKX's taker fee; each asset its own process,
the three averaged at the end:

    rule                   trades  gross bp  net bp (se)    sd bp  B bp  W     20 / W  max W of an asset
    zigzag 0.2, L 0.5        160    +15.3    -4.7 (11.0)    139    419   0.96   20.9    1.36
    shiryaev 0.5, L 0.6      121    +20.0    -0.0 (15.8)    173    360   0.86   23.3    1.10

The trades reproduce §17 (zigzag +12.5 on 93 development trades and +19.2 on the hold-out's 67;
Shiryaev +15.8 on 81 and +28.4 on 40); the gate closes every zigzag trade before its stop, and
six of Shiryaev's are stopped. `B` is the stop distance at the entry plus the round trip; two of
those six lost a hair more, a bar opening past the stop. Net of the fee both means are
at or under zero in units of `B` (-0.014 and -0.028), so at their measured mean and sd the test
never gets there. Even at the gross, no fee at all, a single stream needs a median of 380 trades
(zigzag) and 906 (Shiryaev), and the average of three assets 896 and 1,913 in all: from 3 to 21
years at the 120 and 90 trades a year these rows make. A strategy worth paper trading has to
arrive with a mean that clears the fee by several of its errors, and these rows were the best of a
grid on development, so their means are an upper bound.

    uv run python -m tradingvision.sequential --sim
    uv run python -m tradingvision.sequential --trades
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tradingvision.oracle import FEE

ALPHA = 0.05
PRIOR = 20  # pseudo-trades the bet is shrunk by, the appendix's: chosen, not measured
SCALE = 0.5  # the prior sd of x / B, the appendix's 150 bp over B = 300 bp
SEED = 17  # the appendix's


def bets(y: np.ndarray, scale: float = SCALE, prior: int = PRIOR) -> np.ndarray:
    """`lambda_i` for each trade along the last axis, from the trades before it only.

    The plug-in Kelly bet on `1 + lambda * y`: the running mean of `y` over its running second
    moment, both over the trades before `i` and shrunk by `prior` pseudo-trades towards 0 and
    `scale**2`, clipped to [0, 1/2]. The first bet is 0: nothing has been seen.
    """
    y = np.asarray(y, dtype=float)
    n = np.arange(1, y.shape[-1] + 1)
    zero = np.zeros(y.shape[:-1] + (1,))
    s1 = np.concatenate([zero, np.cumsum(y, axis=-1)[..., :-1]], axis=-1)
    s2 = np.concatenate([zero, np.cumsum(y**2, axis=-1)[..., :-1]], axis=-1)
    return np.clip((s1 / (n + prior)) / ((s2 + prior * scale**2) / (n + prior)), 0, 0.5)


def wealth(y: np.ndarray, scale: float = SCALE, prior: int = PRIOR) -> np.ndarray:
    """log W after each trade along the last axis, `y = x / B >= -1` the trades in units of their bound."""
    y = np.asarray(y, dtype=float)
    return np.cumsum(np.log1p(bets(y, scale, prior) * y), axis=-1)


def crossing(logw: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """The stopping time: the count of trades at which W first reaches 1/alpha, NaN if it never does."""
    hit = np.asarray(logw) >= np.log(1 / alpha)
    return np.where(hit.any(axis=-1), hit.argmax(axis=-1) + 1.0, np.nan)


def average(e) -> float:
    """The mean of e-values: an e-value under any dependence between them."""
    return float(np.mean(e))


def ebh(e, alpha: float = ALPHA) -> np.ndarray:
    """e-BH: which of K e-values to reject at a false-discovery rate `alpha`, under any dependence.

    The k largest are rejected for the largest k with e_(k) >= K / (alpha k).
    """
    e = np.asarray(e, dtype=float)
    order = np.argsort(-e)
    k = np.arange(1, len(e) + 1)
    ok = np.flatnonzero(e[order] >= len(e) / (alpha * k))
    out = np.zeros(len(e), dtype=bool)
    if len(ok):
        out[order[: ok.max() + 1]] = True
    return out


def draws(rng: np.random.Generator, mu: float, sd: float, bound: float, shape) -> np.ndarray:
    """Trades `max(c + sd e, -bound)`, e a t(4) of unit variance, `c` such that the clipped mean is `mu`.

    `c` comes from a bisection on two million more draws of the same generator, as the appendix
    does: the order of the calls is part of what fixes the numbers.
    """
    e = rng.standard_t(4, shape) / np.sqrt(2)
    big = rng.standard_t(4, 2_000_000) / np.sqrt(2)
    lo, hi = mu - 100 * sd / 150, mu + 100 * sd / 150  # the appendix's +-100 bp at its 150 bp sd
    for _ in range(60):
        c = (lo + hi) / 2
        lo, hi = (c, hi) if np.maximum(c + sd * big, -bound).mean() < mu else (lo, c)
    return np.maximum(c + sd * e, -bound)


def tstat(x: np.ndarray) -> np.ndarray:
    """The one-sample t of the mean after each trade along the last axis."""
    n = np.arange(1, x.shape[-1] + 1)
    m = np.cumsum(x, axis=-1) / n
    v = (np.cumsum(x**2, axis=-1) - n * m**2) / np.maximum(n - 1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return m / np.sqrt(v / n)


def simulate(
    mus=(0.0, 7.5, 15.0, 30.0), sigma=150.0, bound=300.0, nmax=3000, reps=4000, alpha=ALPHA, seed=SEED
) -> pd.DataFrame:
    """The table of `#conferma`: how often each test declares an edge, by true mean a trade (bp).

    The fixed-sample t-test at 160 and 1,000 trades, the same t-test read after every trade from
    the 10th to `nmax` (stopping at the first t > 1.645), and the betting test within 1,000 and
    `nmax` trades, with its median trades to reach 1/alpha among the runs that do.
    """
    rng, z = np.random.default_rng(seed), 1.645
    rows = []
    for mu in mus:
        x = draws(rng, mu, sigma, bound, (reps, nmax))
        logw, t = wealth(x / bound, sigma / bound), tstat(x)
        hit = logw >= np.log(1 / alpha)
        rows.append(
            {
                "mu_bp": mu,
                "mean_x": x.mean(),
                "t-test n=160": (t[:, 159] > z).mean(),
                "t-test n=1000": (t[:, 999] > z).mean(),
                f"t peeking, any n<={nmax}": (t[:, 9:] > z).any(axis=1).mean(),
                "betting by 1000": hit[:, :1000].any(axis=1).mean(),
                f"betting by {nmax}": hit.any(axis=1).mean(),
                "median trades": np.nanmedian(crossing(logw, alpha)),
            }
        )
    return pd.DataFrame(rows).set_index("mu_bp")


def needed(mean: float, sd: float, streams: int = 1, nmax: int = 20_000, reps: int = 400, seed: int = SEED) -> float:
    """Median trades, over all `streams`, for the average of `streams` processes to reach 1/alpha.

    Each stream's trades are `draws` in units of their bound (`y`, clipped at -1) with this mean and
    sd; the streams take turns, so the count is the total. NaN when under half the runs get there
    within `nmax` trades per stream. In chunks of 100 runs, to stay in memory.
    """
    rng, out = np.random.default_rng(seed), []
    for _ in range(reps // 100):
        w = np.exp(wealth(draws(rng, mean, sd, 1.0, (100, streams, nmax)))).mean(axis=1)
        out.append(crossing(np.log(w)) * streams)
    t = np.concatenate(out)
    return float(np.median(t)) if np.isfinite(t).mean() > 0.5 else np.nan


def detector_trades() -> dict[str, pd.DataFrame]:
    """The two rows of HANDOFF §17 positive on all four folds, as the chart page runs them.

    v2's predictions for ETH, BTC and SOL over the sixteen months; the zigzag at 0.2 and Shiryaev
    at 0.5 on `detect.V2_FIT`, gated at `chart.LEVEL` (0.5 and 0.6) read at the alarm, a rejected
    alarm closing; a 6 ATR stop (ATR at v2's 12 bars) and flat after it until the opposite signal.
    Each trade carries `x`, its net log return at OKX's fee, and `B`, its stop distance at the entry
    plus the round trip: the most it can lose unless a bar gaps through the stop.
    """
    from tradingvision import detect, stops, strategy

    pred, _, _ = strategy.load()
    bars = strategy.ohlc(pred.index)
    stop = ("atr", 6.0)
    sl = stops.width(stop, stops.atr_pct(bars, strategy.WINDOW))
    out = {}
    for name, found, level in (
        ("zigzag 0.2, L 0.5", detect.alarms(pred, detect.zigzag, 0.2), 0.5),
        ("shiryaev 0.5, L 0.6", detect.alarms(pred, detect.shiryaev, detect.V2_FIT, 0.5), 0.6),
    ):
        kept, on = detect.gate(found, pred, level, "alarm", True)
        held = strategy.walked(kept, bars, stop=stop, after="opposite", on=on)[2].reset_index()
        at = pd.MultiIndex.from_arrays([held.entry, held.symbol])
        held["x"] = held.gross - 2 * FEE
        held["B"] = sl.reindex(at).to_numpy() + 2 * FEE
        out[name] = held.sort_values("exit_time", kind="stable").reset_index(drop=True)
    return out


def test_trades(held: pd.DataFrame, alpha: float = ALPHA) -> dict:
    """One betting process per asset in the order its trades close, averaged across assets at the end."""
    y = held.x / held.B
    logw = {s: wealth(y[held.symbol == s].to_numpy()) for s in held.symbol.unique()}
    final = {s: float(np.exp(w[-1])) for s, w in logw.items()}
    w = average(list(final.values()))
    return {
        "trades": len(held),
        "net_bp": held.x.mean() * 1e4,
        "se_bp": held.x.std() / np.sqrt(len(held)) * 1e4,
        "sd_bp": held.x.std() * 1e4,
        "B_bp": held.B.median() * 1e4,
        "gross_bp": (held.x.mean() + 2 * FEE) * 1e4,
        "y_mean": y.mean(),
        "y_gross": y.mean() + (2 * FEE / held.B).mean(),
        "y_sd": y.std(),
        "below_-1": int((y < -1).sum()),
        "W": w,
        "W_max_asset": max(float(np.exp(v.max())) for v in logw.values()),
        "1/alpha / W": 1 / alpha / w,
    } | {f"W {s}": v for s, v in final.items()}


def _selfcheck() -> None:
    """A bounded zero-mean null stays under alpha, a planted edge is found, and the bet never looks ahead.

    200 runs of 2,000 trades each for the null and 200 of 1,000 for the edge, seeded: under a
    second of CI. Ville bounds the null's crossing rate by alpha; the appendix's 4,000 runs put it
    at 2.4% within 3,000 trades, so 200 runs leave room for the noise of the estimate.
    """
    rng = np.random.default_rng(0)
    null = rng.uniform(-1, 1, (200, 2000))
    assert np.isfinite(crossing(wealth(null))).mean() <= ALPHA
    # A bounded null that is not symmetric: a fair two-point bet, -1 a third of the time and +1/2 otherwise.
    skew = np.where(rng.uniform(size=(200, 2000)) < 1 / 3, -1.0, 0.5)
    assert np.isfinite(crossing(wealth(skew))).mean() <= ALPHA
    # A planted edge of 0.2 in units of the bound, sd about 0.58: every run gets there within 1,000
    # (the slowest of these 200 in 234 trades; at 0.1 one run in 200 does not).
    edge = rng.uniform(-1, 1, (200, 1000)) + 0.2
    assert np.isfinite(crossing(wealth(np.maximum(edge, -1)))).all()
    # Predictable: changing trade i changes no bet up to and including its own, and does change the next.
    y = rng.uniform(-1, 1, 50) + 0.05  # a bet inside (0, 1/2), so a change can show
    b, z = bets(y), y.copy()
    z[20] = -1.0
    assert np.array_equal(bets(z)[:21], b[:21]) and bets(z)[21] != b[21]
    assert bets(y)[0] == 0 and (bets(y) >= 0).all() and (bets(y) <= 0.5).all()
    # The bet on x / B is the appendix's on x: lambda = m B / v, shrunk by 20 trades at sigma^2.
    x, B, sigma = 300 * y, 300.0, 150.0
    n = np.arange(1, len(x) + 1)
    m = np.concatenate([[0], np.cumsum(x)[:-1]]) / (n + 20)
    v = (np.concatenate([[0], np.cumsum(x**2)[:-1]]) + 20 * sigma**2) / (n + 20)
    assert np.allclose(bets(x / B, sigma / B), np.clip(m * B / v, 0, 0.5))
    # Combining: the mean, and e-BH rejects the k largest for the largest k with e_(k) >= K / (alpha k).
    assert average([10.0, 30.0]) == 20.0
    # K = 4: the bars are 80, 40, 26.7, 20; 30 would clear the third, 25 does not.
    assert list(ebh([100.0, 30.0, 1.0, 50.0], 0.05)) == [True, True, False, True]
    assert list(ebh([100.0, 25.0, 1.0, 50.0], 0.05)) == [True, False, False, True]
    assert not ebh([10.0, 10.0]).any() and ebh([40.0, 40.0]).all()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim", action="store_true", help="the table of false_alarms.html #conferma, 4,000 runs")
    ap.add_argument("--trades", action="store_true", help="the test on the two rows of HANDOFF §17")
    args = ap.parse_args()
    _selfcheck()
    pd.set_option("display.width", 200)
    if args.sim:
        print("share of 4,000 runs declaring an edge at alpha 5%; sd 150 bp, t(4), losses cut at 300 bp\n")
        print(simulate().T.round(3).to_string())
    if args.trades:
        rows = {}
        for name, held in detector_trades().items():
            r = test_trades(held)
            # At the net mean, and at the gross: what the test would need with no fee at all.
            r["needed"] = needed(r["y_mean"], r["y_sd"])
            r["needed at gross"] = needed(r["y_gross"], r["y_sd"])
            r["needed at gross, 3 averaged"] = needed(r["y_gross"], r["y_sd"], streams=3)
            rows[name] = r
        print("the betting test on v2's two gated detectors, 16 months, OKX fee, B = stop + round trip\n")
        print(pd.DataFrame(rows).T.round(3).to_string())


if __name__ == "__main__":
    main()
