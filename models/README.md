# Deployed checkpoints

The chart page reads `swing.pt` and `swing-v2.pt` from here when the `data/` store does not carry
them, which is the case in the Docker image: `data/` is gitignored and never copied, so a checkpoint
only reaches Render by being committed to this directory.

```bash
uv run python -m tradingvision.swing --save                  # writes data/swing.pt
uv run python -m tradingvision.swing --timeframe 15m --window 12 --smoothing 0.5 --inputs reduced \
    --steps 48 --stage label --test-start 2025-06 --save data/swing-v2.pt
cp data/swing.pt data/swing-v2.pt models/
```

Commit both and redeploy. Each model is drawn only on the timeframe it was fitted on and needs
enough history behind the window to score a bar (`swing.MIN_BARS` for the step-7 model); the sidebar
says which of these is missing rather than drawing nothing.

`swing-v2.pt` is the **Swing Leg Position v2** checkbox: 15m, pivots and features at 12, time
weight 0.5, the 15 inputs of `swing.reduced(12)` over 48 bars, label stage. Ticked, it pins the
label's two sliders to those values. It carries each training pair's scaler, so the page feeds a
known pair the inputs the walk-forward measured.

`gru.pt`, the GRU on the swing label at 0.7 / 24, left on 2026-10-03 with the first pipeline it was
built on: it is in `OLD/models/`, and `OLD/README.md` says why.

The files are small — a 48-unit encoder — so they belong in git rather than in LFS.
`TRADINGVISION_MODELS=/some/disk` moves the lookup to a mounted disk if a checkpoint ever outgrows
that.
