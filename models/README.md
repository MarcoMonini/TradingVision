# Deployed checkpoints

The chart page reads `gru.pt` and `swing.pt` from here when the `data/` store does not carry them,
which is the case in the Docker image: `data/` is gitignored and never copied, so a checkpoint only
reaches Render by being committed to this directory.

```bash
uv run python -m tradingvision.gru --features all --save     # writes data/gru.pt
uv run python -m tradingvision.swing --save                  # writes data/swing.pt
cp data/gru.pt data/swing.pt models/
```

Commit both and redeploy. Two things decide whether the page can actually draw them, and they are
the page's own rules and not a deployment detail:

- the GRU is drawn only on the timeframe of the branch it was trained on and against the label it
  was fitted to, and a checkpoint saved with `--label cross` is never drawn on one pair at all —
  a cross-sectional rank says nothing about a symbol read alone;
- the swing model is drawn only on its own timeframe and against the retrospective label, and it
  needs enough history behind the window to score a bar (`swing.MIN_BARS`).

The sidebar says which of these is missing rather than drawing nothing.

Both files are small — a 32-unit GRU and a 48-unit encoder — so they belong in git rather than in
LFS. `TRADINGVISION_MODELS=/some/disk` moves the lookup to a mounted disk if a checkpoint ever
outgrows that.
