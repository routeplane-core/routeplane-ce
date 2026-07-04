# Brand assets

- `social-preview.png` — the repo's 1280×640 social-preview card (GitHub Settings →
  Social preview; also correct for link unfurls). Every number on it links back to
  a published harness: the perf numbers to `benchmarks/perf/RESULTS.md`, the RTK
  number to `benchmarks/rtk-eval/RESULTS.md`.
- `social-preview.html` — the card's source. Regenerate after a numbers refresh:
  `chromium --headless --screenshot=social-preview.png --window-size=1280,640 --hide-scrollbars file://$PWD/social-preview.html`
- `mark.svg` — the hub-and-spoke mark (canonical source lives in the marketing site).
