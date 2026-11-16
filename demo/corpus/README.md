# Project Kestrel - field trial notes

Project Kestrel is a (fictional) acoustic monitoring classifier that identifies bird calls in
recordings from unattended field microphones. This corpus is the trial's working folder.

- Trial period: 1 to 30 September 2026, four sites: Ashby Fen, Ashby Moor, Carrow Wood and
  Dunmere Point.
- Three model versions were evaluated: v1 (baseline), v2 (noise-robust front end) and v3
  (v2 plus a larger classifier head).
- Every model was scored on a validation split and a held-out test split at each site;
  results are in `runs.csv` (accuracy and false positive rate are fractions, not percentages).
- Trial lead: Dr Imogen Hale. Field engineer: Tomasz Wrona.

See `notes/` for the field log and the design review, and `web-cache/` for pages saved from
the web during the trial (unverified).
