# Design review, 25 September 2026

Attendees: Dr Imogen Hale, Tomasz Wrona, Priya Nandakumar.

## Deployment criterion

We will deploy a model only if its mean test-split accuracy across all four sites is at
least 0.870 AND its worst single-site test-split false positive rate is below 0.060.
Validation-split numbers do not count towards this criterion.

## Decisions

1. v1 is retired.
2. v2 remains the production candidate until v3 is re-scored at Ashby Fen.
3. "No model ships on validation numbers alone." (P. Nandakumar)

## Open questions

- Is the v3 head worth its 2.4x inference cost on the field hardware?
- Should Ashby Moor be excluded from the false positive criterion during high wind?
