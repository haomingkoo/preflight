# Preflight design

Extends the shared kooexperience design core (`~/.agent-memory/design/DESIGN-core.md`). Where this file is silent, the core applies.

## Audience and job
Analysts and students who need to know whether a CSV is usable before they model it: gaps, column types, distributions, then a baseline score.

## Mood
A pre-flight checklist for data. Terse, ordered, dense, nothing decorative. Light theme, because this is desk work in daylight next to spreadsheets and notebooks.

## Tokens (`assets/preflight.css`)
- Ground `#f4f6f6`, surface `#ffffff`, surface-2 `#f0f3f3`, rule `#dbe1e1`, control border `#8a9494` (3.1:1 on white).
- Ink `#141a1d`, soft `#3a4448`, muted `#56616a` (6.3:1 on white, 5.8:1 on ground).
- One accent, teal `#0f766e` (5.5:1 on white; white on teal 5.5:1): links, focus ring, active tab, primary buttons, the dashed upload edge. Danger `#b42318` only for "Delete my data now" and errors.
- Charts: the `preflight` Plotly template in `app.py`. Teal leads the colourway; sequential scale is teal, correlation uses brown-to-teal diverging fixed at -1 to 1; missing cells are the one dark mark.
- Type: Hanken Grotesk (variable, self-hosted), chosen in #7 for compact, even figures. Sizes 13, 15, 18, 20, 32. Tabular numerals everywhere.
- Radius 6 controls, 10 cards. One shadow, only on open dropdown menus.

## Signature move
The numbered workflow: tabs "1. Guide" to "6. Model" read as a checklist you work through in order, and the guide is the same six steps.

## Rules for this app
- Consent comes before upload, inside one "Upload a file" block. A locked dropzone says why ("Tick the box above to turn on upload.").
- Tables: plain headers ("Missing %", "Unique values"), text left, numbers right, fixed decimals.
- Every message names the problem and the next step ("Pick a target column in Data health first."). No "Please", no error codes.
- Empty panes say what will appear after upload; charts show a sentence, not a blank grid.
- Legal pages (`legal.py`) share the header, footer and palette.

## Dials
ENERGY 1 / RHYTHM 2 / MOTION 1. One motion: panes rise in 240 ms when a tab opens.
